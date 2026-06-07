//! Extension actions — runtime operations extensions can perform.
//!
//! [`ExtensionActions`] holds the resources needed to *do* things: send
//! messages to the agent's processing queues and persist session entries.
//! It is the action counterpart to
//! [`ExtensionRunner`](super::ExtensionRunner)'s event-bus role.
//!
//! # Architecture
//!
//! ```text
//! ExtensionApi             ← facade extensions see
//!     ├── Arc<ExtensionRunner>    ← handler registration + dispatch
//!     └── Arc<ExtensionActions>   ← this struct (Weak<Agent> + session manager)
//! ```
//!
//! # Why `Weak<Agent>`
//!
//! Extensions capture `Arc<ExtensionApi>` in their handlers, which are stored
//! in `ExtensionRunner`. A strong `Arc<Agent>` would create a reference cycle:
//!
//! ```text
//! Actions → Arc<Agent> → hooks → Arc<Runner> → hooks capture → Arc<ExtensionApi> → Arc<Actions>
//! ```
//!
//! `Weak<Agent>` breaks this cycle. When `AgentSession` drops, the agent
//! deallocates and [`send_user_message`](ExtensionActions::send_user_message)
//! / [`send_custom_message`](ExtensionActions::send_custom_message) silently
//! do nothing.

use crate::extension::events::MessageMode;
use crate::session_manager::{SessionError, SessionManager};
use ameli_agent_core::types::{AgentMessage, CustomMessage};
use ameli_ai::types::{UserContent, UserMessage};
use std::sync::{Arc, Weak};

// ---------------------------------------------------------------------------
// ExtensionActions
// ---------------------------------------------------------------------------

/// Runtime action performer for extensions.
///
/// Holds a [`Weak<Agent>] for message injection and an
/// [`Arc<dyn SessionManager>`] for session persistence. The weak reference is
/// set after the agent is constructed (via [`set_agent`](Self::set_agent)),
/// and safely degrades when the session is torn down.
pub struct ExtensionActions {
    /// Weak reference to the agent — avoids reference cycles.
    agent: parking_lot::RwLock<Option<Weak<ameli_agent_core::agent::Agent>>>,
    /// Session storage backend for entry persistence.
    session_manager: Arc<dyn SessionManager>,
}

impl ExtensionActions {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create empty actions with no agent reference.
    ///
    /// Call [`set_agent`](Self::set_agent) after constructing the agent to
    /// wire up message injection.
    pub fn new(session_manager: Arc<dyn SessionManager>) -> Self {
        Self {
            agent: parking_lot::RwLock::new(None),
            session_manager,
        }
    }

    // -----------------------------------------------------------------------
    // Agent wiring
    // -----------------------------------------------------------------------

    /// Set (or replace) the agent reference.
    ///
    /// Stores a [`Weak`] derived from the given `Arc<Agent>`. Call this after
    /// constructing the agent, before any extensions might try to send
    /// messages.
    pub fn set_agent(&self, agent: &Arc<ameli_agent_core::agent::Agent>) {
        let mut guard = self.agent.write();
        *guard = Some(Arc::downgrade(agent));
    }

    // -----------------------------------------------------------------------
    // Message injection
    // -----------------------------------------------------------------------

    /// Send a user message to the agent's processing queue.
    ///
    /// Builds a [`UserMessage`] from the given content with the current
    /// timestamp, wraps it in [`AgentMessage::User`], and injects it into
    /// the queue determined by `mode`:
    ///
    /// - [`MessageMode::Steer`]: injected after the current assistant turn
    ///   finishes executing its tool calls, before the next LLM call.
    /// - [`MessageMode::FollowUp`]: injected after the agent would otherwise
    ///   stop, potentially triggering a new round of processing.
    ///
    /// Silently does nothing if the agent has been dropped (session torn down).
    pub async fn send_user_message(&self, content: UserContent, mode: MessageMode) {
        let message = UserMessage {
            content,
            timestamp: now_ms(),
        };
        self.enqueue_message(AgentMessage::User(message), mode)
            .await;
    }

    /// Send a custom message to the agent's processing queue.
    ///
    /// Builds a [`CustomMessage`] from the given parameters with the current
    /// timestamp, wraps it in [`AgentMessage::Custom`], and injects it into
    /// the queue determined by `mode`.
    ///
    /// Custom messages are converted to LLM-compatible messages by registered
    /// custom message formatters during the `convert_to_llm` pipeline.
    ///
    /// Silently does nothing if the agent has been dropped (session torn down).
    pub async fn send_custom_message(
        &self,
        custom_type: &str,
        data: Option<serde_json::Value>,
        display: bool,
        details: Option<serde_json::Value>,
        mode: MessageMode,
    ) {
        let message = CustomMessage {
            custom_type: custom_type.to_string(),
            data,
            display,
            details,
            timestamp: now_ms(),
        };
        self.enqueue_message(AgentMessage::Custom(message), mode)
            .await;
    }

    // -----------------------------------------------------------------------
    // Session entry persistence
    // -----------------------------------------------------------------------

    /// Append a custom entry to the session tree.
    ///
    /// Delegates to [`SessionManager::append_custom_entry`]. On error, the
    /// error is propagated to the caller (the `ExtensionApi` wrapper reports
    /// it to registered error listeners before propagating).
    ///
    /// Returns the ID of the created entry.
    pub async fn append_custom_entry(
        &self,
        custom_type: &str,
        data: Option<serde_json::Value>,
    ) -> Result<String, SessionError> {
        self.session_manager
            .append_custom_entry(custom_type, data)
            .await
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Enqueue a message on the appropriate agent queue.
    ///
    /// Upgrades the `Weak<Agent>` and calls `steer()` or `follow_up()`
    /// depending on mode. Silently does nothing if the agent has been dropped.
    async fn enqueue_message(&self, msg: AgentMessage, mode: MessageMode) {
        let agent = {
            let guard = self.agent.read();
            match guard.as_ref() {
                Some(weak) => match weak.upgrade() {
                    Some(arc) => arc,
                    None => return, // Agent dropped — silently do nothing
                },
                None => return, // No agent set yet — silently do nothing
            }
        };

        match mode {
            MessageMode::Steer => agent.steer(msg).await,
            MessageMode::FollowUp => agent.follow_up(msg).await,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_manager::InMemorySessionManager;
    use ameli_agent_core::types::AgentState;
    use ameli_agent_core::AgentOptions;
    use ameli_ai::types::{Cost, InputType, Model};
    use std::collections::HashSet;

    fn test_model() -> Model {
        Model {
            id: "test-model".into(),
            name: "Test Model".into(),
            api: "test-api".into(),
            provider: "test-provider".into(),
            base_url: "http://localhost".into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![InputType::Text],
            cost: Cost::default(),
            context_window: 128_000,
            max_tokens: 16_384,
            compat: None,
        }
    }

    fn test_agent() -> Arc<ameli_agent_core::agent::Agent> {
        ameli_agent_core::agent::Agent::new_arc(AgentOptions {
            initial_state: Some(AgentState {
                system_prompt: String::new(),
                model: test_model(),
                thinking_level: ameli_agent_core::types::ThinkingLevel::Off,
                tools: vec![],
                messages: vec![],
                is_streaming: false,
                streaming_message: None,
                pending_tool_calls: HashSet::new(),
                error_message: None,
            }),
            ..Default::default()
        })
    }

    fn test_session_manager() -> Arc<InMemorySessionManager> {
        Arc::new(InMemorySessionManager::new())
    }

    // -- send_user_message tests --

    #[tokio::test]
    async fn send_user_message_steer() {
        let agent = test_agent();
        let actions = ExtensionActions::new(test_session_manager());
        actions.set_agent(&agent);

        actions
            .send_user_message(
                ameli_ai::types::UserContent::Text("steer this".into()),
                MessageMode::Steer,
            )
            .await;

        assert!(agent.has_queued_messages().await);
    }

    #[tokio::test]
    async fn send_user_message_follow_up() {
        let agent = test_agent();
        let actions = ExtensionActions::new(test_session_manager());
        actions.set_agent(&agent);

        actions
            .send_user_message(
                ameli_ai::types::UserContent::Text("follow up".into()),
                MessageMode::FollowUp,
            )
            .await;

        assert!(agent.has_queued_messages().await);
    }

    #[tokio::test]
    async fn send_user_message_silent_noop_when_agent_dropped() {
        let actions = ExtensionActions::new(test_session_manager());

        // No agent set — should silently do nothing
        actions
            .send_user_message(
                ameli_ai::types::UserContent::Text("no agent".into()),
                MessageMode::Steer,
            )
            .await;
        // No panic = success
    }

    #[tokio::test]
    async fn send_user_message_silent_noop_when_weak_dropped() {
        let agent = test_agent();
        let actions = ExtensionActions::new(test_session_manager());
        actions.set_agent(&agent);

        // Drop the agent
        drop(agent);

        actions
            .send_user_message(
                ameli_ai::types::UserContent::Text("dropped agent".into()),
                MessageMode::Steer,
            )
            .await;
        // No panic = success
    }

    // -- send_custom_message tests --

    #[tokio::test]
    async fn send_custom_message_steer() {
        let agent = test_agent();
        let actions = ExtensionActions::new(test_session_manager());
        actions.set_agent(&agent);

        actions
            .send_custom_message(
                "instruction",
                Some(serde_json::json!({"message": "be helpful"})),
                true,
                None,
                MessageMode::Steer,
            )
            .await;

        assert!(agent.has_queued_messages().await);
    }

    #[tokio::test]
    async fn send_custom_message_follow_up() {
        let agent = test_agent();
        let actions = ExtensionActions::new(test_session_manager());
        actions.set_agent(&agent);

        actions
            .send_custom_message(
                "context",
                Some(serde_json::json!({"key": "value"})),
                false,
                None,
                MessageMode::FollowUp,
            )
            .await;

        assert!(agent.has_queued_messages().await);
    }

    #[tokio::test]
    async fn send_custom_message_silent_noop_when_no_agent() {
        let actions = ExtensionActions::new(test_session_manager());

        actions
            .send_custom_message("type", None, true, None, MessageMode::Steer)
            .await;
        // No panic = success
    }

    // -- append_custom_entry tests --

    #[tokio::test]
    async fn append_custom_entry_delegates_to_session_manager() {
        let sm = test_session_manager();
        let actions = ExtensionActions::new(sm.clone());

        let entry_id = actions
            .append_custom_entry("my_type", Some(serde_json::json!({"key": "value"})))
            .await
            .unwrap();

        assert!(!entry_id.is_empty());

        let entries = sm.entries().await.unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0] {
            crate::session_manager::types::SessionEntry::Custom(ce) => {
                assert_eq!(ce.custom_type, "my_type");
                assert_eq!(ce.data, Some(serde_json::json!({"key": "value"})));
            }
            other => panic!("Expected Custom entry, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn append_custom_entry_with_none_data() {
        let sm = test_session_manager();
        let actions = ExtensionActions::new(sm.clone());

        let entry_id = actions.append_custom_entry("simple", None).await.unwrap();
        assert!(!entry_id.is_empty());

        let entries = sm.entries().await.unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0] {
            crate::session_manager::types::SessionEntry::Custom(ce) => {
                assert_eq!(ce.custom_type, "simple");
                assert!(ce.data.is_none());
            }
            other => panic!("Expected Custom entry, got {other:?}"),
        }
    }

    // -- set_agent tests --

    #[tokio::test]
    async fn set_agent_can_be_called_multiple_times() {
        let agent1 = test_agent();
        let agent2 = test_agent();
        let actions = ExtensionActions::new(test_session_manager());

        actions.set_agent(&agent1);
        actions.set_agent(&agent2);

        // Send to agent2 (last set_agent wins)
        actions
            .send_user_message(
                ameli_ai::types::UserContent::Text("test".into()),
                MessageMode::Steer,
            )
            .await;

        assert!(agent2.has_queued_messages().await);
        // agent1 should NOT have queued messages
        assert!(!agent1.has_queued_messages().await);
    }
}
