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
/// Holds a [`Weak<Agent>`] for message injection and an
/// [`Arc<dyn SessionManager>`] for session persistence. The weak reference
/// safely degrades when the session is torn down.
///
pub struct ExtensionActions {
    /// Weak reference to the agent — avoids reference cycles.
    agent: Weak<ameli_agent_core::agent::Agent>,
    /// Session storage backend for entry persistence.
    session_manager: Arc<dyn SessionManager>,
}

impl ExtensionActions {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create actions with a live agent reference.
    ///
    /// The agent reference is stored as [`Weak`] so no reference cycle is
    /// created. When the agent is dropped, message injection methods silently
    /// do nothing.
    pub fn new(
        session_manager: Arc<dyn SessionManager>,
        agent: &Arc<ameli_agent_core::agent::Agent>,
    ) -> Self {
        Self {
            agent: Arc::downgrade(agent),
            session_manager,
        }
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
        let agent = match self.agent.upgrade() {
            Some(arc) => arc,
            None => return, // Agent dropped — silently do nothing
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
        let actions = ExtensionActions::new(test_session_manager(), &agent);

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
        let actions = ExtensionActions::new(test_session_manager(), &agent);

        actions
            .send_user_message(
                ameli_ai::types::UserContent::Text("follow up".into()),
                MessageMode::FollowUp,
            )
            .await;

        assert!(agent.has_queued_messages().await);
    }

    #[tokio::test]
    async fn send_user_message_silent_noop_when_weak_dropped() {
        let agent = test_agent();
        let actions = ExtensionActions::new(test_session_manager(), &agent);

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
        let actions = ExtensionActions::new(test_session_manager(), &agent);

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
        let actions = ExtensionActions::new(test_session_manager(), &agent);

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
    async fn append_custom_entry_with_none_data() {
        let sm = test_session_manager();
        let agent = test_agent();
        let actions = ExtensionActions::new(sm.clone(), &agent);

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

    #[tokio::test]
    async fn append_custom_entry_delegates_to_session_manager() {
        let sm = test_session_manager();
        let agent = test_agent();
        let actions = ExtensionActions::new(sm.clone(), &agent);

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
}
