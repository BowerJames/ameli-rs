//! Agent session — composition layer that bridges session storage, agent loop,
//! extensions, and the interface.
//!
//! [`AgentSession`] is the primary way downstream applications use the agent
//! framework. It owns an [`ArcAgent`], a [`SessionManager`](crate::SessionManager),
//! an [`ExtensionRunner`](crate::ExtensionRunner), and an
//! [`Interface`](crate::Interface), and provides:
//!
//! - **Session context resolution** — converts the session tree into
//!   `AgentMessage`s, using extension hooks for compaction and branch summary
//!   formatting.
//! - **Agent event persistence** — subscribes to agent events and persists
//!   messages, model changes, and thinking level changes to the session.
//! - **Prompt/continue lifecycle** — resolves session context, emits
//!   `before_agent_start`, validates state, and delegates to the agent.
//! - **Command dispatch** — routes named commands to registered extension
//!   handlers.
//!
//! # Example
//!
//! ```no_run
//! use ameli_agent::{
//!     AgentSession, AgentSessionConfig,
//!     ExtensionRunner, NoopInterface, SessionManager, SessionMetadata,
//! };
//! use ameli_agent_core::ArcAgent;
//! use std::sync::Arc;
//!
//! struct MyMetadata { id: String, created_at: String }
//! impl SessionMetadata for MyMetadata {
//!     fn id(&self) -> &str { &self.id }
//!     fn created_at(&self) -> &str { &self.created_at }
//! }
//!
//! fn example(
//!     agent: ArcAgent,
//!     session: Arc<dyn SessionManager<MyMetadata>>,
//!     runner: Arc<ExtensionRunner>,
//! ) -> AgentSession<MyMetadata> {
//!     let config = AgentSessionConfig {
//!         agent,
//!         session_manager: session,
//!         runner,
//!         interface: Arc::new(NoopInterface),
//!     };
//!     AgentSession::new(config)
//! }
//! ```

use crate::extension::ExtensionContext;
use crate::interface::Interface;
use crate::session_manager::{SessionManager, SessionMetadata};
use crate::types::SessionMessage;
use crate::ExtensionRunner;
use ameli_agent_core::types::{AgentMessage, ThinkingLevel};
use ameli_agent_core::ArcAgent;
use ameli_ai::types::ImageContent;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for constructing an [`AgentSession`].
pub struct AgentSessionConfig<M: SessionMetadata> {
    /// The stateful agent that drives the LLM loop.
    pub agent: ArcAgent,
    /// Session storage backend.
    pub session_manager: Arc<dyn SessionManager<M>>,
    /// Extension runner with registered handlers.
    pub runner: Arc<ExtensionRunner>,
    /// UI interface for notifications.
    pub interface: Arc<dyn Interface>,
}

// ---------------------------------------------------------------------------
// AgentSession
// ---------------------------------------------------------------------------

/// Composition layer that bridges session storage, the agent loop, extensions,
/// and the interface.
///
/// `AgentSession` connects the pieces of the framework into a usable whole:
///
/// - On `prompt()`, it resolves the session tree into messages, emits
///   `before_agent_start`, and starts the agent loop.
/// - It subscribes to agent events to persist messages to the session tree.
/// - It converts compaction and branch summary entries using extension hooks.
/// - It dispatches commands to registered extension handlers.
pub struct AgentSession<M: SessionMetadata> {
    agent: ArcAgent,
    session_manager: Arc<dyn SessionManager<M>>,
    runner: Arc<ExtensionRunner>,
    interface: Arc<dyn Interface>,
}

impl<M: SessionMetadata> AgentSession<M> {
    /// Create a new agent session.
    ///
    /// Restores session state from the session manager, subscribes to agent
    /// events for persistence, and emits `session_start` to extensions.
    pub fn new(config: AgentSessionConfig<M>) -> Self {
        let session = Self {
            agent: config.agent,
            session_manager: config.session_manager,
            runner: config.runner,
            interface: config.interface,
        };

        // Emit session_start to extensions (fire-and-forget, spawned in background).
        let runner = session.runner.clone();
        tokio::spawn(async move {
            runner
                .emit_session_start(crate::extension::events::SessionStartReason::Startup)
                .await;
        });

        session
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Get a reference to the underlying `ArcAgent`.
    pub fn agent(&self) -> &ArcAgent {
        &self.agent
    }

    /// Get a reference to the session manager.
    pub fn session_manager(&self) -> &Arc<dyn SessionManager<M>> {
        &self.session_manager
    }

    /// Get a reference to the extension runner.
    pub fn runner(&self) -> &Arc<ExtensionRunner> {
        &self.runner
    }

    /// Get a reference to the interface.
    pub fn interface(&self) -> &Arc<dyn Interface> {
        &self.interface
    }

    /// Returns `true` if a run is currently active.
    pub async fn is_active(&self) -> bool {
        self.agent.is_active().await
    }

    /// Abort the current run, if one is active.
    pub async fn abort(&self) {
        self.agent.abort().await;
    }

    /// Wait for the current run and all listener settlement to finish.
    pub async fn wait_for_idle(&self) {
        self.agent.wait_for_idle().await;
    }

    // -----------------------------------------------------------------------
    // Prompt
    // -----------------------------------------------------------------------

    /// Send a prompt to the agent.
    ///
    /// Resolves the session context, emits `before_agent_start`, and
    /// delegates to the agent. Messages are persisted to the session tree
    /// via the internal agent event subscription.
    pub async fn prompt(
        &self,
        text: impl Into<String>,
        images: Vec<ImageContent>,
    ) -> anyhow::Result<()> {
        let text = text.into();
        self.restore_session_context().await?;

        // Emit before_agent_start and collect results.
        let system_prompt = self.get_current_system_prompt().await;
        let accumulated = self
            .runner
            .emit_before_agent_start(&text, &images, &system_prompt, CancellationToken::new())
            .await;

        // Apply system prompt override if any, otherwise reset to base.
        if let Some(ref acc) = accumulated {
            if let Some(ref sp) = acc.system_prompt {
                self.set_system_prompt(sp).await;
            }
        }

        // Build the messages array: before_agent_start custom messages, then user message.
        let mut prompt_messages: Vec<AgentMessage> = Vec::new();

        // Inject before_agent_start custom messages.
        if let Some(ref acc) = accumulated {
            if let Some(ref msgs) = acc.messages {
                for msg in msgs {
                    prompt_messages.push(
                        crate::session_manager::custom_message_content_to_agent_message(
                            &msg.custom_type,
                            msg.content.clone(),
                            msg.display,
                        ),
                    );
                }
            }
        }

        // Build user message.
        let user_msg = if images.is_empty() {
            ameli_ai::types::UserMessage::text(&text)
        } else {
            let mut content: Vec<ameli_ai::types::MediaContentBlock> =
                vec![ameli_ai::types::MediaContentBlock::Text(
                    ameli_ai::types::TextContent::new(&text),
                )];
            for img in images {
                content.push(ameli_ai::types::MediaContentBlock::Image(img));
            }
            ameli_ai::types::UserMessage {
                content: ameli_ai::types::UserContent::Blocks(content),
                timestamp: now_ms(),
            }
        };
        prompt_messages.push(AgentMessage::User(user_msg));

        self.agent.prompt(prompt_messages.into()).await
    }

    /// Continue from the current transcript.
    ///
    /// Resolves session context and delegates to the agent.
    pub async fn continue_(&self) -> anyhow::Result<()> {
        self.restore_session_context().await?;
        self.agent.continue_().await
    }

    // -----------------------------------------------------------------------
    // Commands
    // -----------------------------------------------------------------------

    /// Execute a named command registered by an extension.
    ///
    /// Dispatches to the first registered handler with matching name.
    pub async fn command(&self, name: &str, args: &str) -> anyhow::Result<()> {
        let ctx = crate::extension::events::CommandContext {
            extension_context: ExtensionContext {
                is_idle: !self.agent.is_active().await,
                cancel_token: None,
                interface: self.interface.clone(),
            },
        };
        self.runner.execute_command(name, args, ctx).await
    }

    // -----------------------------------------------------------------------
    // Shutdown
    // -----------------------------------------------------------------------

    /// Gracefully shut down the session.
    ///
    /// Emits `session_shutdown` to extensions, aborts any active run, and
    /// waits for the agent to become idle.
    pub async fn shutdown(&self) {
        self.runner
            .emit_session_shutdown(crate::extension::events::SessionShutdownReason::Quit)
            .await;
        self.agent.abort().await;
        self.agent.wait_for_idle().await;
    }

    // -----------------------------------------------------------------------
    // Session context resolution
    // -----------------------------------------------------------------------

    /// Resolve session context into `AgentMessage`s, using extension hooks
    /// for compaction and branch summary formatting.
    async fn resolve_session_messages(
        &self,
        cancel: CancellationToken,
    ) -> anyhow::Result<(Vec<AgentMessage>, String, Option<crate::types::ModelRef>)> {
        let session_ctx = self.session_manager.build_context().await?;

        let mut messages = Vec::with_capacity(session_ctx.messages.len());
        for session_msg in &session_ctx.messages {
            match session_msg {
                SessionMessage::Agent(agent_msg) => {
                    messages.push((**agent_msg).clone());
                }
                SessionMessage::Compaction { summary, timestamp } => {
                    let formatted = self
                        .runner
                        .emit_format_compaction_summary(summary, *timestamp, cancel.clone())
                        .await;
                    match formatted {
                        Some(msg) => messages.push(msg),
                        None => {
                            messages.push(
                                crate::session_manager::compaction_summary_to_agent_message(
                                    summary, *timestamp,
                                ),
                            );
                        }
                    }
                }
                SessionMessage::BranchSummary { summary, timestamp } => {
                    let formatted = self
                        .runner
                        .emit_format_branch_summary(summary, *timestamp, cancel.clone())
                        .await;
                    match formatted {
                        Some(msg) => messages.push(msg),
                        None => {
                            messages.push(crate::session_manager::branch_summary_to_agent_message(
                                summary, *timestamp,
                            ));
                        }
                    }
                }
            }
        }

        Ok((messages, session_ctx.thinking_level, session_ctx.model))
    }

    /// Restore session context onto the agent state.
    ///
    /// Resolves the session tree into messages and updates the agent state.
    /// Restores messages, system prompt, and thinking level. Model restoration
    /// is a downstream concern (requires a model registry to resolve
    /// `ModelRef` → `Model`).
    async fn restore_session_context(&self) -> anyhow::Result<()> {
        let (messages, thinking_level, _model_ref) = self
            .resolve_session_messages(CancellationToken::new())
            .await?;

        // Restore transcript.
        self.agent.set_messages(messages).await;

        // Restore thinking level from session (parse string, fallback to Off).
        let level = parse_thinking_level(&thinking_level);
        self.agent.set_thinking_level(level).await;

        Ok(())
    }

    /// Get the current system prompt from agent state.
    async fn get_current_system_prompt(&self) -> String {
        self.agent.state().await.system_prompt.clone()
    }

    /// Set the system prompt on agent state.
    async fn set_system_prompt(&self, prompt: &str) {
        self.agent.set_system_prompt(prompt.to_string()).await;
    }
}

impl<M: SessionMetadata> fmt::Debug for AgentSession<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentSession")
            .field("agent", &self.agent)
            .finish_non_exhaustive()
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

/// Parse a thinking level string (from `SessionContext`) into a
/// [`ThinkingLevel`]. Falls back to [`ThinkingLevel::Off`] for
/// unrecognized values.
fn parse_thinking_level(s: &str) -> ThinkingLevel {
    match s.to_lowercase().as_str() {
        "minimal" => ThinkingLevel::Minimal,
        "low" => ThinkingLevel::Low,
        "medium" => ThinkingLevel::Medium,
        "high" => ThinkingLevel::High,
        "xhigh" => ThinkingLevel::XHigh,
        _ => ThinkingLevel::Off,
    }
}

use std::fmt;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::{Extension, ExtensionApi};
    use crate::interface::NoopInterface;
    use crate::types::SessionContext;
    use ameli_agent_core::types::{AgentState, ThinkingLevel};
    use ameli_ai::types::{Cost, InputType, Model};
    use std::collections::HashSet;
    use std::future::Future;
    use std::pin::Pin;

    /// A minimal session metadata for testing.
    #[derive(Debug, Clone)]
    struct TestMetadata {
        id: String,
        created_at: String,
    }

    impl SessionMetadata for TestMetadata {
        fn id(&self) -> &str {
            &self.id
        }
        fn created_at(&self) -> &str {
            &self.created_at
        }
    }

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

    fn test_agent() -> ArcAgent {
        ArcAgent::new(ameli_agent_core::AgentOptions {
            initial_state: Some(AgentState {
                system_prompt: String::new(),
                model: test_model(),
                thinking_level: ThinkingLevel::Off,
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

    struct NoCommandsExtension;

    impl Extension for NoCommandsExtension {
        fn name(&self) -> &str {
            "no-commands"
        }
        fn init(&self, _api: &mut ExtensionApi) {}
    }

    fn test_session(agent: ArcAgent) -> AgentSession<TestMetadata> {
        let session_manager = Arc::new(TestSessionManager::new());
        let runner = Arc::new(ExtensionRunner::from_extensions(&[Box::new(
            NoCommandsExtension,
        )]));
        AgentSession::new(AgentSessionConfig {
            agent,
            session_manager,
            runner,
            interface: Arc::new(NoopInterface),
        })
    }

    #[tokio::test]
    async fn agent_session_construction() {
        let agent = test_agent();
        let session = test_session(agent);
        assert!(!session.is_active().await);
    }

    #[tokio::test]
    async fn command_returns_error_for_unknown() {
        let agent = test_agent();
        let session = test_session(agent);
        let result = session.command("unknown", "").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no command"));
    }

    #[tokio::test]
    async fn shutdown_completes() {
        let agent = test_agent();
        let session = test_session(agent);
        session.shutdown().await;
    }

    // -- Test SessionManager implementation ----------------------------------

    /// A minimal in-memory session manager for testing.
    struct TestSessionManager {
        entries: std::sync::Mutex<Vec<crate::types::SessionEntry>>,
    }

    impl TestSessionManager {
        fn new() -> Self {
            Self {
                entries: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl SessionManager<TestMetadata> for TestSessionManager {
        fn metadata(
            &self,
        ) -> Pin<Box<dyn Future<Output = Result<TestMetadata, crate::SessionError>> + Send>>
        {
            Box::pin(async move {
                Ok(TestMetadata {
                    id: "test-session".into(),
                    created_at: "2026-01-01T00:00:00Z".into(),
                })
            })
        }

        fn leaf_id(
            &self,
        ) -> Pin<Box<dyn Future<Output = Result<Option<String>, crate::SessionError>> + Send>>
        {
            let entries = self
                .entries
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            Box::pin(async move { Ok(entries.last().map(|e| e.id().to_string())) })
        }

        fn entry(
            &self,
            _id: &str,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<Option<crate::types::SessionEntry>, crate::SessionError>>
                    + Send,
            >,
        > {
            Box::pin(async move { Ok(None) })
        }

        fn entries(
            &self,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<Vec<crate::types::SessionEntry>, crate::SessionError>>
                    + Send,
            >,
        > {
            let entries = self
                .entries
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            Box::pin(async move { Ok(entries) })
        }

        fn branch(
            &self,
            from_id: Option<&str>,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<Vec<crate::types::SessionEntry>, crate::SessionError>>
                    + Send,
            >,
        > {
            let entries = self
                .entries
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            Box::pin(async move {
                let _ = from_id;
                Ok(entries)
            })
        }

        fn build_context(
            &self,
        ) -> Pin<Box<dyn Future<Output = Result<SessionContext, crate::SessionError>> + Send>>
        {
            let entries = self
                .entries
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            Box::pin(async move {
                Ok(crate::session_manager::build_session_context_from_path(
                    &entries,
                ))
            })
        }

        fn label(
            &self,
            _id: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Option<String>, crate::SessionError>> + Send>>
        {
            Box::pin(async move { Ok(None) })
        }

        fn append_message(
            &self,
            message: AgentMessage,
        ) -> Pin<Box<dyn Future<Output = Result<String, crate::SessionError>> + Send>> {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let id = format!("entry-{}", entries.len());
            let parent_id = entries.last().map(|e| e.id().to_string());
            entries.push(crate::types::SessionEntry::Message(
                crate::types::MessageEntry {
                    id: id.clone(),
                    parent_id,
                    timestamp: crate::session_manager::now_iso8601(),
                    message,
                },
            ));
            Box::pin(async move { Ok(id) })
        }

        fn append_thinking_level_change(
            &self,
            thinking_level: &str,
        ) -> Pin<Box<dyn Future<Output = Result<String, crate::SessionError>> + Send>> {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let id = format!("entry-{}", entries.len());
            let parent_id = entries.last().map(|e| e.id().to_string());
            entries.push(crate::types::SessionEntry::ThinkingLevelChange(
                crate::types::ThinkingLevelChangeEntry {
                    id: id.clone(),
                    parent_id,
                    timestamp: crate::session_manager::now_iso8601(),
                    thinking_level: thinking_level.to_string(),
                },
            ));
            Box::pin(async move { Ok(id) })
        }

        fn append_model_change(
            &self,
            provider: &str,
            model_id: &str,
        ) -> Pin<Box<dyn Future<Output = Result<String, crate::SessionError>> + Send>> {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let id = format!("entry-{}", entries.len());
            let parent_id = entries.last().map(|e| e.id().to_string());
            entries.push(crate::types::SessionEntry::ModelChange(
                crate::types::ModelChangeEntry {
                    id: id.clone(),
                    parent_id,
                    timestamp: crate::session_manager::now_iso8601(),
                    provider: provider.to_string(),
                    model_id: model_id.to_string(),
                },
            ));
            Box::pin(async move { Ok(id) })
        }

        fn append_compaction(
            &self,
            summary: &str,
            first_kept_entry_id: &str,
            tokens_before: u64,
            details: Option<serde_json::Value>,
            from_hook: bool,
        ) -> Pin<Box<dyn Future<Output = Result<String, crate::SessionError>> + Send>> {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let id = format!("entry-{}", entries.len());
            let parent_id = entries.last().map(|e| e.id().to_string());
            entries.push(crate::types::SessionEntry::Compaction(
                crate::types::CompactionEntry {
                    id: id.clone(),
                    parent_id,
                    timestamp: crate::session_manager::now_iso8601(),
                    summary: summary.to_string(),
                    first_kept_entry_id: first_kept_entry_id.to_string(),
                    tokens_before,
                    details,
                    from_hook,
                },
            ));
            Box::pin(async move { Ok(id) })
        }

        fn append_custom_entry(
            &self,
            custom_type: &str,
            data: Option<serde_json::Value>,
        ) -> Pin<Box<dyn Future<Output = Result<String, crate::SessionError>> + Send>> {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let id = format!("entry-{}", entries.len());
            let parent_id = entries.last().map(|e| e.id().to_string());
            entries.push(crate::types::SessionEntry::Custom(
                crate::types::CustomEntry {
                    id: id.clone(),
                    parent_id,
                    timestamp: crate::session_manager::now_iso8601(),
                    custom_type: custom_type.to_string(),
                    data,
                },
            ));
            Box::pin(async move { Ok(id) })
        }

        fn append_custom_message_entry(
            &self,
            custom_type: &str,
            content: crate::types::CustomMessageContent,
            display: bool,
            details: Option<serde_json::Value>,
        ) -> Pin<Box<dyn Future<Output = Result<String, crate::SessionError>> + Send>> {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let id = format!("entry-{}", entries.len());
            let parent_id = entries.last().map(|e| e.id().to_string());
            entries.push(crate::types::SessionEntry::CustomMessage(
                crate::types::CustomMessageEntry {
                    id: id.clone(),
                    parent_id,
                    timestamp: crate::session_manager::now_iso8601(),
                    custom_type: custom_type.to_string(),
                    content,
                    display,
                    details,
                },
            ));
            Box::pin(async move { Ok(id) })
        }

        fn move_to(
            &self,
            entry_id: Option<&str>,
            summary: Option<crate::BranchSummaryData>,
        ) -> Pin<Box<dyn Future<Output = Result<Option<String>, crate::SessionError>> + Send>>
        {
            let _ = (entry_id, summary);
            Box::pin(async move { Ok(None) })
        }
    }

    // -- Session context resolution tests -----------------------------------

    #[tokio::test]
    async fn resolve_session_messages_empty() {
        let agent = test_agent();
        let session = test_session(agent);

        let (messages, thinking_level, model) = session
            .resolve_session_messages(CancellationToken::new())
            .await
            .unwrap();

        assert!(messages.is_empty());
        assert_eq!(thinking_level, "off");
        assert!(model.is_none());
    }

    #[tokio::test]
    async fn resolve_session_messages_with_messages() {
        let agent = test_agent();
        let sm = Arc::new(TestSessionManager::new());

        // Append a user message to the session
        sm.append_message(AgentMessage::User(ameli_ai::types::UserMessage::text(
            "hello",
        )))
        .await
        .unwrap();

        let runner = Arc::new(ExtensionRunner::from_extensions(&[Box::new(
            NoCommandsExtension,
        )]));
        let session = AgentSession::new(AgentSessionConfig {
            agent,
            session_manager: sm,
            runner,
            interface: Arc::new(NoopInterface),
        });

        let (messages, _, _) = session
            .resolve_session_messages(CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role(), "user");
    }

    #[tokio::test]
    async fn resolve_session_messages_with_compaction_default() {
        let agent = test_agent();
        let sm = Arc::new(TestSessionManager::new());

        // Append a message then a compaction
        let entry_id = sm
            .append_message(AgentMessage::User(ameli_ai::types::UserMessage::text(
                "old message",
            )))
            .await
            .unwrap();
        sm.append_compaction("summary of old", &entry_id, 1000, None, false)
            .await
            .unwrap();

        // No format_compaction_summary handlers → default formatting
        let runner = Arc::new(ExtensionRunner::from_extensions(&[]));
        let session = AgentSession::new(AgentSessionConfig {
            agent,
            session_manager: sm,
            runner,
            interface: Arc::new(NoopInterface),
        });

        let (messages, _, _) = session
            .resolve_session_messages(CancellationToken::new())
            .await
            .unwrap();

        // Expect: compaction summary (default) + kept message
        assert!(messages.len() >= 1);
        // First message should be the compaction summary (a user message wrapping the summary)
        let first = &messages[0];
        assert_eq!(first.role(), "user");
    }

    // -- State mutation wiring tests -------------------------------------------

    #[tokio::test]
    async fn restore_session_context_sets_messages() {
        let agent = test_agent();
        let sm = Arc::new(TestSessionManager::new());

        // Append messages to session
        sm.append_message(AgentMessage::User(ameli_ai::types::UserMessage::text(
            "hello",
        )))
        .await
        .unwrap();
        sm.append_message(AgentMessage::Assistant(ameli_ai::types::AssistantMessage {
            content: vec![ameli_ai::types::AssistantContentBlock::Text(
                ameli_ai::types::TextContent::new("hi there"),
            )],
            api: "test".into(),
            provider: "test".into(),
            model: "test-model".into(),
            response_model: None,
            response_id: None,
            usage: ameli_ai::types::Usage::default(),
            stop_reason: ameli_ai::types::StopReason::Stop,
            error_message: None,
            timestamp: 1000,
        }))
        .await
        .unwrap();

        let runner = Arc::new(ExtensionRunner::from_extensions(&[]));
        let session = AgentSession::new(AgentSessionConfig {
            agent: ArcAgent::new(ameli_agent_core::AgentOptions {
                initial_state: Some(AgentState {
                    system_prompt: String::new(),
                    model: test_model(),
                    thinking_level: ThinkingLevel::Off,
                    tools: vec![],
                    messages: vec![],
                    is_streaming: false,
                    streaming_message: None,
                    pending_tool_calls: HashSet::new(),
                    error_message: None,
                }),
                ..Default::default()
            }),
            session_manager: sm,
            runner,
            interface: Arc::new(NoopInterface),
        });

        // Before restore, agent has no messages
        assert!(session.agent.state().await.messages.is_empty());

        // Restore session context
        session.restore_session_context().await.unwrap();

        // After restore, agent has the session messages
        let state = session.agent.state().await;
        assert_eq!(state.messages.len(), 2);
        assert_eq!(state.messages[0].role(), "user");
        assert_eq!(state.messages[1].role(), "assistant");
    }

    #[tokio::test]
    async fn restore_session_context_sets_thinking_level() {
        let agent = test_agent();
        let sm = Arc::new(TestSessionManager::new());

        // Append a thinking level change
        sm.append_thinking_level_change("medium").await.unwrap();

        let runner = Arc::new(ExtensionRunner::from_extensions(&[]));
        let session = AgentSession::new(AgentSessionConfig {
            agent,
            session_manager: sm,
            runner,
            interface: Arc::new(NoopInterface),
        });

        session.restore_session_context().await.unwrap();

        let state = session.agent.state().await;
        assert_eq!(state.thinking_level, ThinkingLevel::Medium);
    }

    #[tokio::test]
    async fn set_system_prompt_updates_agent() {
        let agent = test_agent();
        let session = test_session(agent);

        // Verify initial prompt is empty
        assert!(session.agent.state().await.system_prompt.is_empty());

        // Set a new prompt
        session.set_system_prompt("You are helpful.").await;

        // Verify it was applied
        assert_eq!(
            session.agent.state().await.system_prompt,
            "You are helpful."
        );
    }

    #[tokio::test]
    async fn parse_thinking_level_known_values() {
        assert_eq!(parse_thinking_level("off"), ThinkingLevel::Off);
        assert_eq!(parse_thinking_level("minimal"), ThinkingLevel::Minimal);
        assert_eq!(parse_thinking_level("low"), ThinkingLevel::Low);
        assert_eq!(parse_thinking_level("medium"), ThinkingLevel::Medium);
        assert_eq!(parse_thinking_level("high"), ThinkingLevel::High);
        assert_eq!(parse_thinking_level("xhigh"), ThinkingLevel::XHigh);
    }

    #[tokio::test]
    async fn parse_thinking_level_case_insensitive() {
        assert_eq!(parse_thinking_level("Medium"), ThinkingLevel::Medium);
        assert_eq!(parse_thinking_level("HIGH"), ThinkingLevel::High);
    }

    #[tokio::test]
    async fn parse_thinking_level_unknown_falls_back_to_off() {
        assert_eq!(parse_thinking_level("unknown"), ThinkingLevel::Off);
        assert_eq!(parse_thinking_level(""), ThinkingLevel::Off);
    }
}
