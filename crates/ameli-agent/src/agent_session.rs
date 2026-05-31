//! Agent session — composition layer that bridges session storage, agent loop,
//! extensions, and the interface.
//!
//! [`AgentSession`] is the primary way downstream applications use the agent
//! framework. It owns an [`ArcAgent`], a [`SessionManager`],
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
//! async fn example(
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
//!     AgentSession::new(config).await
//! }
//! ```

use crate::error::CreateAgentSessionError;
use crate::extension::{init_extensions, Extension, ExtensionContext, ExtensionRunner};
use crate::interface::Interface;
use ameli_agent_core::types::{
    AgentEvent, AgentMessage, AgentState, CustomAgentMessage, ThinkingLevel,
};
use ameli_agent_core::{AgentOptions, ArcAgent, Subscription};
use ameli_ai::types::{ImageContent, MediaContentBlock, TextContent};
use ameli_auth_storage::AuthStorage;
use ameli_model_registry::ModelRegistry;
use ameli_session_manager::{
    CustomMessageContent, ModelRef, SessionContext, SessionManager, SessionMessage, SessionMetadata,
};
use std::collections::HashSet;
use std::fmt;
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
    _subscription: Subscription,
}

// ---------------------------------------------------------------------------
// Agent event handler (runs inside subscriber closure)
// ---------------------------------------------------------------------------

/// Handle an [`AgentEvent`] from the agent: dispatch extension notifications
/// and persist messages to the session.
async fn handle_agent_event<M: SessionMetadata>(
    event: AgentEvent,
    session_manager: &Arc<dyn SessionManager<M>>,
    runner: &Arc<ExtensionRunner>,
    cancel: CancellationToken,
) {
    match &event {
        AgentEvent::MessageEnd { message } => {
            // Run the message_end hook chain — handlers may replace the message.
            // This is the sole dispatch path (matching pi's emitMessageEnd).
            let event = crate::extension::events::MessageEndEvent {
                message: message.clone(),
            };
            let final_msg = runner
                .emit_message_end(event, cancel)
                .await
                .unwrap_or_else(|| message.clone());

            // Persist to session.
            persist_message(&final_msg, session_manager).await;
        }
        _ => {
            // All other events: dispatch to extensions as notifications.
            runner.dispatch_agent_event(event, cancel).await;
        }
    }
}

/// Persist a finalized message to the session tree.
async fn persist_message<M: SessionMetadata>(
    message: &AgentMessage,
    session_manager: &Arc<dyn SessionManager<M>>,
) {
    match message {
        AgentMessage::User(_) | AgentMessage::Assistant(_) | AgentMessage::ToolResult(_) => {
            if let Err(e) = session_manager.append_message(message.clone()).await {
                tracing::warn!("Failed to persist message to session: {e}");
            }
        }
        AgentMessage::Custom(custom_msg) => {
            // Extract fields from the custom message via to_json().
            let json = custom_msg.to_json();
            if let (Some(custom_type), Some(content_val)) = (
                json.get("customType").and_then(|v| v.as_str()),
                json.get("content"),
            ) {
                let content =
                    match serde_json::from_value::<CustomMessageContent>(content_val.clone()) {
                        Ok(c) => c,
                        Err(_) => CustomMessageContent::Text(content_val.to_string()),
                    };
                let display = json
                    .get("display")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let details = json.get("details").cloned();
                if let Err(e) = session_manager
                    .append_custom_message_entry(custom_type, content, display, details)
                    .await
                {
                    tracing::warn!("Failed to persist custom message to session: {e}");
                }
            }
        }
    }
}

impl<M: SessionMetadata> AgentSession<M> {
    /// Create a new agent session.
    ///
    /// Restores session state from the session manager, subscribes to agent
    /// events for persistence, and emits `session_start` to extensions.
    pub async fn new(config: AgentSessionConfig<M>) -> Self {
        let agent = config.agent;
        let session_manager = config.session_manager;
        let runner = config.runner;
        let interface = config.interface;

        // Subscribe to agent events for extension dispatch and persistence.
        let sm = session_manager.clone();
        let runner_clone = runner.clone();
        let subscription = agent
            .subscribe(Arc::new(move |event, cancel| {
                let sm = sm.clone();
                let runner = runner_clone.clone();
                Box::pin(async move {
                    handle_agent_event(event, &sm, &runner, cancel).await;
                })
            }))
            .await;

        // Emit session_start to extensions.
        runner
            .emit_session_start(crate::extension::events::SessionStartReason::Startup)
            .await;

        Self {
            agent,
            session_manager,
            runner,
            interface,
            _subscription: subscription,
        }
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
    /// Emits `before_agent_start`, and delegates to the agent. Messages are
    /// persisted to the session tree via the internal agent event subscription.
    ///
    /// The caller is responsible for ensuring session context has been restored
    /// (e.g., via [`create_agent_session`]) before calling this method.
    pub async fn prompt(
        &self,
        text: impl Into<String>,
        images: Vec<ImageContent>,
    ) -> anyhow::Result<()> {
        let text = text.into();

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
                    prompt_messages.push(custom_message_content_to_agent_message(
                        &msg.custom_type,
                        msg.content.clone(),
                        msg.display,
                        msg.details.clone(),
                    ));
                }
            }
        }

        // Build user message.
        let user_msg = if images.is_empty() {
            ameli_ai::types::UserMessage::text(&text)
        } else {
            let mut content: Vec<MediaContentBlock> =
                vec![MediaContentBlock::Text(TextContent::new(&text))];
            for img in images {
                content.push(MediaContentBlock::Image(img));
            }
            ameli_ai::types::UserMessage {
                content: ameli_ai::types::UserContent::Blocks(content),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            }
        };
        prompt_messages.push(AgentMessage::User(user_msg));

        self.agent.prompt(prompt_messages.into()).await
    }

    /// Continue from the current transcript.
    ///
    /// Delegates to the agent. The caller is responsible for ensuring session
    /// context has been restored (e.g., via [`create_agent_session`])
    /// before calling this method.
    pub async fn continue_(&self) -> anyhow::Result<()> {
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

    /// Convert `SessionMessage`s from a pre-built context into `AgentMessage`s,
    /// consulting extension hooks for compaction and branch summary formatting.
    ///
    /// This does **not** call `build_context()` — the caller provides the
    /// already-built context.
    async fn resolve_messages_from_context(
        &self,
        session_ctx: &SessionContext,
        cancel: CancellationToken,
    ) -> Vec<AgentMessage> {
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
                            messages.push(compaction_summary_to_agent_message(summary, *timestamp));
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
                            messages.push(branch_summary_to_agent_message(summary, *timestamp));
                        }
                    }
                }
            }
        }
        messages
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

/// Convert a [`ThinkingLevel`] to its session storage string.
fn thinking_level_to_str(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Off => "off",
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::XHigh => "xhigh",
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers (relocated from session_manager module)
// ---------------------------------------------------------------------------

/// Extension-injected message content wrapped as a custom agent message.
#[derive(Clone)]
struct ExtensionCustomMessage {
    custom_type: String,
    content: CustomMessageContent,
    display: bool,
    details: Option<serde_json::Value>,
}

impl CustomAgentMessage for ExtensionCustomMessage {
    fn message_type(&self) -> &str {
        &self.custom_type
    }
    fn clone_boxed(&self) -> Box<dyn CustomAgentMessage> {
        Box::new(self.clone())
    }
    fn to_json(&self) -> serde_json::Value {
        let base = match &self.content {
            CustomMessageContent::Text(t) => serde_json::json!({
                "customType": self.custom_type,
                "content": t,
                "display": self.display,
            }),
            CustomMessageContent::Rich(blocks) => serde_json::json!({
                "customType": self.custom_type,
                "content": blocks,
                "display": self.display,
            }),
        };
        if let Some(details) = &self.details {
            let mut map = base.as_object().cloned().unwrap_or_default();
            map.insert("details".to_string(), details.clone());
            serde_json::Value::Object(map)
        } else {
            base
        }
    }
    fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionCustomMessage")
            .field("custom_type", &self.custom_type)
            .field("display", &self.display)
            .finish_non_exhaustive()
    }
}

/// Convert a [`CustomMessageContent`] to an [`AgentMessage::Custom`].
///
/// Used by [`AgentSession`] to inject `before_agent_start` extension messages
/// into the LLM context.
fn custom_message_content_to_agent_message(
    custom_type: &str,
    content: CustomMessageContent,
    display: bool,
    details: Option<serde_json::Value>,
) -> AgentMessage {
    let ext_msg = ExtensionCustomMessage {
        custom_type: custom_type.to_string(),
        content,
        display,
        details,
    };
    AgentMessage::Custom(Box::new(ext_msg))
}

/// Default formatting for a compaction summary as a synthetic user message.
///
/// `AgentSession` uses this as the fallback when no extension overrides
/// `on_format_compaction_summary`.
fn compaction_summary_to_agent_message(summary: &str, timestamp: u64) -> AgentMessage {
    let text = format!(
        "The conversation history before this point was compacted into the following summary:\n\n\
         <summary>\n{summary}\n</summary>",
    );
    let content = vec![MediaContentBlock::Text(TextContent::new(text))];
    AgentMessage::User(ameli_ai::types::UserMessage {
        content: ameli_ai::types::UserContent::Blocks(content),
        timestamp,
    })
}

/// Default formatting for a branch summary as a synthetic user message.
///
/// `AgentSession` uses this as the fallback when no extension overrides
/// `on_format_branch_summary`.
fn branch_summary_to_agent_message(summary: &str, timestamp: u64) -> AgentMessage {
    let text = format!(
        "The following is a summary of a branch that this conversation came back from:\n\n\
         <summary>\n{summary}\n</summary>",
    );
    let content = vec![MediaContentBlock::Text(TextContent::new(text))];
    AgentMessage::User(ameli_ai::types::UserMessage {
        content: ameli_ai::types::UserContent::Blocks(content),
        timestamp,
    })
}

// ---------------------------------------------------------------------------
// create_agent_session — factory function
// ---------------------------------------------------------------------------

/// Inputs for [`create_agent_session`].
///
/// Collects all dependencies needed to construct a fully loaded, idle
/// [`AgentSession`]. The generic parameter `M` is the session metadata type
/// defined by the downstream application's storage backend.
pub struct CreateAgentSessionOptions<M: SessionMetadata> {
    /// Which model to use. Resolved to a full [`Model`] via `model_registry`.
    pub model: ModelRef,
    /// Model registry for resolving [`ModelRef`] → [`Model`].
    pub model_registry: Arc<dyn ModelRegistry>,
    /// Auth storage for resolving API keys at stream time.
    pub auth_storage: Arc<dyn AuthStorage>,
    /// Session storage backend.
    pub session_manager: Arc<dyn SessionManager<M>>,
    /// UI interface for notifications.
    pub interface: Arc<dyn Interface>,
    /// Extensions to register.
    pub extensions: Vec<Box<dyn Extension>>,
    /// Optional thinking level override. Defaults to [`ThinkingLevel::Off`].
    pub thinking_level: Option<ThinkingLevel>,
    /// Optional system prompt. Defaults to empty.
    pub system_prompt: Option<String>,
}

/// Result of [`create_agent_session`].
///
/// The returned session is fully loaded and idle — ready for
/// [`prompt`](AgentSession::prompt) or [`continue_`](AgentSession::continue_).
pub struct CreateAgentSessionResult<M: SessionMetadata> {
    /// The created session.
    pub session: AgentSession<M>,
    /// Non-fatal warnings collected during session creation.
    pub warnings: Vec<String>,
}

impl<M: SessionMetadata> fmt::Debug for CreateAgentSessionResult<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CreateAgentSessionResult")
            .field("session", &self.session)
            .field("warnings", &self.warnings)
            .finish()
    }
}

/// Create a fully loaded, idle [`AgentSession`].
///
/// This is the primary way to construct an agent session. It:
///
/// 1. Resolves [`ModelRef`] → [`Model`] via the model registry
/// 2. Validates that an API key is available for the model's provider
/// 3. Initializes extensions and wires their hooks into the agent
/// 4. Wires auth storage into the agent so API keys are resolved per-call
/// 5. Restores session context (messages, thinking level) from storage
/// 6. Persists initial model and thinking level for new sessions
///
/// The returned session is idle and ready for
/// [`prompt`](AgentSession::prompt) or [`continue_`](AgentSession::continue_).
///
/// # Errors
///
/// Returns [`CreateAgentSessionError::ModelNotFound`] if the model registry
/// cannot resolve the requested model, [`CreateAgentSessionError::ApiKeyNotFound`]
/// if no API key is configured for the model's provider, or
/// [`CreateAgentSessionError::Session`] if session storage fails.
///
/// # Example
///
/// ```no_run
/// use ameli_agent::{
///     create_agent_session, CreateAgentSessionOptions, NoopInterface,
///     SessionManager, SessionMetadata, InMemorySessionManager,
/// };
/// use ameli_agent::ModelRef;
/// use ameli_auth_storage::InMemoryAuthStorage;
/// use ameli_model_registry::DefaultModelRegistry;
/// use std::sync::Arc;
///
/// async fn example() -> Result<(), ameli_agent::CreateAgentSessionError> {
///     let auth_storage = Arc::new(InMemoryAuthStorage::new());
///     let model_registry = Arc::new(DefaultModelRegistry::new());
///     let session_manager = Arc::new(InMemorySessionManager::new());
///
///     let result = create_agent_session(CreateAgentSessionOptions {
///         model: ModelRef { provider: "openai".into(), model_id: "gpt-4o".into() },
///         model_registry,
///         auth_storage,
///         session_manager,
///         interface: Arc::new(NoopInterface),
///         extensions: vec![],
///         thinking_level: None,
///         system_prompt: None,
///     }).await?;
///
///     // Session is ready for prompt, continue, command, etc.
///     Ok(())
/// }
/// ```
pub async fn create_agent_session<M: SessionMetadata>(
    options: CreateAgentSessionOptions<M>,
) -> Result<CreateAgentSessionResult<M>, CreateAgentSessionError> {
    // 1. Resolve ModelRef → Model via the registry.
    let model = options
        .model_registry
        .get_model(&options.model.provider, &options.model.model_id)?;

    // 2. Validate that an API key exists (fail-fast).
    options
        .auth_storage
        .get_api_key(&model.provider)
        .await
        .map_err(|_| CreateAgentSessionError::ApiKeyNotFound {
            provider: model.provider.clone(),
        })?;

    // 3. Initialize extensions.
    let handlers = init_extensions(&options.extensions);
    let runner = Arc::new(ExtensionRunner::with_interface(
        handlers,
        options.interface.clone(),
    ));

    // 4. Build AgentOptions.
    let thinking_level = options.thinking_level.unwrap_or(ThinkingLevel::Off);
    let tools = runner.get_registered_tools();
    let auth_storage = options.auth_storage.clone();

    let mut agent_options = AgentOptions {
        initial_state: Some(AgentState {
            system_prompt: options.system_prompt.unwrap_or_default(),
            model: model.clone(),
            thinking_level,
            tools,
            messages: Vec::new(),
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        }),
        get_api_key: Some(Arc::new(move |provider: &str| {
            let auth_storage = auth_storage.clone();
            let provider = provider.to_string();
            Box::pin(async move { auth_storage.get_api_key(&provider).await.ok() })
        })),
        api_registry: Some(ameli_ai::api::DEFAULT_API_REGISTRY.clone()),
        ..Default::default()
    };

    // 5. Install extension hooks (before_tool_call, after_tool_call, transform_context).
    runner.install_hooks(&mut agent_options);

    // 6. Construct ArcAgent.
    let agent = ArcAgent::new(agent_options);

    // 7. Construct AgentSession.
    let session = AgentSession::new(AgentSessionConfig {
        agent,
        session_manager: options.session_manager.clone(),
        runner: runner.clone(),
        interface: options.interface.clone(),
    })
    .await;

    // 8. Restore session context from storage (only if session has existing data).
    let session_ctx = options.session_manager.build_context().await?;
    let has_existing_session = !session_ctx.messages.is_empty();

    if has_existing_session {
        // Existing session: restore messages and thinking level.
        let messages = session
            .resolve_messages_from_context(&session_ctx, CancellationToken::new())
            .await;
        session.agent.set_messages(messages).await;
        let level = parse_thinking_level(&session_ctx.thinking_level);
        session.agent.set_thinking_level(level).await;
    } else {
        // New session: persist initial model and thinking level.
        options
            .session_manager
            .append_model_change(&model.provider, &model.id)
            .await?;
        options
            .session_manager
            .append_thinking_level_change(thinking_level_to_str(thinking_level))
            .await?;
    }

    Ok(CreateAgentSessionResult {
        session,
        warnings: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::{Extension, ExtensionApi};
    use crate::interface::NoopInterface;
    use ameli_ai::types::{Cost, InputType, Model};
    use ameli_session_manager::{InMemoryMetadata, InMemorySessionManager, SessionEntry};

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

    async fn test_session(agent: ArcAgent) -> AgentSession<InMemoryMetadata> {
        let session_manager = Arc::new(InMemorySessionManager::new());
        let runner = Arc::new(ExtensionRunner::from_extensions(&[Box::new(
            NoCommandsExtension,
        )]));
        AgentSession::new(AgentSessionConfig {
            agent,
            session_manager,
            runner,
            interface: Arc::new(NoopInterface),
        })
        .await
    }

    #[tokio::test]
    async fn agent_session_construction() {
        let agent = test_agent();
        let session = test_session(agent).await;
        assert!(!session.is_active().await);
    }

    #[tokio::test]
    async fn command_returns_error_for_unknown() {
        let agent = test_agent();
        let session = test_session(agent).await;
        let result = session.command("unknown", "").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no command"));
    }

    #[tokio::test]
    async fn shutdown_completes() {
        let agent = test_agent();
        let session = test_session(agent).await;
        session.shutdown().await;
    }

    // -- Session context resolution tests -----------------------------------

    #[tokio::test]
    async fn resolve_messages_from_context_empty() {
        let agent = test_agent();
        let session = test_session(agent).await;

        let ctx = SessionContext {
            messages: vec![],
            thinking_level: "off".into(),
            model: None,
        };
        let messages = session
            .resolve_messages_from_context(&ctx, CancellationToken::new())
            .await;
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn resolve_messages_from_context_with_agent_message() {
        let agent = test_agent();
        let session = test_session(agent).await;

        let ctx = SessionContext {
            messages: vec![SessionMessage::Agent(Box::new(AgentMessage::User(
                ameli_ai::types::UserMessage::text("hello"),
            )))],
            thinking_level: "off".into(),
            model: None,
        };
        let messages = session
            .resolve_messages_from_context(&ctx, CancellationToken::new())
            .await;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role(), "user");
    }

    #[tokio::test]
    async fn resolve_messages_from_context_with_compaction_default() {
        let agent = test_agent();
        let session = test_session(agent).await;

        // No format_compaction_summary handlers → default formatting
        let ctx = SessionContext {
            messages: vec![SessionMessage::Compaction {
                summary: "summary of old".into(),
                timestamp: 1000,
            }],
            thinking_level: "off".into(),
            model: None,
        };
        let messages = session
            .resolve_messages_from_context(&ctx, CancellationToken::new())
            .await;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role(), "user");
    }

    // -- State mutation wiring tests -------------------------------------------

    #[tokio::test]
    async fn create_agent_session_restores_messages_from_existing_session() {
        let sm = Arc::new(InMemorySessionManager::new());

        // Pre-populate session with messages
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
            provider: "test-provider".into(),
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

        let result = create_agent_session(CreateAgentSessionOptions {
            model: ModelRef {
                provider: "test-provider".into(),
                model_id: "test-model".into(),
            },
            model_registry: test_model_registry(),
            auth_storage: test_auth_storage(),
            session_manager: sm.clone(),
            interface: Arc::new(NoopInterface),
            extensions: vec![],
            thinking_level: None,
            system_prompt: None,
        })
        .await
        .unwrap();

        let state = result.session.agent().state().await;
        assert_eq!(state.messages.len(), 2);
        assert_eq!(state.messages[0].role(), "user");
        assert_eq!(state.messages[1].role(), "assistant");

        // No model_change should be persisted (session already had data)
        let entries = sm.entries().await.unwrap();
        let model_changes: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, SessionEntry::ModelChange(_)))
            .collect();
        assert_eq!(
            model_changes.len(),
            0,
            "should not persist model change for existing session"
        );
    }

    #[tokio::test]
    async fn create_agent_session_restores_thinking_level_from_existing_session() {
        let sm = Arc::new(InMemorySessionManager::new());

        // Pre-populate with a message and thinking level change
        sm.append_message(AgentMessage::User(ameli_ai::types::UserMessage::text(
            "hello",
        )))
        .await
        .unwrap();
        sm.append_thinking_level_change("medium").await.unwrap();

        let result = create_agent_session(CreateAgentSessionOptions {
            model: ModelRef {
                provider: "test-provider".into(),
                model_id: "test-model".into(),
            },
            model_registry: test_model_registry(),
            auth_storage: test_auth_storage(),
            session_manager: sm,
            interface: Arc::new(NoopInterface),
            extensions: vec![],
            thinking_level: None,
            system_prompt: None,
        })
        .await
        .unwrap();

        let state = result.session.agent().state().await;
        assert_eq!(state.thinking_level, ThinkingLevel::Medium);
    }

    #[tokio::test]
    async fn set_system_prompt_updates_agent() {
        let agent = test_agent();
        let session = test_session(agent).await;

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

    // -- Event persistence tests -----------------------------------------------

    #[tokio::test]
    async fn persist_standard_message_appends_to_session() {
        let sm: Arc<dyn SessionManager<InMemoryMetadata>> = Arc::new(InMemorySessionManager::new());

        let msg = AgentMessage::User(ameli_ai::types::UserMessage::text("hello"));
        persist_message(&msg, &sm).await;

        let entries = sm.entries().await.unwrap();
        assert_eq!(entries.len(), 1);
        if let SessionEntry::Message(me) = &entries[0] {
            assert_eq!(me.message.role(), "user");
        } else {
            panic!("Expected Message entry");
        }
    }

    #[tokio::test]
    async fn persist_custom_message_appends_to_session() {
        let sm: Arc<dyn SessionManager<InMemoryMetadata>> = Arc::new(InMemorySessionManager::new());

        let msg = custom_message_content_to_agent_message(
            "context",
            CustomMessageContent::Text("some context".into()),
            true,
            None,
        );
        persist_message(&msg, &sm).await;

        let entries = sm.entries().await.unwrap();
        assert_eq!(entries.len(), 1);
        if let SessionEntry::CustomMessage(cme) = &entries[0] {
            assert_eq!(cme.custom_type, "context");
            assert!(cme.display);
        } else {
            panic!("Expected CustomMessage entry");
        }
    }

    #[tokio::test]
    async fn handle_agent_event_persists_message_end() {
        let sm: Arc<dyn SessionManager<InMemoryMetadata>> = Arc::new(InMemorySessionManager::new());
        let runner = Arc::new(ExtensionRunner::from_extensions(&[]));

        let event = AgentEvent::MessageEnd {
            message: AgentMessage::User(ameli_ai::types::UserMessage::text("hello")),
        };
        handle_agent_event(event, &sm, &runner, CancellationToken::new()).await;

        let entries = sm.entries().await.unwrap();
        assert_eq!(entries.len(), 1);
        if let SessionEntry::Message(me) = &entries[0] {
            assert_eq!(me.message.role(), "user");
        } else {
            panic!("Expected Message entry");
        }
    }

    // -- create_agent_session tests ---------------------------------------------

    use ameli_auth_storage::InMemoryAuthStorage;
    use ameli_model_registry::DefaultModelRegistry;

    /// Helper: register a test model and return the registry.
    fn test_model_registry() -> Arc<DefaultModelRegistry> {
        let registry = Arc::new(DefaultModelRegistry::new());
        registry.register(test_model());
        registry
    }

    /// Helper: create auth storage with a test API key.
    fn test_auth_storage() -> Arc<InMemoryAuthStorage> {
        let storage = Arc::new(InMemoryAuthStorage::new());
        storage.set_api_key("test-provider", "test-key".to_string());
        storage
    }

    #[tokio::test]
    async fn create_agent_session_succeeds_with_valid_inputs() {
        let result = create_agent_session(CreateAgentSessionOptions {
            model: ModelRef {
                provider: "test-provider".into(),
                model_id: "test-model".into(),
            },
            model_registry: test_model_registry(),
            auth_storage: test_auth_storage(),
            session_manager: Arc::new(InMemorySessionManager::new()),
            interface: Arc::new(NoopInterface),
            extensions: vec![],
            thinking_level: None,
            system_prompt: None,
        })
        .await;

        assert!(result.is_ok(), "expected Ok, got {:?}", result.err());
        let result = result.unwrap();
        assert!(!result.session.is_active().await);
        assert!(result.warnings.is_empty());
    }

    #[tokio::test]
    async fn create_agent_session_fails_for_unknown_model() {
        let result = create_agent_session(CreateAgentSessionOptions {
            model: ModelRef {
                provider: "unknown-provider".into(),
                model_id: "unknown-model".into(),
            },
            model_registry: test_model_registry(),
            auth_storage: test_auth_storage(),
            session_manager: Arc::new(InMemorySessionManager::new()),
            interface: Arc::new(NoopInterface),
            extensions: vec![],
            thinking_level: None,
            system_prompt: None,
        })
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, CreateAgentSessionError::ModelNotFound(_)));
    }

    #[tokio::test]
    async fn create_agent_session_fails_for_missing_api_key() {
        // Auth storage with no key for test-provider
        let auth_storage = Arc::new(InMemoryAuthStorage::new());

        let result = create_agent_session(CreateAgentSessionOptions {
            model: ModelRef {
                provider: "test-provider".into(),
                model_id: "test-model".into(),
            },
            model_registry: test_model_registry(),
            auth_storage,
            session_manager: Arc::new(InMemorySessionManager::new()),
            interface: Arc::new(NoopInterface),
            extensions: vec![],
            thinking_level: None,
            system_prompt: None,
        })
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, CreateAgentSessionError::ApiKeyNotFound { provider } if provider == "test-provider")
        );
    }

    #[tokio::test]
    async fn create_agent_session_sets_model_and_thinking_level() {
        let result = create_agent_session(CreateAgentSessionOptions {
            model: ModelRef {
                provider: "test-provider".into(),
                model_id: "test-model".into(),
            },
            model_registry: test_model_registry(),
            auth_storage: test_auth_storage(),
            session_manager: Arc::new(InMemorySessionManager::new()),
            interface: Arc::new(NoopInterface),
            extensions: vec![],
            thinking_level: Some(ThinkingLevel::High),
            system_prompt: Some("You are helpful.".into()),
        })
        .await
        .unwrap();

        let state = result.session.agent().state().await;
        assert_eq!(state.model.id, "test-model");
        assert_eq!(state.thinking_level, ThinkingLevel::High);
        assert_eq!(state.system_prompt, "You are helpful.");
    }

    #[tokio::test]
    async fn create_agent_session_persists_model_for_new_session() {
        let sm = Arc::new(InMemorySessionManager::new());

        let result = create_agent_session(CreateAgentSessionOptions {
            model: ModelRef {
                provider: "test-provider".into(),
                model_id: "test-model".into(),
            },
            model_registry: test_model_registry(),
            auth_storage: test_auth_storage(),
            session_manager: sm.clone(),
            interface: Arc::new(NoopInterface),
            extensions: vec![],
            thinking_level: Some(ThinkingLevel::Medium),
            system_prompt: None,
        })
        .await
        .unwrap();

        // Session should be idle and have persisted model + thinking level
        assert!(!result.session.is_active().await);

        let entries = sm.entries().await.unwrap();
        // Should have a model_change and thinking_level_change entry
        let model_changes: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, SessionEntry::ModelChange(_)))
            .collect();
        let thinking_changes: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, SessionEntry::ThinkingLevelChange(_)))
            .collect();
        assert_eq!(model_changes.len(), 1);
        assert_eq!(thinking_changes.len(), 1);
    }

    #[tokio::test]
    async fn create_agent_session_with_extensions() {
        struct TestExtension;
        impl Extension for TestExtension {
            fn name(&self) -> &str {
                "test-ext"
            }
            fn init(&self, _api: &mut ExtensionApi) {}
        }

        let result = create_agent_session(CreateAgentSessionOptions {
            model: ModelRef {
                provider: "test-provider".into(),
                model_id: "test-model".into(),
            },
            model_registry: test_model_registry(),
            auth_storage: test_auth_storage(),
            session_manager: Arc::new(InMemorySessionManager::new()),
            interface: Arc::new(NoopInterface),
            extensions: vec![Box::new(TestExtension)],
            thinking_level: None,
            system_prompt: None,
        })
        .await;

        assert!(result.is_ok());
    }
}
