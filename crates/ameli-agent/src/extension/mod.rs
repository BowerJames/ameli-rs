//! Extension system for ameli-agent.
//!
//! This module defines the core extension API — the trait extensions implement,
//! the event types they subscribe to, and the registration surface for tools,
//! hooks, and commands.
//!
//! # Architecture
//!
//! ```text
//! ExtensionRunner          ← pure event bus (handler storage + dispatch)
//! ExtensionActions         ← action performer (Weak<Agent> + session manager)
//!     ↓
//! ExtensionApi             ← facade wrapping both, shared with all extensions
//!     ↓
//! Extension trait          → impl Extension for MyExt { fn init(&self, api) }
//!     ↓
//! ExtensionRunner          → wires handlers to ArcAgent + AgentLoopConfig
//! ```
//!
//! # Extension lifecycle
//!
//! 1. Create an [`ExtensionRunner`] (pure event bus).
//! 2. Create an `ArcAgent` and unconditionally install all extension hook
//!    closures via [`ExtensionRunner::install_hooks_on_agent`]. The closures
//!    capture `Arc<ExtensionRunner>` and iterate handler lists at runtime.
//! 3. Create [`ExtensionActions`] with a `Weak<Agent>` from the agent.
//! 4. Create an [`ExtensionApi`] wrapping both runner and actions.
//! 5. Call [`Extension::init`] on each extension with `&Arc<ExtensionApi>`.
//!    Extensions call registration methods to subscribe to events, register
//!    tools, and register commands. They may also call action methods like
//!    [`ExtensionApi::send_user_message`] from handlers and tools.
//! 6. Set tools collected by the runner on the agent.
//!
//! # Events
//!
//! Three categories based on dispatch semantics:
//!
//! - **Fire-and-forget** — observe but don't modify. Errors are caught.
//! - **Sequential chain** — handlers run in order, each seeing accumulated
//!   state from prior handlers.
//! - **First-to-return** — handlers run in order; first `Some` result wins.
//!
//! # Design note
//!
//! This module is inspired by pi's extension system but deliberately minimal
//! for the headless first pass. UI-specific extensions (shortcuts, flags,
//! rendering) and model/provider events are deferred to future work.

pub mod actions;
pub mod context;
pub mod events;
pub mod runner;

pub use actions::ExtensionActions;
pub use context::ExtensionContext;
pub use events::*;
pub use runner::{ExtensionError, ExtensionRunner};

use ameli_agent_core::types::AgentTool;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Pinned, boxed, sendable future returned by extension handlers.
type BoxFuture<T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send>>;

// ---------------------------------------------------------------------------
// Extension trait
// ---------------------------------------------------------------------------

/// Trait for ameli-agent extensions.
///
/// Extensions implement this trait, call typed registration methods on
/// [`ExtensionApi`] during [`init`](Extension::init), and the runtime wires
/// them to the agent loop.
///
/// The `api` is an `Arc`, so extensions can clone it and pass it to tools
/// or commands they construct. The `Arc` remains valid for the lifetime of
/// the session.
///
/// # Examples
///
/// ```
/// use ameli_agent::extension::{Extension, ExtensionApi};
/// use std::sync::Arc;
///
/// struct LoggingExtension;
///
/// impl Extension for LoggingExtension {
///     fn init(&self, api: &Arc<ExtensionApi>) {
///         api.on_agent_start(|_event, _ctx| {
///             Box::pin(async move {
///                 println!("Agent started");
///                 Ok(())
///             })
///         });
///     }
/// }
/// ```
pub trait Extension: Send + Sync {
    /// Called once during extension registration.
    ///
    /// Use `api` to subscribe to events and register tools. The `api`
    /// can be cloned and stored for later use in tools or commands.
    fn init(&self, api: &Arc<ExtensionApi>);
}

// ---------------------------------------------------------------------------
// ExtensionApi
// ---------------------------------------------------------------------------

/// Registration surface and action facade passed to extensions during
/// [`Extension::init`].
///
/// Wraps `Arc<ExtensionRunner>` for handler registration and
/// `Arc<ExtensionActions>` for runtime actions (sending messages, persisting
/// session entries). Extensions receive `&Arc<Self>` so they can clone it
/// and pass it to tools/commands they build.
///
/// # Handler contract
///
/// Handlers must not panic. Notification handlers return `anyhow::Result<()>`;
/// errors are reported to registered error listeners and do not stop dispatch
/// to subsequent handlers. Hook handlers return `Option<ResultType>` — return
/// `None` to allow default behavior.
pub struct ExtensionApi {
    runner: Arc<ExtensionRunner>,
    actions: Arc<ExtensionActions>,
}

impl ExtensionApi {
    /// Create a new API surface backed by the given runner and actions.
    pub fn new(runner: Arc<ExtensionRunner>, actions: Arc<ExtensionActions>) -> Self {
        Self { runner, actions }
    }

    // -----------------------------------------------------------------------
    // Notification event registration (fire-and-forget)
    // -----------------------------------------------------------------------

    /// Subscribe to agent loop start.
    pub fn on_agent_start(
        &self,
        handler: impl Fn(AgentStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner.add_agent_start_handler(Arc::new(handler));
    }

    /// Subscribe to agent loop end.
    pub fn on_agent_end(
        &self,
        handler: impl Fn(AgentEndEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner.add_agent_end_handler(Arc::new(handler));
    }

    /// Subscribe to turn start.
    pub fn on_turn_start(
        &self,
        handler: impl Fn(TurnStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner.add_turn_start_handler(Arc::new(handler));
    }

    /// Subscribe to turn end.
    pub fn on_turn_end(
        &self,
        handler: impl Fn(TurnEndEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner.add_turn_end_handler(Arc::new(handler));
    }

    /// Subscribe to message start (user, assistant, or tool result).
    pub fn on_message_start(
        &self,
        handler: impl Fn(MessageStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner.add_message_start_handler(Arc::new(handler));
    }

    /// Subscribe to message streaming updates (assistant messages only).
    pub fn on_message_update(
        &self,
        handler: impl Fn(MessageUpdateEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner.add_message_update_handler(Arc::new(handler));
    }

    /// Subscribe to message end (user, assistant, or tool result).
    ///
    /// This is a **sequential chain hook**: handlers run in registration order
    /// and can return a replacement message that preserves the original role.
    /// Each handler sees the result of prior handlers.
    pub fn on_message_end(
        &self,
        handler: impl Fn(MessageEndEvent, ExtensionContext) -> BoxFuture<Option<MessageEndResult>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner.add_message_end_handler(Arc::new(handler));
    }

    /// Subscribe to tool execution start.
    pub fn on_tool_execution_start(
        &self,
        handler: impl Fn(ToolExecutionStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner
            .add_tool_execution_start_handler(Arc::new(handler));
    }

    /// Subscribe to tool execution updates (partial/streaming output).
    pub fn on_tool_execution_update(
        &self,
        handler: impl Fn(ToolExecutionUpdateEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner
            .add_tool_execution_update_handler(Arc::new(handler));
    }

    /// Subscribe to tool execution end.
    pub fn on_tool_execution_end(
        &self,
        handler: impl Fn(ToolExecutionEndEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner
            .add_tool_execution_end_handler(Arc::new(handler));
    }

    /// Subscribe to session start (emitted when AgentSession is created).
    pub fn on_session_start(
        &self,
        handler: impl Fn(SessionStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner.add_session_start_handler(Arc::new(handler));
    }

    /// Subscribe to session shutdown (emitted when AgentSession is shutting
    /// down).
    pub fn on_session_shutdown(
        &self,
        handler: impl Fn(SessionShutdownEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner.add_session_shutdown_handler(Arc::new(handler));
    }

    // -----------------------------------------------------------------------
    // Hook event registration
    // -----------------------------------------------------------------------

    /// Register a hook called before a tool executes.
    ///
    /// Handlers run in registration order. If any handler returns
    /// `Some(ToolCallResult { block: true, .. })`, the tool is blocked.
    pub fn on_tool_call(
        &self,
        handler: impl Fn(ToolCallEvent, ExtensionContext) -> BoxFuture<Option<ToolCallResult>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner.add_tool_call_handler(Arc::new(handler));
    }

    /// Register a hook called after a tool finishes executing.
    ///
    /// Handlers run in registration order. Each handler sees the result after
    /// previous handler changes. Return `Some(ToolResultPatch)` to override
    /// parts of the result.
    pub fn on_tool_result(
        &self,
        handler: impl Fn(ToolResultEvent, ExtensionContext) -> BoxFuture<Option<ToolResultPatch>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner.add_tool_result_handler(Arc::new(handler));
    }

    /// Register a hook called before each LLM call to modify the context.
    ///
    /// Handlers run in registration order. If any handler returns
    /// `Some(ContextResult)`, the messages are replaced for subsequent
    /// handlers and the LLM call.
    pub fn on_context(
        &self,
        handler: impl Fn(ContextEvent, ExtensionContext) -> BoxFuture<Option<ContextResult>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner.add_context_handler(Arc::new(handler));
    }

    /// Register a hook called before the agent loop starts processing a
    /// prompt.
    ///
    /// **Sequential accumulate**: all handler results are collected. Custom
    /// messages are accumulated in order. The last non-`None` `system_prompt`
    /// wins.
    pub fn on_before_agent_start(
        &self,
        handler: impl Fn(BeforeAgentStartEvent, ExtensionContext) -> BoxFuture<Option<BeforeAgentStartResult>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner
            .add_before_agent_start_handler(Arc::new(handler));
    }

    /// Register a hook called when a compaction summary needs formatting into
    /// an [`AgentMessage`](ameli_agent_core::types::AgentMessage).
    ///
    /// Handlers run in registration order. The first handler to return
    /// `Some(...)` wins. If no handler returns `Some`, the default
    /// conversion wraps the summary in a synthetic user message.
    pub fn on_format_compaction_summary(
        &self,
        handler: impl Fn(
                FormatCompactionSummaryEvent,
                ExtensionContext,
            ) -> BoxFuture<Option<FormatCompactionSummaryResult>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner
            .add_format_compaction_summary_handler(Arc::new(handler));
    }

    /// Register a hook called when a branch summary needs formatting into
    /// an [`AgentMessage`](ameli_agent_core::types::AgentMessage).
    ///
    /// Handlers run in registration order. The first handler to return
    /// `Some(...)` wins. If no handler returns `Some`, the default
    /// conversion wraps the summary in a synthetic user message.
    pub fn on_format_branch_summary(
        &self,
        handler: impl Fn(
                FormatBranchSummaryEvent,
                ExtensionContext,
            ) -> BoxFuture<Option<FormatBranchSummaryResult>>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner
            .add_format_branch_summary_handler(Arc::new(handler));
    }

    // -----------------------------------------------------------------------
    // Command registration
    // -----------------------------------------------------------------------

    /// Register a named command that can be invoked via
    /// [`AgentSession::command`](crate::AgentSession::command).
    ///
    /// Commands are identified by name. The first extension to register a
    /// name wins.
    pub fn register_command(
        &self,
        name: impl Into<String>,
        description: Option<String>,
        handler: Arc<dyn Fn(String, CommandContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync>,
    ) {
        self.runner.add_command(RegisteredCommand {
            name: name.into(),
            description,
            handler,
        });
    }

    // -----------------------------------------------------------------------
    // Tool registration
    // -----------------------------------------------------------------------

    /// Register an LLM-callable tool.
    ///
    /// The tool will be available for the model to invoke during agent runs.
    pub fn register_tool(&self, tool: Arc<dyn AgentTool>) {
        self.runner.add_tool(tool);
    }

    // -----------------------------------------------------------------------
    // Custom message formatter registration
    // -----------------------------------------------------------------------

    /// Register a formatter that converts a custom message type to an
    /// LLM-compatible [`Message`](ameli_ai::types::Message).
    ///
    /// The formatter is looked up by `custom_type` when the agent's
    /// `convert_to_llm` pipeline encounters an
    /// [`AgentMessage::Custom`](ameli_agent_core::types::AgentMessage::Custom).
    /// If the formatter returns `Some(Message)`, the message is included in
    /// the LLM context. If it returns `None`, the message is skipped.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// api.register_custom_message_formatter("instruction", |custom_type, data| {
    ///     let msg = data["message"].as_str().unwrap_or("");
    ///     Some(ameli_ai::types::Message::User(
    ///         ameli_ai::types::UserMessage::text(&format!("<instruction>{msg}</instruction>"))
    ///     ))
    /// });
    /// ```
    pub fn register_custom_message_formatter(
        &self,
        custom_type: impl Into<String>,
        handler: impl Fn(&str, &serde_json::Value) -> Option<ameli_ai::types::Message>
            + Send
            + Sync
            + 'static,
    ) {
        self.runner
            .add_custom_message_formatter(custom_type.into(), Arc::new(handler));
    }

    // -----------------------------------------------------------------------
    // Session entry persistence
    // -----------------------------------------------------------------------

    /// Append a custom entry to the session tree.
    ///
    /// The entry is persisted via the session manager and does **not** appear
    /// in LLM context. Extensions can call this from event handlers or tool
    /// execution by cloning the `Arc<ExtensionApi>` into closures.
    ///
    /// Returns a boxed future that resolves to the ID of the created entry.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError`](crate::session_manager::SessionError) on
    /// storage failure. The error is also reported to registered error
    /// listeners.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// struct MyExtension;
    ///
    /// impl Extension for MyExtension {
    ///     fn init(&self, api: &Arc<ExtensionApi>) {
    ///         let api = api.clone();
    ///         // Store `api` in a tool or use from a handler:
    ///         let entry_id = api.append_custom_entry("my_type", Some(json!({"key": "value"}))).await?;
    ///     }
    /// }
    /// ```
    pub fn append_custom_entry(
        &self,
        custom_type: &str,
        data: Option<serde_json::Value>,
    ) -> crate::session_manager::manager::AsyncResult<String> {
        let custom_type = custom_type.to_string();
        let actions = self.actions.clone();
        let runner = self.runner.clone();
        Box::pin(async move {
            let result = actions.append_custom_entry(&custom_type, data).await;
            if let Err(ref e) = result {
                runner.report_error(ExtensionError {
                    event: "append_custom_entry".to_string(),
                    error: e.to_string(),
                });
            }
            result
        })
    }

    // -----------------------------------------------------------------------
    // Message injection
    // -----------------------------------------------------------------------

    /// Send a user message to the agent's processing queue.
    ///
    /// Builds a [`UserMessage`](ameli_ai::types::UserMessage) from the given
    /// content and injects it into the queue determined by `mode`:
    ///
    /// - [`MessageMode::Steer`]: injected after the current assistant turn
    ///   finishes executing its tool calls, before the next LLM call.
    /// - [`MessageMode::FollowUp`]: injected after the agent would otherwise
    ///   stop, potentially triggering a new round of processing.
    ///
    /// Silently does nothing if the agent has been dropped (session torn down).
    pub async fn send_user_message(
        &self,
        content: ameli_ai::types::UserContent,
        mode: MessageMode,
    ) {
        self.actions.send_user_message(content, mode).await;
    }

    /// Send a custom message to the agent's processing queue.
    ///
    /// Builds a [`CustomMessage`](ameli_agent_core::types::CustomMessage) from
    /// the given parameters and injects it into the queue determined by `mode`.
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
        self.actions
            .send_custom_message(custom_type, data, display, details, mode)
            .await;
    }
}

impl std::fmt::Debug for ExtensionApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionApi").finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// init_extensions helper
// ---------------------------------------------------------------------------

/// Initialize a list of extensions using a shared [`ExtensionApi`].
///
/// Calls [`Extension::init`] on each extension with `&Arc<ExtensionApi>`.
/// All registrations are forwarded to the runner's interior-mutable storage.
///
/// # Examples
///
/// ```ignore
/// use ameli_agent::extension::{Extension, ExtensionApi, ExtensionRunner, ExtensionActions, init_extensions};
/// use ameli_agent::interface::NoopInterface;
/// use ameli_agent::session_manager::InMemorySessionManager;
/// use std::sync::Arc;
///
/// struct MyExt;
/// impl Extension for MyExt {
///     fn init(&self, _api: &Arc<ExtensionApi>) {}
/// }
///
/// // Assume runner, actions, and agent are already constructed
/// let api = Arc::new(ExtensionApi::new(runner, actions));
/// let extensions: Vec<Box<dyn Extension>> = vec![Box::new(MyExt)];
/// init_extensions(&api, &extensions);
/// ```
pub fn init_extensions(api: &Arc<ExtensionApi>, extensions: &[Box<dyn Extension>]) {
    for ext in extensions {
        ext.init(api);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interface::NoopInterface;
    use crate::session_manager::manager::SessionManager;
    use ameli_agent_core::types::AgentToolResult;
    use ameli_ai::types::Tool;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    struct BlockBashExtension;

    impl Extension for BlockBashExtension {
        fn init(&self, api: &Arc<ExtensionApi>) {
            api.on_tool_call(|event, _ctx| {
                let tool_name = event.tool_name.clone();
                Box::pin(async move {
                    if tool_name == "bash" {
                        return Some(ToolCallResult::block("bash is blocked"));
                    }
                    None
                })
            });
        }
    }

    struct LoggingExtension;

    impl Extension for LoggingExtension {
        fn init(&self, api: &Arc<ExtensionApi>) {
            api.on_agent_start(|_event, _ctx| Box::pin(async { Ok(()) }));
            api.on_turn_end(|_event, _ctx| Box::pin(async { Ok(()) }));
            api.on_session_start(|_event, _ctx| Box::pin(async { Ok(()) }));
            api.on_session_shutdown(|_event, _ctx| Box::pin(async { Ok(()) }));
        }
    }

    struct BeforeAgentStartExtension;

    impl Extension for BeforeAgentStartExtension {
        fn init(&self, api: &Arc<ExtensionApi>) {
            api.on_before_agent_start(|event, _ctx| {
                let prompt = event.prompt.clone();
                Box::pin(async move {
                    if prompt == "override" {
                        return Some(BeforeAgentStartResult {
                            system_prompt: Some("overridden prompt".into()),
                            message: None,
                        });
                    }
                    None
                })
            });
        }
    }

    struct MessageEndExtension;

    impl Extension for MessageEndExtension {
        fn init(&self, api: &Arc<ExtensionApi>) {
            api.on_message_end(|_event, _ctx| Box::pin(async move { None }));
        }
    }

    struct ToolUpdateExtension;

    impl Extension for ToolUpdateExtension {
        fn init(&self, api: &Arc<ExtensionApi>) {
            api.on_tool_execution_update(|_event, _ctx| Box::pin(async { Ok(()) }));
        }
    }

    struct CommandExtension;

    impl Extension for CommandExtension {
        fn init(&self, api: &Arc<ExtensionApi>) {
            api.register_command(
                "greet",
                Some("Say hello".into()),
                Arc::new(|args, _ctx| {
                    let args = args.to_string();
                    Box::pin(async move {
                        let _ = args;
                        Ok(())
                    })
                }),
            );
        }
    }

    struct EchoTool;

    impl AgentTool for EchoTool {
        fn label(&self) -> &str {
            "Echo"
        }
        fn tool_definition(&self) -> Tool {
            Tool {
                name: "echo".into(),
                description: "Echoes input".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "message": { "type": "string" }
                    }
                }),
            }
        }
        fn prepare_arguments(&self, args: serde_json::Value) -> serde_json::Value {
            args
        }
        fn execution_mode(&self) -> Option<ameli_agent_core::types::ToolExecutionMode> {
            None
        }
        fn fmt_debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("EchoTool").finish()
        }
        fn execute(
            &self,
            _tool_call_id: &str,
            params: serde_json::Value,
            _cancel: Option<CancellationToken>,
        ) -> Pin<Box<dyn Future<Output = AgentToolResult> + Send + '_>> {
            Box::pin(async move {
                let message = params.get("message").and_then(|v| v.as_str()).unwrap_or("");
                AgentToolResult::text(message, serde_json::json!({}))
            })
        }
    }

    fn noop_interface() -> Arc<dyn crate::interface::Interface> {
        Arc::new(NoopInterface)
    }

    fn test_session_manager() -> Arc<crate::session_manager::InMemorySessionManager> {
        Arc::new(crate::session_manager::InMemorySessionManager::new())
    }

    fn test_agent() -> Arc<ameli_agent_core::agent::Agent> {
        use ameli_agent_core::types::{AgentState, ThinkingLevel};
        use ameli_agent_core::AgentOptions;
        use std::collections::HashSet;

        ameli_agent_core::agent::Agent::new_arc(AgentOptions {
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

    fn test_model() -> ameli_ai::types::Model {
        ameli_ai::types::Model {
            id: "test-model".into(),
            name: "Test Model".into(),
            api: "test-api".into(),
            provider: "test-provider".into(),
            base_url: "http://localhost".into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![ameli_ai::types::InputType::Text],
            cost: ameli_ai::types::Cost::default(),
            context_window: 128_000,
            max_tokens: 16_384,
            compat: None,
        }
    }

    fn test_actions_with_agent(
        agent: &Arc<ameli_agent_core::agent::Agent>,
    ) -> Arc<ExtensionActions> {
        Arc::new(ExtensionActions::new(test_session_manager(), agent))
    }

    #[test]
    fn register_tool() {
        let agent = test_agent();
        let runner = Arc::new(ExtensionRunner::empty(noop_interface()));
        let actions = test_actions_with_agent(&agent);
        let api = Arc::new(ExtensionApi::new(runner.clone(), actions));
        api.register_tool(Arc::new(EchoTool));
        let tools = runner.get_registered_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(
            tools
                .first()
                .unwrap_or_else(|| panic!("expected at least one tool"))
                .name(),
            "echo"
        );
    }

    #[test]
    fn init_extensions_collects_registrations() {
        let agent = test_agent();
        let extensions: Vec<Box<dyn Extension>> =
            vec![Box::new(BlockBashExtension), Box::new(LoggingExtension)];
        let runner = Arc::new(ExtensionRunner::empty(noop_interface()));
        let actions = test_actions_with_agent(&agent);
        let api = Arc::new(ExtensionApi::new(runner.clone(), actions));
        init_extensions(&api, &extensions);
        assert!(runner.has_tool_call_handlers());
        assert!(runner.has_any_handlers());
    }

    #[test]
    fn register_command() {
        let agent = test_agent();
        let runner = Arc::new(ExtensionRunner::empty(noop_interface()));
        let actions = test_actions_with_agent(&agent);
        let api = Arc::new(ExtensionApi::new(runner.clone(), actions));
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(CommandExtension)];
        init_extensions(&api, &extensions);
        let commands = runner.get_registered_commands();
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "greet");
        assert_eq!(commands[0].description.as_deref(), Some("Say hello"));
    }

    #[test]
    fn register_before_agent_start() {
        let agent = test_agent();
        let runner = Arc::new(ExtensionRunner::empty(noop_interface()));
        let actions = test_actions_with_agent(&agent);
        let api = Arc::new(ExtensionApi::new(runner.clone(), actions));
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(BeforeAgentStartExtension)];
        init_extensions(&api, &extensions);
        assert!(runner.has_before_agent_start_handlers());
    }

    #[test]
    fn register_tool_execution_update() {
        let agent = test_agent();
        let runner = Arc::new(ExtensionRunner::empty(noop_interface()));
        let actions = test_actions_with_agent(&agent);
        let api = Arc::new(ExtensionApi::new(runner.clone(), actions));
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(ToolUpdateExtension)];
        init_extensions(&api, &extensions);
        assert!(runner.has_tool_execution_update_handlers());
    }

    #[test]
    fn register_message_end_hook() {
        let agent = test_agent();
        let runner = Arc::new(ExtensionRunner::empty(noop_interface()));
        let actions = test_actions_with_agent(&agent);
        let api = Arc::new(ExtensionApi::new(runner.clone(), actions));
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(MessageEndExtension)];
        init_extensions(&api, &extensions);
        assert!(runner.has_message_end_handlers());
    }

    #[test]
    fn has_agent_start_handler() {
        let agent = test_agent();
        let runner = Arc::new(ExtensionRunner::empty(noop_interface()));
        let actions = test_actions_with_agent(&agent);
        let api = Arc::new(ExtensionApi::new(runner.clone(), actions));
        assert!(!runner.has_agent_start_handlers());
        api.on_agent_start(|_, _| Box::pin(async { Ok(()) }));
        assert!(runner.has_agent_start_handlers());
    }

    #[test]
    fn has_tool_call_handler() {
        let agent = test_agent();
        let runner = Arc::new(ExtensionRunner::empty(noop_interface()));
        let actions = test_actions_with_agent(&agent);
        let api = Arc::new(ExtensionApi::new(runner.clone(), actions));
        assert!(!runner.has_tool_call_handlers());
        api.on_tool_call(|_, _| Box::pin(async { None }));
        assert!(runner.has_tool_call_handlers());
    }

    #[tokio::test]
    async fn tool_call_handler_blocks() {
        let agent = test_agent();
        let runner = Arc::new(ExtensionRunner::empty(noop_interface()));
        let actions = test_actions_with_agent(&agent);
        let api = Arc::new(ExtensionApi::new(runner.clone(), actions));
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(BlockBashExtension)];
        init_extensions(&api, &extensions);

        let event = ToolCallEvent {
            tool_call_id: "tc_1".into(),
            tool_name: "bash".into(),
            args: serde_json::json!({"command": "rm -rf /"}),
        };
        let ctx = ExtensionContext::for_testing();

        let result = runner.emit_tool_call_for_test(event, ctx).await;
        let Some(result) = result else {
            panic!("tool_call handler for 'bash' should return Some");
        };
        assert!(result.block);
        assert_eq!(result.reason.as_deref(), Some("bash is blocked"));
    }

    #[tokio::test]
    async fn tool_call_handler_allows_other_tools() {
        let agent = test_agent();
        let runner = Arc::new(ExtensionRunner::empty(noop_interface()));
        let actions = test_actions_with_agent(&agent);
        let api = Arc::new(ExtensionApi::new(runner.clone(), actions));
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(BlockBashExtension)];
        init_extensions(&api, &extensions);

        let event = ToolCallEvent {
            tool_call_id: "tc_2".into(),
            tool_name: "read".into(),
            args: serde_json::json!({"path": "/etc/hosts"}),
        };
        let ctx = ExtensionContext::for_testing();

        let result = runner.emit_tool_call_for_test(event, ctx).await;
        assert!(result.is_none());
    }

    #[test]
    fn register_multiple_tools() {
        let agent = test_agent();
        let runner = Arc::new(ExtensionRunner::empty(noop_interface()));
        let actions = test_actions_with_agent(&agent);
        let api = Arc::new(ExtensionApi::new(runner.clone(), actions));
        api.register_tool(Arc::new(EchoTool));
        api.register_tool(Arc::new(EchoTool));
        assert_eq!(runner.get_registered_tools().len(), 2);
    }

    #[test]
    fn init_extensions_with_no_extensions() {
        let agent = test_agent();
        let runner = Arc::new(ExtensionRunner::empty(noop_interface()));
        let actions = test_actions_with_agent(&agent);
        let api = Arc::new(ExtensionApi::new(runner.clone(), actions));
        let extensions: Vec<Box<dyn Extension>> = vec![];
        init_extensions(&api, &extensions);
        assert!(!runner.has_any_handlers());
    }

    #[test]
    fn extension_can_clone_api() {
        let agent = test_agent();
        let runner = Arc::new(ExtensionRunner::empty(noop_interface()));
        let actions = test_actions_with_agent(&agent);
        let api = Arc::new(ExtensionApi::new(runner.clone(), actions));
        let cloned = api.clone();
        cloned.register_tool(Arc::new(EchoTool));
        assert_eq!(runner.get_registered_tools().len(), 1);
    }

    // -- append_custom_entry tests ------------------------------------------

    #[tokio::test]
    async fn append_custom_entry_via_api() {
        let sm = test_session_manager();
        let agent = test_agent();
        let runner = Arc::new(ExtensionRunner::empty(noop_interface()));
        let actions = Arc::new(ExtensionActions::new(sm.clone(), &agent));
        let api = Arc::new(ExtensionApi::new(runner.clone(), actions));

        let entry_id = api
            .append_custom_entry("test_type", Some(serde_json::json!({"key": 42})))
            .await
            .unwrap();

        assert!(!entry_id.is_empty());

        // Verify the entry was persisted
        let entries = sm.entries().await.unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[tokio::test]
    async fn append_custom_entry_from_extension_handler() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let sm = test_session_manager();
        let agent = test_agent();
        let runner = Arc::new(ExtensionRunner::empty(noop_interface()));
        let actions = Arc::new(ExtensionActions::new(sm.clone(), &agent));
        let api = Arc::new(ExtensionApi::new(runner.clone(), actions));

        let entry_count = Arc::new(AtomicUsize::new(0));

        // Register a handler that appends a custom entry via the API
        let api_clone = api.clone();
        let count = entry_count.clone();
        runner.add_agent_start_handler(Arc::new(move |_event, _ctx| {
            let api = api_clone.clone();
            let count = count.clone();
            Box::pin(async move {
                let result = api
                    .append_custom_entry("from_handler", Some(serde_json::json!({"worked": true})))
                    .await;
                if result.is_ok() {
                    count.fetch_add(1, Ordering::SeqCst);
                }
                Ok(())
            })
        }));

        // Dispatch agent_start to trigger the handler
        runner
            .dispatch_agent_event(
                ameli_agent_core::types::AgentEvent::AgentStart,
                tokio_util::sync::CancellationToken::new(),
            )
            .await;

        assert_eq!(entry_count.load(Ordering::SeqCst), 1);

        // Verify the entry was persisted
        let entries = sm.entries().await.unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0] {
            crate::session_manager::types::SessionEntry::Custom(ce) => {
                assert_eq!(ce.custom_type, "from_handler");
            }
            other => panic!("Expected Custom entry, got {other:?}"),
        }
    }
}
