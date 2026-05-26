//! Extension system for ameli-agent.
//!
//! This module defines the core extension API — the trait extensions implement,
//! the event types they subscribe to, and the registration surface for tools
//! and hooks.
//!
//! # Architecture
//!
//! ```text
//! Extension trait     →  impl Extension for MyExt { fn init(&self, api) }
//!                            ↓
//! ExtensionApi        →  api.on_tool_call(handler), api.register_tool(tool)
//!                            ↓
//! ExtensionRunner     →  (future) wires handlers to ArcAgent + AgentLoopConfig
//! ```
//!
//! The first pass provides type definitions only. The `ExtensionRunner` that
//! bridges to `ameli-agent-core` will be implemented in a subsequent step.
//!
//! # Extension lifecycle
//!
//! Extensions implement [`Extension`] and receive an [`ExtensionApi`] during
//! [`init`](Extension::init). They call typed registration methods to subscribe
//! to events and register tools. Adding a new event type is non-breaking — it
//! just adds a new method on `ExtensionApi`.
//!
//! # Events
//!
//! Two categories:
//!
//! - **Notification events** — observe but don't modify (e.g., `on_agent_start`).
//! - **Hook events** — can return a result to influence agent behaviour
//!   (e.g., `on_tool_call` can block execution).
//!
//! # Design note
//!
//! This module is inspired by pi's extension system but deliberately minimal
//! for the headless first pass. UI-specific extensions (commands, shortcuts,
//! flags, rendering) and session/model/provider events are deferred to future
//! work.

pub mod context;
pub mod events;

pub use context::ExtensionContext;
pub use events::*;

use ameli_agent_core::types::AgentTool;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Pinned, boxed, sendable future returned by extension handlers.
type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

// Handler function types for notification events.
type AgentStartHandler =
    Box<dyn Fn(AgentStartEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync>;
type AgentEndHandler = Box<dyn Fn(AgentEndEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync>;
type TurnStartHandler =
    Box<dyn Fn(TurnStartEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync>;
type TurnEndHandler = Box<dyn Fn(TurnEndEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync>;
type MessageStartHandler =
    Box<dyn Fn(MessageStartEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync>;
type MessageUpdateHandler =
    Box<dyn Fn(MessageUpdateEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync>;
type MessageEndHandler =
    Box<dyn Fn(MessageEndEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync>;
type ToolExecutionStartHandler =
    Box<dyn Fn(ToolExecutionStartEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync>;
type ToolExecutionEndHandler =
    Box<dyn Fn(ToolExecutionEndEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync>;

// Handler function types for hook events.
type ToolCallHandler =
    Box<dyn Fn(ToolCallEvent, ExtensionContext) -> BoxFuture<Option<ToolCallResult>> + Send + Sync>;
type ToolResultHandler = Box<
    dyn Fn(ToolResultEvent, ExtensionContext) -> BoxFuture<Option<ToolResultPatch>> + Send + Sync,
>;
type ContextHandler =
    Box<dyn Fn(ContextEvent, ExtensionContext) -> BoxFuture<Option<ContextResult>> + Send + Sync>;

// ---------------------------------------------------------------------------
// Extension trait
// ---------------------------------------------------------------------------

/// Trait for ameli-agent extensions.
///
/// Extensions implement this trait, call typed registration methods on
/// [`ExtensionApi`] during [`init`](Extension::init), and the runtime wires
/// them to the agent loop.
///
/// # Examples
///
/// ```
/// use ameli_agent::extension::{Extension, ExtensionApi};
///
/// struct LoggingExtension;
///
/// impl Extension for LoggingExtension {
///     fn name(&self) -> &str { "logging" }
///
///     fn init(&self, api: &mut ExtensionApi) {
///         api.on_agent_start(|_event, _ctx| {
///             Box::pin(async move {
///                 println!("Agent started");
///             })
///         });
///     }
/// }
/// ```
pub trait Extension: Send + Sync {
    /// Stable name for this extension (used for logging and diagnostics).
    fn name(&self) -> &str;

    /// Called once during extension registration.
    ///
    /// Use `api` to subscribe to events and register tools.
    fn init(&self, api: &mut ExtensionApi);
}

// ---------------------------------------------------------------------------
// ExtensionApi
// ---------------------------------------------------------------------------

/// Registration surface passed to extensions during [`Extension::init`].
///
/// Extensions call typed `on_xxx()` methods to subscribe to events and
/// `register_tool()` to add LLM-callable tools. The future
/// `ExtensionRunner` extracts these registrations and wires them to the
/// agent loop.
///
/// # Handler contract
///
/// Handlers must not panic. If a handler fails, it should return a safe
/// fallback value (e.g., `None` for hooks, or simply return for
/// notifications). The runtime logs handler errors and continues.
pub struct ExtensionApi {
    // Notification handlers
    agent_start_handlers: Vec<AgentStartHandler>,
    agent_end_handlers: Vec<AgentEndHandler>,
    turn_start_handlers: Vec<TurnStartHandler>,
    turn_end_handlers: Vec<TurnEndHandler>,
    message_start_handlers: Vec<MessageStartHandler>,
    message_update_handlers: Vec<MessageUpdateHandler>,
    message_end_handlers: Vec<MessageEndHandler>,
    tool_execution_start_handlers: Vec<ToolExecutionStartHandler>,
    tool_execution_end_handlers: Vec<ToolExecutionEndHandler>,

    // Hook handlers
    tool_call_handlers: Vec<ToolCallHandler>,
    tool_result_handlers: Vec<ToolResultHandler>,
    context_handlers: Vec<ContextHandler>,

    // Registered tools
    tools: Vec<Arc<dyn AgentTool>>,
}

impl ExtensionApi {
    /// Create a new, empty API surface.
    pub fn new() -> Self {
        Self {
            agent_start_handlers: Vec::new(),
            agent_end_handlers: Vec::new(),
            turn_start_handlers: Vec::new(),
            turn_end_handlers: Vec::new(),
            message_start_handlers: Vec::new(),
            message_update_handlers: Vec::new(),
            message_end_handlers: Vec::new(),
            tool_execution_start_handlers: Vec::new(),
            tool_execution_end_handlers: Vec::new(),
            tool_call_handlers: Vec::new(),
            tool_result_handlers: Vec::new(),
            context_handlers: Vec::new(),
            tools: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Notification event registration
    // -----------------------------------------------------------------------

    /// Subscribe to agent loop start.
    pub fn on_agent_start(
        &mut self,
        handler: impl Fn(AgentStartEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync + 'static,
    ) {
        self.agent_start_handlers.push(Box::new(handler));
    }

    /// Subscribe to agent loop end.
    pub fn on_agent_end(
        &mut self,
        handler: impl Fn(AgentEndEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync + 'static,
    ) {
        self.agent_end_handlers.push(Box::new(handler));
    }

    /// Subscribe to turn start.
    pub fn on_turn_start(
        &mut self,
        handler: impl Fn(TurnStartEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync + 'static,
    ) {
        self.turn_start_handlers.push(Box::new(handler));
    }

    /// Subscribe to turn end.
    pub fn on_turn_end(
        &mut self,
        handler: impl Fn(TurnEndEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync + 'static,
    ) {
        self.turn_end_handlers.push(Box::new(handler));
    }

    /// Subscribe to message start (user, assistant, or tool result).
    pub fn on_message_start(
        &mut self,
        handler: impl Fn(MessageStartEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync + 'static,
    ) {
        self.message_start_handlers.push(Box::new(handler));
    }

    /// Subscribe to message streaming updates (assistant messages only).
    pub fn on_message_update(
        &mut self,
        handler: impl Fn(MessageUpdateEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync + 'static,
    ) {
        self.message_update_handlers.push(Box::new(handler));
    }

    /// Subscribe to message end (user, assistant, or tool result).
    pub fn on_message_end(
        &mut self,
        handler: impl Fn(MessageEndEvent, ExtensionContext) -> BoxFuture<()> + Send + Sync + 'static,
    ) {
        self.message_end_handlers.push(Box::new(handler));
    }

    /// Subscribe to tool execution start.
    pub fn on_tool_execution_start(
        &mut self,
        handler: impl Fn(ToolExecutionStartEvent, ExtensionContext) -> BoxFuture<()>
            + Send
            + Sync
            + 'static,
    ) {
        self.tool_execution_start_handlers.push(Box::new(handler));
    }

    /// Subscribe to tool execution end.
    pub fn on_tool_execution_end(
        &mut self,
        handler: impl Fn(ToolExecutionEndEvent, ExtensionContext) -> BoxFuture<()>
            + Send
            + Sync
            + 'static,
    ) {
        self.tool_execution_end_handlers.push(Box::new(handler));
    }

    // -----------------------------------------------------------------------
    // Hook event registration
    // -----------------------------------------------------------------------

    /// Register a hook called before a tool executes.
    ///
    /// Handlers run in registration order. If any handler returns
    /// `Some(ToolCallResult { block: true, .. })`, the tool is blocked.
    pub fn on_tool_call(
        &mut self,
        handler: impl Fn(ToolCallEvent, ExtensionContext) -> BoxFuture<Option<ToolCallResult>>
            + Send
            + Sync
            + 'static,
    ) {
        self.tool_call_handlers.push(Box::new(handler));
    }

    /// Register a hook called after a tool finishes executing.
    ///
    /// Handlers run in registration order. Each handler sees the result after
    /// previous handler changes. Return `Some(ToolResultPatch)` to override
    /// parts of the result.
    pub fn on_tool_result(
        &mut self,
        handler: impl Fn(ToolResultEvent, ExtensionContext) -> BoxFuture<Option<ToolResultPatch>>
            + Send
            + Sync
            + 'static,
    ) {
        self.tool_result_handlers.push(Box::new(handler));
    }

    /// Register a hook called before each LLM call to modify the context.
    ///
    /// Handlers run in registration order. If any handler returns
    /// `Some(ContextResult)`, the messages are replaced for subsequent
    /// handlers and the LLM call.
    pub fn on_context(
        &mut self,
        handler: impl Fn(ContextEvent, ExtensionContext) -> BoxFuture<Option<ContextResult>>
            + Send
            + Sync
            + 'static,
    ) {
        self.context_handlers.push(Box::new(handler));
    }

    // -----------------------------------------------------------------------
    // Tool registration
    // -----------------------------------------------------------------------

    /// Register an LLM-callable tool.
    ///
    /// The tool will be available for the model to invoke during agent runs.
    pub fn register_tool(&mut self, tool: Arc<dyn AgentTool>) {
        self.tools.push(tool);
    }

    // -----------------------------------------------------------------------
    // Accessors (used by the future ExtensionRunner)
    // -----------------------------------------------------------------------

    /// Returns `true` if any notification or hook handlers have been registered.
    pub fn has_handlers(&self) -> bool {
        !self.agent_start_handlers.is_empty()
            || !self.agent_end_handlers.is_empty()
            || !self.turn_start_handlers.is_empty()
            || !self.turn_end_handlers.is_empty()
            || !self.message_start_handlers.is_empty()
            || !self.message_update_handlers.is_empty()
            || !self.message_end_handlers.is_empty()
            || !self.tool_execution_start_handlers.is_empty()
            || !self.tool_execution_end_handlers.is_empty()
            || !self.tool_call_handlers.is_empty()
            || !self.tool_result_handlers.is_empty()
            || !self.context_handlers.is_empty()
    }

    /// Returns `true` if any handlers are registered for the given event type.
    pub fn has_handlers_for(&self, event_type: &ExtensionEvent) -> bool {
        match event_type {
            ExtensionEvent::AgentStart(_) => !self.agent_start_handlers.is_empty(),
            ExtensionEvent::AgentEnd(_) => !self.agent_end_handlers.is_empty(),
            ExtensionEvent::TurnStart(_) => !self.turn_start_handlers.is_empty(),
            ExtensionEvent::TurnEnd(_) => !self.turn_end_handlers.is_empty(),
            ExtensionEvent::MessageStart(_) => !self.message_start_handlers.is_empty(),
            ExtensionEvent::MessageUpdate(_) => !self.message_update_handlers.is_empty(),
            ExtensionEvent::MessageEnd(_) => !self.message_end_handlers.is_empty(),
            ExtensionEvent::ToolExecutionStart(_) => !self.tool_execution_start_handlers.is_empty(),
            ExtensionEvent::ToolExecutionEnd(_) => !self.tool_execution_end_handlers.is_empty(),
            ExtensionEvent::ToolCall(_) => !self.tool_call_handlers.is_empty(),
            ExtensionEvent::ToolResult(_) => !self.tool_result_handlers.is_empty(),
            ExtensionEvent::Context(_) => !self.context_handlers.is_empty(),
        }
    }

    /// Take all registered notification handlers, leaving empty vectors.
    ///
    /// Used by the runner to extract handlers after all extensions have been
    /// initialized.
    pub fn into_handlers(self) -> ExtensionHandlers {
        ExtensionHandlers {
            agent_start_handlers: self.agent_start_handlers,
            agent_end_handlers: self.agent_end_handlers,
            turn_start_handlers: self.turn_start_handlers,
            turn_end_handlers: self.turn_end_handlers,
            message_start_handlers: self.message_start_handlers,
            message_update_handlers: self.message_update_handlers,
            message_end_handlers: self.message_end_handlers,
            tool_execution_start_handlers: self.tool_execution_start_handlers,
            tool_execution_end_handlers: self.tool_execution_end_handlers,
            tool_call_handlers: self.tool_call_handlers,
            tool_result_handlers: self.tool_result_handlers,
            context_handlers: self.context_handlers,
            tools: self.tools,
        }
    }
}

impl Default for ExtensionApi {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ExtensionHandlers — extracted handlers from all extensions
// ---------------------------------------------------------------------------

/// All handlers and tools extracted from extensions after initialization.
///
/// The `ExtensionRunner` will consume this to wire handlers into the agent
/// loop. Produced by [`ExtensionApi::into_handlers`].
pub struct ExtensionHandlers {
    // Notification
    pub agent_start_handlers: Vec<AgentStartHandler>,
    pub agent_end_handlers: Vec<AgentEndHandler>,
    pub turn_start_handlers: Vec<TurnStartHandler>,
    pub turn_end_handlers: Vec<TurnEndHandler>,
    pub message_start_handlers: Vec<MessageStartHandler>,
    pub message_update_handlers: Vec<MessageUpdateHandler>,
    pub message_end_handlers: Vec<MessageEndHandler>,
    pub tool_execution_start_handlers: Vec<ToolExecutionStartHandler>,
    pub tool_execution_end_handlers: Vec<ToolExecutionEndHandler>,

    // Hooks
    pub tool_call_handlers: Vec<ToolCallHandler>,
    pub tool_result_handlers: Vec<ToolResultHandler>,
    pub context_handlers: Vec<ContextHandler>,

    // Tools
    pub tools: Vec<Arc<dyn AgentTool>>,
}

impl ExtensionHandlers {
    /// Returns `true` if no handlers or tools were registered.
    pub fn is_empty(&self) -> bool {
        self.agent_start_handlers.is_empty()
            && self.agent_end_handlers.is_empty()
            && self.turn_start_handlers.is_empty()
            && self.turn_end_handlers.is_empty()
            && self.message_start_handlers.is_empty()
            && self.message_update_handlers.is_empty()
            && self.message_end_handlers.is_empty()
            && self.tool_execution_start_handlers.is_empty()
            && self.tool_execution_end_handlers.is_empty()
            && self.tool_call_handlers.is_empty()
            && self.tool_result_handlers.is_empty()
            && self.context_handlers.is_empty()
            && self.tools.is_empty()
    }
}

// ---------------------------------------------------------------------------
// init_extensions helper
// ---------------------------------------------------------------------------

/// Initialize a list of extensions and collect all registrations.
///
/// Calls [`Extension::init`] on each extension with a fresh [`ExtensionApi`],
/// then returns the extracted [`ExtensionHandlers`].
///
/// # Examples
///
/// ```
/// use ameli_agent::extension::{Extension, ExtensionApi, init_extensions};
///
/// struct MyExt;
/// impl Extension for MyExt {
///     fn name(&self) -> &str { "my-ext" }
///     fn init(&self, _api: &mut ExtensionApi) {}
/// }
///
/// let extensions: Vec<Box<dyn Extension>> = vec![Box::new(MyExt)];
/// let handlers = init_extensions(&extensions);
/// ```
pub fn init_extensions(extensions: &[Box<dyn Extension>]) -> ExtensionHandlers {
    let mut api = ExtensionApi::new();
    for ext in extensions {
        ext.init(&mut api);
    }
    api.into_handlers()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ameli_agent_core::types::AgentToolResult;
    use ameli_ai::types::Tool;

    /// A minimal Extension that registers a tool_call hook.
    struct BlockBashExtension;

    impl Extension for BlockBashExtension {
        fn name(&self) -> &str {
            "block-bash"
        }

        fn init(&self, api: &mut ExtensionApi) {
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

    /// A minimal Extension that registers a notification handler.
    struct LoggingExtension;

    impl Extension for LoggingExtension {
        fn name(&self) -> &str {
            "logging"
        }

        fn init(&self, api: &mut ExtensionApi) {
            api.on_agent_start(|_event, _ctx| {
                Box::pin(async move {
                    // Log agent start
                })
            });
            api.on_turn_end(|_event, _ctx| {
                Box::pin(async move {
                    // Log turn end
                })
            });
        }
    }

    /// A minimal tool for testing register_tool.
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
            _cancel: Option<tokio_util::sync::CancellationToken>,
        ) -> Pin<Box<dyn Future<Output = AgentToolResult> + Send + '_>> {
            Box::pin(async move {
                AgentToolResult::text(
                    params["message"].as_str().unwrap_or(""),
                    serde_json::json!({}),
                )
            })
        }
    }

    #[test]
    fn api_starts_empty() {
        let api = ExtensionApi::new();
        assert!(!api.has_handlers());
        assert!(api.into_handlers().is_empty());
    }

    #[test]
    fn register_tool() {
        let mut api = ExtensionApi::new();
        api.register_tool(Arc::new(EchoTool));
        let handlers = api.into_handlers();
        assert_eq!(handlers.tools.len(), 1);
        assert_eq!(handlers.tools[0].name(), "echo");
    }

    #[test]
    fn init_extensions_collects_registrations() {
        let extensions: Vec<Box<dyn Extension>> =
            vec![Box::new(BlockBashExtension), Box::new(LoggingExtension)];
        let handlers = init_extensions(&extensions);
        assert_eq!(handlers.tool_call_handlers.len(), 1);
        assert_eq!(handlers.agent_start_handlers.len(), 1);
        assert_eq!(handlers.turn_end_handlers.len(), 1);
        assert!(!handlers.is_empty());
    }

    #[test]
    fn has_handlers_for_checks_specific_event() {
        let mut api = ExtensionApi::new();
        assert!(!api.has_handlers_for(&ExtensionEvent::AgentStart(AgentStartEvent)));
        assert!(
            !api.has_handlers_for(&ExtensionEvent::ToolCall(ToolCallEvent {
                tool_call_id: String::new(),
                tool_name: String::new(),
                args: serde_json::Value::Null,
            }))
        );

        api.on_agent_start(|_, _| Box::pin(async {}));
        assert!(api.has_handlers_for(&ExtensionEvent::AgentStart(AgentStartEvent)));
        assert!(
            !api.has_handlers_for(&ExtensionEvent::AgentEnd(AgentEndEvent {
                messages: vec![]
            }))
        );
    }

    #[tokio::test]
    async fn tool_call_handler_blocks() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(BlockBashExtension)];
        let handlers = init_extensions(&extensions);

        let event = ToolCallEvent {
            tool_call_id: "tc_1".into(),
            tool_name: "bash".into(),
            args: serde_json::json!({"command": "rm -rf /"}),
        };
        let ctx = ExtensionContext::for_testing();

        let result = (handlers.tool_call_handlers[0])(event, ctx).await;
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.block);
        assert_eq!(result.reason.as_deref(), Some("bash is blocked"));
    }

    #[tokio::test]
    async fn tool_call_handler_allows_other_tools() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(BlockBashExtension)];
        let handlers = init_extensions(&extensions);

        let event = ToolCallEvent {
            tool_call_id: "tc_2".into(),
            tool_name: "read".into(),
            args: serde_json::json!({"path": "/etc/hosts"}),
        };
        let ctx = ExtensionContext::for_testing();

        let result = (handlers.tool_call_handlers[0])(event, ctx).await;
        assert!(result.is_none());
    }

    #[test]
    fn register_multiple_tools() {
        let mut api = ExtensionApi::new();
        api.register_tool(Arc::new(EchoTool));
        api.register_tool(Arc::new(EchoTool));
        let handlers = api.into_handlers();
        assert_eq!(handlers.tools.len(), 2);
    }

    #[test]
    fn init_extensions_with_no_extensions() {
        let extensions: Vec<Box<dyn Extension>> = vec![];
        let handlers = init_extensions(&extensions);
        assert!(handlers.is_empty());
    }
}
