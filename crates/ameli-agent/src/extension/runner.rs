//! Extension runner — bridges extension handlers to the agent loop.
//!
//! [`ExtensionRunner`] is the runtime that connects extension event handlers
//! to the agent via hook closures installed into [`AgentOptions`].
//! [`AgentSession`](crate::AgentSession) handles event subscription and
//! dispatches to the runner's emit methods.
//!
//! # Lifecycle
//!
//! 1. Create an empty runner with [`ExtensionRunner::empty`].
//! 2. Create an [`ExtensionApi`](super::ExtensionApi) wrapping `Arc<ExtensionRunner>`.
//! 3. Initialize extensions — they register handlers via the API.
//! 4. Call [`ExtensionRunner::install_hooks`] to install hook closures into
//!    [`AgentOptions`](ameli_agent_core::AgentOptions).
//! 5. Construct an [`AgentSession`](crate::AgentSession) (or an `ArcAgent`
//!    from those options) to handle event subscription and persistence.
//!
//! # Error handling
//!
//! Notification handlers return `anyhow::Result<()>`. Errors are caught by
//! the runner, reported to registered error listeners, and dispatch continues
//! to subsequent handlers — errors never stop notification dispatch.
//!
//! Hook handlers return `Option<ResultType>`. If a hook handler panics, it
//! propagates.
//!
//! The error listener infrastructure ([`ExtensionRunner::on_error`]) is
//! provided for structured error reporting.

use crate::extension::context::ExtensionContext;
use crate::extension::events::*;
use crate::extension::Extension;
use crate::interface::Interface;
use ameli_agent_core::types::{
    AfterToolCallContext, AfterToolCallResult, AgentMessage, AgentTool, BeforeToolCallContext,
    BeforeToolCallResult,
};

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Handler function type aliases
// ---------------------------------------------------------------------------

/// Pinned, boxed, sendable future returned by extension handlers.
type BoxFuture<T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send>>;

// Notification handler types (fire-and-forget).
type AgentStartHandler =
    Arc<dyn Fn(AgentStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync>;
type AgentEndHandler =
    Arc<dyn Fn(AgentEndEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync>;
type TurnStartHandler =
    Arc<dyn Fn(TurnStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync>;
type TurnEndHandler =
    Arc<dyn Fn(TurnEndEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync>;
type MessageStartHandler =
    Arc<dyn Fn(MessageStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync>;
type MessageUpdateHandler = Arc<
    dyn Fn(MessageUpdateEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync,
>;
type ToolExecutionStartHandler = Arc<
    dyn Fn(ToolExecutionStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
        + Send
        + Sync,
>;
type ToolExecutionUpdateHandler = Arc<
    dyn Fn(ToolExecutionUpdateEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
        + Send
        + Sync,
>;
type ToolExecutionEndHandler = Arc<
    dyn Fn(ToolExecutionEndEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync,
>;
type SessionStartHandler =
    Arc<dyn Fn(SessionStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync>;
type SessionShutdownHandler = Arc<
    dyn Fn(SessionShutdownEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync,
>;

// Hook handler types.
type ToolCallHandler =
    Arc<dyn Fn(ToolCallEvent, ExtensionContext) -> BoxFuture<Option<ToolCallResult>> + Send + Sync>;
type ToolResultHandler = Arc<
    dyn Fn(ToolResultEvent, ExtensionContext) -> BoxFuture<Option<ToolResultPatch>> + Send + Sync,
>;
type ContextHandler =
    Arc<dyn Fn(ContextEvent, ExtensionContext) -> BoxFuture<Option<ContextResult>> + Send + Sync>;
type BeforeAgentStartHandler = Arc<
    dyn Fn(BeforeAgentStartEvent, ExtensionContext) -> BoxFuture<Option<BeforeAgentStartResult>>
        + Send
        + Sync,
>;
type MessageEndHandler = Arc<
    dyn Fn(MessageEndEvent, ExtensionContext) -> BoxFuture<Option<MessageEndResult>> + Send + Sync,
>;
type FormatCompactionSummaryHandler = Arc<
    dyn Fn(
            FormatCompactionSummaryEvent,
            ExtensionContext,
        ) -> BoxFuture<Option<FormatCompactionSummaryResult>>
        + Send
        + Sync,
>;
type FormatBranchSummaryHandler = Arc<
    dyn Fn(
            FormatBranchSummaryEvent,
            ExtensionContext,
        ) -> BoxFuture<Option<FormatBranchSummaryResult>>
        + Send
        + Sync,
>;

// Custom message formatter handler type.
type CustomMessageFormatterHandler =
    Arc<dyn Fn(&str, &serde_json::Value) -> Option<ameli_ai::types::Message> + Send + Sync>;

// ---------------------------------------------------------------------------
// ExtensionHandlers — interior-mutable handler storage
// ---------------------------------------------------------------------------

/// All handlers and tools registered by extensions.
///
/// Stored behind a `RwLock` inside [`ExtensionRunner`] for interior
/// mutability. Registration (write) and dispatch (read) are lock-protected.
struct ExtensionHandlers {
    // Notification (fire-and-forget)
    agent_start_handlers: Vec<AgentStartHandler>,
    agent_end_handlers: Vec<AgentEndHandler>,
    turn_start_handlers: Vec<TurnStartHandler>,
    turn_end_handlers: Vec<TurnEndHandler>,
    message_start_handlers: Vec<MessageStartHandler>,
    message_update_handlers: Vec<MessageUpdateHandler>,
    tool_execution_start_handlers: Vec<ToolExecutionStartHandler>,
    tool_execution_update_handlers: Vec<ToolExecutionUpdateHandler>,
    tool_execution_end_handlers: Vec<ToolExecutionEndHandler>,
    session_start_handlers: Vec<SessionStartHandler>,
    session_shutdown_handlers: Vec<SessionShutdownHandler>,

    // Hooks
    tool_call_handlers: Vec<ToolCallHandler>,
    tool_result_handlers: Vec<ToolResultHandler>,
    context_handlers: Vec<ContextHandler>,
    before_agent_start_handlers: Vec<BeforeAgentStartHandler>,
    message_end_handlers: Vec<MessageEndHandler>,
    format_compaction_summary_handlers: Vec<FormatCompactionSummaryHandler>,
    format_branch_summary_handlers: Vec<FormatBranchSummaryHandler>,

    // Custom message formatters
    custom_message_formatters: Vec<(String, CustomMessageFormatterHandler)>,

    // Commands
    commands: Vec<RegisteredCommand>,

    // Tools
    tools: Vec<Arc<dyn AgentTool>>,
}

impl ExtensionHandlers {
    fn empty() -> Self {
        Self {
            agent_start_handlers: Vec::new(),
            agent_end_handlers: Vec::new(),
            turn_start_handlers: Vec::new(),
            turn_end_handlers: Vec::new(),
            message_start_handlers: Vec::new(),
            message_update_handlers: Vec::new(),
            tool_execution_start_handlers: Vec::new(),
            tool_execution_update_handlers: Vec::new(),
            tool_execution_end_handlers: Vec::new(),
            session_start_handlers: Vec::new(),
            session_shutdown_handlers: Vec::new(),
            tool_call_handlers: Vec::new(),
            tool_result_handlers: Vec::new(),
            context_handlers: Vec::new(),
            before_agent_start_handlers: Vec::new(),
            message_end_handlers: Vec::new(),
            format_compaction_summary_handlers: Vec::new(),
            format_branch_summary_handlers: Vec::new(),
            custom_message_formatters: Vec::new(),
            commands: Vec::new(),
            tools: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.agent_start_handlers.is_empty()
            && self.agent_end_handlers.is_empty()
            && self.turn_start_handlers.is_empty()
            && self.turn_end_handlers.is_empty()
            && self.message_start_handlers.is_empty()
            && self.message_update_handlers.is_empty()
            && self.message_end_handlers.is_empty()
            && self.tool_execution_start_handlers.is_empty()
            && self.tool_execution_update_handlers.is_empty()
            && self.tool_execution_end_handlers.is_empty()
            && self.session_start_handlers.is_empty()
            && self.session_shutdown_handlers.is_empty()
            && self.tool_call_handlers.is_empty()
            && self.tool_result_handlers.is_empty()
            && self.context_handlers.is_empty()
            && self.before_agent_start_handlers.is_empty()
            && self.format_compaction_summary_handlers.is_empty()
            && self.format_branch_summary_handlers.is_empty()
            && self.custom_message_formatters.is_empty()
            && self.commands.is_empty()
            && self.tools.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ExtensionError
// ---------------------------------------------------------------------------

/// Error reported to error listeners when an extension handler fails.
///
/// Produced when notification handlers return `Err` or when hook handlers
/// encounter errors.
#[derive(Debug, Clone)]
pub struct ExtensionError {
    /// Event type being dispatched when the error occurred.
    pub event: String,
    /// Human-readable error message.
    pub error: String,
}

// ---------------------------------------------------------------------------
// ExtensionErrorListener
// ---------------------------------------------------------------------------

/// Listener function called when an extension handler fails.
///
/// Listeners are called synchronously and should not panic.
pub type ExtensionErrorListener = Arc<dyn Fn(ExtensionError) + Send + Sync>;

// ---------------------------------------------------------------------------
// ExtensionRunner
// ---------------------------------------------------------------------------

/// Runtime that bridges extension handlers to the agent loop.
///
/// Handler storage is behind a `RwLock` for interior mutability —
/// [`ExtensionApi`](super::ExtensionApi) registers handlers via the write
/// lock during init, while dispatch reads through the read lock at runtime.
///
/// Construct via [`ExtensionRunner::empty`] or
/// [`ExtensionRunner::from_extensions`]. Share via `Arc<Self>` so hook
/// closures and the subscriber can both reference it.
pub struct ExtensionRunner {
    /// Handlers registered by extensions, behind `RwLock` for interior mutability.
    handlers: parking_lot::RwLock<ExtensionHandlers>,
    /// Error listeners.
    error_listeners: parking_lot::Mutex<Vec<ExtensionErrorListener>>,
    /// UI interface for creating ExtensionContext.
    interface: Arc<dyn Interface>,
    /// Turn counter for providing turn_index in events.
    turn_index: AtomicU32,
}

impl ExtensionRunner {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create an empty runner with the given interface.
    ///
    /// Handlers are registered later via the [`ExtensionApi`](super::ExtensionApi)
    /// and forwarded to this runner's interior-mutable storage.
    pub fn empty(interface: Arc<dyn Interface>) -> Self {
        Self {
            handlers: parking_lot::RwLock::new(ExtensionHandlers::empty()),
            error_listeners: parking_lot::Mutex::new(Vec::new()),
            interface,
            turn_index: AtomicU32::new(0),
        }
    }

    /// Create a runner by initializing a list of extensions.
    ///
    /// Convenience that creates an empty runner, builds an `ExtensionApi`,
    /// initializes all extensions, and returns the `Arc<ExtensionRunner>`.
    /// Uses [`NoopInterface`](crate::interface::NoopInterface).
    pub fn from_extensions(extensions: &[Box<dyn Extension>]) -> Arc<Self> {
        let runner = Arc::new(Self::empty(Arc::new(crate::interface::NoopInterface)));
        crate::extension::init_extensions(extensions, &runner);
        runner
    }

    /// Create a runner by initializing a list of extensions with a custom
    /// interface.
    pub fn from_extensions_with_interface(
        extensions: &[Box<dyn Extension>],
        interface: Arc<dyn Interface>,
    ) -> Arc<Self> {
        let runner = Arc::new(Self::empty(interface));
        crate::extension::init_extensions(extensions, &runner);
        runner
    }

    // -----------------------------------------------------------------------
    // Registration methods (called by ExtensionApi, pub(crate))
    // -----------------------------------------------------------------------

    pub(crate) fn add_agent_start_handler(&self, handler: AgentStartHandler) {
        let mut h = self.handlers.write();
        h.agent_start_handlers.push(handler);
    }

    pub(crate) fn add_agent_end_handler(&self, handler: AgentEndHandler) {
        let mut h = self.handlers.write();
        h.agent_end_handlers.push(handler);
    }

    pub(crate) fn add_turn_start_handler(&self, handler: TurnStartHandler) {
        let mut h = self.handlers.write();
        h.turn_start_handlers.push(handler);
    }

    pub(crate) fn add_turn_end_handler(&self, handler: TurnEndHandler) {
        let mut h = self.handlers.write();
        h.turn_end_handlers.push(handler);
    }

    pub(crate) fn add_message_start_handler(&self, handler: MessageStartHandler) {
        let mut h = self.handlers.write();
        h.message_start_handlers.push(handler);
    }

    pub(crate) fn add_message_update_handler(&self, handler: MessageUpdateHandler) {
        let mut h = self.handlers.write();
        h.message_update_handlers.push(handler);
    }

    pub(crate) fn add_message_end_handler(&self, handler: MessageEndHandler) {
        let mut h = self.handlers.write();
        h.message_end_handlers.push(handler);
    }

    pub(crate) fn add_tool_execution_start_handler(&self, handler: ToolExecutionStartHandler) {
        let mut h = self.handlers.write();
        h.tool_execution_start_handlers.push(handler);
    }

    pub(crate) fn add_tool_execution_update_handler(&self, handler: ToolExecutionUpdateHandler) {
        let mut h = self.handlers.write();
        h.tool_execution_update_handlers.push(handler);
    }

    pub(crate) fn add_tool_execution_end_handler(&self, handler: ToolExecutionEndHandler) {
        let mut h = self.handlers.write();
        h.tool_execution_end_handlers.push(handler);
    }

    pub(crate) fn add_session_start_handler(&self, handler: SessionStartHandler) {
        let mut h = self.handlers.write();
        h.session_start_handlers.push(handler);
    }

    pub(crate) fn add_session_shutdown_handler(&self, handler: SessionShutdownHandler) {
        let mut h = self.handlers.write();
        h.session_shutdown_handlers.push(handler);
    }

    pub(crate) fn add_tool_call_handler(&self, handler: ToolCallHandler) {
        let mut h = self.handlers.write();
        h.tool_call_handlers.push(handler);
    }

    pub(crate) fn add_tool_result_handler(&self, handler: ToolResultHandler) {
        let mut h = self.handlers.write();
        h.tool_result_handlers.push(handler);
    }

    pub(crate) fn add_context_handler(&self, handler: ContextHandler) {
        let mut h = self.handlers.write();
        h.context_handlers.push(handler);
    }

    pub(crate) fn add_before_agent_start_handler(&self, handler: BeforeAgentStartHandler) {
        let mut h = self.handlers.write();
        h.before_agent_start_handlers.push(handler);
    }

    pub(crate) fn add_format_compaction_summary_handler(
        &self,
        handler: FormatCompactionSummaryHandler,
    ) {
        let mut h = self.handlers.write();
        h.format_compaction_summary_handlers.push(handler);
    }

    pub(crate) fn add_format_branch_summary_handler(&self, handler: FormatBranchSummaryHandler) {
        let mut h = self.handlers.write();
        h.format_branch_summary_handlers.push(handler);
    }

    pub(crate) fn add_command(&self, cmd: RegisteredCommand) {
        let mut h = self.handlers.write();
        h.commands.push(cmd);
    }

    pub(crate) fn add_tool(&self, tool: Arc<dyn AgentTool>) {
        let mut h = self.handlers.write();
        h.tools.push(tool);
    }

    pub(crate) fn add_custom_message_formatter(
        &self,
        custom_type: String,
        handler: CustomMessageFormatterHandler,
    ) {
        let mut h = self.handlers.write();
        h.custom_message_formatters.push((custom_type, handler));
    }

    // -----------------------------------------------------------------------
    // Error listeners
    // -----------------------------------------------------------------------

    /// Register an error listener called when an extension handler fails.
    pub fn on_error(&self, listener: ExtensionErrorListener) {
        let mut listeners = self.error_listeners.lock();
        listeners.push(listener);
    }

    // -----------------------------------------------------------------------
    // Handler queries (read through lock)
    // -----------------------------------------------------------------------

    /// Returns `true` if any extension registered a tool_call hook handler.
    pub fn has_tool_call_handlers(&self) -> bool {
        let h = self.handlers.read();
        !h.tool_call_handlers.is_empty()
    }

    /// Returns `true` if any extension registered a tool_result hook handler.
    pub fn has_tool_result_handlers(&self) -> bool {
        let h = self.handlers.read();
        !h.tool_result_handlers.is_empty()
    }

    /// Returns `true` if any extension registered a context hook handler.
    pub fn has_context_handlers(&self) -> bool {
        let h = self.handlers.read();
        !h.context_handlers.is_empty()
    }

    /// Returns `true` if any extension registered a before_agent_start hook.
    pub fn has_before_agent_start_handlers(&self) -> bool {
        let h = self.handlers.read();
        !h.before_agent_start_handlers.is_empty()
    }

    /// Returns `true` if any extension registered a message_end hook handler.
    pub fn has_message_end_handlers(&self) -> bool {
        let h = self.handlers.read();
        !h.message_end_handlers.is_empty()
    }

    /// Returns `true` if any extension registered a format_compaction_summary
    /// hook handler.
    pub fn has_format_compaction_summary_handlers(&self) -> bool {
        let h = self.handlers.read();
        !h.format_compaction_summary_handlers.is_empty()
    }

    /// Returns `true` if any extension registered a format_branch_summary
    /// hook handler.
    pub fn has_format_branch_summary_handlers(&self) -> bool {
        let h = self.handlers.read();
        !h.format_branch_summary_handlers.is_empty()
    }

    /// Returns `true` if any extension registered an agent_start handler.
    pub fn has_agent_start_handlers(&self) -> bool {
        let h = self.handlers.read();
        !h.agent_start_handlers.is_empty()
    }

    /// Returns `true` if any extension registered a tool_execution_update
    /// handler.
    pub fn has_tool_execution_update_handlers(&self) -> bool {
        let h = self.handlers.read();
        !h.tool_execution_update_handlers.is_empty()
    }

    /// Returns `true` if any extension registered a custom message formatter.
    pub fn has_custom_message_formatters(&self) -> bool {
        let h = self.handlers.read();
        !h.custom_message_formatters.is_empty()
    }

    /// Returns `true` if any handlers are registered for any event type.
    pub fn has_any_handlers(&self) -> bool {
        let h = self.handlers.read();
        !h.is_empty()
    }

    // -----------------------------------------------------------------------
    // Tool and command collection (read through lock)
    // -----------------------------------------------------------------------

    /// Collect all tools registered by extensions.
    pub fn get_registered_tools(&self) -> Vec<Arc<dyn AgentTool>> {
        let h = self.handlers.read();
        h.tools.clone()
    }

    /// Collect all commands registered by extensions.
    pub fn get_registered_commands(&self) -> Vec<RegisteredCommand> {
        let h = self.handlers.read();
        h.commands.clone()
    }

    // -----------------------------------------------------------------------
    // Custom message formatting
    // -----------------------------------------------------------------------

    /// Look up a formatter by `custom_type` and convert `(custom_type, data)`
    /// to an LLM-compatible [`Message`](ameli_ai::types::Message).
    ///
    /// Returns `None` if no formatter is registered for the given type, or
    /// if the formatter returns `None` (skip the message).
    pub fn format_custom_message(
        &self,
        custom_type: &str,
        data: &serde_json::Value,
    ) -> Option<ameli_ai::types::Message> {
        let handlers = self.handlers.read().custom_message_formatters.clone();
        for (registered_type, handler) in &handlers {
            if registered_type == custom_type {
                return handler(custom_type, data);
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Hook installation
    // -----------------------------------------------------------------------

    /// Install hook closures into [`AgentOptions`](ameli_agent_core::AgentOptions)
    /// for tool call interception and context transformation.
    ///
    /// Only installs a hook if there are registered handlers for the
    /// corresponding event type. This avoids unnecessary overhead.
    ///
    /// Call this **before** constructing the [`ArcAgent`](ameli_agent_core::ArcAgent).
    pub fn install_hooks(self: &Arc<Self>, options: &mut ameli_agent_core::AgentOptions) {
        if self.has_tool_call_handlers() {
            let runner = self.clone();
            options.before_tool_call = Some(Arc::new(move |ctx, cancel| {
                let runner = runner.clone();
                let ctx = ctx.clone();
                Box::pin(async move { runner.handle_before_tool_call(&ctx, cancel).await })
            }));
        }

        if self.has_tool_result_handlers() {
            let runner = self.clone();
            options.after_tool_call = Some(Arc::new(move |ctx, cancel| {
                let runner = runner.clone();
                let ctx = ctx.clone();
                Box::pin(async move { runner.handle_after_tool_call(&ctx, cancel).await })
            }));
        }

        if self.has_context_handlers() {
            let runner = self.clone();
            options.transform_context = Some(Arc::new(move |messages, cancel| {
                let runner = runner.clone();
                let messages = messages.to_vec();
                Box::pin(async move { runner.handle_transform_context(&messages, cancel).await })
            }));
        }

        if self.has_custom_message_formatters() {
            let runner = self.clone();
            let original = options.convert_to_llm.clone();
            options.convert_to_llm = Some(Arc::new(move |messages: &[AgentMessage]| {
                let runner = runner.clone();
                let original = original.clone();
                let messages = messages.to_vec();
                Box::pin(async move {
                    runner
                        .handle_convert_to_llm_with_custom_messages(&messages, original.as_ref())
                        .await
                })
            }));
        }
    }

    // -----------------------------------------------------------------------
    // Summary formatting hooks
    // -----------------------------------------------------------------------

    /// Dispatch format_compaction_summary hook to all handlers. First handler
    /// returning `Some` wins. Returns `None` if no handler overrides.
    pub async fn emit_format_compaction_summary(
        &self,
        summary: &str,
        timestamp: u64,
        cancel: CancellationToken,
    ) -> Option<AgentMessage> {
        let ctx = self.make_context(cancel);
        let event = FormatCompactionSummaryEvent {
            summary: summary.to_string(),
            timestamp,
        };
        let handlers = self
            .handlers
            .read()
            .format_compaction_summary_handlers
            .clone();
        for handler in &handlers {
            if let Some(result) = handler(event.clone(), ctx.clone()).await {
                return Some(result.message);
            }
        }
        None
    }

    /// Dispatch format_branch_summary hook to all handlers. First handler
    /// returning `Some` wins. Returns `None` if no handler overrides.
    pub async fn emit_format_branch_summary(
        &self,
        summary: &str,
        timestamp: u64,
        cancel: CancellationToken,
    ) -> Option<AgentMessage> {
        let ctx = self.make_context(cancel);
        let event = FormatBranchSummaryEvent {
            summary: summary.to_string(),
            timestamp,
        };
        let handlers = self.handlers.read().format_branch_summary_handlers.clone();
        for handler in &handlers {
            if let Some(result) = handler(event.clone(), ctx.clone()).await {
                return Some(result.message);
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Session lifecycle hooks
    // -----------------------------------------------------------------------

    /// Dispatch session_start event to all handlers (fire-and-forget).
    pub async fn emit_session_start(&self, reason: SessionStartReason) {
        let ctx = self.make_context(CancellationToken::new());
        let event = SessionStartEvent { reason };
        let handlers = self.handlers.read().session_start_handlers.clone();
        for handler in &handlers {
            if let Err(e) = handler(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    event: "session_start".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    /// Dispatch session_shutdown event to all handlers (fire-and-forget).
    /// Returns `true` if any handlers were registered.
    pub async fn emit_session_shutdown(&self, reason: SessionShutdownReason) -> bool {
        let handlers = self.handlers.read().session_shutdown_handlers.clone();
        if handlers.is_empty() {
            return false;
        }
        let ctx = self.make_context(CancellationToken::new());
        let event = SessionShutdownEvent { reason };
        for handler in &handlers {
            if let Err(e) = handler(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    event: "session_shutdown".to_string(),
                    error: e.to_string(),
                });
            }
        }
        true
    }

    // -----------------------------------------------------------------------
    // Before agent start hook (sequential accumulate)
    // -----------------------------------------------------------------------

    /// Dispatch before_agent_start hook to all handlers.
    ///
    /// All handler results are collected. Custom messages are accumulated in
    /// order. The last non-`None` `system_prompt` wins.
    pub async fn emit_before_agent_start(
        &self,
        prompt: &str,
        images: &[ameli_ai::types::ImageContent],
        audio: &[ameli_ai::types::AudioContent],
        system_prompt: &str,
        cancel: CancellationToken,
    ) -> Option<BeforeAgentStartAccumulated> {
        let handlers = self.handlers.read().before_agent_start_handlers.clone();
        if handlers.is_empty() {
            return None;
        }

        let ctx = self.make_context(cancel);
        let mut current_system_prompt = system_prompt.to_string();
        let mut messages: Vec<BeforeAgentStartMessage> = Vec::new();
        let mut modified = false;

        for handler in &handlers {
            let event = BeforeAgentStartEvent {
                prompt: prompt.to_string(),
                images: images.to_vec(),
                audio: audio.to_vec(),
                system_prompt: current_system_prompt.clone(),
            };
            if let Some(result) = handler(event, ctx.clone()).await {
                if let Some(msg) = result.message {
                    messages.push(msg);
                    modified = true;
                }
                if let Some(sp) = result.system_prompt {
                    current_system_prompt = sp;
                    modified = true;
                }
            }
        }

        if modified {
            Some(BeforeAgentStartAccumulated {
                messages: if messages.is_empty() {
                    None
                } else {
                    Some(messages)
                },
                system_prompt: if current_system_prompt != system_prompt {
                    Some(current_system_prompt)
                } else {
                    None
                },
            })
        } else {
            None
        }
    }

    // -----------------------------------------------------------------------
    // Message end hook (sequential chain)
    // -----------------------------------------------------------------------

    /// Dispatch message_end hook to all handlers.
    ///
    /// Handlers run in order. Each can return a replacement message that
    /// preserves the original role. Returns the final replacement, or `None`
    /// if no handler modified the message.
    pub async fn emit_message_end(
        &self,
        event: MessageEndEvent,
        cancel: CancellationToken,
    ) -> Option<AgentMessage> {
        let handlers = self.handlers.read().message_end_handlers.clone();
        if handlers.is_empty() {
            return None;
        }

        let ctx = self.make_context(cancel);
        let original_role = event.message.role().to_string();
        let mut current_message = event.message;
        let mut modified = false;

        for handler in &handlers {
            let current_event = MessageEndEvent {
                message: current_message.clone(),
            };
            if let Some(result) = handler(current_event, ctx.clone()).await {
                if result.message.role() != original_role {
                    self.report_error(ExtensionError {
                        event: "message_end".to_string(),
                        error: format!(
                            "message_end handlers must return a message with the same role (expected: {original_role}, got: {})",
                            result.message.role()
                        ),
                    });
                    continue;
                }
                current_message = result.message;
                modified = true;
            }
        }

        if modified {
            Some(current_message)
        } else {
            None
        }
    }

    // -----------------------------------------------------------------------
    // Command dispatch
    // -----------------------------------------------------------------------

    /// Execute a registered command by name.
    ///
    /// First registered command with matching name wins.
    pub async fn execute_command(
        &self,
        name: &str,
        args: &str,
        ctx: CommandContext,
    ) -> anyhow::Result<()> {
        let handlers = self.handlers.read().commands.clone();
        for cmd in &handlers {
            if cmd.name == name {
                return (cmd.handler)(args.to_string(), ctx).await;
            }
        }
        anyhow::bail!("no command registered with name: {name}")
    }

    // -----------------------------------------------------------------------
    // Agent event → extension event dispatch
    // -----------------------------------------------------------------------

    /// Map an [`AgentEvent`](ameli_agent_core::types::AgentEvent) to extension
    /// notification events and dispatch to registered handlers.
    pub async fn dispatch_agent_event(
        &self,
        event: ameli_agent_core::types::AgentEvent,
        cancel: CancellationToken,
    ) {
        use ameli_agent_core::types::AgentEvent;

        match event {
            AgentEvent::AgentStart => {
                self.turn_index.store(0, AtomicOrdering::SeqCst);
                self.dispatch_agent_start(cancel).await;
            }
            AgentEvent::AgentEnd { messages } => {
                self.dispatch_agent_end(messages, cancel).await;
            }
            AgentEvent::TurnStart => {
                self.dispatch_turn_start(cancel).await;
            }
            AgentEvent::TurnEnd {
                message,
                tool_results,
            } => {
                self.dispatch_turn_end(message, tool_results, cancel).await;
            }
            AgentEvent::MessageStart { message } => {
                self.dispatch_message_start(message, cancel).await;
            }
            AgentEvent::MessageUpdate {
                message,
                assistant_message_event,
            } => {
                self.dispatch_message_update(message, assistant_message_event, cancel)
                    .await;
            }
            AgentEvent::MessageEnd { message: _ } => {
                // MessageEnd is handled by AgentSession via emit_message_end()
                // (sequential chain with replacement).
            }
            AgentEvent::ToolExecutionStart {
                tool_call_id,
                tool_name,
                args,
            } => {
                self.dispatch_tool_execution_start(tool_call_id, tool_name, args, cancel)
                    .await;
            }
            AgentEvent::ToolExecutionUpdate {
                tool_call_id,
                tool_name,
                args,
                partial_result,
            } => {
                self.dispatch_tool_execution_update(
                    tool_call_id,
                    tool_name,
                    args,
                    partial_result,
                    cancel,
                )
                .await;
            }
            AgentEvent::ToolExecutionEnd {
                tool_call_id,
                tool_name,
                result,
                is_error,
            } => {
                self.dispatch_tool_execution_end(tool_call_id, tool_name, result, is_error, cancel)
                    .await;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Notification dispatchers (one per notification event type)
    // -----------------------------------------------------------------------

    async fn dispatch_agent_start(&self, cancel: CancellationToken) {
        let ctx = self.make_context(cancel);
        let event = AgentStartEvent;
        let handlers = self.handlers.read().agent_start_handlers.clone();
        for handler in &handlers {
            if let Err(e) = handler(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    event: "agent_start".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    async fn dispatch_agent_end(&self, messages: Vec<AgentMessage>, cancel: CancellationToken) {
        let ctx = self.make_context(cancel);
        let event = AgentEndEvent { messages };
        let handlers = self.handlers.read().agent_end_handlers.clone();
        for handler in &handlers {
            if let Err(e) = handler(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    event: "agent_end".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    async fn dispatch_turn_start(&self, cancel: CancellationToken) {
        let turn_index = self.turn_index.load(AtomicOrdering::SeqCst);
        let ctx = self.make_context(cancel);
        let event = TurnStartEvent {
            turn_index,
            timestamp: now_ms(),
        };
        let handlers = self.handlers.read().turn_start_handlers.clone();
        for handler in &handlers {
            if let Err(e) = handler(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    event: "turn_start".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    async fn dispatch_turn_end(
        &self,
        message: AgentMessage,
        tool_results: Vec<ameli_ai::types::ToolResultMessage>,
        cancel: CancellationToken,
    ) {
        let turn_index = self.turn_index.load(AtomicOrdering::SeqCst);
        self.turn_index.fetch_add(1, AtomicOrdering::SeqCst);

        let ctx = self.make_context(cancel);
        let event = TurnEndEvent {
            turn_index,
            message,
            tool_results,
        };
        let handlers = self.handlers.read().turn_end_handlers.clone();
        for handler in &handlers {
            if let Err(e) = handler(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    event: "turn_end".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    async fn dispatch_message_start(&self, message: AgentMessage, cancel: CancellationToken) {
        let ctx = self.make_context(cancel);
        let event = MessageStartEvent { message };
        let handlers = self.handlers.read().message_start_handlers.clone();
        for handler in &handlers {
            if let Err(e) = handler(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    event: "message_start".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    async fn dispatch_message_update(
        &self,
        message: AgentMessage,
        assistant_message_event: Box<ameli_ai::types::AssistantMessageEvent>,
        cancel: CancellationToken,
    ) {
        let ctx = self.make_context(cancel);
        let event = MessageUpdateEvent {
            message,
            assistant_message_event,
        };
        let handlers = self.handlers.read().message_update_handlers.clone();
        for handler in &handlers {
            if let Err(e) = handler(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    event: "message_update".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    async fn dispatch_tool_execution_start(
        &self,
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
        cancel: CancellationToken,
    ) {
        let ctx = self.make_context(cancel);
        let event = ToolExecutionStartEvent {
            tool_call_id,
            tool_name,
            args,
        };
        let handlers = self.handlers.read().tool_execution_start_handlers.clone();
        for handler in &handlers {
            if let Err(e) = handler(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    event: "tool_execution_start".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    async fn dispatch_tool_execution_update(
        &self,
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
        partial_result: ameli_agent_core::types::AgentToolResult<serde_json::Value>,
        cancel: CancellationToken,
    ) {
        let ctx = self.make_context(cancel);
        let event = ToolExecutionUpdateEvent {
            tool_call_id,
            tool_name,
            args,
            partial_result,
        };
        let handlers = self.handlers.read().tool_execution_update_handlers.clone();
        for handler in &handlers {
            if let Err(e) = handler(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    event: "tool_execution_update".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    async fn dispatch_tool_execution_end(
        &self,
        tool_call_id: String,
        tool_name: String,
        result: ameli_agent_core::types::AgentToolResult<serde_json::Value>,
        is_error: bool,
        cancel: CancellationToken,
    ) {
        let ctx = self.make_context(cancel);
        let event = ToolExecutionEndEvent {
            tool_call_id,
            tool_name,
            result,
            is_error,
        };
        let handlers = self.handlers.read().tool_execution_end_handlers.clone();
        for handler in &handlers {
            if let Err(e) = handler(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    event: "tool_execution_end".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    // -----------------------------------------------------------------------
    // Hook dispatchers (internal)
    // -----------------------------------------------------------------------

    /// Dispatch tool_call hook to all handlers. Returns the first blocking
    /// result, or `None` if no handler blocks.
    async fn emit_tool_call(
        &self,
        event: ToolCallEvent,
        cancel: CancellationToken,
    ) -> Option<ToolCallResult> {
        let ctx = self.make_context(cancel);
        let handlers = self.handlers.read().tool_call_handlers.clone();
        for handler in &handlers {
            if let Some(result) = handler(event.clone(), ctx.clone()).await {
                if result.block {
                    return Some(result);
                }
            }
        }
        None
    }

    /// Dispatch tool_result hook to all handlers. Merges patches sequentially.
    async fn emit_tool_result(
        &self,
        event: ToolResultEvent,
        cancel: CancellationToken,
    ) -> Option<ToolResultPatch> {
        let ctx = self.make_context(cancel);
        let mut current_event = event;
        let mut combined = ToolResultPatch::default();
        let mut modified = false;

        let handlers = self.handlers.read().tool_result_handlers.clone();
        for handler in &handlers {
            if let Some(patch) = handler(current_event.clone(), ctx.clone()).await {
                if let Some(content) = patch.content {
                    current_event.content = content.clone();
                    combined.content = Some(content);
                    modified = true;
                }
                if let Some(details) = patch.details {
                    current_event.details = details.clone();
                    combined.details = Some(details);
                    modified = true;
                }
                if let Some(is_error) = patch.is_error {
                    current_event.is_error = is_error;
                    combined.is_error = Some(is_error);
                    modified = true;
                }
                if let Some(terminate) = patch.terminate {
                    combined.terminate = Some(terminate);
                    modified = true;
                }
            }
        }

        if modified {
            Some(combined)
        } else {
            None
        }
    }

    /// Dispatch context hook to all handlers. Chains message transformations.
    async fn emit_context(
        &self,
        messages: Vec<AgentMessage>,
        cancel: CancellationToken,
    ) -> Vec<AgentMessage> {
        let ctx = self.make_context(cancel);
        let mut current = messages;

        let handlers = self.handlers.read().context_handlers.clone();
        for handler in &handlers {
            let event = ContextEvent {
                messages: current.clone(),
            };
            if let Some(result) = handler(event, ctx.clone()).await {
                current = result.messages;
            }
        }

        current
    }

    // -----------------------------------------------------------------------
    // AgentLoopConfig hook implementations
    // -----------------------------------------------------------------------

    async fn handle_before_tool_call(
        &self,
        ctx: &BeforeToolCallContext,
        cancel: Option<CancellationToken>,
    ) -> Option<BeforeToolCallResult> {
        let event = ToolCallEvent {
            tool_call_id: ctx.tool_call.id.clone(),
            tool_name: ctx.tool_call.name.clone(),
            args: ctx.args.clone(),
        };

        let result = self
            .emit_tool_call(event, cancel.unwrap_or_default())
            .await?;

        Some(BeforeToolCallResult {
            block: result.block,
            reason: result.reason,
        })
    }

    async fn handle_after_tool_call(
        &self,
        ctx: &AfterToolCallContext,
        cancel: Option<CancellationToken>,
    ) -> Option<AfterToolCallResult> {
        let event = ToolResultEvent {
            tool_call_id: ctx.tool_call.id.clone(),
            tool_name: ctx.tool_call.name.clone(),
            args: ctx.args.clone(),
            content: ctx.result.content.clone(),
            details: ctx.result.details.clone(),
            is_error: ctx.is_error,
        };

        let patch = self
            .emit_tool_result(event, cancel.unwrap_or_default())
            .await?;

        Some(AfterToolCallResult {
            content: patch.content,
            details: patch.details,
            is_error: patch.is_error,
            terminate: patch.terminate,
        })
    }

    async fn handle_transform_context(
        &self,
        messages: &[AgentMessage],
        cancel: Option<CancellationToken>,
    ) -> Vec<AgentMessage> {
        self.emit_context(messages.to_vec(), cancel.unwrap_or_default())
            .await
    }

    /// Wrapped `convert_to_llm` that converts custom messages via registered
    /// formatters before delegating standard messages to the original function.
    ///
    /// For each [`AgentMessage::Custom`], looks up a formatter by
    /// `custom_type`. If the formatter returns `Some(Message)`, the message
    /// is included in the LLM context. If `None`, the message is skipped.
    /// Standard messages are delegated to the original `convert_to_llm`.
    /// Results are merged in original message order.
    async fn handle_convert_to_llm_with_custom_messages(
        &self,
        messages: &[AgentMessage],
        original: Option<&Arc<ameli_agent_core::agent::ConvertToLlmFn>>,
    ) -> Vec<ameli_ai::types::Message> {
        let mut custom_results: Vec<(usize, ameli_ai::types::Message)> = Vec::new();
        let mut non_custom: Vec<AgentMessage> = Vec::new();
        let mut non_custom_indices: Vec<usize> = Vec::new();

        for (i, msg) in messages.iter().enumerate() {
            match msg {
                AgentMessage::Custom(custom) => {
                    let data = custom.data.as_ref().unwrap_or(&serde_json::Value::Null);
                    if let Some(llm_msg) = self.format_custom_message(&custom.custom_type, data) {
                        custom_results.push((i, llm_msg));
                    }
                    // else skip — formatter returned None
                }
                other => {
                    non_custom.push(other.clone());
                    non_custom_indices.push(i);
                }
            }
        }

        // Delegate non-custom messages to original convert_to_llm
        let non_custom_llm = match original {
            Some(original_fn) => original_fn(&non_custom).await,
            None => {
                // Default filter: keep only standard LLM messages
                non_custom.iter().filter_map(|m| m.as_message()).collect()
            }
        };

        // Merge results in original message order
        let mut merged: Vec<(usize, ameli_ai::types::Message)> = custom_results;
        for (offset, llm_msg) in non_custom_llm.into_iter().enumerate() {
            if let Some(&original_idx) = non_custom_indices.get(offset) {
                merged.push((original_idx, llm_msg));
            }
        }
        merged.sort_by_key(|(i, _)| *i);
        merged.into_iter().map(|(_, msg)| msg).collect()
    }

    /// Report an error to all registered error listeners.
    ///
    /// This is provided as a public API for callers that want to report
    /// extension-related errors through the listener infrastructure.
    pub fn report_error(&self, error: ExtensionError) {
        let listeners = self.error_listeners.lock();
        for listener in listeners.iter() {
            listener(error.clone());
        }
    }

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Dispatch a tool_call hook for testing. Returns the first blocking
    /// result, or `None` if no handler blocks.
    #[cfg(test)]
    pub(crate) async fn emit_tool_call_for_test(
        &self,
        event: ToolCallEvent,
        ctx: ExtensionContext,
    ) -> Option<ToolCallResult> {
        let handlers = self.handlers.read().tool_call_handlers.clone();
        for handler in &handlers {
            if let Some(result) = handler(event.clone(), ctx.clone()).await {
                if result.block {
                    return Some(result);
                }
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build an [`ExtensionContext`] for handler dispatch.
    fn make_context(&self, cancel: CancellationToken) -> ExtensionContext {
        ExtensionContext {
            is_idle: false,
            cancel_token: Some(cancel),
            interface: self.interface.clone(),
        }
    }
}

impl fmt::Debug for ExtensionRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionRunner")
            .field("has_any_handlers", &self.has_any_handlers())
            .field("tools", &self.get_registered_tools().len())
            .field("commands", &self.get_registered_commands().len())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Accumulated result types
// ---------------------------------------------------------------------------

/// Combined result from all `before_agent_start` handlers.
#[derive(Debug, Clone)]
pub struct BeforeAgentStartAccumulated {
    /// Custom messages to inject alongside the user message.
    pub messages: Option<Vec<BeforeAgentStartMessage>>,
    /// Replacement system prompt. Last handler's value wins.
    pub system_prompt: Option<String>,
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
    use crate::extension::ExtensionApi;
    use ameli_agent_core::types::{AgentContext, AgentToolResult, ToolExecutionMode};
    use ameli_ai::types::{
        AssistantMessage, MediaContentBlock, TextContent, Tool, ToolCall, Usage,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn noop_interface() -> Arc<dyn crate::interface::Interface> {
        Arc::new(crate::interface::NoopInterface)
    }

    // -- Test extensions ----------------------------------------------------

    struct LoggingExtension;

    impl Extension for LoggingExtension {
        fn init(&self, api: &Arc<ExtensionApi>) {
            api.on_agent_start(|_event, _ctx| Box::pin(async { Ok(()) }));
            api.on_turn_end(|_event, _ctx| Box::pin(async { Ok(()) }));
            api.on_session_start(|_event, _ctx| Box::pin(async { Ok(()) }));
            api.on_session_shutdown(|_event, _ctx| Box::pin(async { Ok(()) }));
        }
    }

    struct BlockBashExtension;

    impl Extension for BlockBashExtension {
        fn init(&self, api: &Arc<ExtensionApi>) {
            api.on_tool_call(|event, _ctx| {
                let tool_name = event.tool_name.clone();
                Box::pin(async move {
                    if tool_name == "bash" {
                        Some(ToolCallResult::block("bash is blocked"))
                    } else {
                        None
                    }
                })
            });
        }
    }

    struct ContextTransformExtension;

    impl Extension for ContextTransformExtension {
        fn init(&self, api: &Arc<ExtensionApi>) {
            api.on_context(|event, _ctx| {
                let messages = event.messages.clone();
                Box::pin(async move {
                    let filtered: Vec<AgentMessage> = messages
                        .into_iter()
                        .filter(|m| match m {
                            AgentMessage::User(u) => {
                                !matches!(&u.content, ameli_ai::types::UserContent::Text(t) if t.is_empty())
                            }
                            _ => true,
                        })
                        .collect();
                    Some(ContextResult {
                        messages: filtered,
                    })
                })
            });
        }
    }

    struct ToolResultModifierExtension;

    impl Extension for ToolResultModifierExtension {
        fn init(&self, api: &Arc<ExtensionApi>) {
            api.on_tool_result(|event, _ctx| {
                let tool_name = event.tool_name.clone();
                Box::pin(async move {
                    if tool_name == "echo" {
                        Some(ToolResultPatch {
                            content: Some(vec![MediaContentBlock::Text(TextContent::new(
                                "modified by extension",
                            ))]),
                            details: None,
                            is_error: None,
                            terminate: None,
                        })
                    } else {
                        None
                    }
                })
            });
        }
    }

    struct CompactionFormatExtension;

    impl Extension for CompactionFormatExtension {
        fn init(&self, api: &Arc<ExtensionApi>) {
            api.on_format_compaction_summary(|event, _ctx| {
                let summary = event.summary.clone();
                let timestamp = event.timestamp;
                Box::pin(async move {
                    let text = format!("[CUSTOM COMPACT] {summary}");
                    let content = vec![MediaContentBlock::Text(TextContent::new(text))];
                    Some(FormatCompactionSummaryResult {
                        message: AgentMessage::User(ameli_ai::types::UserMessage {
                            content: ameli_ai::types::UserContent::Blocks(content),
                            timestamp,
                        }),
                    })
                })
            });
        }
    }

    struct BeforeAgentStartOverrideExtension;

    impl Extension for BeforeAgentStartOverrideExtension {
        fn init(&self, api: &Arc<ExtensionApi>) {
            api.on_before_agent_start(|_event, _ctx| {
                Box::pin(async move {
                    Some(BeforeAgentStartResult {
                        system_prompt: Some("overridden".into()),
                        message: None,
                    })
                })
            });
        }
    }

    struct MessageEndReplaceExtension;

    impl Extension for MessageEndReplaceExtension {
        fn init(&self, api: &Arc<ExtensionApi>) {
            api.on_message_end(|event, _ctx| {
                let msg = event.message.clone();
                Box::pin(async move { Some(MessageEndResult { message: msg }) })
            });
        }
    }

    struct EchoTool;

    impl ameli_agent_core::types::AgentTool for EchoTool {
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
        fn execution_mode(&self) -> Option<ToolExecutionMode> {
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
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AgentToolResult> + Send + '_>>
        {
            let message = params
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Box::pin(async move { AgentToolResult::text(message, serde_json::json!({})) })
        }
    }

    struct ToolRegisteringExtension;

    impl Extension for ToolRegisteringExtension {
        fn init(&self, api: &Arc<ExtensionApi>) {
            api.register_tool(Arc::new(EchoTool));
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

    // -- Construction tests -------------------------------------------------

    #[test]
    fn from_extensions_collects_handlers() {
        let extensions: Vec<Box<dyn Extension>> =
            vec![Box::new(LoggingExtension), Box::new(BlockBashExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);
        assert!(runner.has_tool_call_handlers());
        assert!(runner.has_any_handlers());
    }

    #[test]
    fn no_handlers_when_empty() {
        let runner = ExtensionRunner::from_extensions(&[]);
        assert!(!runner.has_tool_call_handlers());
        assert!(!runner.has_tool_result_handlers());
        assert!(!runner.has_context_handlers());
        assert!(!runner.has_before_agent_start_handlers());
        assert!(!runner.has_message_end_handlers());
        assert!(!runner.has_format_compaction_summary_handlers());
        assert!(!runner.has_format_branch_summary_handlers());
        assert!(!runner.has_any_handlers());
    }

    // -- Tool collection tests ----------------------------------------------

    #[test]
    fn get_registered_tools() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(ToolRegisteringExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);
        let tools = runner.get_registered_tools();
        assert_eq!(tools.len(), 1);
        let first = tools.first();
        assert!(first.is_some(), "expected at least one tool");
        assert_eq!(first.map(|t| t.name()).as_deref(), Some("echo"));
    }

    #[test]
    fn get_registered_commands() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(CommandExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);
        let commands = runner.get_registered_commands();
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "greet");
    }

    // -- Error listener tests -----------------------------------------------

    #[test]
    fn error_listener_receives_errors() {
        let error_count = Arc::new(AtomicUsize::new(0));
        let error_count_clone = error_count.clone();

        let runner = ExtensionRunner::from_extensions(&[]);
        runner.on_error(Arc::new(move |_err| {
            error_count_clone.fetch_add(1, Ordering::SeqCst);
        }));

        runner.report_error(ExtensionError {
            event: "test_event".to_string(),
            error: "something failed".to_string(),
        });

        assert_eq!(error_count.load(Ordering::SeqCst), 1);
    }

    // -- Hook installation tests --------------------------------------------

    #[test]
    fn install_hooks_with_tool_call_handler() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(BlockBashExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);

        let mut options = ameli_agent_core::AgentOptions::default();
        runner.install_hooks(&mut options);

        assert!(options.before_tool_call.is_some());
        assert!(options.after_tool_call.is_none());
        assert!(options.transform_context.is_none());
    }

    #[test]
    fn install_hooks_with_context_handler() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(ContextTransformExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);

        let mut options = ameli_agent_core::AgentOptions::default();
        runner.install_hooks(&mut options);

        assert!(options.before_tool_call.is_none());
        assert!(options.after_tool_call.is_none());
        assert!(options.transform_context.is_some());
    }

    #[test]
    fn install_hooks_with_tool_result_handler() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(ToolResultModifierExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);

        let mut options = ameli_agent_core::AgentOptions::default();
        runner.install_hooks(&mut options);

        assert!(options.before_tool_call.is_none());
        assert!(options.after_tool_call.is_some());
        assert!(options.transform_context.is_none());
    }

    #[test]
    fn install_hooks_skips_when_no_handlers() {
        let runner = ExtensionRunner::from_extensions(&[]);

        let mut options = ameli_agent_core::AgentOptions::default();
        runner.install_hooks(&mut options);

        assert!(options.before_tool_call.is_none());
        assert!(options.after_tool_call.is_none());
        assert!(options.transform_context.is_none());
    }

    // -- Hook handler mapping tests -----------------------------------------

    #[tokio::test]
    async fn before_tool_call_blocks_bash() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(BlockBashExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);

        let ctx = BeforeToolCallContext {
            assistant_message: AssistantMessage {
                content: vec![],
                api: "test".into(),
                provider: "test".into(),
                model: "test".into(),
                response_model: None,
                response_id: None,
                usage: Usage::default(),
                stop_reason: ameli_ai::types::StopReason::Stop,
                error_message: None,
                timestamp: 0,
            },
            tool_call: ToolCall {
                id: "tc_1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({"command": "rm -rf /"}),
                thought_signature: None,
            },
            args: serde_json::json!({"command": "rm -rf /"}),
            context: AgentContext {
                system_prompt: String::new(),
                messages: vec![],
                tools: vec![],
            },
        };

        let result = runner.handle_before_tool_call(&ctx, None).await;
        assert!(result.is_some());
        let Some(result) = result else {
            panic!("expected blocking result");
        };
        assert!(result.block);
        assert_eq!(result.reason.as_deref(), Some("bash is blocked"));
    }

    #[tokio::test]
    async fn before_tool_call_allows_other_tools() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(BlockBashExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);

        let ctx = BeforeToolCallContext {
            assistant_message: AssistantMessage {
                content: vec![],
                api: "test".into(),
                provider: "test".into(),
                model: "test".into(),
                response_model: None,
                response_id: None,
                usage: Usage::default(),
                stop_reason: ameli_ai::types::StopReason::Stop,
                error_message: None,
                timestamp: 0,
            },
            tool_call: ToolCall {
                id: "tc_2".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": "/etc/hosts"}),
                thought_signature: None,
            },
            args: serde_json::json!({"path": "/etc/hosts"}),
            context: AgentContext {
                system_prompt: String::new(),
                messages: vec![],
                tools: vec![],
            },
        };

        let result = runner.handle_before_tool_call(&ctx, None).await;
        assert!(result.is_none());
    }

    // -- Tool result modifier test ------------------------------------------

    #[tokio::test]
    async fn after_tool_call_modifies_result() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(ToolResultModifierExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);

        let ctx = AfterToolCallContext {
            assistant_message: AssistantMessage {
                content: vec![],
                api: "test".into(),
                provider: "test".into(),
                model: "test".into(),
                response_model: None,
                response_id: None,
                usage: Usage::default(),
                stop_reason: ameli_ai::types::StopReason::Stop,
                error_message: None,
                timestamp: 0,
            },
            tool_call: ToolCall {
                id: "tc_1".into(),
                name: "echo".into(),
                arguments: serde_json::json!({"message": "hello"}),
                thought_signature: None,
            },
            args: serde_json::json!({"message": "hello"}),
            result: AgentToolResult::text("hello", serde_json::json!({})),
            is_error: false,
            context: AgentContext {
                system_prompt: String::new(),
                messages: vec![],
                tools: vec![],
            },
        };

        let result = runner.handle_after_tool_call(&ctx, None).await;
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.content.is_some());
        let content = result.content.unwrap();
        assert_eq!(content.len(), 1);
        match &content[0] {
            MediaContentBlock::Text(t) => assert_eq!(t.text, "modified by extension"),
            _ => panic!("expected text block"),
        }
    }

    // -- Context transform test ---------------------------------------------

    #[tokio::test]
    async fn context_transform_filters_messages() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(ContextTransformExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);

        let messages = vec![
            AgentMessage::User(ameli_ai::types::UserMessage::text("hello")),
            AgentMessage::User(ameli_ai::types::UserMessage::text("")),
            AgentMessage::User(ameli_ai::types::UserMessage::text("world")),
        ];
        let result = runner.handle_transform_context(&messages, None).await;
        assert_eq!(result.len(), 2);
    }

    // -- Summary formatting tests -------------------------------------------

    #[tokio::test]
    async fn format_compaction_summary_custom() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(CompactionFormatExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);

        let result = runner
            .emit_format_compaction_summary("old conversation", 1000, CancellationToken::new())
            .await;
        assert!(result.is_some());
        match result.unwrap() {
            AgentMessage::User(msg) => match &msg.content {
                ameli_ai::types::UserContent::Blocks(blocks) => match &blocks[0] {
                    MediaContentBlock::Text(t) => {
                        assert!(t.text.contains("[CUSTOM COMPACT]"));
                    }
                    _ => panic!("expected text block"),
                },
                _ => panic!("expected blocks"),
            },
            _ => panic!("expected user message"),
        }
    }

    #[tokio::test]
    async fn format_compaction_summary_default_when_no_handlers() {
        let runner = ExtensionRunner::from_extensions(&[]);
        let result = runner
            .emit_format_compaction_summary("summary", 1000, CancellationToken::new())
            .await;
        assert!(result.is_none());
    }

    // -- Before agent start tests -------------------------------------------

    #[tokio::test]
    async fn before_agent_start_override() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(BeforeAgentStartOverrideExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);

        let result = runner
            .emit_before_agent_start("hello", &[], &[], "original", CancellationToken::new())
            .await;
        assert!(result.is_some());
        let acc = result.unwrap();
        assert_eq!(acc.system_prompt.as_deref(), Some("overridden"));
    }

    // -- Message end test ---------------------------------------------------

    #[tokio::test]
    async fn message_end_replaces() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(MessageEndReplaceExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);

        let event = MessageEndEvent {
            message: AgentMessage::User(ameli_ai::types::UserMessage::text("hello")),
        };
        let result = runner
            .emit_message_end(event, CancellationToken::new())
            .await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().role(), "user");
    }

    // -- Session lifecycle tests --------------------------------------------

    #[tokio::test]
    async fn session_start_dispatches() {
        let runner = ExtensionRunner::from_extensions(&[Box::new(LoggingExtension)]);
        runner.emit_session_start(SessionStartReason::Startup).await;
    }

    #[tokio::test]
    async fn session_shutdown_dispatches() {
        let runner = ExtensionRunner::from_extensions(&[Box::new(LoggingExtension)]);
        let handled = runner
            .emit_session_shutdown(SessionShutdownReason::Quit)
            .await;
        assert!(handled);
    }

    #[tokio::test]
    async fn session_shutdown_returns_false_when_no_handlers() {
        let runner = ExtensionRunner::from_extensions(&[]);
        let handled = runner
            .emit_session_shutdown(SessionShutdownReason::Quit)
            .await;
        assert!(!handled);
    }

    // -- Command dispatch test ----------------------------------------------

    #[tokio::test]
    async fn execute_command_dispatches() {
        let runner = ExtensionRunner::from_extensions(&[Box::new(CommandExtension)]);
        let ctx = CommandContext {
            extension_context: ExtensionContext::for_testing(),
        };
        let result = runner.execute_command("greet", "world", ctx).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn execute_command_unknown_fails() {
        let runner = ExtensionRunner::from_extensions(&[]);
        let ctx = CommandContext {
            extension_context: ExtensionContext::for_testing(),
        };
        let result = runner.execute_command("unknown", "", ctx).await;
        assert!(result.is_err());
    }

    // -- Error propagation during dispatch ---------------------------------

    #[tokio::test]
    async fn notification_handler_error_reported_to_listener_and_dispatch_continues() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let error_received = Arc::new(AtomicBool::new(false));
        let second_handler_ran = Arc::new(AtomicBool::new(false));

        struct FailingThenSucceedingExtension {
            ran_flag: Arc<AtomicBool>,
        }

        impl Extension for FailingThenSucceedingExtension {
            fn init(&self, api: &Arc<ExtensionApi>) {
                let flag = self.ran_flag.clone();
                // First handler: always fails
                api.on_agent_start(|_event, _ctx| {
                    Box::pin(async { Err(anyhow::anyhow!("handler failed")) })
                });
                // Second handler: succeeds and sets flag
                api.on_agent_start(move |_event, _ctx| {
                    let flag = flag.clone();
                    Box::pin(async move {
                        flag.store(true, Ordering::SeqCst);
                        Ok(())
                    })
                });
            }
        }

        let runner =
            ExtensionRunner::from_extensions(&[Box::new(FailingThenSucceedingExtension {
                ran_flag: second_handler_ran.clone(),
            })]);

        let error_flag = error_received.clone();
        runner.on_error(Arc::new(move |err| {
            assert_eq!(err.event, "agent_start");
            assert!(err.error.contains("handler failed"));
            error_flag.store(true, Ordering::SeqCst);
        }));

        runner
            .dispatch_agent_event(
                ameli_agent_core::types::AgentEvent::AgentStart,
                CancellationToken::new(),
            )
            .await;

        assert!(
            error_received.load(Ordering::SeqCst),
            "error listener should have been called"
        );
        assert!(
            second_handler_ran.load(Ordering::SeqCst),
            "second handler should still have run"
        );
    }

    // -- dispatch_agent_event routing test -----------------------------------

    #[tokio::test]
    async fn dispatch_agent_event_routes_agent_start() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let call_count = Arc::new(AtomicUsize::new(0));

        struct AgentStartCountingExtension {
            count: Arc<AtomicUsize>,
        }

        impl Extension for AgentStartCountingExtension {
            fn init(&self, api: &Arc<ExtensionApi>) {
                let count = self.count.clone();
                api.on_agent_start(move |_event, _ctx| {
                    let count = count.clone();
                    Box::pin(async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                });
            }
        }

        let runner = ExtensionRunner::from_extensions(&[Box::new(AgentStartCountingExtension {
            count: call_count.clone(),
        })]);

        // Dispatch via the public routing method, not the private dispatch_agent_start
        runner
            .dispatch_agent_event(
                ameli_agent_core::types::AgentEvent::AgentStart,
                CancellationToken::new(),
            )
            .await;

        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    // -- Empty runner tests -------------------------------------------------

    #[test]
    fn empty_runner_has_no_handlers() {
        let runner = ExtensionRunner::empty(noop_interface());
        assert!(!runner.has_any_handlers());
        assert!(runner.get_registered_tools().is_empty());
        assert!(runner.get_registered_commands().is_empty());
    }
}
