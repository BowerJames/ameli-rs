//! Extension runner — bridges extension handlers to the agent loop.
//!
//! [`ExtensionRunner`] is the runtime that connects extension event handlers
//! to the agent via hook closures installed into [`AgentOptions`].
//! [`AgentSession`](crate::AgentSession) handles event subscription and
//! dispatches to the runner's emit methods.
//!
//! # Lifecycle
//!
//! 1. Create extensions and build a runner with [`ExtensionRunner::from_extensions`].
//! 2. Call [`ExtensionRunner::install_hooks`] to install hook closures into
//!    [`AgentOptions`].
//! 3. Construct an [`AgentSession`](crate::AgentSession) (or an `ArcAgent`
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
use crate::extension::{init_extensions, Extension, ExtensionHandlers};
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
// ExtensionError
// ---------------------------------------------------------------------------

/// Error reported to error listeners when an extension handler fails.
///
/// Produced when notification handlers return `Err` or when hook handlers
/// encounter errors.
#[derive(Debug, Clone)]
pub struct ExtensionError {
    /// Name of the extension that failed (best-effort; "unknown" if unavailable).
    pub extension_name: String,
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
/// Construct via [`ExtensionRunner::from_extensions`] or
/// [`ExtensionRunner::new`]. Share via `Arc<Self>` so hook closures and the
/// subscriber can both reference it.
pub struct ExtensionRunner {
    /// Handlers collected from all extensions during initialization.
    handlers: ExtensionHandlers,
    /// Error listeners, protected by a std Mutex for interior mutability.
    error_listeners: std::sync::Mutex<Vec<ExtensionErrorListener>>,
    /// UI interface for creating ExtensionContext.
    interface: Arc<dyn Interface>,
    /// Turn counter for providing turn_index in events.
    turn_index: AtomicU32,
}

impl ExtensionRunner {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a runner from pre-collected handlers with a default no-op
    /// interface.
    pub fn new(handlers: ExtensionHandlers) -> Self {
        Self {
            handlers,
            error_listeners: std::sync::Mutex::new(Vec::new()),
            interface: Arc::new(crate::interface::NoopInterface),
            turn_index: AtomicU32::new(0),
        }
    }

    /// Create a runner by initializing a list of extensions and collecting
    /// their registrations. Uses a default no-op interface.
    pub fn from_extensions(extensions: &[Box<dyn Extension>]) -> Self {
        Self::new(init_extensions(extensions))
    }

    /// Create a runner with a custom interface for ExtensionContext.
    pub fn with_interface(handlers: ExtensionHandlers, interface: Arc<dyn Interface>) -> Self {
        Self {
            handlers,
            error_listeners: std::sync::Mutex::new(Vec::new()),
            interface,
            turn_index: AtomicU32::new(0),
        }
    }

    // -----------------------------------------------------------------------
    // Error listeners
    // -----------------------------------------------------------------------

    /// Register an error listener called when an extension handler fails.
    pub fn on_error(&self, listener: ExtensionErrorListener) {
        let mut listeners = self
            .error_listeners
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        listeners.push(listener);
    }

    // -----------------------------------------------------------------------
    // Handler queries
    // -----------------------------------------------------------------------

    /// Returns `true` if any extension registered a tool_call hook handler.
    pub fn has_tool_call_handlers(&self) -> bool {
        !self.handlers.tool_call_handlers.is_empty()
    }

    /// Returns `true` if any extension registered a tool_result hook handler.
    pub fn has_tool_result_handlers(&self) -> bool {
        !self.handlers.tool_result_handlers.is_empty()
    }

    /// Returns `true` if any extension registered a context hook handler.
    pub fn has_context_handlers(&self) -> bool {
        !self.handlers.context_handlers.is_empty()
    }

    /// Returns `true` if any extension registered a before_agent_start hook.
    pub fn has_before_agent_start_handlers(&self) -> bool {
        !self.handlers.before_agent_start_handlers.is_empty()
    }

    /// Returns `true` if any extension registered a message_end hook handler.
    pub fn has_message_end_handlers(&self) -> bool {
        !self.handlers.message_end_handlers.is_empty()
    }

    /// Returns `true` if any extension registered a format_compaction_summary
    /// hook handler.
    pub fn has_format_compaction_summary_handlers(&self) -> bool {
        !self.handlers.format_compaction_summary_handlers.is_empty()
    }

    /// Returns `true` if any extension registered a format_branch_summary
    /// hook handler.
    pub fn has_format_branch_summary_handlers(&self) -> bool {
        !self.handlers.format_branch_summary_handlers.is_empty()
    }

    /// Returns `true` if any handlers are registered for any event type.
    pub fn has_any_handlers(&self) -> bool {
        !self.handlers.is_empty()
    }

    // -----------------------------------------------------------------------
    // Tool and command collection
    // -----------------------------------------------------------------------

    /// Collect all tools registered by extensions.
    pub fn get_registered_tools(&self) -> Vec<Arc<dyn AgentTool>> {
        self.handlers.tools.clone()
    }

    /// Collect all commands registered by extensions.
    pub fn get_registered_commands(&self) -> Vec<RegisteredCommand> {
        self.handlers.commands.clone()
    }

    // -----------------------------------------------------------------------
    // Hook installation
    // -----------------------------------------------------------------------

    /// Install hook closures into [`AgentOptions`] for tool call interception
    /// and context transformation.
    ///
    /// Only installs a hook if there are registered handlers for the
    /// corresponding event type. This avoids unnecessary overhead.
    ///
    /// Call this **before** constructing the [`ArcAgent`].
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
        for handler in &self.handlers.format_compaction_summary_handlers {
            if let Some(result) = (handler.handler)(event.clone(), ctx.clone()).await {
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
        for handler in &self.handlers.format_branch_summary_handlers {
            if let Some(result) = (handler.handler)(event.clone(), ctx.clone()).await {
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
        for handler in &self.handlers.session_start_handlers {
            if let Err(e) = (handler.handler)(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    extension_name: handler.extension_name.clone(),
                    event: "session_start".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    /// Dispatch session_shutdown event to all handlers (fire-and-forget).
    pub async fn emit_session_shutdown(&self, reason: SessionShutdownReason) -> bool {
        if self.handlers.session_shutdown_handlers.is_empty() {
            return false;
        }
        let ctx = self.make_context(CancellationToken::new());
        let event = SessionShutdownEvent { reason };
        for handler in &self.handlers.session_shutdown_handlers {
            if let Err(e) = (handler.handler)(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    extension_name: handler.extension_name.clone(),
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
        system_prompt: &str,
        cancel: CancellationToken,
    ) -> Option<BeforeAgentStartAccumulated> {
        if self.handlers.before_agent_start_handlers.is_empty() {
            return None;
        }

        let ctx = self.make_context(cancel);
        let mut current_system_prompt = system_prompt.to_string();
        let mut messages: Vec<BeforeAgentStartMessage> = Vec::new();
        let mut modified = false;

        for handler in &self.handlers.before_agent_start_handlers {
            let event = BeforeAgentStartEvent {
                prompt: prompt.to_string(),
                images: images.to_vec(),
                system_prompt: current_system_prompt.clone(),
            };
            if let Some(result) = (handler.handler)(event, ctx.clone()).await {
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
        if self.handlers.message_end_handlers.is_empty() {
            return None;
        }

        let ctx = self.make_context(cancel);
        let original_role = event.message.role().to_string();
        let mut current_message = event.message;
        let mut modified = false;

        for handler in &self.handlers.message_end_handlers {
            let current_event = MessageEndEvent {
                message: current_message.clone(),
            };
            if let Some(result) = (handler.handler)(current_event, ctx.clone()).await {
                if result.message.role() != original_role {
                    self.report_error(ExtensionError {
                        extension_name: handler.extension_name.clone(),
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
        for cmd in &self.handlers.commands {
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
                // (sequential chain with replacement). This arm should never be
                // reached in normal operation since AgentSession intercepts
                // MessageEnd before delegating to dispatch_agent_event().
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

    pub async fn dispatch_agent_start(&self, cancel: CancellationToken) {
        let ctx = self.make_context(cancel);
        let event = AgentStartEvent;
        for handler in &self.handlers.agent_start_handlers {
            if let Err(e) = (handler.handler)(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    extension_name: handler.extension_name.clone(),
                    event: "agent_start".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    pub async fn dispatch_agent_end(&self, messages: Vec<AgentMessage>, cancel: CancellationToken) {
        let ctx = self.make_context(cancel);
        let event = AgentEndEvent { messages };
        for handler in &self.handlers.agent_end_handlers {
            if let Err(e) = (handler.handler)(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    extension_name: handler.extension_name.clone(),
                    event: "agent_end".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    pub async fn dispatch_turn_start(&self, cancel: CancellationToken) {
        let turn_index = self.turn_index.load(AtomicOrdering::SeqCst);
        let ctx = self.make_context(cancel);
        let event = TurnStartEvent {
            turn_index,
            timestamp: now_ms(),
        };
        for handler in &self.handlers.turn_start_handlers {
            if let Err(e) = (handler.handler)(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    extension_name: handler.extension_name.clone(),
                    event: "turn_start".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    pub async fn dispatch_turn_end(
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
        for handler in &self.handlers.turn_end_handlers {
            if let Err(e) = (handler.handler)(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    extension_name: handler.extension_name.clone(),
                    event: "turn_end".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    pub async fn dispatch_message_start(&self, message: AgentMessage, cancel: CancellationToken) {
        let ctx = self.make_context(cancel);
        let event = MessageStartEvent { message };
        for handler in &self.handlers.message_start_handlers {
            if let Err(e) = (handler.handler)(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    extension_name: handler.extension_name.clone(),
                    event: "message_start".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    pub async fn dispatch_message_update(
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
        for handler in &self.handlers.message_update_handlers {
            if let Err(e) = (handler.handler)(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    extension_name: handler.extension_name.clone(),
                    event: "message_update".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    pub async fn dispatch_tool_execution_start(
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
        for handler in &self.handlers.tool_execution_start_handlers {
            if let Err(e) = (handler.handler)(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    extension_name: handler.extension_name.clone(),
                    event: "tool_execution_start".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    pub async fn dispatch_tool_execution_update(
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
        for handler in &self.handlers.tool_execution_update_handlers {
            if let Err(e) = (handler.handler)(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    extension_name: handler.extension_name.clone(),
                    event: "tool_execution_update".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    pub async fn dispatch_tool_execution_end(
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
        for handler in &self.handlers.tool_execution_end_handlers {
            if let Err(e) = (handler.handler)(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    extension_name: handler.extension_name.clone(),
                    event: "tool_execution_end".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    // -----------------------------------------------------------------------
    // Hook dispatchers (emit_tool_call, emit_tool_result, emit_context)
    // -----------------------------------------------------------------------

    /// Dispatch tool_call hook to all handlers. Returns the first blocking
    /// result, or `None` if no handler blocks.
    async fn emit_tool_call(
        &self,
        event: ToolCallEvent,
        cancel: CancellationToken,
    ) -> Option<ToolCallResult> {
        let ctx = self.make_context(cancel);
        for handler in &self.handlers.tool_call_handlers {
            if let Some(result) = (handler.handler)(event.clone(), ctx.clone()).await {
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

        for handler in &self.handlers.tool_result_handlers {
            if let Some(patch) = (handler.handler)(current_event.clone(), ctx.clone()).await {
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

        for handler in &self.handlers.context_handlers {
            let event = ContextEvent {
                messages: current.clone(),
            };
            if let Some(result) = (handler.handler)(event, ctx.clone()).await {
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

    /// Report an error to all registered error listeners.
    ///
    /// This is provided as a public API for callers that want to report
    /// extension-related errors through the listener infrastructure.
    pub fn report_error(&self, error: ExtensionError) {
        let listeners = self
            .error_listeners
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for listener in listeners.iter() {
            listener(error.clone());
        }
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
            .field(
                "notification_handlers",
                &format_args!(
                    "agent_start={} agent_end={} turn_start={} turn_end={} message_start={} message_update={} message_end={} tool_exec_start={} tool_exec_update={} tool_exec_end={} session_start={} session_shutdown={}",
                    self.handlers.agent_start_handlers.len(),
                    self.handlers.agent_end_handlers.len(),
                    self.handlers.turn_start_handlers.len(),
                    self.handlers.turn_end_handlers.len(),
                    self.handlers.message_start_handlers.len(),
                    self.handlers.message_update_handlers.len(),
                    self.handlers.message_end_handlers.len(),
                    self.handlers.tool_execution_start_handlers.len(),
                    self.handlers.tool_execution_update_handlers.len(),
                    self.handlers.tool_execution_end_handlers.len(),
                    self.handlers.session_start_handlers.len(),
                    self.handlers.session_shutdown_handlers.len(),
                ),
            )
            .field(
                "hook_handlers",
                &format_args!(
                    "tool_call={} tool_result={} context={} before_agent_start={} format_compaction={} format_branch={}",
                    self.handlers.tool_call_handlers.len(),
                    self.handlers.tool_result_handlers.len(),
                    self.handlers.context_handlers.len(),
                    self.handlers.before_agent_start_handlers.len(),
                    self.handlers.format_compaction_summary_handlers.len(),
                    self.handlers.format_branch_summary_handlers.len(),
                ),
            )
            .field("commands", &self.handlers.commands.len())
            .field("tools", &self.handlers.tools.len())
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

    // -- Test extensions ----------------------------------------------------

    struct LoggingExtension;

    impl Extension for LoggingExtension {
        fn name(&self) -> &str {
            "logging"
        }
        fn init(&self, api: &mut ExtensionApi) {
            api.on_agent_start(|_event, _ctx| Box::pin(async { Ok(()) }));
            api.on_turn_end(|_event, _ctx| Box::pin(async { Ok(()) }));
            api.on_session_start(|_event, _ctx| Box::pin(async { Ok(()) }));
            api.on_session_shutdown(|_event, _ctx| Box::pin(async { Ok(()) }));
        }
    }

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
        fn name(&self) -> &str {
            "context-transform"
        }
        fn init(&self, api: &mut ExtensionApi) {
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
        fn name(&self) -> &str {
            "tool-result-mod"
        }
        fn init(&self, api: &mut ExtensionApi) {
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
        fn name(&self) -> &str {
            "compaction-fmt"
        }
        fn init(&self, api: &mut ExtensionApi) {
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
        fn name(&self) -> &str {
            "before-start-override"
        }
        fn init(&self, api: &mut ExtensionApi) {
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
        fn name(&self) -> &str {
            "msg-end-replace"
        }
        fn init(&self, api: &mut ExtensionApi) {
            api.on_message_end(|event, _ctx| {
                let msg = event.message.clone();
                Box::pin(async move {
                    // Return the same message (no-op replacement for testing)
                    Some(MessageEndResult { message: msg })
                })
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
        fn name(&self) -> &str {
            "tool-registering"
        }
        fn init(&self, api: &mut ExtensionApi) {
            api.register_tool(Arc::new(EchoTool));
        }
    }

    struct CommandExtension;

    impl Extension for CommandExtension {
        fn name(&self) -> &str {
            "command-ext"
        }
        fn init(&self, api: &mut ExtensionApi) {
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
            extension_name: "test-ext".to_string(),
            event: "test_event".to_string(),
            error: "something failed".to_string(),
        });

        assert_eq!(error_count.load(Ordering::SeqCst), 1);
    }

    // -- Hook installation tests --------------------------------------------

    #[test]
    fn install_hooks_with_tool_call_handler() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(BlockBashExtension)];
        let runner = Arc::new(ExtensionRunner::from_extensions(&extensions));

        let mut options = ameli_agent_core::AgentOptions::default();
        runner.install_hooks(&mut options);

        assert!(options.before_tool_call.is_some());
        assert!(options.after_tool_call.is_none());
        assert!(options.transform_context.is_none());
    }

    #[test]
    fn install_hooks_with_context_handler() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(ContextTransformExtension)];
        let runner = Arc::new(ExtensionRunner::from_extensions(&extensions));

        let mut options = ameli_agent_core::AgentOptions::default();
        runner.install_hooks(&mut options);

        assert!(options.before_tool_call.is_none());
        assert!(options.after_tool_call.is_none());
        assert!(options.transform_context.is_some());
    }

    #[test]
    fn install_hooks_with_tool_result_handler() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(ToolResultModifierExtension)];
        let runner = Arc::new(ExtensionRunner::from_extensions(&extensions));

        let mut options = ameli_agent_core::AgentOptions::default();
        runner.install_hooks(&mut options);

        assert!(options.before_tool_call.is_none());
        assert!(options.after_tool_call.is_some());
        assert!(options.transform_context.is_none());
    }

    #[test]
    fn install_hooks_skips_when_no_handlers() {
        let runner = Arc::new(ExtensionRunner::from_extensions(&[]));

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
        let runner = Arc::new(ExtensionRunner::from_extensions(&extensions));

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
        let runner = Arc::new(ExtensionRunner::from_extensions(&extensions));

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

    #[tokio::test]
    async fn after_tool_call_patches_result() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(ToolResultModifierExtension)];
        let runner = Arc::new(ExtensionRunner::from_extensions(&extensions));

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
            result: AgentToolResult::text("original", serde_json::json!({})),
            is_error: false,
            context: AgentContext {
                system_prompt: String::new(),
                messages: vec![],
                tools: vec![],
            },
        };

        let result = runner.handle_after_tool_call(&ctx, None).await;
        assert!(result.is_some());
        let Some(result) = result else {
            panic!("expected patched result");
        };
        let Some(content) = result.content else {
            panic!("expected content");
        };
        assert_eq!(content.len(), 1);
        let Some(first) = content.first() else {
            panic!("expected at least one block");
        };
        match first {
            ameli_ai::types::MediaContentBlock::Text(t) => {
                assert_eq!(t.text, "modified by extension");
            }
            other => panic!("expected text content, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn transform_context_filters_messages() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(ContextTransformExtension)];
        let runner = Arc::new(ExtensionRunner::from_extensions(&extensions));

        let messages = vec![
            AgentMessage::User(ameli_ai::types::UserMessage::text("hello")),
            AgentMessage::User(ameli_ai::types::UserMessage::text("")),
            AgentMessage::User(ameli_ai::types::UserMessage::text("world")),
        ];

        let result = runner.handle_transform_context(&messages, None).await;
        assert_eq!(result.len(), 2);
    }

    // -- Format hook tests --------------------------------------------------

    #[tokio::test]
    async fn format_compaction_summary_custom() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(CompactionFormatExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);

        let result = runner
            .emit_format_compaction_summary(
                "old conversation was about X",
                1000,
                CancellationToken::new(),
            )
            .await;

        assert!(result.is_some());
        let Some(msg) = result else {
            panic!("expected formatted message");
        };
        match &msg {
            AgentMessage::User(u) => match &u.content {
                ameli_ai::types::UserContent::Blocks(blocks) => {
                    assert_eq!(blocks.len(), 1);
                    let Some(first) = blocks.first() else {
                        panic!("expected at least one block");
                    };
                    match first {
                        MediaContentBlock::Text(t) => {
                            assert!(t.text.contains("[CUSTOM COMPACT]"));
                        }
                        other => panic!("expected text block, got {other:?}"),
                    }
                }
                other => panic!("expected blocks, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn format_compaction_summary_default_when_no_handlers() {
        let runner = ExtensionRunner::from_extensions(&[]);
        let result = runner
            .emit_format_compaction_summary("summary text", 1000, CancellationToken::new())
            .await;
        assert!(result.is_none());
    }

    // -- Before agent start tests -------------------------------------------

    #[tokio::test]
    async fn emit_before_agent_start_accumulates() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(BeforeAgentStartOverrideExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);

        let result = runner
            .emit_before_agent_start("hello", &[], "original prompt", CancellationToken::new())
            .await;

        assert!(result.is_some());
        let accumulated = result.unwrap();
        assert_eq!(accumulated.system_prompt.as_deref(), Some("overridden"));
    }

    #[tokio::test]
    async fn emit_before_agent_start_none_when_no_handlers() {
        let runner = ExtensionRunner::from_extensions(&[]);
        let result = runner
            .emit_before_agent_start("hello", &[], "prompt", CancellationToken::new())
            .await;
        assert!(result.is_none());
    }

    // -- Message end hook tests ---------------------------------------------

    #[tokio::test]
    async fn emit_message_end_chains_replacements() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(MessageEndReplaceExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);

        let event = MessageEndEvent {
            message: AgentMessage::User(ameli_ai::types::UserMessage::text("original")),
        };
        let result = runner
            .emit_message_end(event, CancellationToken::new())
            .await;
        assert!(result.is_some());
        let msg = result.unwrap();
        assert_eq!(msg.role(), "user");
    }

    #[tokio::test]
    async fn emit_message_end_none_when_no_handlers() {
        let runner = ExtensionRunner::from_extensions(&[]);
        let event = MessageEndEvent {
            message: AgentMessage::User(ameli_ai::types::UserMessage::text("hi")),
        };
        let result = runner
            .emit_message_end(event, CancellationToken::new())
            .await;
        assert!(result.is_none());
    }

    // -- Session lifecycle tests --------------------------------------------

    #[tokio::test]
    async fn emit_session_start_fires_to_handlers() {
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        let mut api = ExtensionApi::new();
        api.current_extension_name = "test".to_string();
        api.on_session_start(move |_event, _ctx| {
            let c = count_clone.clone();
            Box::pin(async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        });

        let runner = ExtensionRunner::new(api.into_handlers());
        runner.emit_session_start(SessionStartReason::Startup).await;

        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn emit_session_shutdown_fires_to_handlers() {
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        let mut api = ExtensionApi::new();
        api.current_extension_name = "test".to_string();
        api.on_session_shutdown(move |_event, _ctx| {
            let c = count_clone.clone();
            Box::pin(async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        });

        let runner = ExtensionRunner::new(api.into_handlers());
        let handled = runner
            .emit_session_shutdown(SessionShutdownReason::Quit)
            .await;
        assert!(handled);
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn emit_session_shutdown_returns_false_when_no_handlers() {
        let runner = ExtensionRunner::from_extensions(&[]);
        let handled = runner
            .emit_session_shutdown(SessionShutdownReason::Quit)
            .await;
        assert!(!handled);
    }

    // -- Command dispatch tests ---------------------------------------------

    #[tokio::test]
    async fn execute_command_dispatches_to_handler() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(CommandExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);

        let ctx = CommandContext {
            extension_context: ExtensionContext::for_testing(),
        };
        let result = runner.execute_command("greet", "world", ctx).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn execute_command_returns_error_for_unknown() {
        let runner = ExtensionRunner::from_extensions(&[]);
        let ctx = CommandContext {
            extension_context: ExtensionContext::for_testing(),
        };
        let result = runner.execute_command("unknown", "", ctx).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no command"));
    }

    // -- Notification dispatch tests ----------------------------------------

    #[tokio::test]
    async fn agent_event_dispatches_agent_start() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let mut api = ExtensionApi::new();
        api.current_extension_name = "test".to_string();
        api.on_agent_start(move |_event, _ctx| {
            let count = call_count_clone.clone();
            Box::pin(async move {
                count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        });

        let runner = ExtensionRunner::new(api.into_handlers());
        runner
            .dispatch_agent_event(
                ameli_agent_core::types::AgentEvent::AgentStart,
                CancellationToken::new(),
            )
            .await;

        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn agent_event_dispatches_tool_execution_update() {
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        let mut api = ExtensionApi::new();
        api.current_extension_name = "test".to_string();
        api.on_tool_execution_update(move |_event, _ctx| {
            let c = count_clone.clone();
            Box::pin(async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        });

        let runner = ExtensionRunner::new(api.into_handlers());
        runner
            .dispatch_agent_event(
                ameli_agent_core::types::AgentEvent::ToolExecutionUpdate {
                    tool_call_id: "tc_1".into(),
                    tool_name: "echo".into(),
                    args: serde_json::json!({}),
                    partial_result: AgentToolResult::text("partial", serde_json::json!({})),
                },
                CancellationToken::new(),
            )
            .await;

        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn notification_handler_error_reported_to_listener() {
        let error_count = Arc::new(AtomicUsize::new(0));
        let handler2_count = Arc::new(AtomicUsize::new(0));
        let error_count_clone = error_count.clone();
        let handler2_count_clone = handler2_count.clone();

        let mut api = ExtensionApi::new();

        api.current_extension_name = "failing-ext".to_string();
        api.on_agent_start(move |_event, _ctx| {
            Box::pin(async { Err(anyhow::anyhow!("handler 1 failed")) })
        });

        api.current_extension_name = "ok-ext".to_string();
        api.on_agent_start(move |_event, _ctx| {
            let count = handler2_count_clone.clone();
            Box::pin(async move {
                count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        });

        let runner = ExtensionRunner::new(api.into_handlers());
        runner.on_error(Arc::new(move |_err| {
            error_count_clone.fetch_add(1, Ordering::SeqCst);
        }));

        runner
            .dispatch_agent_event(
                ameli_agent_core::types::AgentEvent::AgentStart,
                CancellationToken::new(),
            )
            .await;

        assert_eq!(error_count.load(Ordering::SeqCst), 1);
        assert_eq!(handler2_count.load(Ordering::SeqCst), 1);
    }

    // -- Debug format test --------------------------------------------------

    #[test]
    fn runner_debug_format() {
        let extensions: Vec<Box<dyn Extension>> =
            vec![Box::new(LoggingExtension), Box::new(BlockBashExtension)];
        let runner = ExtensionRunner::from_extensions(&extensions);
        let debug = format!("{runner:?}");
        assert!(debug.contains("ExtensionRunner"));
    }
}
