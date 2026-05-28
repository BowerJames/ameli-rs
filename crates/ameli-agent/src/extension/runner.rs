//! Extension runner — bridges extension handlers to the agent loop.
//!
//! [`ExtensionRunner`] is the runtime that connects extension event handlers
//! to the agent via two mechanisms:
//!
//! 1. **Hook closures** installed into [`AgentOptions`] (`before_tool_call`,
//!    `after_tool_call`, `transform_context`) for hook events.
//! 2. **Event subscription** on [`ArcAgent`] for notification dispatch.
//!
//! # Lifecycle
//!
//! 1. Create extensions and build a runner with [`ExtensionRunner::from_extensions`].
//! 2. Call [`ExtensionRunner::install_hooks`] to install hook closures into
//!    [`AgentOptions`].
//! 3. Create an [`ArcAgent`] from those options.
//! 4. Call [`ExtensionRunner::wire_notifications`] to subscribe to agent events.
//!    Keep the returned [`ExtensionWiring`] alive — dropping it unsubscribes.
//! 5. Use the agent normally.
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
use ameli_agent_core::{ArcAgent, Subscription};
use std::fmt;
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
// ExtensionWiring
// ---------------------------------------------------------------------------

/// Handle that keeps the agent event subscription alive.
///
/// Dropping this unsubscribes from agent events. The caller must hold onto it
/// for as long as the extension runner should receive notification events.
pub struct ExtensionWiring {
    // Holds the ArcAgent subscription. Dropping unsubscribes.
    _subscription: Subscription,
}

impl fmt::Debug for ExtensionWiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionWiring").finish()
    }
}

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
    // Tool collection
    // -----------------------------------------------------------------------

    /// Collect all tools registered by extensions.
    pub fn get_registered_tools(&self) -> Vec<Arc<dyn AgentTool>> {
        self.handlers.tools.clone()
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
    // Notification wiring
    // -----------------------------------------------------------------------

    /// Subscribe to [`ArcAgent`] events and dispatch them as extension
    /// notification events.
    ///
    /// Returns an [`ExtensionWiring`] that holds the subscription. The caller
    /// must keep it alive for as long as notifications should be received.
    /// Dropping it unsubscribes from agent events.
    pub async fn wire_notifications(self: &Arc<Self>, agent: &ArcAgent) -> ExtensionWiring {
        let runner = self.clone();
        let subscription = agent
            .subscribe(Arc::new(move |event, cancel| {
                let runner = runner.clone();
                Box::pin(async move {
                    runner.dispatch_agent_event(event, cancel).await;
                })
            }))
            .await;

        ExtensionWiring {
            _subscription: subscription,
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
    // Agent event → extension event dispatch
    // -----------------------------------------------------------------------

    /// Map an [`AgentEvent`](ameli_agent_core::types::AgentEvent) to extension
    /// notification events and dispatch to registered handlers.
    async fn dispatch_agent_event(
        &self,
        event: ameli_agent_core::types::AgentEvent,
        cancel: CancellationToken,
    ) {
        use ameli_agent_core::types::AgentEvent;

        match event {
            AgentEvent::AgentStart => {
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
            AgentEvent::MessageEnd { message } => {
                self.dispatch_message_end(message, cancel).await;
            }
            AgentEvent::ToolExecutionStart {
                tool_call_id,
                tool_name,
                args,
            } => {
                self.dispatch_tool_execution_start(tool_call_id, tool_name, args, cancel)
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
            // ToolExecutionUpdate has no corresponding extension event.
            AgentEvent::ToolExecutionUpdate { .. } => {}
        }
    }

    // -----------------------------------------------------------------------
    // Notification dispatchers (one per notification event type)
    // -----------------------------------------------------------------------

    async fn dispatch_agent_start(&self, cancel: CancellationToken) {
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

    async fn dispatch_agent_end(&self, messages: Vec<AgentMessage>, cancel: CancellationToken) {
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

    async fn dispatch_turn_start(&self, cancel: CancellationToken) {
        let ctx = self.make_context(cancel);
        let event = TurnStartEvent;
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

    async fn dispatch_turn_end(
        &self,
        message: AgentMessage,
        tool_results: Vec<ameli_ai::types::ToolResultMessage>,
        cancel: CancellationToken,
    ) {
        let ctx = self.make_context(cancel);
        let event = TurnEndEvent {
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

    async fn dispatch_message_start(&self, message: AgentMessage, cancel: CancellationToken) {
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

    async fn dispatch_message_end(&self, message: AgentMessage, cancel: CancellationToken) {
        let ctx = self.make_context(cancel);
        let event = MessageEndEvent { message };
        for handler in &self.handlers.message_end_handlers {
            if let Err(e) = (handler.handler)(event.clone(), ctx.clone()).await {
                self.report_error(ExtensionError {
                    extension_name: handler.extension_name.clone(),
                    event: "message_end".to_string(),
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

    async fn dispatch_tool_execution_end(
        &self,
        tool_call_id: String,
        tool_name: String,
        result: ameli_agent_core::types::AgentToolResult,
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

    /// Handle `before_tool_call` from the agent loop by dispatching to
    /// extension tool_call handlers.
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

    /// Handle `after_tool_call` from the agent loop by dispatching to
    /// extension tool_result handlers.
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

    /// Handle `transform_context` from the agent loop by dispatching to
    /// extension context handlers.
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
            // TODO: Wire to actual agent idle state when runner gains agent reference
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
                    "agent_start={} agent_end={} turn_start={} turn_end={} message_start={} message_update={} message_end={} tool_exec_start={} tool_exec_end={}",
                    self.handlers.agent_start_handlers.len(),
                    self.handlers.agent_end_handlers.len(),
                    self.handlers.turn_start_handlers.len(),
                    self.handlers.turn_end_handlers.len(),
                    self.handlers.message_start_handlers.len(),
                    self.handlers.message_update_handlers.len(),
                    self.handlers.message_end_handlers.len(),
                    self.handlers.tool_execution_start_handlers.len(),
                    self.handlers.tool_execution_end_handlers.len(),
                ),
            )
            .field(
                "hook_handlers",
                &format_args!(
                    "tool_call={} tool_result={} context={} format_compaction={} format_branch={}",
                    self.handlers.tool_call_handlers.len(),
                    self.handlers.tool_result_handlers.len(),
                    self.handlers.context_handlers.len(),
                    self.handlers.format_compaction_summary_handlers.len(),
                    self.handlers.format_branch_summary_handlers.len(),
                ),
            )
            .field("tools", &self.handlers.tools.len())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::ExtensionApi;
    use ameli_agent_core::types::{AgentContext, AgentState, AgentToolResult, ToolExecutionMode};
    use ameli_ai::api::ApiRegistry;
    use ameli_ai::api::StreamFn;
    use ameli_ai::stream::create_assistant_message_event_stream;
    use ameli_ai::types::{
        AssistantContentBlock, AssistantMessage, Cost, InputType, MediaContentBlock, TextContent,
        Tool, ToolCall, Usage,
    };
    use std::collections::HashSet;
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
                    // Filter out empty user messages as a simple transform
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

    // -- Helpers ------------------------------------------------------------

    fn test_model() -> ameli_ai::types::Model {
        ameli_ai::types::Model {
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

    fn test_agent_options(registry: Arc<ApiRegistry>) -> ameli_agent_core::AgentOptions {
        ameli_agent_core::AgentOptions {
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
            api_registry: Some(registry),
            ..Default::default()
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
    fn get_registered_tools_from_multiple_extensions() {
        let extensions: Vec<Box<dyn Extension>> = vec![
            Box::new(ToolRegisteringExtension),
            Box::new(ToolRegisteringExtension),
        ];
        let runner = ExtensionRunner::from_extensions(&extensions);
        let tools = runner.get_registered_tools();
        assert_eq!(tools.len(), 2);
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

    #[test]
    fn multiple_error_listeners() {
        let count_a = Arc::new(AtomicUsize::new(0));
        let count_b = Arc::new(AtomicUsize::new(0));
        let count_a_clone = count_a.clone();
        let count_b_clone = count_b.clone();

        let runner = ExtensionRunner::from_extensions(&[]);
        runner.on_error(Arc::new(move |_err| {
            count_a_clone.fetch_add(1, Ordering::SeqCst);
        }));
        runner.on_error(Arc::new(move |_err| {
            count_b_clone.fetch_add(1, Ordering::SeqCst);
        }));

        runner.report_error(ExtensionError {
            extension_name: "ext".to_string(),
            event: "evt".to_string(),
            error: "err".to_string(),
        });

        assert_eq!(count_a.load(Ordering::SeqCst), 1);
        assert_eq!(count_b.load(Ordering::SeqCst), 1);
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

    #[test]
    fn install_hooks_with_all_handler_types() {
        let extensions: Vec<Box<dyn Extension>> = vec![
            Box::new(BlockBashExtension),
            Box::new(ToolResultModifierExtension),
            Box::new(ContextTransformExtension),
        ];
        let runner = Arc::new(ExtensionRunner::from_extensions(&extensions));

        let mut options = ameli_agent_core::AgentOptions::default();
        runner.install_hooks(&mut options);

        assert!(options.before_tool_call.is_some());
        assert!(options.after_tool_call.is_some());
        assert!(options.transform_context.is_some());
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
    async fn after_tool_call_returns_none_when_no_patch() {
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
                name: "other_tool".into(),
                arguments: serde_json::json!({}),
                thought_signature: None,
            },
            args: serde_json::json!({}),
            result: AgentToolResult::text("original", serde_json::json!({})),
            is_error: false,
            context: AgentContext {
                system_prompt: String::new(),
                messages: vec![],
                tools: vec![],
            },
        };

        let result = runner.handle_after_tool_call(&ctx, None).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn tool_result_handlers_see_accumulated_state() {
        // Two extensions register tool_result handlers.
        // Handler A patches content to "from-a".
        // Handler B sees Handler A's patched content ("from-a") and patches details.
        // This verifies the doc contract: "Each handler sees the result after
        // previous handler changes."
        use std::sync::Arc as StdArc;

        let seen_by_b: StdArc<std::sync::Mutex<Option<Vec<MediaContentBlock>>>> =
            StdArc::new(std::sync::Mutex::new(None));
        let seen_by_b_clone = seen_by_b.clone();

        let mut api = ExtensionApi::new();
        api.current_extension_name = "handler-a".to_string();

        // Handler A: patches content
        api.on_tool_result(move |event, _ctx| {
            let _ = &event;
            Box::pin(async move {
                Some(ToolResultPatch {
                    content: Some(vec![MediaContentBlock::Text(TextContent::new("from-a"))]),
                    details: None,
                    is_error: None,
                    terminate: None,
                })
            })
        });

        api.current_extension_name = "handler-b".to_string();

        // Handler B: captures what it sees and patches details
        api.on_tool_result(move |event, _ctx| {
            let captured = seen_by_b_clone.clone();
            Box::pin(async move {
                *captured.lock().unwrap_or_else(|e| e.into_inner()) = Some(event.content.clone());
                Some(ToolResultPatch {
                    content: None,
                    details: Some(serde_json::json!({"patched": true})),
                    is_error: None,
                    terminate: None,
                })
            })
        });

        let runner = ExtensionRunner::new(api.into_handlers());

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
                arguments: serde_json::json!({}),
                thought_signature: None,
            },
            args: serde_json::json!({}),
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

        // Handler B should have seen Handler A's patched content, not the original
        let Some(seen) = seen_by_b.lock().unwrap_or_else(|e| e.into_inner()).take() else {
            panic!("handler B should have captured content");
        };
        assert_eq!(seen.len(), 1);
        let Some(first) = seen.first() else {
            panic!("expected at least one block");
        };
        match first {
            MediaContentBlock::Text(t) => assert_eq!(t.text, "from-a"),
            other => panic!("expected text content, got {other:?}"),
        }

        // The combined result should have A's content and B's details
        assert!(result.content.is_some());
        assert!(result.details.is_some());
        let Some(details) = result.details else {
            panic!("expected details");
        };
        assert_eq!(details["patched"], serde_json::Value::Bool(true));
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
        // Empty message should be filtered out
        let Some(first) = result.first() else {
            panic!("expected first message");
        };
        match first {
            AgentMessage::User(u) => match &u.content {
                ameli_ai::types::UserContent::Text(t) => assert_eq!(t, "hello"),
                other => panic!("expected text, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
        let Some(second) = result.get(1) else {
            panic!("expected second message");
        };
        match second {
            AgentMessage::User(u) => match &u.content {
                ameli_ai::types::UserContent::Text(t) => assert_eq!(t, "world"),
                other => panic!("expected text, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
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
                            assert!(t.text.contains("old conversation was about X"));
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
    async fn agent_event_dispatches_turn_end() {
        let received_text = Arc::new(std::sync::Mutex::new(None::<String>));
        let received_text_clone = received_text.clone();

        let mut api = ExtensionApi::new();
        api.current_extension_name = "test".to_string();
        api.on_turn_end(move |event, _ctx| {
            let msg = received_text_clone.clone();
            let text = match &event.message {
                AgentMessage::User(u) => match &u.content {
                    ameli_ai::types::UserContent::Text(t) => Some(t.clone()),
                    _ => None,
                },
                _ => None,
            };
            Box::pin(async move {
                *msg.lock().unwrap_or_else(|e| e.into_inner()) = text;
                Ok(())
            })
        });

        let runner = ExtensionRunner::new(api.into_handlers());
        runner
            .dispatch_agent_event(
                ameli_agent_core::types::AgentEvent::TurnEnd {
                    message: AgentMessage::User(ameli_ai::types::UserMessage::text("test-turn")),
                    tool_results: vec![],
                },
                CancellationToken::new(),
            )
            .await;

        let guard = received_text.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(guard.as_deref(), Some("test-turn"));
    }

    #[tokio::test]
    async fn agent_event_ignores_tool_execution_update() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let mut api = ExtensionApi::new();
        api.current_extension_name = "test".to_string();
        api.on_tool_execution_start(move |_event, _ctx| {
            let count = call_count_clone.clone();
            Box::pin(async move {
                count.fetch_add(1, Ordering::SeqCst);
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

        assert_eq!(call_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn agent_event_dispatches_message_end() {
        let received = Arc::new(std::sync::Mutex::new(false));
        let received_clone = received.clone();

        let mut api = ExtensionApi::new();
        api.current_extension_name = "test".to_string();
        api.on_message_end(move |_event, _ctx| {
            let flag = received_clone.clone();
            Box::pin(async move {
                *flag.lock().unwrap_or_else(|e| e.into_inner()) = true;
                Ok(())
            })
        });

        let runner = ExtensionRunner::new(api.into_handlers());
        runner
            .dispatch_agent_event(
                ameli_agent_core::types::AgentEvent::MessageEnd {
                    message: AgentMessage::User(ameli_ai::types::UserMessage::text("done")),
                },
                CancellationToken::new(),
            )
            .await;

        assert!(*received.lock().unwrap_or_else(|e| e.into_inner()));
    }

    // -- Integration test ---------------------------------------------------

    #[tokio::test]
    async fn wire_notifications_receives_events_from_agent() {
        #[derive(Clone)]
        struct ImmediateProvider;

        impl StreamFn for ImmediateProvider {
            fn stream(
                &self,
                model: &ameli_ai::types::Model,
                _context: ameli_ai::types::Context,
                _options: ameli_ai::types::StreamOptions,
            ) -> ameli_ai::stream::AssistantMessageEventStream {
                let (producer, stream) = create_assistant_message_event_stream();
                let msg = AssistantMessage {
                    content: vec![AssistantContentBlock::Text(TextContent::new("hello"))],
                    api: model.api.clone(),
                    provider: model.provider.clone(),
                    model: model.id.clone(),
                    response_model: None,
                    response_id: None,
                    usage: Usage::default(),
                    stop_reason: ameli_ai::types::StopReason::Stop,
                    error_message: None,
                    timestamp: 0,
                };
                producer.push(ameli_ai::types::AssistantMessageEvent::Done {
                    reason: ameli_ai::types::StopReason::Stop,
                    message: msg,
                });
                producer.end();
                stream
            }
        }

        let event_count = Arc::new(AtomicUsize::new(0));
        let event_count_clone = event_count.clone();
        let notify = Arc::new(tokio::sync::Notify::new());
        let notify_clone = notify.clone();

        let mut api = ExtensionApi::new();
        api.current_extension_name = "test".to_string();
        api.on_agent_start(move |_event, _ctx| {
            let count = event_count_clone.clone();
            let notify = notify_clone.clone();
            Box::pin(async move {
                count.fetch_add(1, Ordering::SeqCst);
                notify.notify_one();
                Ok(())
            })
        });

        let runner = Arc::new(ExtensionRunner::new(api.into_handlers()));

        let registry = Arc::new(ApiRegistry::new());
        registry.register("test-api", Box::new(ImmediateProvider));

        let mut options = test_agent_options(registry);
        runner.install_hooks(&mut options);

        let agent = ArcAgent::new(options);
        let _wiring = runner.wire_notifications(&agent).await;

        let _ = agent.prompt("hello".into()).await;

        // Wait for the notification handler to fire (with a timeout to
        // prevent hangs on failure)
        let result =
            tokio::time::timeout(std::time::Duration::from_secs(2), notify.notified()).await;
        assert!(result.is_ok(), "Timed out waiting for agent_start event");

        // Should have received agent_start at minimum
        assert!(event_count.load(Ordering::SeqCst) > 0);
    }

    #[tokio::test]
    async fn notification_handler_error_reported_to_listener() {
        let error_count = Arc::new(AtomicUsize::new(0));
        let handler2_count = Arc::new(AtomicUsize::new(0));
        let error_count_clone = error_count.clone();
        let handler2_count_clone = handler2_count.clone();

        let mut api = ExtensionApi::new();

        // Handler 1: returns an error
        api.current_extension_name = "failing-ext".to_string();
        api.on_agent_start(move |_event, _ctx| {
            Box::pin(async move { Err(anyhow::anyhow!("handler 1 failed")) })
        });

        // Handler 2: should still run despite handler 1's error
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

        // Error should have been reported to the listener
        assert_eq!(error_count.load(Ordering::SeqCst), 1);
        // Handler 2 should still have been called
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
