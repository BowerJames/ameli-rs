//! Extension system for ameli-agent.
//!
//! This module defines the core extension API — the trait extensions implement,
//! the event types they subscribe to, and the registration surface for tools,
//! hooks, and commands.
//!
//! # Architecture
//!
//! ```text
//! Extension trait     →  impl Extension for MyExt { fn init(&self, api) }
//!                            ↓
//! ExtensionApi        →  api.on_tool_call(handler), api.register_tool(tool)
//!                            ↓
//! ExtensionRunner     →  wires handlers to ArcAgent + AgentLoopConfig
//! ```
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

pub mod context;
pub mod events;
pub mod runner;

pub use context::ExtensionContext;
pub use events::*;
pub use runner::{ExtensionError, ExtensionRunner};

use ameli_agent_core::types::AgentTool;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Pinned, boxed, sendable future returned by extension handlers.
type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

// Handler function types for notification events (fire-and-forget).
//
// Notification handlers return `anyhow::Result<()>` so errors can be
// captured by the runner and reported to registered error listeners.
type AgentStartHandler =
    Box<dyn Fn(AgentStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync>;
type AgentEndHandler =
    Box<dyn Fn(AgentEndEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync>;
type TurnStartHandler =
    Box<dyn Fn(TurnStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync>;
type TurnEndHandler =
    Box<dyn Fn(TurnEndEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync>;
type MessageStartHandler =
    Box<dyn Fn(MessageStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync>;
type MessageUpdateHandler = Box<
    dyn Fn(MessageUpdateEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync,
>;
type ToolExecutionStartHandler = Box<
    dyn Fn(ToolExecutionStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
        + Send
        + Sync,
>;
type ToolExecutionUpdateHandler = Box<
    dyn Fn(ToolExecutionUpdateEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
        + Send
        + Sync,
>;
type ToolExecutionEndHandler = Box<
    dyn Fn(ToolExecutionEndEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync,
>;
type SessionStartHandler =
    Box<dyn Fn(SessionStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync>;
type SessionShutdownHandler = Box<
    dyn Fn(SessionShutdownEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync,
>;

// Handler function types for hook events.
type ToolCallHandler =
    Box<dyn Fn(ToolCallEvent, ExtensionContext) -> BoxFuture<Option<ToolCallResult>> + Send + Sync>;
type ToolResultHandler = Box<
    dyn Fn(ToolResultEvent, ExtensionContext) -> BoxFuture<Option<ToolResultPatch>> + Send + Sync,
>;
type ContextHandler =
    Box<dyn Fn(ContextEvent, ExtensionContext) -> BoxFuture<Option<ContextResult>> + Send + Sync>;
type BeforeAgentStartHandler = Box<
    dyn Fn(BeforeAgentStartEvent, ExtensionContext) -> BoxFuture<Option<BeforeAgentStartResult>>
        + Send
        + Sync,
>;
type MessageEndHandler = Box<
    dyn Fn(MessageEndEvent, ExtensionContext) -> BoxFuture<Option<MessageEndResult>> + Send + Sync,
>;
type FormatCompactionSummaryHandler = Box<
    dyn Fn(
            FormatCompactionSummaryEvent,
            ExtensionContext,
        ) -> BoxFuture<Option<FormatCompactionSummaryResult>>
        + Send
        + Sync,
>;
type FormatBranchSummaryHandler = Box<
    dyn Fn(
            FormatBranchSummaryEvent,
            ExtensionContext,
        ) -> BoxFuture<Option<FormatBranchSummaryResult>>
        + Send
        + Sync,
>;

// ---------------------------------------------------------------------------
// Named handler wrappers
// ---------------------------------------------------------------------------

/// Wraps a handler with its registering extension's name for error attribution.
pub(crate) struct Named<H> {
    pub(crate) extension_name: String,
    pub(crate) handler: H,
}

impl<H> Named<H> {
    fn new(extension_name: String, handler: H) -> Self {
        Self {
            extension_name,
            handler,
        }
    }
}

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
///                 Ok(())
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
/// `register_tool()` to add LLM-callable tools. The
/// [`ExtensionRunner`] extracts these registrations and wires them to the
/// agent loop.
///
/// # Handler contract
///
/// Handlers must not panic. Notification handlers return `anyhow::Result<()>`;
/// errors are reported to registered error listeners and do not stop dispatch
/// to subsequent handlers. Hook handlers return `Option<ResultType>` — return
/// `None` to allow default behavior.
pub struct ExtensionApi {
    /// Name of the extension currently being initialized. Set by
    /// [`init_extensions`] before calling each extension's `init`.
    current_extension_name: String,

    // Notification handlers (fire-and-forget)
    agent_start_handlers: Vec<Named<AgentStartHandler>>,
    agent_end_handlers: Vec<Named<AgentEndHandler>>,
    turn_start_handlers: Vec<Named<TurnStartHandler>>,
    turn_end_handlers: Vec<Named<TurnEndHandler>>,
    message_start_handlers: Vec<Named<MessageStartHandler>>,
    message_update_handlers: Vec<Named<MessageUpdateHandler>>,
    tool_execution_start_handlers: Vec<Named<ToolExecutionStartHandler>>,
    tool_execution_update_handlers: Vec<Named<ToolExecutionUpdateHandler>>,
    tool_execution_end_handlers: Vec<Named<ToolExecutionEndHandler>>,
    session_start_handlers: Vec<Named<SessionStartHandler>>,
    session_shutdown_handlers: Vec<Named<SessionShutdownHandler>>,

    // Hook handlers
    tool_call_handlers: Vec<Named<ToolCallHandler>>,
    tool_result_handlers: Vec<Named<ToolResultHandler>>,
    context_handlers: Vec<Named<ContextHandler>>,
    before_agent_start_handlers: Vec<Named<BeforeAgentStartHandler>>,
    message_end_handlers: Vec<Named<MessageEndHandler>>,
    format_compaction_summary_handlers: Vec<Named<FormatCompactionSummaryHandler>>,
    format_branch_summary_handlers: Vec<Named<FormatBranchSummaryHandler>>,

    // Commands
    commands: Vec<RegisteredCommand>,

    // Registered tools
    tools: Vec<Arc<dyn AgentTool>>,
}

impl ExtensionApi {
    /// Create a new, empty API surface.
    pub fn new() -> Self {
        Self {
            current_extension_name: String::new(),
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
            commands: Vec::new(),
            tools: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Notification event registration (fire-and-forget)
    // -----------------------------------------------------------------------

    /// Subscribe to agent loop start.
    pub fn on_agent_start(
        &mut self,
        handler: impl Fn(AgentStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.agent_start_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
    }

    /// Subscribe to agent loop end.
    pub fn on_agent_end(
        &mut self,
        handler: impl Fn(AgentEndEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.agent_end_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
    }

    /// Subscribe to turn start.
    pub fn on_turn_start(
        &mut self,
        handler: impl Fn(TurnStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.turn_start_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
    }

    /// Subscribe to turn end.
    pub fn on_turn_end(
        &mut self,
        handler: impl Fn(TurnEndEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.turn_end_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
    }

    /// Subscribe to message start (user, assistant, or tool result).
    pub fn on_message_start(
        &mut self,
        handler: impl Fn(MessageStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.message_start_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
    }

    /// Subscribe to message streaming updates (assistant messages only).
    pub fn on_message_update(
        &mut self,
        handler: impl Fn(MessageUpdateEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.message_update_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
    }

    /// Subscribe to message end (user, assistant, or tool result).
    ///
    /// This is a **sequential chain hook**: handlers run in registration order
    /// and can return a replacement message that preserves the original role.
    /// Each handler sees the result of prior handlers.
    pub fn on_message_end(
        &mut self,
        handler: impl Fn(MessageEndEvent, ExtensionContext) -> BoxFuture<Option<MessageEndResult>>
            + Send
            + Sync
            + 'static,
    ) {
        self.message_end_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
    }

    /// Subscribe to tool execution start.
    pub fn on_tool_execution_start(
        &mut self,
        handler: impl Fn(ToolExecutionStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.tool_execution_start_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
    }

    /// Subscribe to tool execution updates (partial/streaming output).
    pub fn on_tool_execution_update(
        &mut self,
        handler: impl Fn(ToolExecutionUpdateEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.tool_execution_update_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
    }

    /// Subscribe to tool execution end.
    pub fn on_tool_execution_end(
        &mut self,
        handler: impl Fn(ToolExecutionEndEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.tool_execution_end_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
    }

    /// Subscribe to session start (emitted when AgentSession is created).
    pub fn on_session_start(
        &mut self,
        handler: impl Fn(SessionStartEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.session_start_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
    }

    /// Subscribe to session shutdown (emitted when AgentSession is shutting
    /// down).
    pub fn on_session_shutdown(
        &mut self,
        handler: impl Fn(SessionShutdownEvent, ExtensionContext) -> BoxFuture<anyhow::Result<()>>
            + Send
            + Sync
            + 'static,
    ) {
        self.session_shutdown_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
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
        self.tool_call_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
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
        self.tool_result_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
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
        self.context_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
    }

    /// Register a hook called before the agent loop starts processing a
    /// prompt.
    ///
    /// **Sequential accumulate**: all handler results are collected. Custom
    /// messages are accumulated in order. The last non-`None` `system_prompt`
    /// wins.
    pub fn on_before_agent_start(
        &mut self,
        handler: impl Fn(BeforeAgentStartEvent, ExtensionContext) -> BoxFuture<Option<BeforeAgentStartResult>>
            + Send
            + Sync
            + 'static,
    ) {
        self.before_agent_start_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
    }

    /// Register a hook called when a compaction summary needs formatting into
    /// an [`AgentMessage`].
    ///
    /// Handlers run in registration order. The first handler to return
    /// `Some(...)` wins. If no handler returns `Some`, the default
    /// conversion wraps the summary in a synthetic user message.
    pub fn on_format_compaction_summary(
        &mut self,
        handler: impl Fn(
                FormatCompactionSummaryEvent,
                ExtensionContext,
            ) -> BoxFuture<Option<FormatCompactionSummaryResult>>
            + Send
            + Sync
            + 'static,
    ) {
        self.format_compaction_summary_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
    }

    /// Register a hook called when a branch summary needs formatting into
    /// an [`AgentMessage`].
    ///
    /// Handlers run in registration order. The first handler to return
    /// `Some(...)` wins. If no handler returns `Some`, the default
    /// conversion wraps the summary in a synthetic user message.
    pub fn on_format_branch_summary(
        &mut self,
        handler: impl Fn(
                FormatBranchSummaryEvent,
                ExtensionContext,
            ) -> BoxFuture<Option<FormatBranchSummaryResult>>
            + Send
            + Sync
            + 'static,
    ) {
        self.format_branch_summary_handlers.push(Named::new(
            self.current_extension_name.clone(),
            Box::new(handler),
        ));
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
        &mut self,
        name: impl Into<String>,
        description: Option<String>,
        handler: Arc<dyn Fn(String, CommandContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync>,
    ) {
        self.commands.push(RegisteredCommand {
            name: name.into(),
            description,
            extension_name: self.current_extension_name.clone(),
            handler,
        });
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
    // Extraction (used by the ExtensionRunner)
    // -----------------------------------------------------------------------

    /// Take all registered handlers and tools, leaving empty vectors.
    ///
    /// Used by the runner to extract registrations after all extensions have
    /// been initialized. The returned [`ExtensionHandlers`] exposes handler
    /// vectors the runner can check directly (e.g.,
    /// `handlers.tool_call_handlers.is_empty()`).
    pub fn into_handlers(self) -> ExtensionHandlers {
        ExtensionHandlers {
            // Notification
            agent_start_handlers: self.agent_start_handlers,
            agent_end_handlers: self.agent_end_handlers,
            turn_start_handlers: self.turn_start_handlers,
            turn_end_handlers: self.turn_end_handlers,
            message_start_handlers: self.message_start_handlers,
            message_update_handlers: self.message_update_handlers,
            tool_execution_start_handlers: self.tool_execution_start_handlers,
            tool_execution_update_handlers: self.tool_execution_update_handlers,
            tool_execution_end_handlers: self.tool_execution_end_handlers,
            session_start_handlers: self.session_start_handlers,
            session_shutdown_handlers: self.session_shutdown_handlers,
            // Hooks
            tool_call_handlers: self.tool_call_handlers,
            tool_result_handlers: self.tool_result_handlers,
            context_handlers: self.context_handlers,
            before_agent_start_handlers: self.before_agent_start_handlers,
            message_end_handlers: self.message_end_handlers,
            format_compaction_summary_handlers: self.format_compaction_summary_handlers,
            format_branch_summary_handlers: self.format_branch_summary_handlers,
            // Commands
            commands: self.commands,
            // Tools
            tools: self.tools,
        }
    }
}

impl Default for ExtensionApi {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ExtensionApi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionApi")
            .field("agent_start", &self.agent_start_handlers.len())
            .field("agent_end", &self.agent_end_handlers.len())
            .field("turn_start", &self.turn_start_handlers.len())
            .field("turn_end", &self.turn_end_handlers.len())
            .field("message_start", &self.message_start_handlers.len())
            .field("message_update", &self.message_update_handlers.len())
            .field("message_end", &self.message_end_handlers.len())
            .field(
                "tool_execution_start",
                &self.tool_execution_start_handlers.len(),
            )
            .field(
                "tool_execution_update",
                &self.tool_execution_update_handlers.len(),
            )
            .field(
                "tool_execution_end",
                &self.tool_execution_end_handlers.len(),
            )
            .field("session_start", &self.session_start_handlers.len())
            .field("session_shutdown", &self.session_shutdown_handlers.len())
            .field("tool_call", &self.tool_call_handlers.len())
            .field("tool_result", &self.tool_result_handlers.len())
            .field("context", &self.context_handlers.len())
            .field(
                "before_agent_start",
                &self.before_agent_start_handlers.len(),
            )
            .field(
                "format_compaction_summary",
                &self.format_compaction_summary_handlers.len(),
            )
            .field(
                "format_branch_summary",
                &self.format_branch_summary_handlers.len(),
            )
            .field("commands", &self.commands.len())
            .field("tools", &self.tools.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ExtensionHandlers — extracted handlers from all extensions
// ---------------------------------------------------------------------------

/// All handlers and tools extracted from extensions after initialization.
///
/// The `ExtensionRunner` consumes this to wire handlers into the agent
/// loop. Produced by [`ExtensionApi::into_handlers`].
pub struct ExtensionHandlers {
    // Notification (fire-and-forget)
    pub(crate) agent_start_handlers: Vec<Named<AgentStartHandler>>,
    pub(crate) agent_end_handlers: Vec<Named<AgentEndHandler>>,
    pub(crate) turn_start_handlers: Vec<Named<TurnStartHandler>>,
    pub(crate) turn_end_handlers: Vec<Named<TurnEndHandler>>,
    pub(crate) message_start_handlers: Vec<Named<MessageStartHandler>>,
    pub(crate) message_update_handlers: Vec<Named<MessageUpdateHandler>>,
    pub(crate) tool_execution_start_handlers: Vec<Named<ToolExecutionStartHandler>>,
    pub(crate) tool_execution_update_handlers: Vec<Named<ToolExecutionUpdateHandler>>,
    pub(crate) tool_execution_end_handlers: Vec<Named<ToolExecutionEndHandler>>,
    pub(crate) session_start_handlers: Vec<Named<SessionStartHandler>>,
    pub(crate) session_shutdown_handlers: Vec<Named<SessionShutdownHandler>>,

    // Hooks
    pub(crate) tool_call_handlers: Vec<Named<ToolCallHandler>>,
    pub(crate) tool_result_handlers: Vec<Named<ToolResultHandler>>,
    pub(crate) context_handlers: Vec<Named<ContextHandler>>,
    pub(crate) before_agent_start_handlers: Vec<Named<BeforeAgentStartHandler>>,
    pub(crate) message_end_handlers: Vec<Named<MessageEndHandler>>,
    pub(crate) format_compaction_summary_handlers: Vec<Named<FormatCompactionSummaryHandler>>,
    pub(crate) format_branch_summary_handlers: Vec<Named<FormatBranchSummaryHandler>>,

    // Commands
    pub(crate) commands: Vec<RegisteredCommand>,

    // Tools
    pub(crate) tools: Vec<Arc<dyn AgentTool>>,
}

impl ExtensionHandlers {
    /// Returns `true` if no handlers, commands, or tools were registered.
    pub fn is_empty(&self) -> bool {
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
            && self.commands.is_empty()
            && self.tools.is_empty()
    }
}

impl fmt::Debug for ExtensionHandlers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionHandlers")
            .field("agent_start", &self.agent_start_handlers.len())
            .field("agent_end", &self.agent_end_handlers.len())
            .field("turn_start", &self.turn_start_handlers.len())
            .field("turn_end", &self.turn_end_handlers.len())
            .field("message_start", &self.message_start_handlers.len())
            .field("message_update", &self.message_update_handlers.len())
            .field("message_end", &self.message_end_handlers.len())
            .field(
                "tool_execution_start",
                &self.tool_execution_start_handlers.len(),
            )
            .field(
                "tool_execution_update",
                &self.tool_execution_update_handlers.len(),
            )
            .field(
                "tool_execution_end",
                &self.tool_execution_end_handlers.len(),
            )
            .field("session_start", &self.session_start_handlers.len())
            .field("session_shutdown", &self.session_shutdown_handlers.len())
            .field("tool_call", &self.tool_call_handlers.len())
            .field("tool_result", &self.tool_result_handlers.len())
            .field("context", &self.context_handlers.len())
            .field(
                "before_agent_start",
                &self.before_agent_start_handlers.len(),
            )
            .field(
                "format_compaction_summary",
                &self.format_compaction_summary_handlers.len(),
            )
            .field(
                "format_branch_summary",
                &self.format_branch_summary_handlers.len(),
            )
            .field("commands", &self.commands.len())
            .field("tools", &self.tools.len())
            .finish()
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
        api.current_extension_name = ext.name().to_string();
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

    struct BeforeAgentStartExtension;

    impl Extension for BeforeAgentStartExtension {
        fn name(&self) -> &str {
            "before-start"
        }

        fn init(&self, api: &mut ExtensionApi) {
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
        fn name(&self) -> &str {
            "message-end"
        }

        fn init(&self, api: &mut ExtensionApi) {
            api.on_message_end(|_event, _ctx| Box::pin(async move { None }));
        }
    }

    struct ToolUpdateExtension;

    impl Extension for ToolUpdateExtension {
        fn name(&self) -> &str {
            "tool-update"
        }

        fn init(&self, api: &mut ExtensionApi) {
            api.on_tool_execution_update(|_event, _ctx| Box::pin(async { Ok(()) }));
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
                let message = params.get("message").and_then(|v| v.as_str()).unwrap_or("");
                AgentToolResult::text(message, serde_json::json!({}))
            })
        }
    }

    #[test]
    fn api_starts_empty() {
        let api = ExtensionApi::new();
        assert!(api.agent_start_handlers.is_empty());
        assert!(api.tool_call_handlers.is_empty());
        assert!(api.tools.is_empty());
        assert!(api.commands.is_empty());
        assert!(api.into_handlers().is_empty());
    }

    #[test]
    fn register_tool() {
        let mut api = ExtensionApi::new();
        api.current_extension_name = "test".to_string();
        api.register_tool(Arc::new(EchoTool));
        let handlers = api.into_handlers();
        assert_eq!(handlers.tools.len(), 1);
        assert_eq!(
            handlers
                .tools
                .first()
                .unwrap_or_else(|| panic!("expected at least one tool"))
                .name(),
            "echo"
        );
    }

    #[test]
    fn init_extensions_collects_registrations() {
        let extensions: Vec<Box<dyn Extension>> =
            vec![Box::new(BlockBashExtension), Box::new(LoggingExtension)];
        let handlers = init_extensions(&extensions);
        assert_eq!(handlers.tool_call_handlers.len(), 1);
        assert_eq!(handlers.agent_start_handlers.len(), 1);
        assert_eq!(handlers.turn_end_handlers.len(), 1);
        assert_eq!(handlers.session_start_handlers.len(), 1);
        assert_eq!(handlers.session_shutdown_handlers.len(), 1);
        assert!(!handlers.is_empty());
    }

    #[test]
    fn register_command() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(CommandExtension)];
        let handlers = init_extensions(&extensions);
        assert_eq!(handlers.commands.len(), 1);
        assert_eq!(handlers.commands[0].name, "greet");
        assert_eq!(
            handlers.commands[0].description.as_deref(),
            Some("Say hello")
        );
        assert_eq!(handlers.commands[0].extension_name, "command-ext");
    }

    #[test]
    fn register_before_agent_start() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(BeforeAgentStartExtension)];
        let handlers = init_extensions(&extensions);
        assert_eq!(handlers.before_agent_start_handlers.len(), 1);
    }

    #[test]
    fn register_tool_execution_update() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(ToolUpdateExtension)];
        let handlers = init_extensions(&extensions);
        assert_eq!(handlers.tool_execution_update_handlers.len(), 1);
    }

    #[test]
    fn register_message_end_hook() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(MessageEndExtension)];
        let handlers = init_extensions(&extensions);
        assert_eq!(handlers.message_end_handlers.len(), 1);
    }

    #[test]
    fn has_agent_start_handler() {
        let mut api = ExtensionApi::new();
        api.current_extension_name = "test".to_string();
        assert!(api.agent_start_handlers.is_empty());
        api.on_agent_start(|_, _| Box::pin(async { Ok(()) }));
        assert_eq!(api.agent_start_handlers.len(), 1);
    }

    #[test]
    fn has_tool_call_handler() {
        let mut api = ExtensionApi::new();
        api.current_extension_name = "test".to_string();
        assert!(api.tool_call_handlers.is_empty());
        api.on_tool_call(|_, _| Box::pin(async { None }));
        assert_eq!(api.tool_call_handlers.len(), 1);
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

        let handler = handlers
            .tool_call_handlers
            .first()
            .unwrap_or_else(|| panic!("expected at least one tool_call handler"));
        let result = (handler.handler)(event, ctx).await;
        let Some(result) = result else {
            panic!("tool_call handler for 'bash' should return Some");
        };
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

        let handler = handlers
            .tool_call_handlers
            .first()
            .unwrap_or_else(|| panic!("expected at least one tool_call handler"));
        let result = (handler.handler)(event, ctx).await;
        assert!(result.is_none());
    }

    #[test]
    fn register_multiple_tools() {
        let mut api = ExtensionApi::new();
        api.current_extension_name = "test".to_string();
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

    #[test]
    fn named_handler_carries_extension_name() {
        let extensions: Vec<Box<dyn Extension>> = vec![Box::new(BlockBashExtension)];
        let handlers = init_extensions(&extensions);
        let named = handlers
            .tool_call_handlers
            .first()
            .unwrap_or_else(|| panic!("expected at least one tool_call handler"));
        assert_eq!(named.extension_name, "block-bash");
    }
}
