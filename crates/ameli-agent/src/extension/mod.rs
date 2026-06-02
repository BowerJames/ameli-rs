//! Extension system — registration, runtime actions, and runner.
//!
//! ```text
//! Extension trait     →  impl Extension for MyExt { fn init(&self, api) }
//!                            ↓
//! ExtensionApi        →  api.on_tool_call(handler), api.register_tool(tool)
//!                     →  api.send_message(...), api.set_model(...)
//!                            ↓
//! ExtensionRunner     →  wires handlers to ArcAgent + AgentLoopConfig
//! ```
//!
//! # Extension lifecycle
//!
//! Extensions implement [`Extension`] and receive an [`Arc<ExtensionApi>`] during
//! [`init`](Extension::init). The `Arc` is long-lived — extensions keep it for
//! the entire session and can call runtime methods from any context (event
//! handlers, background tasks, etc.).
//!
//! `ExtensionApi` serves two roles:
//!
//! - **Registration surface** — `on_xxx()`, `register_tool()`,
//!   `register_command()` push into interior-mutable vectors.
//! - **Runtime action surface** — `send_message()`, `set_model()`,
//!   `get_active_tools()`, etc. delegate to a bound [`ExtensionActions`]
//!   backend.
//!
//! # Events
//!
//! Extension event handlers use one of two invocation modes:
//!
//! - **Sequential chain** — handlers run in order, each seeing accumulated
//!   state from prior handlers.
//! - **First-to-return** — handlers run in order; first `Some` result wins.
//!
//! # Design note
//!
//! `ExtensionApi` is generic over the concrete agent/session types via the
//! [`ExtensionActions`] trait. The ameli-agent crate provides `SessionActions<M>`
//! as the concrete implementation; the extension crate remains agnostic.

pub mod actions;
pub mod context;
pub mod events;
pub mod runner;

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};

use ameli_agent_core::types::{AgentMessage, AgentTool, ThinkingLevel};
use ameli_ai::types::{ImageContent, Model};

use crate::interface::Interface;

pub use actions::{
    AsyncResult, ExtensionActionError, ExtensionActions, MessageDelivery, NoopExtensionActions,
    ToolInfo,
};
pub use context::ExtensionContext;
pub use events::{
    AgentEndEvent, AgentStartEvent, BeforeAgentStartEvent, BeforeAgentStartMessage,
    BeforeAgentStartResult, CommandContext, ContextEvent, ContextResult, ExtensionEvent,
    FormatBranchSummaryEvent, FormatBranchSummaryResult, FormatCompactionSummaryEvent,
    FormatCompactionSummaryResult, MessageEndEvent, MessageEndResult, MessageStartEvent,
    MessageUpdateEvent, RegisteredCommand, SessionShutdownEvent, SessionShutdownReason,
    SessionStartEvent, SessionStartReason, ToolCallEvent, ToolCallResult, ToolExecutionEndEvent,
    ToolExecutionStartEvent, ToolExecutionUpdateEvent, ToolResultEvent, ToolResultPatch,
    TurnEndEvent, TurnStartEvent,
};
pub use runner::{ExtensionError, ExtensionRunner};

/// Named handler wrapper — associates a handler with the extension that
/// registered it.
#[derive(Debug)]
pub struct Named<T> {
    /// Name of the extension that registered this handler.
    pub name: String,
    /// The handler.
    pub handler: T,
}

impl<T> Named<T> {
    fn new(name: String, handler: T) -> Self {
        Self { name, handler }
    }
}

// ---------------------------------------------------------------------------
// Extension trait
// ---------------------------------------------------------------------------

/// Trait for implementing extensions.
///
/// Extensions receive an [`Arc<ExtensionApi>`] during [`init`](Extension::init).
/// The API serves as both a registration surface (subscribe to events, register
/// tools) and a runtime action surface (send messages, query state).
///
/// Extensions should capture the `Arc<ExtensionApi>` and store it for later use
/// — it remains valid for the entire session.
pub trait Extension: Send + Sync {
    /// Initialize the extension.
    ///
    /// Called once during session creation. Use the `api` parameter to register
    /// event handlers, tools, and commands.
    fn init(&self, api: Arc<ExtensionApi>);

    /// Human-readable name for this extension (used in diagnostics).
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Handler type aliases (reduces type_complexity clippy warnings)
// ---------------------------------------------------------------------------

/// Notification handler: `Fn(Event, ExtensionContext) -> AsyncResult<(), anyhow::Error>`.
type NotificationHandler<E> =
    Box<dyn Fn(E, ExtensionContext) -> AsyncResult<(), anyhow::Error> + Send + Sync>;

/// Hook handler: `Fn(Event, ExtensionContext) -> Pin<Box<dyn Future<Output = Option<R>> + Send>>`.
type HookHandler<E, R> = Box<
    dyn Fn(E, ExtensionContext) -> Pin<Box<dyn Future<Output = Option<R>> + Send>> + Send + Sync,
>;

/// Named notification handler.
type NamedNotification<E> = Named<NotificationHandler<E>>;

/// Named hook handler.
type NamedHook<E, R> = Named<HookHandler<E, R>>;

// ---------------------------------------------------------------------------
// Registrations — accumulated during Extension::init()
// ---------------------------------------------------------------------------

/// Interior-mutable registration state. Each `on_xxx` / `register_xxx` call
/// pushes into the appropriate vector.
struct Registrations {
    /// Name of the extension currently being initialized.
    current_extension_name: String,

    // Notification handlers
    agent_start_handlers: Vec<NamedNotification<AgentStartEvent>>,
    agent_end_handlers: Vec<NamedNotification<AgentEndEvent>>,
    turn_start_handlers: Vec<NamedNotification<TurnStartEvent>>,
    turn_end_handlers: Vec<NamedNotification<TurnEndEvent>>,
    message_start_handlers: Vec<NamedNotification<MessageStartEvent>>,
    message_update_handlers: Vec<NamedNotification<MessageUpdateEvent>>,
    tool_execution_start_handlers: Vec<NamedNotification<ToolExecutionStartEvent>>,
    tool_execution_update_handlers: Vec<NamedNotification<ToolExecutionUpdateEvent>>,
    tool_execution_end_handlers: Vec<NamedNotification<ToolExecutionEndEvent>>,
    session_start_handlers: Vec<NamedNotification<SessionStartEvent>>,
    session_shutdown_handlers: Vec<NamedNotification<SessionShutdownEvent>>,

    // Hook handlers (return Option<ResultType>)
    message_end_handlers: Vec<NamedHook<MessageEndEvent, MessageEndResult>>,
    tool_call_handlers: Vec<NamedHook<ToolCallEvent, ToolCallResult>>,
    tool_result_handlers: Vec<NamedHook<ToolResultEvent, ToolResultPatch>>,
    context_handlers: Vec<NamedHook<ContextEvent, ContextResult>>,
    before_agent_start_handlers:
        Vec<NamedHook<events::BeforeAgentStartEvent, events::BeforeAgentStartResult>>,
    format_compaction_summary_handlers:
        Vec<NamedHook<FormatCompactionSummaryEvent, FormatCompactionSummaryResult>>,
    format_branch_summary_handlers:
        Vec<NamedHook<FormatBranchSummaryEvent, FormatBranchSummaryResult>>,

    // Commands
    commands: Vec<Named<CommandHandler>>,

    // Tools
    tools: Vec<Named<Arc<dyn AgentTool>>>,
}

impl Registrations {
    fn new() -> Self {
        Self {
            current_extension_name: String::new(),
            agent_start_handlers: Vec::new(),
            agent_end_handlers: Vec::new(),
            turn_start_handlers: Vec::new(),
            turn_end_handlers: Vec::new(),
            message_start_handlers: Vec::new(),
            message_update_handlers: Vec::new(),
            message_end_handlers: Vec::new(),
            tool_execution_start_handlers: Vec::new(),
            tool_execution_update_handlers: Vec::new(),
            tool_execution_end_handlers: Vec::new(),
            session_start_handlers: Vec::new(),
            session_shutdown_handlers: Vec::new(),
            tool_call_handlers: Vec::new(),
            tool_result_handlers: Vec::new(),
            context_handlers: Vec::new(),
            before_agent_start_handlers: Vec::new(),
            format_compaction_summary_handlers: Vec::new(),
            format_branch_summary_handlers: Vec::new(),
            commands: Vec::new(),
            tools: Vec::new(),
        }
    }
}

/// Type alias for command handler functions.
type CommandHandler =
    Box<dyn Fn(String, events::CommandContext) -> AsyncResult<(), anyhow::Error> + Send + Sync>;

/// Extracted handlers from an `ExtensionApi` after all extensions are initialized.
/// Used to construct an [`ExtensionRunner`].
pub struct ExtensionHandlers {
    pub agent_start_handlers: Vec<NamedNotification<AgentStartEvent>>,
    pub agent_end_handlers: Vec<NamedNotification<AgentEndEvent>>,
    pub turn_start_handlers: Vec<NamedNotification<TurnStartEvent>>,
    pub turn_end_handlers: Vec<NamedNotification<TurnEndEvent>>,
    pub message_start_handlers: Vec<NamedNotification<MessageStartEvent>>,
    pub message_update_handlers: Vec<NamedNotification<MessageUpdateEvent>>,
    pub tool_execution_start_handlers: Vec<NamedNotification<ToolExecutionStartEvent>>,
    pub tool_execution_update_handlers: Vec<NamedNotification<ToolExecutionUpdateEvent>>,
    pub tool_execution_end_handlers: Vec<NamedNotification<ToolExecutionEndEvent>>,
    pub session_start_handlers: Vec<NamedNotification<SessionStartEvent>>,
    pub session_shutdown_handlers: Vec<NamedNotification<SessionShutdownEvent>>,

    pub message_end_handlers: Vec<NamedHook<MessageEndEvent, MessageEndResult>>,
    pub tool_call_handlers: Vec<NamedHook<ToolCallEvent, ToolCallResult>>,
    pub tool_result_handlers: Vec<NamedHook<ToolResultEvent, ToolResultPatch>>,
    pub context_handlers: Vec<NamedHook<ContextEvent, ContextResult>>,
    pub before_agent_start_handlers:
        Vec<NamedHook<events::BeforeAgentStartEvent, events::BeforeAgentStartResult>>,
    pub format_compaction_summary_handlers:
        Vec<NamedHook<FormatCompactionSummaryEvent, FormatCompactionSummaryResult>>,
    pub format_branch_summary_handlers:
        Vec<NamedHook<FormatBranchSummaryEvent, FormatBranchSummaryResult>>,

    pub commands: Vec<Named<CommandHandler>>,
    pub tools: Vec<Named<Arc<dyn AgentTool>>>,
}

impl ExtensionHandlers {
    /// Returns `true` if no handlers of any kind have been registered.
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
            && self.commands.is_empty()
            && self.tools.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ExtensionApi — the main API surface for extensions
// ---------------------------------------------------------------------------

/// The primary API surface for extensions.
///
/// `ExtensionApi` is passed to [`Extension::init`] wrapped in an `Arc`.
/// Extensions use it for two purposes:
///
/// 1. **Registration** — subscribe to events (`on_agent_start`, `on_tool_call`,
///    etc.), register tools and commands.
/// 2. **Runtime actions** — send messages, query model state, control the agent.
///
/// Runtime actions delegate to an [`ExtensionActions`] backend that is bound at
/// construction time. The backend is always bound (never in a "not bound" state).
pub struct ExtensionApi {
    /// Interior-mutable registration state.
    registrations: Mutex<Registrations>,

    /// Actions backend — always bound.
    actions: Arc<dyn ExtensionActions>,

    /// UI interface — set after construction, used to create ExtensionContext.
    interface: RwLock<Option<Arc<dyn Interface>>>,

    /// Next-turn message queue for deferred message delivery.
    next_turn_queue: Mutex<Vec<AgentMessage>>,
}

impl ExtensionApi {
    /// Create a new `ExtensionApi` with the given actions backend.
    pub fn new(actions: Arc<dyn ExtensionActions>) -> Self {
        Self {
            registrations: Mutex::new(Registrations::new()),
            actions,
            interface: RwLock::new(None),
            next_turn_queue: Mutex::new(Vec::new()),
        }
    }

    // -----------------------------------------------------------------------
    // Registration methods
    // -----------------------------------------------------------------------

    /// Register a handler for `AgentStart` events.
    pub fn on_agent_start(
        &self,
        handler: impl Fn(AgentStartEvent, ExtensionContext) -> AsyncResult<(), anyhow::Error>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.agent_start_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `AgentEnd` events.
    pub fn on_agent_end(
        &self,
        handler: impl Fn(AgentEndEvent, ExtensionContext) -> AsyncResult<(), anyhow::Error>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.agent_end_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `TurnStart` events.
    pub fn on_turn_start(
        &self,
        handler: impl Fn(TurnStartEvent, ExtensionContext) -> AsyncResult<(), anyhow::Error>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.turn_start_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `TurnEnd` events.
    pub fn on_turn_end(
        &self,
        handler: impl Fn(TurnEndEvent, ExtensionContext) -> AsyncResult<(), anyhow::Error>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.turn_end_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `MessageStart` events.
    pub fn on_message_start(
        &self,
        handler: impl Fn(MessageStartEvent, ExtensionContext) -> AsyncResult<(), anyhow::Error>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.message_start_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `MessageUpdate` events.
    pub fn on_message_update(
        &self,
        handler: impl Fn(MessageUpdateEvent, ExtensionContext) -> AsyncResult<(), anyhow::Error>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.message_update_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `MessageEnd` events (hook — can modify message).
    pub fn on_message_end(
        &self,
        handler: impl Fn(
                MessageEndEvent,
                ExtensionContext,
            ) -> Pin<Box<dyn Future<Output = Option<MessageEndResult>> + Send>>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.message_end_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `ToolExecutionStart` events.
    pub fn on_tool_execution_start(
        &self,
        handler: impl Fn(ToolExecutionStartEvent, ExtensionContext) -> AsyncResult<(), anyhow::Error>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.tool_execution_start_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `ToolExecutionUpdate` events.
    pub fn on_tool_execution_update(
        &self,
        handler: impl Fn(ToolExecutionUpdateEvent, ExtensionContext) -> AsyncResult<(), anyhow::Error>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.tool_execution_update_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `ToolExecutionEnd` events.
    pub fn on_tool_execution_end(
        &self,
        handler: impl Fn(ToolExecutionEndEvent, ExtensionContext) -> AsyncResult<(), anyhow::Error>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.tool_execution_end_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `SessionStart` events.
    pub fn on_session_start(
        &self,
        handler: impl Fn(SessionStartEvent, ExtensionContext) -> AsyncResult<(), anyhow::Error>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.session_start_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `SessionShutdown` events.
    pub fn on_session_shutdown(
        &self,
        handler: impl Fn(SessionShutdownEvent, ExtensionContext) -> AsyncResult<(), anyhow::Error>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.session_shutdown_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `ToolCall` events (hook — can block tool).
    pub fn on_tool_call(
        &self,
        handler: impl Fn(
                ToolCallEvent,
                ExtensionContext,
            ) -> Pin<Box<dyn Future<Output = Option<ToolCallResult>> + Send>>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.tool_call_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `ToolResult` events (hook — can patch result).
    pub fn on_tool_result(
        &self,
        handler: impl Fn(
                ToolResultEvent,
                ExtensionContext,
            ) -> Pin<Box<dyn Future<Output = Option<ToolResultPatch>> + Send>>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.tool_result_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `Context` events (hook — can modify context).
    pub fn on_context(
        &self,
        handler: impl Fn(
                ContextEvent,
                ExtensionContext,
            ) -> Pin<Box<dyn Future<Output = Option<ContextResult>> + Send>>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.context_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `BeforeAgentStart` events.
    pub fn on_before_agent_start(
        &self,
        handler: impl Fn(
                events::BeforeAgentStartEvent,
                ExtensionContext,
            )
                -> Pin<Box<dyn Future<Output = Option<events::BeforeAgentStartResult>> + Send>>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.before_agent_start_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `FormatCompactionSummary` events (first-to-return mode).
    pub fn on_format_compaction_summary(
        &self,
        handler: impl Fn(
                FormatCompactionSummaryEvent,
                ExtensionContext,
            )
                -> Pin<Box<dyn Future<Output = Option<FormatCompactionSummaryResult>> + Send>>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.format_compaction_summary_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a handler for `FormatBranchSummary` events (first-to-return mode).
    pub fn on_format_branch_summary(
        &self,
        handler: impl Fn(
                FormatBranchSummaryEvent,
                ExtensionContext,
            )
                -> Pin<Box<dyn Future<Output = Option<FormatBranchSummaryResult>> + Send>>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.format_branch_summary_handlers
            .push(Named::new(name, Box::new(handler)));
    }

    /// Register a named command.
    pub fn register_command(
        &self,
        command_name: &str,
        handler: impl Fn(String, events::CommandContext) -> AsyncResult<(), anyhow::Error>
            + Send
            + Sync
            + 'static,
    ) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        regs.commands
            .push(Named::new(command_name.to_string(), Box::new(handler)));
    }

    /// Register a tool.
    pub fn register_tool(&self, tool: Arc<dyn AgentTool>) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        let name = regs.current_extension_name.clone();
        regs.tools.push(Named::new(name, tool));
    }

    // -----------------------------------------------------------------------
    // Runtime action methods
    // -----------------------------------------------------------------------

    /// Send a message to the agent with the specified delivery mode.
    ///
    /// See [`MessageDelivery`] for the semantics of each mode.
    pub fn send_message(
        &self,
        msg: AgentMessage,
        delivery: MessageDelivery,
    ) -> AsyncResult<(), ExtensionActionError> {
        self.actions.send_message(msg, delivery)
    }

    /// Send a text message from the user.
    pub fn send_user_message(
        &self,
        text: String,
        images: Vec<ImageContent>,
        delivery: MessageDelivery,
    ) -> AsyncResult<(), ExtensionActionError> {
        self.actions.send_user_message(text, images, delivery)
    }

    /// Append a custom entry to the session history.
    pub fn append_entry(
        &self,
        custom_type: &str,
        data: Option<serde_json::Value>,
    ) -> AsyncResult<(), ExtensionActionError> {
        self.actions.append_entry(custom_type, data)
    }

    /// Get the names of currently active tools.
    pub fn get_active_tools(&self) -> AsyncResult<Vec<String>, ExtensionActionError> {
        self.actions.get_active_tools()
    }

    /// Get info for all registered tools (including inactive ones).
    pub fn get_all_tools(&self) -> Vec<ToolInfo> {
        self.actions.get_all_tools()
    }

    /// Set which tools are active by name.
    pub fn set_active_tools(&self, names: Vec<String>) -> AsyncResult<(), ExtensionActionError> {
        self.actions.set_active_tools(names)
    }

    /// Get the current model.
    pub fn model(&self) -> AsyncResult<Option<Model>, ExtensionActionError> {
        self.actions.model()
    }

    /// Set the model. Returns `Ok(true)` if the model was accepted.
    pub fn set_model(&self, model: Model) -> AsyncResult<bool, ExtensionActionError> {
        self.actions.set_model(model)
    }

    /// Get the current thinking level.
    pub fn get_thinking_level(&self) -> AsyncResult<ThinkingLevel, ExtensionActionError> {
        self.actions.get_thinking_level()
    }

    /// Set the thinking level.
    pub fn set_thinking_level(
        &self,
        level: ThinkingLevel,
    ) -> AsyncResult<(), ExtensionActionError> {
        self.actions.set_thinking_level(level)
    }

    /// Get the current system prompt.
    pub fn get_system_prompt(&self) -> AsyncResult<String, ExtensionActionError> {
        self.actions.get_system_prompt()
    }

    /// Check if there are pending messages in the agent's queues.
    pub fn has_pending_messages(&self) -> AsyncResult<bool, ExtensionActionError> {
        self.actions.has_pending_messages()
    }

    /// Request the agent to abort the current run.
    pub fn abort(&self) {
        self.actions.abort();
    }

    /// Check if the agent is currently idle (no active run).
    pub fn is_idle(&self) -> AsyncResult<bool, ExtensionActionError> {
        self.actions.is_idle()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Set the UI interface for ExtensionContext creation.
    pub fn set_interface(&self, interface: Arc<dyn Interface>) {
        let mut guard = self.interface.write().unwrap_or_else(|e| e.into_inner());
        *guard = Some(interface);
    }

    /// Set the current extension name (for testing purposes).
    #[cfg(test)]
    pub fn set_current_extension_name(&self, name: &str) {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        regs.current_extension_name = name.to_string();
    }

    /// Get a reference to the interface, if set.
    pub(crate) fn get_interface(&self) -> Option<Arc<dyn Interface>> {
        self.interface
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Queue a message for delivery on the next turn.
    pub(crate) fn queue_next_turn_message(&self, msg: AgentMessage) {
        let mut guard = self
            .next_turn_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        guard.push(msg);
    }

    /// Drain all queued next-turn messages.
    pub(crate) fn drain_next_turn_messages(&self) -> Vec<AgentMessage> {
        let mut guard = self
            .next_turn_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *guard)
    }

    /// Take all registered handlers, leaving empty vectors in their place.
    /// Called once after all extensions are initialized.
    fn take_handlers(&self) -> ExtensionHandlers {
        let mut regs = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        ExtensionHandlers {
            agent_start_handlers: std::mem::take(&mut regs.agent_start_handlers),
            agent_end_handlers: std::mem::take(&mut regs.agent_end_handlers),
            turn_start_handlers: std::mem::take(&mut regs.turn_start_handlers),
            turn_end_handlers: std::mem::take(&mut regs.turn_end_handlers),
            message_start_handlers: std::mem::take(&mut regs.message_start_handlers),
            message_update_handlers: std::mem::take(&mut regs.message_update_handlers),
            message_end_handlers: std::mem::take(&mut regs.message_end_handlers),
            tool_execution_start_handlers: std::mem::take(&mut regs.tool_execution_start_handlers),
            tool_execution_update_handlers: std::mem::take(
                &mut regs.tool_execution_update_handlers,
            ),
            tool_execution_end_handlers: std::mem::take(&mut regs.tool_execution_end_handlers),
            session_start_handlers: std::mem::take(&mut regs.session_start_handlers),
            session_shutdown_handlers: std::mem::take(&mut regs.session_shutdown_handlers),
            tool_call_handlers: std::mem::take(&mut regs.tool_call_handlers),
            tool_result_handlers: std::mem::take(&mut regs.tool_result_handlers),
            context_handlers: std::mem::take(&mut regs.context_handlers),
            before_agent_start_handlers: std::mem::take(&mut regs.before_agent_start_handlers),
            format_compaction_summary_handlers: std::mem::take(
                &mut regs.format_compaction_summary_handlers,
            ),
            format_branch_summary_handlers: std::mem::take(
                &mut regs.format_branch_summary_handlers,
            ),
            commands: std::mem::take(&mut regs.commands),
            tools: std::mem::take(&mut regs.tools),
        }
    }
}

// ---------------------------------------------------------------------------
// init_extensions — entry point
// ---------------------------------------------------------------------------

/// Initialize a list of extensions, returning the API and extracted handlers.
///
/// For each extension:
/// 1. Set `current_extension_name` in the registrations.
/// 2. Call `extension.init(api.clone())`.
///
/// After all extensions are initialized, extract the accumulated handlers.
pub fn init_extensions(
    extensions: &[Box<dyn Extension>],
    actions: Arc<dyn ExtensionActions>,
) -> (Arc<ExtensionApi>, ExtensionHandlers) {
    let api = Arc::new(ExtensionApi::new(actions));

    for ext in extensions {
        {
            let mut regs = api.registrations.lock().unwrap_or_else(|e| e.into_inner());
            regs.current_extension_name = ext.name().to_string();
        }
        ext.init(api.clone());
    }

    let handlers = api.take_handlers();
    (api, handlers)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ameli_agent_core::types::AgentMessage;
    use ameli_ai::types::Model;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A simple no-op actions backend for testing.
    struct TestActions;

    impl ExtensionActions for TestActions {
        fn send_message(
            &self,
            _msg: AgentMessage,
            _delivery: MessageDelivery,
        ) -> AsyncResult<(), ExtensionActionError> {
            Box::pin(async { Ok(()) })
        }

        fn send_user_message(
            &self,
            _text: String,
            _images: Vec<ImageContent>,
            _delivery: MessageDelivery,
        ) -> AsyncResult<(), ExtensionActionError> {
            Box::pin(async { Ok(()) })
        }

        fn append_entry(
            &self,
            _custom_type: &str,
            _data: Option<serde_json::Value>,
        ) -> AsyncResult<(), ExtensionActionError> {
            Box::pin(async { Ok(()) })
        }

        fn get_active_tools(&self) -> AsyncResult<Vec<String>, ExtensionActionError> {
            Box::pin(async { Ok(Vec::new()) })
        }

        fn get_all_tools(&self) -> Vec<ToolInfo> {
            Vec::new()
        }

        fn set_active_tools(&self, _names: Vec<String>) -> AsyncResult<(), ExtensionActionError> {
            Box::pin(async { Ok(()) })
        }

        fn model(&self) -> AsyncResult<Option<Model>, ExtensionActionError> {
            Box::pin(async { Ok(None) })
        }

        fn set_model(&self, _model: Model) -> AsyncResult<bool, ExtensionActionError> {
            Box::pin(async { Ok(true) })
        }

        fn get_thinking_level(&self) -> AsyncResult<ThinkingLevel, ExtensionActionError> {
            Box::pin(async { Ok(ThinkingLevel::Off) })
        }

        fn set_thinking_level(
            &self,
            _level: ThinkingLevel,
        ) -> AsyncResult<(), ExtensionActionError> {
            Box::pin(async { Ok(()) })
        }

        fn get_system_prompt(&self) -> AsyncResult<String, ExtensionActionError> {
            Box::pin(async { Ok(String::new()) })
        }

        fn has_pending_messages(&self) -> AsyncResult<bool, ExtensionActionError> {
            Box::pin(async { Ok(false) })
        }

        fn abort(&self) {}

        fn is_idle(&self) -> AsyncResult<bool, ExtensionActionError> {
            Box::pin(async { Ok(true) })
        }
    }

    fn test_api() -> Arc<ExtensionApi> {
        Arc::new(ExtensionApi::new(Arc::new(TestActions)))
    }

    struct TestExtension {
        name: String,
    }

    impl TestExtension {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    impl Extension for TestExtension {
        fn init(&self, api: Arc<ExtensionApi>) {
            api.on_agent_start(move |_event, _ctx| Box::pin(async { Ok(()) }));
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[tokio::test]
    async fn test_register_handler() {
        let api = test_api();
        let call_count = Arc::new(AtomicUsize::new(0));
        let count_clone = call_count.clone();

        api.on_agent_start(move |_event, _ctx| {
            let count = count_clone.clone();
            Box::pin(async move {
                count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        });

        let handlers = api.take_handlers();
        assert_eq!(handlers.agent_start_handlers.len(), 1);

        // Call the handler (with a dummy context)
        let ctx = ExtensionContext::new(api.clone(), None);
        let _ = (handlers.agent_start_handlers[0].handler)(AgentStartEvent, ctx).await;
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_init_extensions() {
        let ext1: Box<dyn Extension> = Box::new(TestExtension::new("ext1"));
        let ext2: Box<dyn Extension> = Box::new(TestExtension::new("ext2"));
        let extensions: Vec<Box<dyn Extension>> = vec![ext1, ext2];

        let actions: Arc<dyn ExtensionActions> = Arc::new(TestActions);
        let (_api, handlers) = init_extensions(&extensions, actions);

        assert_eq!(handlers.agent_start_handlers.len(), 2);
        assert_eq!(handlers.agent_start_handlers[0].name, "ext1");
        assert_eq!(handlers.agent_start_handlers[1].name, "ext2");
    }

    #[tokio::test]
    async fn test_multiple_handlers_same_event() {
        let api = test_api();
        let count = Arc::new(AtomicUsize::new(0));

        let c1 = count.clone();
        api.on_agent_start(move |_event, _ctx| {
            let c = c1.clone();
            Box::pin(async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        });

        let c2 = count.clone();
        api.on_agent_start(move |_event, _ctx| {
            let c = c2.clone();
            Box::pin(async move {
                c.fetch_add(10, Ordering::SeqCst);
                Ok(())
            })
        });

        let handlers = api.take_handlers();
        assert_eq!(handlers.agent_start_handlers.len(), 2);

        let ctx = ExtensionContext::new(api.clone(), None);
        for h in &handlers.agent_start_handlers {
            let _ = (h.handler)(AgentStartEvent, ctx.clone()).await;
        }
        assert_eq!(count.load(Ordering::SeqCst), 11);
    }

    #[test]
    fn test_register_tool() {
        use ameli_agent_core::types::{AgentTool, AgentToolResult, ToolExecutionMode};
        use ameli_ai::types::Tool;
        use std::future::Future;
        use std::pin::Pin;

        struct DummyTool;
        impl AgentTool for DummyTool {
            fn name(&self) -> String {
                "test_tool".to_string()
            }
            fn label(&self) -> &str {
                "test_tool"
            }
            fn tool_definition(&self) -> Tool {
                Tool {
                    name: "test_tool".to_string(),
                    description: "A test tool".to_string(),
                    parameters: serde_json::json!({"type": "object"}),
                }
            }
            fn prepare_arguments(&self, args: serde_json::Value) -> serde_json::Value {
                args
            }
            fn execute(
                &self,
                _tool_name: &str,
                _args: serde_json::Value,
                _cancel: Option<tokio_util::sync::CancellationToken>,
            ) -> Pin<Box<dyn Future<Output = AgentToolResult> + Send>> {
                Box::pin(async { AgentToolResult::text("ok", serde_json::json!(true)) })
            }
            fn fmt_debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "DummyTool")
            }
            fn execution_mode(&self) -> Option<ToolExecutionMode> {
                None
            }
        }

        let api = test_api();
        api.register_tool(Arc::new(DummyTool));

        let handlers = api.take_handlers();
        assert_eq!(handlers.tools.len(), 1);
        assert_eq!(handlers.tools[0].handler.name(), "test_tool");
    }

    #[tokio::test]
    async fn test_runtime_actions_delegate() {
        let api = test_api();

        // These should not panic — they delegate to TestActions (no-ops)
        assert!(api.get_active_tools().await.unwrap().is_empty());
        assert!(api.get_all_tools().is_empty());
        assert!(api.model().await.unwrap().is_none());
        assert_eq!(api.get_thinking_level().await.unwrap(), ThinkingLevel::Off);
        assert!(api.get_system_prompt().await.unwrap().is_empty());
        assert!(!api.has_pending_messages().await.unwrap());
        assert!(api.is_idle().await.unwrap());
    }

    #[test]
    fn test_next_turn_queue() {
        let api = test_api();

        assert!(api.drain_next_turn_messages().is_empty());

        api.queue_next_turn_message(AgentMessage::User(ameli_ai::types::UserMessage::text(
            "hello",
        )));
        api.queue_next_turn_message(AgentMessage::User(ameli_ai::types::UserMessage::text(
            "world",
        )));

        let msgs = api.drain_next_turn_messages();
        assert_eq!(msgs.len(), 2);

        // After drain, queue should be empty
        assert!(api.drain_next_turn_messages().is_empty());
    }
}
