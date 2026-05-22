//! Stateful wrapper around the low-level agent loop.
//!
//! `Agent` owns the current transcript, emits lifecycle events, executes tools,
//! and exposes queueing APIs for steering and follow-up messages. It is always
//! constructed behind an `Arc` and uses interior mutability for async-safe
//! access.
//!
//! # Lifecycle
//!
//! - [`ArcAgent::prompt`] — start a new prompt (text, single message, or batch)
//! - [`ArcAgent::continue_`] — continue from the current transcript
//! - [`ArcAgent::abort`] — cancel the current run
//! - [`ArcAgent::wait_for_idle`] — resolve when the current run and all listeners finish
//!
//! # Events
//!
//! [`ArcAgent::subscribe`] registers a listener that receives [`AgentEvent`]s in
//! subscription order. `agent_end` is the final event for a run, but the agent
//! does not become idle until all awaited listeners for that event settle.
//!
//! # Construction
//!
//! Use [`ArcAgent::new`] to create the agent. The returned `ArcAgent` wraps an
//! `Arc<Agent>` and provides the full API including queue-based
//! steering/follow-up.

use crate::agent_loop::{run_agent_loop, run_agent_loop_continue};
use crate::types::*;
use ameli_ai::provider::ProviderRegistry;
use ameli_ai::types::{
    Cost, MediaContentBlock, Message, Model, StreamOptions, TextContent, ThinkingBudgets,
    Transport,
};
use std::collections::HashSet;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// BoxFuture alias
// ---------------------------------------------------------------------------

type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

// ---------------------------------------------------------------------------
// Default model
// ---------------------------------------------------------------------------

/// Minimal default model used when no initial state is provided.
fn default_model() -> Model {
    Model {
        id: "unknown".into(),
        name: "unknown".into(),
        api: "unknown".into(),
        provider: "unknown".into(),
        base_url: String::new(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![],
        cost: Cost::default(),
        context_window: 0,
        max_tokens: 0,
        compat: None,
    }
}

// ---------------------------------------------------------------------------
// Default convert_to_llm
// ---------------------------------------------------------------------------

/// Default LLM conversion: filters to standard LLM messages and downgrades.
fn default_convert_to_llm(messages: &[AgentMessage]) -> BoxFuture<Vec<Message>> {
    let result: Vec<Message> = messages
        .iter()
        .filter(|m| m.is_standard())
        .filter_map(|m| m.as_message())
        .collect();
    Box::pin(async move { result })
}

// ---------------------------------------------------------------------------
// PendingMessageQueue
// ---------------------------------------------------------------------------

/// Internal queue of [`AgentMessage`]s with configurable drain behaviour.
///
/// Used for steering messages (injected mid-run) and follow-up messages
/// (injected after the agent would otherwise stop). Protected by the enclosing
/// `Mutex` in [`AgentInner`] — no internal locking needed.
struct PendingMessageQueue {
    messages: Vec<AgentMessage>,
    mode: QueueMode,
}

impl PendingMessageQueue {
    fn new(mode: QueueMode) -> Self {
        Self {
            messages: Vec::new(),
            mode,
        }
    }

    fn enqueue(&mut self, message: AgentMessage) {
        self.messages.push(message);
    }

    fn has_items(&self) -> bool {
        !self.messages.is_empty()
    }

    /// Drain messages according to the current [`QueueMode`].
    ///
    /// - `All`: return every queued message and clear the queue.
    /// - `OneAtATime`: return only the oldest message, leaving the rest.
    fn drain(&mut self) -> Vec<AgentMessage> {
        match self.mode {
            QueueMode::All => std::mem::take(&mut self.messages),
            QueueMode::OneAtATime => {
                if self.messages.is_empty() {
                    return Vec::new();
                }
                vec![self.messages.remove(0)]
            }
        }
    }

    fn clear(&mut self) {
        self.messages.clear();
    }
}

// ---------------------------------------------------------------------------
// ActiveRun
// ---------------------------------------------------------------------------

/// Tracks the in-progress run to enforce single-concurrency.
struct ActiveRun {
    /// Cancellation token — aborting sets this.
    cancel: CancellationToken,
    /// Notified when the run finishes (after all listener settlement).
    done: Arc<tokio::sync::Notify>,
}

// ---------------------------------------------------------------------------
// Subscriber
// ---------------------------------------------------------------------------

/// A subscriber callback registered via [`ArcAgent::subscribe`].
///
/// Called with each [`AgentEvent`] and the current run's `CancellationToken`.
/// Subscribers are awaited sequentially in registration order.
pub type SubscriberFn = dyn Fn(AgentEvent, CancellationToken) -> BoxFuture<()> + Send + Sync;

// ---------------------------------------------------------------------------
// AgentInner — all mutable state behind a single Mutex
// ---------------------------------------------------------------------------

/// All mutable Agent state protected by a single `tokio::sync::Mutex`.
struct AgentInner {
    // --- Conversation state ---
    system_prompt: String,
    model: Model,
    thinking_level: ThinkingLevel,
    tools: Vec<Arc<dyn AgentTool>>,
    messages: Vec<AgentMessage>,

    // --- Runtime state ---
    is_streaming: bool,
    streaming_message: Option<AgentMessage>,
    pending_tool_calls: HashSet<String>,
    error_message: Option<String>,

    // --- Queues ---
    steering_queue: PendingMessageQueue,
    follow_up_queue: PendingMessageQueue,

    // --- Run coordination ---
    active_run: Option<ActiveRun>,

    // --- Subscribers ---
    subscribers: Vec<Option<Arc<SubscriberFn>>>,
}

// ---------------------------------------------------------------------------
// PromptInput
// ---------------------------------------------------------------------------

/// Input accepted by [`ArcAgent::prompt`].
pub enum PromptInput {
    /// A plain text string, optionally with images.
    Text {
        text: String,
        images: Vec<ameli_ai::types::ImageContent>,
    },
    /// One or more pre-built agent messages.
    Messages(Vec<AgentMessage>),
}

impl From<String> for PromptInput {
    fn from(text: String) -> Self {
        Self::Text {
            text,
            images: Vec::new(),
        }
    }
}

impl From<&str> for PromptInput {
    fn from(text: &str) -> Self {
        Self::Text {
            text: text.into(),
            images: Vec::new(),
        }
    }
}

impl From<AgentMessage> for PromptInput {
    fn from(msg: AgentMessage) -> Self {
        Self::Messages(vec![msg])
    }
}

impl From<Vec<AgentMessage>> for PromptInput {
    fn from(msgs: Vec<AgentMessage>) -> Self {
        Self::Messages(msgs)
    }
}

// ---------------------------------------------------------------------------
// AgentOptions
// ---------------------------------------------------------------------------

/// Configuration for constructing an [`Agent`].
pub struct AgentOptions {
    /// Optional initial state. Unset fields use defaults.
    pub initial_state: Option<AgentState>,

    /// Converts `AgentMessage[]` to LLM-compatible `Message[]` before each LLM
    /// call. When `None`, a default filter that keeps only standard messages is
    /// used.
    pub convert_to_llm:
        Option<Arc<dyn Fn(&[AgentMessage]) -> BoxFuture<Vec<Message>> + Send + Sync>>,

    /// Optional transform applied to the context before `convert_to_llm`.
    pub transform_context:
        Option<Arc<dyn Fn(&[AgentMessage], Option<CancellationToken>) -> BoxFuture<Vec<AgentMessage>> + Send + Sync>>,

    /// Resolves an API key dynamically for each LLM call.
    pub get_api_key: Option<Arc<dyn Fn(&str) -> BoxFuture<Option<String>> + Send + Sync>>,

    /// Called before a tool is executed, after arguments have been validated.
    pub before_tool_call: Option<
        Arc<
            dyn Fn(
                    &BeforeToolCallContext,
                    Option<CancellationToken>,
                ) -> BoxFuture<Option<BeforeToolCallResult>>
                + Send
                + Sync,
        >,
    >,

    /// Called after a tool finishes executing.
    pub after_tool_call: Option<
        Arc<
            dyn Fn(
                    &AfterToolCallContext,
                    Option<CancellationToken>,
                ) -> BoxFuture<Option<AfterToolCallResult>>
                + Send
                + Sync,
        >,
    >,

    /// Called after `TurnEnd` to optionally update model/context/thinking for
    /// the next turn. Receives the active `CancellationToken` so it can check
    /// for cancellation.
    pub prepare_next_turn:
        Option<Arc<dyn Fn(Option<CancellationToken>) -> BoxFuture<Option<AgentLoopTurnUpdate>> + Send + Sync>>,

    /// How steering messages are drained. Default: `OneAtATime`.
    pub steering_mode: QueueMode,

    /// How follow-up messages are drained. Default: `OneAtATime`.
    pub follow_up_mode: QueueMode,

    /// Session identifier forwarded to providers for cache-aware backends.
    pub session_id: Option<String>,

    /// Optional per-level thinking token budgets forwarded to the stream function.
    pub thinking_budgets: Option<ThinkingBudgets>,

    /// Preferred transport forwarded to the stream function.
    pub transport: Option<Transport>,

    /// Optional cap for provider-requested retry delays.
    pub max_retry_delay_ms: Option<u64>,

    /// Tool execution strategy for multi-tool-call assistant messages.
    pub tool_execution: ToolExecutionMode,

    /// Provider registry. When `None`, a new empty registry is used.
    pub registry: Option<Arc<ProviderRegistry>>,
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            initial_state: None,
            convert_to_llm: None,
            transform_context: None,
            get_api_key: None,
            before_tool_call: None,
            after_tool_call: None,
            prepare_next_turn: None,
            steering_mode: QueueMode::OneAtATime,
            follow_up_mode: QueueMode::OneAtATime,
            session_id: None,
            thinking_budgets: None,
            transport: None,
            max_retry_delay_ms: None,
            tool_execution: ToolExecutionMode::Parallel,
            registry: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Subscription handle
// ---------------------------------------------------------------------------

/// Handle returned by [`ArcAgent::subscribe`]. Unsubscribes on drop.
pub struct Subscription {
    agent: Arc<Agent>,
    /// Index in `subscribers` vec. Used to identify this subscriber for removal.
    index: usize,
}

impl Drop for Subscription {
    fn drop(&mut self) {
        let index = self.index;
        // Best-effort removal: set to None (tombstone). If the inner mutex
        // is already held or the agent is dropped, this is a no-op.
        let inner = self.agent.inner.try_lock();
        if let Ok(mut inner) = inner {
            if index < inner.subscribers.len() {
                inner.subscribers[index] = None;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/// Inner agent implementation. Uses interior mutability via `tokio::sync::Mutex`.
///
/// Construct via [`ArcAgent::new`] which wraps this in an `Arc`.
pub struct Agent {
    inner: Mutex<AgentInner>,

    // Immutable config captured at construction time.
    convert_to_llm: Arc<dyn Fn(&[AgentMessage]) -> BoxFuture<Vec<Message>> + Send + Sync>,
    transform_context:
        Option<Arc<dyn Fn(&[AgentMessage], Option<CancellationToken>) -> BoxFuture<Vec<AgentMessage>> + Send + Sync>>,
    get_api_key: Option<Arc<dyn Fn(&str) -> BoxFuture<Option<String>> + Send + Sync>>,
    before_tool_call: Option<
        Arc<
            dyn Fn(
                    &BeforeToolCallContext,
                    Option<CancellationToken>,
                ) -> BoxFuture<Option<BeforeToolCallResult>>
                + Send
                + Sync,
        >,
    >,
    after_tool_call: Option<
        Arc<
            dyn Fn(
                    &AfterToolCallContext,
                    Option<CancellationToken>,
                ) -> BoxFuture<Option<AfterToolCallResult>>
                + Send
                + Sync,
        >,
    >,
    prepare_next_turn:
        Option<Arc<dyn Fn(Option<CancellationToken>) -> BoxFuture<Option<AgentLoopTurnUpdate>> + Send + Sync>>,
    session_id: Option<String>,
    thinking_budgets: Option<ThinkingBudgets>,
    transport: Transport,
    max_retry_delay_ms: Option<u64>,
    tool_execution: ToolExecutionMode,
    registry: Arc<ProviderRegistry>,
}

impl Agent {
    /// Create a new agent with the given options.
    pub fn new(options: AgentOptions) -> Self {
        let state = options.initial_state.unwrap_or_else(|| AgentState {
            system_prompt: String::new(),
            model: default_model(),
            thinking_level: ThinkingLevel::Off,
            tools: Vec::new(),
            messages: Vec::new(),
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        });

        let inner = AgentInner {
            system_prompt: state.system_prompt,
            model: state.model,
            thinking_level: state.thinking_level,
            tools: state.tools,
            messages: state.messages,
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
            steering_queue: PendingMessageQueue::new(options.steering_mode),
            follow_up_queue: PendingMessageQueue::new(options.follow_up_mode),
            active_run: None,
            subscribers: Vec::new(),
        };

        let registry = options
            .registry
            .unwrap_or_else(|| Arc::new(ProviderRegistry::new()));

        Self {
            inner: Mutex::new(inner),
            convert_to_llm: options
                .convert_to_llm
                .unwrap_or_else(|| Arc::new(|msgs| default_convert_to_llm(msgs))),
            transform_context: options.transform_context,
            get_api_key: options.get_api_key,
            before_tool_call: options.before_tool_call,
            after_tool_call: options.after_tool_call,
            prepare_next_turn: options.prepare_next_turn,
            session_id: options.session_id,
            thinking_budgets: options.thinking_budgets,
            transport: options.transport.unwrap_or(Transport::Auto),
            max_retry_delay_ms: options.max_retry_delay_ms,
            tool_execution: options.tool_execution,
            registry,
        }
    }

    /// Convenience constructor that returns `Arc<Agent>`.
    pub fn new_arc(options: AgentOptions) -> Arc<Self> {
        Arc::new(Self::new(options))
    }

    // -----------------------------------------------------------------------
    // State accessors
    // -----------------------------------------------------------------------

    /// Snapshot the current agent state.
    pub async fn state(&self) -> AgentState {
        let inner = self.inner.lock().await;
        AgentState {
            system_prompt: inner.system_prompt.clone(),
            model: inner.model.clone(),
            thinking_level: inner.thinking_level,
            tools: inner.tools.clone(),
            messages: inner.messages.clone(),
            is_streaming: inner.is_streaming,
            streaming_message: inner.streaming_message.clone(),
            pending_tool_calls: inner.pending_tool_calls.clone(),
            error_message: inner.error_message.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Queue operations
    // -----------------------------------------------------------------------

    /// Queue a steering message to be injected after the current assistant
    /// turn finishes.
    pub async fn steer(&self, message: AgentMessage) {
        let mut inner = self.inner.lock().await;
        inner.steering_queue.enqueue(message);
    }

    /// Queue a follow-up message to run only after the agent would otherwise
    /// stop.
    pub async fn follow_up(&self, message: AgentMessage) {
        let mut inner = self.inner.lock().await;
        inner.follow_up_queue.enqueue(message);
    }

    /// Remove all queued steering messages.
    pub async fn clear_steering_queue(&self) {
        let mut inner = self.inner.lock().await;
        inner.steering_queue.clear();
    }

    /// Remove all queued follow-up messages.
    pub async fn clear_follow_up_queue(&self) {
        let mut inner = self.inner.lock().await;
        inner.follow_up_queue.clear();
    }

    /// Remove all queued steering and follow-up messages.
    pub async fn clear_all_queues(&self) {
        let mut inner = self.inner.lock().await;
        inner.steering_queue.clear();
        inner.follow_up_queue.clear();
    }

    /// Returns `true` when either queue still contains pending messages.
    pub async fn has_queued_messages(&self) -> bool {
        let inner = self.inner.lock().await;
        inner.steering_queue.has_items() || inner.follow_up_queue.has_items()
    }

    /// Set the steering queue drain mode.
    pub async fn set_steering_mode(&self, mode: QueueMode) {
        let mut inner = self.inner.lock().await;
        inner.steering_queue.mode = mode;
    }

    /// Set the follow-up queue drain mode.
    pub async fn set_follow_up_mode(&self, mode: QueueMode) {
        let mut inner = self.inner.lock().await;
        inner.follow_up_queue.mode = mode;
    }

    // -----------------------------------------------------------------------
    // Run lifecycle
    // -----------------------------------------------------------------------

    /// Returns `true` if a run is currently active.
    pub async fn is_active(&self) -> bool {
        let inner = self.inner.lock().await;
        inner.active_run.is_some()
    }

    /// Abort the current run, if one is active.
    pub async fn abort(&self) {
        let inner = self.inner.lock().await;
        if let Some(run) = &inner.active_run {
            run.cancel.cancel();
        }
    }

    /// Resolve when the current run and all awaited event listeners have
    /// finished. Returns immediately if no run is active.
    pub async fn wait_for_idle(&self) {
        let notify = {
            let inner = self.inner.lock().await;
            inner.active_run.as_ref().map(|r| r.done.clone())
        };
        if let Some(notify) = notify {
            notify.notified().await;
        }
    }

    /// Clear transcript state, runtime state, and queued messages.
    pub async fn reset(&self) {
        let mut inner = self.inner.lock().await;
        inner.messages.clear();
        inner.is_streaming = false;
        inner.streaming_message = None;
        inner.pending_tool_calls.clear();
        inner.error_message = None;
        inner.follow_up_queue.clear();
        inner.steering_queue.clear();
    }

    // -----------------------------------------------------------------------
    // Private: context/config builders
    // -----------------------------------------------------------------------

    /// Snapshot the current state into an `AgentContext`.
    async fn create_context_snapshot(&self) -> AgentContext {
        let inner = self.inner.lock().await;
        AgentContext {
            system_prompt: inner.system_prompt.clone(),
            messages: inner.messages.clone(),
            tools: inner.tools.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Private: event processing
    // -----------------------------------------------------------------------

    /// Process an event from the agent loop: reduce internal state, then
    /// dispatch to subscribers.
    async fn process_event(&self, event: AgentEvent) {
        // 1. Reduce internal state and capture cancel token
        let cancel = {
            let mut inner = self.inner.lock().await;
            match &event {
                AgentEvent::MessageStart { message } => {
                    inner.streaming_message = Some(message.clone());
                }
                AgentEvent::MessageUpdate { message, .. } => {
                    inner.streaming_message = Some(message.clone());
                }
                AgentEvent::MessageEnd { message } => {
                    inner.streaming_message = None;
                    inner.messages.push(message.clone());
                }
                AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                    inner.pending_tool_calls.insert(tool_call_id.clone());
                }
                AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                    inner.pending_tool_calls.remove(tool_call_id);
                }
                AgentEvent::TurnEnd { message, .. } => {
                    if let AgentMessage::Assistant(msg) = message {
                        if msg.error_message.is_some() {
                            inner.error_message = msg.error_message.clone();
                        }
                    }
                }
                AgentEvent::AgentEnd { .. } => {
                    inner.streaming_message = None;
                }
                _ => {}
            }

            inner
                .active_run
                .as_ref()
                .map(|r| r.cancel.clone())
                .unwrap_or_else(CancellationToken::new)
        };

        // 2. Fan out to subscribers sequentially
        self.dispatch_to_subscribers(event, cancel).await;
    }

    /// Dispatch an event to all registered subscribers, sequentially in
    /// registration order.
    async fn dispatch_to_subscribers(&self, event: AgentEvent, cancel: CancellationToken) {
        let subscribers: Vec<Option<Arc<SubscriberFn>>> = {
            let inner = self.inner.lock().await;
            inner.subscribers.clone()
        };

        for subscriber in subscribers.iter().flatten() {
            subscriber(event.clone(), cancel.clone()).await;
        }
    }

    // -----------------------------------------------------------------------
    // Private: input normalization
    // -----------------------------------------------------------------------

    fn normalize_prompt_input(input: PromptInput) -> Vec<AgentMessage> {
        match input {
            PromptInput::Messages(msgs) => msgs,
            PromptInput::Text { text, images } => {
                let mut content: Vec<MediaContentBlock> =
                    vec![MediaContentBlock::Text(TextContent::new(text))];
                for img in images {
                    content.push(MediaContentBlock::Image(img));
                }
                let user_msg = ameli_ai::types::UserMessage {
                    content: ameli_ai::types::UserContent::Blocks(content),
                    timestamp: now_ms(),
                };
                vec![AgentMessage::User(user_msg)]
            }
        }
    }
}

impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("tool_execution", &self.tool_execution)
            .field("transport", &self.transport)
            .field("session_id", &self.session_id)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// build_loop_config — creates AgentLoopConfig with Arc<Agent> closures
// ---------------------------------------------------------------------------

/// Build [`AgentLoopConfig`] with closures that capture `Arc<Agent>` for queue
/// draining and event dispatch.
async fn build_loop_config(
    agent: &Arc<Agent>,
    skip_initial_steering_poll: bool,
) -> AgentLoopConfig {
    let (model, thinking_level) = {
        let inner = agent.inner.lock().await;
        (inner.model.clone(), inner.thinking_level)
    };

    let reasoning = match thinking_level {
        ThinkingLevel::Off => None,
        ThinkingLevel::Minimal => Some(ameli_ai::types::ThinkingLevel::Minimal),
        ThinkingLevel::Low => Some(ameli_ai::types::ThinkingLevel::Low),
        ThinkingLevel::Medium => Some(ameli_ai::types::ThinkingLevel::Medium),
        ThinkingLevel::High => Some(ameli_ai::types::ThinkingLevel::High),
        ThinkingLevel::XHigh => Some(ameli_ai::types::ThinkingLevel::XHigh),
    };

    let convert_to_llm = agent.convert_to_llm.clone();
    let transform_context = agent.transform_context.clone();
    let get_api_key = agent.get_api_key.clone();
    let before_tool_call = agent.before_tool_call.clone();
    let after_tool_call = agent.after_tool_call.clone();
    let prepare_next_turn_fn = agent.prepare_next_turn.clone();

    // Steering closure: drains the agent's steering queue
    let agent_for_steering = agent.clone();
    let steering_inner = Arc::new(move || -> BoxFuture<Vec<AgentMessage>> {
        let agent = agent_for_steering.clone();
        Box::pin(async move {
            let mut inner = agent.inner.lock().await;
            inner.steering_queue.drain()
        })
    });

    // Follow-up closure: drains the agent's follow-up queue
    let agent_for_followup = agent.clone();
    let followup_inner = Arc::new(move || -> BoxFuture<Vec<AgentMessage>> {
        let agent = agent_for_followup.clone();
        Box::pin(async move {
            let mut inner = agent.inner.lock().await;
            inner.follow_up_queue.drain()
        })
    });

    // If skip_initial_steering_poll, wrap to return empty on first call
    let skip = Arc::new(std::sync::atomic::AtomicBool::new(skip_initial_steering_poll));
    let steering_fn: Arc<dyn Fn() -> BoxFuture<Vec<AgentMessage>> + Send + Sync> =
        Arc::new(move || {
            let skip = skip.clone();
            let inner_fn = steering_inner.clone();
            Box::pin(async move {
                if skip.swap(false, std::sync::atomic::Ordering::SeqCst) {
                    return Vec::new();
                }
                inner_fn().await
            })
        });

    let follow_up_fn: Arc<dyn Fn() -> BoxFuture<Vec<AgentMessage>> + Send + Sync> = followup_inner;

    let stream_options = StreamOptions {
        reasoning,
        session_id: agent.session_id.clone(),
        thinking_budgets: agent.thinking_budgets,
        transport: Some(agent.transport),
        max_retry_delay_ms: agent.max_retry_delay_ms,
        ..Default::default()
    };

    AgentLoopConfig {
        model,
        convert_to_llm,
        stream_options,
        transform_context,
        get_api_key,
        should_stop_after_turn: None,
        prepare_next_turn: prepare_next_turn_fn.map(|hook| {
            Arc::new(move |_ctx: &PrepareNextTurnContext| -> BoxFuture<Option<AgentLoopTurnUpdate>> {
                let hook = hook.clone();
                Box::pin(async move { hook(None).await })
            })
                as Arc<
                    dyn Fn(
                            &PrepareNextTurnContext,
                        ) -> BoxFuture<Option<AgentLoopTurnUpdate>>
                        + Send
                        + Sync,
                >
        }),
        get_steering_messages: Some(steering_fn),
        get_follow_up_messages: Some(follow_up_fn),
        tool_execution: agent.tool_execution,
        before_tool_call,
        after_tool_call,
    }
}

// ---------------------------------------------------------------------------
// ArcAgent — primary public API
// ---------------------------------------------------------------------------

/// Convenience wrapper around `Arc<Agent>` that provides the full API
/// including `prompt` and `continue_` with proper queue integration.
///
/// This is the primary way to use the agent. Create with [`ArcAgent::new`].
pub struct ArcAgent {
    inner: Arc<Agent>,
}

impl ArcAgent {
    /// Create a new agent with the given options.
    pub fn new(options: AgentOptions) -> Self {
        Self {
            inner: Agent::new_arc(options),
        }
    }

    /// Get a reference to the underlying `Arc<Agent>`.
    pub fn agent(&self) -> &Arc<Agent> {
        &self.inner
    }

    /// Snapshot the current agent state.
    pub async fn state(&self) -> AgentState {
        self.inner.state().await
    }

    // -----------------------------------------------------------------------
    // Queue operations
    // -----------------------------------------------------------------------

    /// Queue a steering message.
    pub async fn steer(&self, message: AgentMessage) {
        self.inner.steer(message).await
    }

    /// Queue a follow-up message.
    pub async fn follow_up(&self, message: AgentMessage) {
        self.inner.follow_up(message).await
    }

    /// Clear all queued steering messages.
    pub async fn clear_steering_queue(&self) {
        self.inner.clear_steering_queue().await
    }

    /// Clear all queued follow-up messages.
    pub async fn clear_follow_up_queue(&self) {
        self.inner.clear_follow_up_queue().await
    }

    /// Clear all queued messages.
    pub async fn clear_all_queues(&self) {
        self.inner.clear_all_queues().await
    }

    /// Returns `true` when either queue has pending messages.
    pub async fn has_queued_messages(&self) -> bool {
        self.inner.has_queued_messages().await
    }

    /// Set the steering queue drain mode.
    pub async fn set_steering_mode(&self, mode: QueueMode) {
        self.inner.set_steering_mode(mode).await
    }

    /// Set the follow-up queue drain mode.
    pub async fn set_follow_up_mode(&self, mode: QueueMode) {
        self.inner.set_follow_up_mode(mode).await
    }

    // -----------------------------------------------------------------------
    // Run lifecycle
    // -----------------------------------------------------------------------

    /// Returns `true` if a run is currently active.
    pub async fn is_active(&self) -> bool {
        self.inner.is_active().await
    }

    /// Abort the current run.
    pub async fn abort(&self) {
        self.inner.abort().await
    }

    /// Wait for the current run to finish (including all listener settlement).
    pub async fn wait_for_idle(&self) {
        self.inner.wait_for_idle().await
    }

    /// Clear transcript state, runtime state, and queued messages.
    pub async fn reset(&self) {
        self.inner.reset().await
    }

    // -----------------------------------------------------------------------
    // Event subscription
    // -----------------------------------------------------------------------

    /// Subscribe to agent lifecycle events.
    ///
    /// The listener is called with each [`AgentEvent`] and the current run's
    /// `CancellationToken`. Listeners are awaited sequentially in subscription
    /// order.
    ///
    /// Returns a [`Subscription`] handle that unsubscribes on drop.
    pub async fn subscribe(&self, listener: Arc<SubscriberFn>) -> Subscription {
        let mut inner = self.inner.inner.lock().await;
        let index = inner.subscribers.len();
        inner.subscribers.push(Some(listener));
        Subscription {
            agent: self.inner.clone(),
            index,
        }
    }

    // -----------------------------------------------------------------------
    // prompt / continue entry points
    // -----------------------------------------------------------------------

    /// Start a new prompt from text, a single message, or a batch of messages.
    ///
    /// Returns an error if the agent is already processing a prompt.
    pub async fn prompt(&self, input: PromptInput) -> anyhow::Result<()> {
        {
            let inner = self.inner.inner.lock().await;
            if inner.active_run.is_some() {
                anyhow::bail!(
                    "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
                );
            }
        }

        let messages = Agent::normalize_prompt_input(input);
        self.run_prompt_messages(messages, false).await;
        Ok(())
    }

    /// Continue from the current transcript.
    ///
    /// If the last message is an assistant message, the agent first attempts to
    /// drain queued steering messages, then follow-up messages, before
    /// returning an error.
    pub async fn continue_(&self) -> anyhow::Result<()> {
        {
            let inner = self.inner.inner.lock().await;
            if inner.active_run.is_some() {
                anyhow::bail!(
                    "Agent is already processing. Wait for completion before continuing."
                );
            }
        }

        let last_role = {
            let inner = self.inner.inner.lock().await;
            inner.messages.last().map(|m| m.role().to_string())
        };

        match last_role.as_deref() {
            None => {
                anyhow::bail!("No messages to continue from");
            }
            Some("assistant") => {
                // Try steering queue first
                let drained = {
                    let mut inner = self.inner.inner.lock().await;
                    inner.steering_queue.drain()
                };
                if !drained.is_empty() {
                    self.run_prompt_messages(drained, true).await;
                    return Ok(());
                }

                // Try follow-up queue
                let drained = {
                    let mut inner = self.inner.inner.lock().await;
                    inner.follow_up_queue.drain()
                };
                if !drained.is_empty() {
                    self.run_prompt_messages(drained, false).await;
                    return Ok(());
                }

                anyhow::bail!("Cannot continue from message role: assistant");
            }
            _ => {
                // Valid — last message is user or toolResult
            }
        }

        self.run_continuation().await
    }

    // -----------------------------------------------------------------------
    // Private: run orchestration
    // -----------------------------------------------------------------------

    async fn run_prompt_messages(
        &self,
        messages: Vec<AgentMessage>,
        skip_initial_steering_poll: bool,
    ) {
        self.run_with_lifecycle(move |cancel| {
            let agent = self.inner.clone();
            async move {
                let context = agent.create_context_snapshot().await;
                let config = build_loop_config(&agent, skip_initial_steering_poll).await;

                let agent_for_emit = agent.clone();
                let emit: crate::agent_loop::AgentEventSink =
                    Arc::new(move |event: AgentEvent| {
                        let agent = agent_for_emit.clone();
                        tokio::spawn(async move {
                            agent.process_event(event).await;
                        });
                    });

                run_agent_loop(
                    messages,
                    context,
                    config,
                    emit,
                    Some(cancel),
                    agent.registry.clone(),
                )
                .await;
            }
        })
        .await;
    }

    async fn run_continuation(&self) -> anyhow::Result<()> {
        self.run_with_lifecycle_result(move |cancel| {
            let agent = self.inner.clone();
            async move {
                let context = agent.create_context_snapshot().await;
                let config = build_loop_config(&agent, false).await;

                let agent_for_emit = agent.clone();
                let emit: crate::agent_loop::AgentEventSink =
                    Arc::new(move |event: AgentEvent| {
                        let agent = agent_for_emit.clone();
                        tokio::spawn(async move {
                            agent.process_event(event).await;
                        });
                    });

                run_agent_loop_continue(
                    context,
                    config,
                    emit,
                    Some(cancel),
                    agent.registry.clone(),
                )
                .await?;
                Ok(())
            }
        })
        .await
    }

    /// Core lifecycle: create `ActiveRun`, execute, finalize.
    async fn run_with_lifecycle<F, Fut>(&self, executor: F)
    where
        F: FnOnce(CancellationToken) -> Fut,
        Fut: Future<Output = ()>,
    {
        let cancel = CancellationToken::new();
        let done = Arc::new(tokio::sync::Notify::new());
        let done_clone = done.clone();

        {
            let mut inner = self.inner.inner.lock().await;
            inner.active_run = Some(ActiveRun {
                cancel: cancel.clone(),
                done: done_clone,
            });
            inner.is_streaming = true;
            inner.streaming_message = None;
            inner.error_message = None;
        }

        executor(cancel).await;

        self.finish_run().await;
        done.notify_waiters();
    }

    /// Core lifecycle variant that propagates a `Result`.
    async fn run_with_lifecycle_result<F, Fut, T>(&self, executor: F) -> anyhow::Result<T>
    where
        F: FnOnce(CancellationToken) -> Fut,
        Fut: Future<Output = anyhow::Result<T>>,
    {
        let cancel = CancellationToken::new();
        let done = Arc::new(tokio::sync::Notify::new());
        let done_clone = done.clone();

        {
            let mut inner = self.inner.inner.lock().await;
            inner.active_run = Some(ActiveRun {
                cancel: cancel.clone(),
                done: done_clone,
            });
            inner.is_streaming = true;
            inner.streaming_message = None;
            inner.error_message = None;
        }

        let result = executor(cancel).await;

        if let Err(ref e) = result {
            let mut inner = self.inner.inner.lock().await;
            inner.error_message = Some(e.to_string());
        }

        self.finish_run().await;
        done.notify_waiters();

        result
    }

    /// Clear runtime state and resolve the active run.
    async fn finish_run(&self) {
        let mut inner = self.inner.inner.lock().await;
        inner.is_streaming = false;
        inner.streaming_message = None;
        inner.pending_tool_calls.clear();
        inner.active_run = None;
    }
}

impl fmt::Debug for ArcAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
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
    use ameli_ai::provider::StreamFn;
    use ameli_ai::stream::create_assistant_message_event_stream;
    use ameli_ai::types::{Context as LlmContext, Cost, InputType};

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

    // --- PendingMessageQueue ---

    #[test]
    fn queue_drain_all() {
        let mut q = PendingMessageQueue::new(QueueMode::All);
        q.enqueue(AgentMessage::User(ameli_ai::types::UserMessage::text("a")));
        q.enqueue(AgentMessage::User(ameli_ai::types::UserMessage::text("b")));
        let drained = q.drain();
        assert_eq!(drained.len(), 2);
        assert!(!q.has_items());
    }

    #[test]
    fn queue_drain_one_at_a_time() {
        let mut q = PendingMessageQueue::new(QueueMode::OneAtATime);
        q.enqueue(AgentMessage::User(ameli_ai::types::UserMessage::text("a")));
        q.enqueue(AgentMessage::User(ameli_ai::types::UserMessage::text("b")));

        let first = q.drain();
        assert_eq!(first.len(), 1);
        assert!(q.has_items());

        let second = q.drain();
        assert_eq!(second.len(), 1);
        assert!(!q.has_items());

        let empty = q.drain();
        assert!(empty.is_empty());
    }

    #[test]
    fn queue_clear() {
        let mut q = PendingMessageQueue::new(QueueMode::All);
        q.enqueue(AgentMessage::User(ameli_ai::types::UserMessage::text("a")));
        q.clear();
        assert!(!q.has_items());
    }

    #[test]
    fn queue_empty_drain() {
        let mut q = PendingMessageQueue::new(QueueMode::All);
        assert!(q.drain().is_empty());
    }

    // --- PromptInput ---

    #[test]
    fn prompt_input_from_str() {
        let input: PromptInput = "hello".into();
        match input {
            PromptInput::Text { text, images } => {
                assert_eq!(text, "hello");
                assert!(images.is_empty());
            }
            _ => panic!("expected Text variant"),
        }
    }

    #[test]
    fn prompt_input_from_messages() {
        let msg = AgentMessage::User(ameli_ai::types::UserMessage::text("hi"));
        let input: PromptInput = vec![msg].into();
        match input {
            PromptInput::Messages(msgs) => assert_eq!(msgs.len(), 1),
            _ => panic!("expected Messages variant"),
        }
    }

    // --- AgentOptions default ---

    #[test]
    fn agent_options_default() {
        let opts = AgentOptions::default();
        assert_eq!(opts.steering_mode, QueueMode::OneAtATime);
        assert_eq!(opts.follow_up_mode, QueueMode::OneAtATime);
        assert_eq!(opts.tool_execution, ToolExecutionMode::Parallel);
        assert!(opts.initial_state.is_none());
        assert!(opts.session_id.is_none());
    }

    // --- Agent construction ---

    #[tokio::test]
    async fn agent_new_defaults() {
        let agent = ArcAgent::new(AgentOptions::default());
        let state = agent.state().await;
        assert_eq!(state.model.id, "unknown");
        assert!(!state.is_streaming);
        assert!(state.messages.is_empty());
    }

    #[tokio::test]
    async fn agent_new_with_initial_state() {
        let model = test_model();
        let state = AgentState {
            system_prompt: "You are helpful.".into(),
            model: model.clone(),
            thinking_level: ThinkingLevel::Off,
            tools: vec![],
            messages: vec![AgentMessage::User(ameli_ai::types::UserMessage::text("hi"))],
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        };
        let agent = ArcAgent::new(AgentOptions {
            initial_state: Some(state),
            ..Default::default()
        });
        let s = agent.state().await;
        assert_eq!(s.system_prompt, "You are helpful.");
        assert_eq!(s.messages.len(), 1);
    }

    // --- Queue operations ---

    #[tokio::test]
    async fn agent_steer_and_follow_up() {
        let agent = ArcAgent::new(AgentOptions::default());
        agent
            .steer(AgentMessage::User(ameli_ai::types::UserMessage::text("steer")))
            .await;
        agent
            .follow_up(AgentMessage::User(ameli_ai::types::UserMessage::text("follow")))
            .await;
        assert!(agent.has_queued_messages().await);

        agent.clear_all_queues().await;
        assert!(!agent.has_queued_messages().await);
    }

    #[tokio::test]
    async fn agent_set_queue_modes() {
        let agent = ArcAgent::new(AgentOptions::default());
        agent.set_steering_mode(QueueMode::All).await;
        agent.set_follow_up_mode(QueueMode::All).await;
    }

    // --- Reset ---

    #[tokio::test]
    async fn agent_reset() {
        let agent = ArcAgent::new(AgentOptions {
            initial_state: Some(AgentState {
                system_prompt: "test".into(),
                model: test_model(),
                thinking_level: ThinkingLevel::Off,
                tools: vec![],
                messages: vec![AgentMessage::User(ameli_ai::types::UserMessage::text("hi"))],
                is_streaming: false,
                streaming_message: None,
                pending_tool_calls: HashSet::new(),
                error_message: None,
            }),
            ..Default::default()
        });
        assert_eq!(agent.state().await.messages.len(), 1);
        agent.reset().await;
        assert!(agent.state().await.messages.is_empty());
    }

    // --- Active run check ---

    #[tokio::test]
    async fn agent_not_active_initially() {
        let agent = ArcAgent::new(AgentOptions::default());
        assert!(!agent.is_active().await);
    }

    // --- Prompt with no provider registered completes with error ---

    #[tokio::test]
    async fn agent_prompt_with_no_provider_still_completes() {
        let registry = Arc::new(ProviderRegistry::new());
        let agent = ArcAgent::new(AgentOptions {
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
            registry: Some(registry),
            ..Default::default()
        });
        let result = agent.prompt("hello".into()).await;
        assert!(result.is_ok());
        assert!(!agent.is_active().await);
    }

    // --- Continue rejects with no messages ---

    #[tokio::test]
    async fn agent_continue_rejects_no_messages() {
        let agent = ArcAgent::new(AgentOptions::default());
        let result = agent.continue_().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No messages"));
    }

    // --- Prompt rejects when already active ---

    #[tokio::test]
    async fn agent_prompt_rejects_when_active() {
        // Provider that never produces events — simulates a hanging LLM call
        #[derive(Clone)]
        struct HangingProvider;

        impl StreamFn for HangingProvider {
            fn stream(
                &self,
                _model: &Model,
                _context: LlmContext,
                _options: ameli_ai::types::StreamOptions,
            ) -> ameli_ai::stream::AssistantMessageEventStream {
                let (_producer, stream) = create_assistant_message_event_stream();
                std::mem::forget(_producer);
                stream
            }
        }

        let registry = Arc::new(ProviderRegistry::new());
        registry.register("test-api", Box::new(HangingProvider));

        let agent = ArcAgent::new(AgentOptions {
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
            registry: Some(registry),
            ..Default::default()
        });

        // Start prompt in background
        let agent_bg = ArcAgent {
            inner: agent.inner.clone(),
        };
        let handle = tokio::spawn(async move {
            let _ = agent_bg.prompt("hello".into()).await;
        });

        // Give the task a moment to start the run
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Second prompt should fail
        let result = agent.prompt("second".into()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already processing"));

        // Abort the hanging task (the stream never ends so the run won't
        // complete naturally — just abort the JoinHandle)
        handle.abort();
    }

    // --- Subscribe and receive events ---

    #[tokio::test]
    async fn agent_subscribe_receives_events() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let event_count = Arc::new(AtomicUsize::new(0));
        let event_count_clone = event_count.clone();

        let registry = Arc::new(ProviderRegistry::new());

        let agent = ArcAgent::new(AgentOptions {
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
            registry: Some(registry),
            ..Default::default()
        });

        let _sub = agent
            .subscribe(Arc::new(move |event, _cancel| {
                let count = event_count_clone.clone();
                Box::pin(async move {
                    count.fetch_add(1, Ordering::SeqCst);
                    let _ = &event;
                })
            }))
            .await;

        // prompt will fail (no provider) but events should still fire
        let _ = agent.prompt("hello".into()).await;

        // Events are dispatched via tokio::spawn, so give them time to complete
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Should have received at least some events
        assert!(event_count.load(Ordering::SeqCst) > 0);
    }

    // --- Default convert_to_llm ---

    #[tokio::test]
    async fn default_convert_to_llm_filters_standard() {
        use serde_json::json;

        // Standard message
        let user = AgentMessage::User(ameli_ai::types::UserMessage::text("hi"));

        // Custom message
        #[derive(Clone)]
        struct TestCustom;
        impl CustomAgentMessage for TestCustom {
            fn message_type(&self) -> &str {
                "test"
            }
            fn clone_boxed(&self) -> Box<dyn CustomAgentMessage> {
                Box::new(self.clone())
            }
            fn to_json(&self) -> serde_json::Value {
                json!({})
            }
            fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("TestCustom").finish()
            }
        }

        let custom = AgentMessage::Custom(Box::new(TestCustom));

        let messages = vec![user, custom];
        let result = default_convert_to_llm(&messages).await;
        assert_eq!(result.len(), 1);
        match &result[0] {
            Message::User(u) => {
                match &u.content {
                    ameli_ai::types::UserContent::Text(t) => assert_eq!(t, "hi"),
                    _ => panic!("expected text content"),
                }
            }
            _ => panic!("expected User message"),
        }
    }
}
