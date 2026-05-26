//! Agent-level types for the ameli agent loop and stateful Agent.
//!
//! This module defines the shared vocabulary used by the agent orchestration
//! layer: tools, messages, events, context, configuration callbacks, and
//! state. These types are a restricted, lightweight port of the pi-agent
//! TypeScript package's `types.ts`.
//!
//! # Key abstractions
//!
//! - [`AgentMessage`] — union of LLM messages and custom app messages
//! - [`AgentTool`] — trait for tools the agent can execute
//! - [`AgentEvent`] — lifecycle events emitted by the agent loop
//! - [`AgentLoopConfig`] — configuration including async callbacks
//! - [`AgentContext`] / [`AgentState`] — context snapshots and mutable state

use ameli_ai::types::{
    AssistantMessage, AssistantMessageEvent, MediaContentBlock, Message, Model, StreamOptions,
    TextContent, Tool, ToolCall, ToolResultMessage,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Simple enums
// ---------------------------------------------------------------------------

/// How tool calls from a single assistant message are executed.
///
/// - `Sequential`: each tool call is prepared, executed, and finalized before
///   the next one starts.
/// - `Parallel`: tool calls are prepared sequentially, then allowed tools
///   execute concurrently. `tool_execution_end` is emitted in completion order,
///   while tool-result message artifacts are emitted later in source order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ToolExecutionMode {
    Sequential,
    #[default]
    Parallel,
}

/// How many queued messages are injected when the agent loop drains a queue.
///
/// - `All`: drain and inject every queued message at that point.
/// - `OneAtATime`: drain only the oldest queued message, leaving the rest for
///   later drain points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum QueueMode {
    All,
    #[default]
    OneAtATime,
}

/// Thinking/reasoning level for models that support extended thinking.
///
/// Mirrors the TS `ThinkingLevel` from pi-agent. This is the same set of
/// values as [`ameli_ai::types::ModelThinkingLevel`] but scoped to the agent
/// layer for clarity.
pub type ThinkingLevel = ameli_ai::types::ModelThinkingLevel;

// ---------------------------------------------------------------------------
// CustomAgentMessage trait
// ---------------------------------------------------------------------------

/// Trait for custom agent message types that extend the standard LLM messages.
///
/// Apps implement this trait to add their own message types (e.g., artifacts,
/// notifications, status messages) while maintaining compatibility with the
/// agent loop.
///
/// # Examples
///
/// ```
/// use ameli_agent_core::types::CustomAgentMessage;
/// use serde_json::json;
/// use std::fmt;
///
/// #[derive(Clone)]
/// struct ArtifactMessage {
///     content: String,
///     timestamp: u64,
/// }
///
/// impl CustomAgentMessage for ArtifactMessage {
///     fn message_type(&self) -> &str { "artifact" }
///     fn clone_boxed(&self) -> Box<dyn CustomAgentMessage> {
///         Box::new(self.clone())
///     }
///     fn to_json(&self) -> serde_json::Value {
///         json!({ "content": self.content, "timestamp": self.timestamp })
///     }
///     fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         f.debug_struct("ArtifactMessage")
///             .field("content", &self.content)
///             .field("timestamp", &self.timestamp)
///             .finish()
///     }
/// }
/// ```
pub trait CustomAgentMessage: Send + Sync {
    /// Discriminant for the custom message type (display/logging).
    fn message_type(&self) -> &str;

    /// Clone into a boxed trait object.
    fn clone_boxed(&self) -> Box<dyn CustomAgentMessage>;

    /// Serialize the custom message for persistence/debugging.
    fn to_json(&self) -> serde_json::Value;

    /// Format the custom message for debugging.
    fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl fmt::Debug for dyn CustomAgentMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_debug(f)
    }
}

// ---------------------------------------------------------------------------
// AgentMessage
// ---------------------------------------------------------------------------

/// A message in the agent's conversation.
///
/// Union of standard LLM messages ([`UserMessage`], [`AssistantMessage`],
/// [`ToolResultMessage`]) and custom app messages via [`CustomAgentMessage`].
///
/// The standard variants are zero-cost wrappers around the `ameli_ai` message
/// types. The `Custom` variant holds a boxed trait object for extensibility.
pub enum AgentMessage {
    User(ameli_ai::types::UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
    Custom(Box<dyn CustomAgentMessage>),
}

impl AgentMessage {
    /// Returns the role string for this message.
    ///
    /// Standard messages return `"user"`, `"assistant"`, or `"toolResult"`.
    /// Custom messages return their [`CustomAgentMessage::message_type`].
    pub fn role(&self) -> &str {
        match self {
            Self::User(_) => "user",
            Self::Assistant(_) => "assistant",
            Self::ToolResult(_) => "toolResult",
            Self::Custom(msg) => msg.message_type(),
        }
    }

    /// Returns the Unix timestamp in milliseconds, if available.
    ///
    /// Standard messages always have a timestamp. Custom messages return `None`
    /// unless they carry their own timestamp semantics.
    pub fn timestamp(&self) -> Option<u64> {
        match self {
            Self::User(m) => Some(m.timestamp),
            Self::Assistant(m) => Some(m.timestamp),
            Self::ToolResult(m) => Some(m.timestamp),
            Self::Custom(_) => None,
        }
    }

    /// Downgrade to a standard LLM [`Message`], if this is a standard variant.
    ///
    /// Returns `None` for custom messages that have no LLM representation.
    pub fn as_message(&self) -> Option<Message> {
        match self {
            Self::User(m) => Some(Message::User(m.clone())),
            Self::Assistant(m) => Some(Message::Assistant(m.clone())),
            Self::ToolResult(m) => Some(Message::ToolResult(m.clone())),
            Self::Custom(_) => None,
        }
    }

    /// Returns `true` if this is a standard LLM message.
    pub fn is_standard(&self) -> bool {
        !matches!(self, Self::Custom(_))
    }
}

impl Clone for AgentMessage {
    fn clone(&self) -> Self {
        match self {
            Self::User(m) => Self::User(m.clone()),
            Self::Assistant(m) => Self::Assistant(m.clone()),
            Self::ToolResult(m) => Self::ToolResult(m.clone()),
            Self::Custom(m) => Self::Custom(m.clone_boxed()),
        }
    }
}

impl fmt::Debug for AgentMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User(m) => f.debug_tuple("User").field(m).finish(),
            Self::Assistant(m) => f.debug_tuple("Assistant").field(m).finish(),
            Self::ToolResult(m) => f.debug_tuple("ToolResult").field(m).finish(),
            Self::Custom(m) => f.debug_tuple("Custom").field(m).finish(),
        }
    }
}

impl fmt::Display for AgentMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User(m) => write!(f, "User({})", m.timestamp),
            Self::Assistant(m) => write!(f, "Assistant({})", m.timestamp),
            Self::ToolResult(m) => write!(f, "ToolResult({})", m.timestamp),
            Self::Custom(msg) => write!(f, "Custom({})", msg.message_type()),
        }
    }
}

impl From<ameli_ai::types::UserMessage> for AgentMessage {
    fn from(m: ameli_ai::types::UserMessage) -> Self {
        Self::User(m)
    }
}

impl From<AssistantMessage> for AgentMessage {
    fn from(m: AssistantMessage) -> Self {
        Self::Assistant(m)
    }
}

impl From<ToolResultMessage> for AgentMessage {
    fn from(m: ToolResultMessage) -> Self {
        Self::ToolResult(m)
    }
}

impl From<Message> for AgentMessage {
    fn from(m: Message) -> Self {
        match m {
            Message::User(u) => Self::User(u),
            Message::Assistant(a) => Self::Assistant(a),
            Message::ToolResult(t) => Self::ToolResult(t),
        }
    }
}

// ---------------------------------------------------------------------------
// AgentToolResult
// ---------------------------------------------------------------------------

/// Result produced by a tool execution.
///
/// Generic over the `details` type `T` for structured tool-specific data.
/// The `content` is what gets sent back to the model.
#[derive(Debug, Clone)]
pub struct AgentToolResult<T = serde_json::Value> {
    /// Text or image content returned to the model.
    pub content: Vec<MediaContentBlock>,
    /// Arbitrary structured details for logs or UI rendering.
    pub details: T,
    /// Hint that the agent should stop after the current tool batch.
    /// Early termination only happens when *every* finalized tool result in
    /// the batch sets this to `true`.
    pub terminate: bool,
}

impl<T> AgentToolResult<T> {
    /// Create a new tool result.
    pub fn new(content: Vec<MediaContentBlock>, details: T) -> Self {
        Self {
            content,
            details,
            terminate: false,
        }
    }

    /// Create a simple text result.
    pub fn text(text: impl Into<String>, details: T) -> Self {
        Self {
            content: vec![MediaContentBlock::Text(TextContent::new(text))],
            details,
            terminate: false,
        }
    }

    /// Create an error result with a text message.
    pub fn error(message: impl Into<String>) -> AgentToolResult<serde_json::Value> {
        AgentToolResult {
            content: vec![MediaContentBlock::Text(TextContent::new(message))],
            details: serde_json::Value::Object(serde_json::Map::new()),
            terminate: false,
        }
    }
}

/// Callback used by tools to stream partial execution updates.
pub type AgentToolUpdateCallback<T> = Box<dyn Fn(AgentToolResult<T>) + Send + Sync>;

// ---------------------------------------------------------------------------
// AgentTool trait
// ---------------------------------------------------------------------------

/// Tool definition used by the agent runtime.
///
/// Extends the base LLM [`Tool`] concept with execution, labeling, argument
/// preparation, and per-tool execution mode.
///
/// The trait is object-safe so tools can be stored as `Box<dyn AgentTool>`.
///
/// # Examples
///
/// ```
/// use ameli_agent_core::types::{AgentTool, AgentToolResult, ToolExecutionMode};
/// use ameli_ai::types::Tool;
/// use serde_json::{json, Value};
/// use std::fmt;
/// use std::future::Future;
/// use std::pin::Pin;
/// use tokio_util::sync::CancellationToken;
///
/// struct EchoTool;
///
/// impl AgentTool for EchoTool {
///     fn label(&self) -> &str { "Echo" }
///
///     fn tool_definition(&self) -> Tool {
///         Tool {
///             name: "echo".into(),
///             description: "Echoes back the input".into(),
///             parameters: json!({
///                 "type": "object",
///                 "properties": {
///                     "message": { "type": "string" }
///                 },
///                 "required": ["message"]
///             }),
///         }
///     }
///
///     fn prepare_arguments(&self, args: Value) -> Value {
///         args
///     }
///
///     fn execution_mode(&self) -> Option<ToolExecutionMode> {
///         None
///     }
///
///     fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         f.debug_struct("EchoTool").finish()
///     }
///
///     fn execute(
///         &self,
///         _tool_call_id: &str,
///         params: Value,
///         _cancel: Option<CancellationToken>,
///     ) -> Pin<Box<dyn Future<Output = AgentToolResult<Value>> + Send + '_>> {
///         Box::pin(async move {
///             let msg = params["message"].as_str().unwrap_or("").to_string();
///             AgentToolResult::text(msg, json!({}))
///         })
///     }
/// }
/// ```
pub trait AgentTool: Send + Sync {
    /// Human-readable label for UI display.
    fn label(&self) -> &str;

    /// The base LLM tool definition (name, description, JSON Schema parameters).
    fn tool_definition(&self) -> Tool;

    /// Convenience: the tool's registered name.
    fn name(&self) -> String {
        self.tool_definition().name
    }

    /// Optional compatibility shim for raw tool-call arguments before schema
    /// validation. Must return an object that matches the tool's parameter
    /// schema.
    fn prepare_arguments(&self, args: serde_json::Value) -> serde_json::Value;

    /// Execute the tool call.
    ///
    /// Throw on failure instead of encoding errors in the result content.
    /// The agent loop catches errors and wraps them in error tool results.
    ///
    /// The `cancel` token should be checked during long-running operations.
    fn execute(
        &self,
        tool_call_id: &str,
        params: serde_json::Value,
        cancel: Option<CancellationToken>,
    ) -> Pin<Box<dyn Future<Output = AgentToolResult<serde_json::Value>> + Send + '_>>;

    /// Format the tool for debugging.
    fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;

    /// Per-tool execution mode override.
    ///
    /// - `Some(Sequential)`: this tool must execute one at a time.
    /// - `Some(Parallel)`: this tool can execute concurrently.
    /// - `None`: use the default execution mode from config.
    fn execution_mode(&self) -> Option<ToolExecutionMode>;
}

impl fmt::Debug for dyn AgentTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_debug(f)
    }
}

// ---------------------------------------------------------------------------
// AgentEvent
// ---------------------------------------------------------------------------

/// Events emitted by the agent loop for UI updates and lifecycle observation.
///
/// `AgentEnd` is the last event emitted for a run. The agent becomes idle
/// only after listeners for that event finish settling.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    // Agent lifecycle
    AgentStart,
    AgentEnd {
        messages: Vec<AgentMessage>,
    },

    // Turn lifecycle — a turn is one assistant response + any tool calls/results
    TurnStart,
    TurnEnd {
        message: AgentMessage,
        tool_results: Vec<ToolResultMessage>,
    },

    // Message lifecycle — emitted for user, assistant, and toolResult messages
    MessageStart {
        message: AgentMessage,
    },
    /// Only emitted for assistant messages during streaming.
    MessageUpdate {
        message: AgentMessage,
        assistant_message_event: Box<AssistantMessageEvent>,
    },
    MessageEnd {
        message: AgentMessage,
    },

    // Tool execution lifecycle
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
        partial_result: AgentToolResult<serde_json::Value>,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: AgentToolResult<serde_json::Value>,
        is_error: bool,
    },
}

// ---------------------------------------------------------------------------
// AgentContext
// ---------------------------------------------------------------------------

/// Context snapshot passed into the low-level agent loop.
#[derive(Clone)]
pub struct AgentContext {
    /// System prompt included with the request.
    pub system_prompt: String,
    /// Transcript visible to the model.
    pub messages: Vec<AgentMessage>,
    /// Tools available for this run.
    pub tools: Vec<Arc<dyn AgentTool>>,
}

impl fmt::Debug for AgentContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentContext")
            .field("system_prompt", &self.system_prompt)
            .field("messages", &self.messages)
            .field("tools", &self.tools.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Callback context structs
// ---------------------------------------------------------------------------

/// Context passed to `before_tool_call`.
#[derive(Debug, Clone)]
pub struct BeforeToolCallContext {
    /// The assistant message that requested the tool call.
    pub assistant_message: AssistantMessage,
    /// The raw tool call block from the assistant message.
    pub tool_call: ToolCall,
    /// Validated tool arguments for the target tool schema.
    pub args: serde_json::Value,
    /// Current agent context at the time the tool call is prepared.
    pub context: AgentContext,
}

/// Result returned from `before_tool_call`.
///
/// Returning a value with `block: true` prevents the tool from executing.
/// The loop emits an error tool result instead.
#[derive(Debug, Clone, Default)]
pub struct BeforeToolCallResult {
    /// Whether to block execution.
    pub block: bool,
    /// Reason shown in the error result if blocked.
    pub reason: Option<String>,
}

impl BeforeToolCallResult {
    /// Create a blocking result with a reason.
    pub fn block(reason: impl Into<String>) -> Self {
        Self {
            block: true,
            reason: Some(reason.into()),
        }
    }
}

/// Context passed to `after_tool_call`.
#[derive(Debug, Clone)]
pub struct AfterToolCallContext {
    /// The assistant message that requested the tool call.
    pub assistant_message: AssistantMessage,
    /// The raw tool call block from the assistant message.
    pub tool_call: ToolCall,
    /// Validated tool arguments for the target tool schema.
    pub args: serde_json::Value,
    /// The executed tool result before any `after_tool_call` overrides.
    pub result: AgentToolResult<serde_json::Value>,
    /// Whether the executed tool result is currently treated as an error.
    pub is_error: bool,
    /// Current agent context at the time the tool call is finalized.
    pub context: AgentContext,
}

/// Partial override returned from `after_tool_call`.
///
/// Merge semantics are field-by-field:
/// - `content`: if provided, replaces the full content array
/// - `details`: if provided, replaces the full details payload
/// - `is_error`: if provided, replaces the error flag
/// - `terminate`: if provided, replaces the early-termination hint
///
/// Omitted fields (`None`) keep their original values. No deep merge.
#[derive(Debug, Clone, Default)]
pub struct AfterToolCallResult {
    pub content: Option<Vec<MediaContentBlock>>,
    pub details: Option<serde_json::Value>,
    pub is_error: Option<bool>,
    /// Hint that the agent should stop after the current tool batch.
    pub terminate: Option<bool>,
}

/// Context passed to `should_stop_after_turn`.
#[derive(Debug, Clone)]
pub struct ShouldStopAfterTurnContext {
    /// The assistant message that completed the turn.
    pub message: AssistantMessage,
    /// Tool result messages from the turn.
    pub tool_results: Vec<ToolResultMessage>,
    /// Current agent context after the turn.
    pub context: AgentContext,
    /// Messages that this loop invocation will return if it exits now.
    pub new_messages: Vec<AgentMessage>,
}

/// Context passed to `prepare_next_turn`.
pub type PrepareNextTurnContext = ShouldStopAfterTurnContext;

/// Replacement runtime state for the next provider request.
#[derive(Debug, Clone)]
pub struct AgentLoopTurnUpdate {
    /// Replacement context for the next turn.
    pub context: Option<AgentContext>,
    /// Replacement model for the next turn.
    pub model: Option<Model>,
    /// Replacement thinking level for the next turn.
    pub thinking_level: Option<ThinkingLevel>,
}

// ---------------------------------------------------------------------------
// Async callback type aliases
// ---------------------------------------------------------------------------

/// Type alias for pinned, boxed, sendable futures returned by callbacks.
type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

// ---------------------------------------------------------------------------
// Callback type aliases (keeps struct definitions clean)
// ---------------------------------------------------------------------------

type ConvertToLlmCb = dyn Fn(&[AgentMessage]) -> BoxFuture<Vec<Message>> + Send + Sync;
type TransformContextCb = dyn Fn(&[AgentMessage], Option<CancellationToken>) -> BoxFuture<Vec<AgentMessage>>
    + Send
    + Sync;
type GetApiKeyCb = dyn Fn(&str) -> BoxFuture<Option<String>> + Send + Sync;
type ShouldStopAfterTurnCb = dyn Fn(&ShouldStopAfterTurnContext) -> BoxFuture<bool> + Send + Sync;
type PrepareNextTurnCb =
    dyn Fn(&PrepareNextTurnContext) -> BoxFuture<Option<AgentLoopTurnUpdate>> + Send + Sync;
type GetMessagesCb = dyn Fn() -> BoxFuture<Vec<AgentMessage>> + Send + Sync;
type BeforeToolCallCb = dyn Fn(&BeforeToolCallContext, Option<CancellationToken>) -> BoxFuture<Option<BeforeToolCallResult>>
    + Send
    + Sync;
type AfterToolCallCb = dyn Fn(&AfterToolCallContext, Option<CancellationToken>) -> BoxFuture<Option<AfterToolCallResult>>
    + Send
    + Sync;

// ---------------------------------------------------------------------------
// AgentLoopConfig
// ---------------------------------------------------------------------------

/// Configuration for the agent loop.
///
/// Combines model/stream settings with optional async callbacks for tool
/// lifecycle hooks, context transformation, and message queuing.
pub struct AgentLoopConfig {
    // --- Required ---
    /// The model to use for LLM requests.
    pub model: Model,

    /// Converts `AgentMessage[]` to LLM-compatible `Message[]` before each
    /// LLM call.
    ///
    /// Each `AgentMessage` must be converted to a `UserMessage`,
    /// `AssistantMessage`, or `ToolResultMessage` that the LLM can understand.
    /// Messages that cannot be converted (e.g., UI-only notifications) should
    /// be filtered out.
    ///
    /// **Contract:** must not panic. Return a safe fallback value instead.
    pub convert_to_llm: Arc<ConvertToLlmCb>,

    // --- Stream options (from SimpleStreamOptions / StreamOptions) ---
    pub stream_options: StreamOptions,

    // --- Optional callbacks ---
    /// Optional transform applied to the context before `convert_to_llm`.
    ///
    /// Use for context window management (pruning old messages) or injecting
    /// context from external sources.
    ///
    /// **Contract:** must not panic. Return the original messages on error.
    pub transform_context: Option<Arc<TransformContextCb>>,

    /// Resolves an API key dynamically for each LLM call.
    ///
    /// Useful for short-lived OAuth tokens that may expire during long-running
    /// tool execution.
    ///
    /// **Contract:** must not panic. Return `None` when no key is available.
    pub get_api_key: Option<Arc<GetApiKeyCb>>,

    /// Called after each turn fully completes. If it returns `true`, the loop
    /// emits `AgentEnd` and exits without starting another LLM call.
    ///
    /// **Contract:** must not panic.
    pub should_stop_after_turn: Option<Arc<ShouldStopAfterTurnCb>>,

    /// Called after `TurnEnd` and before the loop decides whether another
    /// provider request should start. Return replacement state to affect the
    /// next turn, or `None` to keep current state.
    pub prepare_next_turn: Option<Arc<PrepareNextTurnCb>>,

    /// Returns steering messages to inject mid-run.
    ///
    /// Called after the current assistant turn finishes executing its tool
    /// calls. If messages are returned, they are added to the context before
    /// the next LLM call.
    ///
    /// **Contract:** must not panic. Return an empty vec when no messages.
    pub get_steering_messages: Option<Arc<GetMessagesCb>>,

    /// Returns follow-up messages to process after the agent would otherwise
    /// stop.
    ///
    /// **Contract:** must not panic. Return an empty vec when no messages.
    pub get_follow_up_messages: Option<Arc<GetMessagesCb>>,

    /// Tool execution mode. Default: `Parallel`.
    pub tool_execution: ToolExecutionMode,

    /// Called before a tool is executed, after arguments have been validated.
    ///
    /// Return a [`BeforeToolCallResult`] with `block: true` to prevent
    /// execution.
    pub before_tool_call: Option<Arc<BeforeToolCallCb>>,

    /// Called after a tool finishes executing, before `ToolExecutionEnd` and
    /// tool-result message events are emitted.
    ///
    /// Return an [`AfterToolCallResult`] to override parts of the result.
    pub after_tool_call: Option<Arc<AfterToolCallCb>>,
}

// ---------------------------------------------------------------------------
// AgentState
// ---------------------------------------------------------------------------

/// Mutable agent state.
///
/// Tracks the conversation transcript, current model, streaming status, and
/// pending tool calls. Fields are public for direct access — the agent loop
/// and the stateful `Agent` struct manage transitions.
#[derive(Clone)]
pub struct AgentState {
    /// System prompt sent with each model request.
    pub system_prompt: String,
    /// Active model used for future turns.
    pub model: Model,
    /// Requested reasoning level for future turns.
    pub thinking_level: ThinkingLevel,
    /// Available tools.
    pub tools: Vec<Arc<dyn AgentTool>>,
    /// Conversation transcript.
    pub messages: Vec<AgentMessage>,
    /// True while the agent is processing a prompt or continuation.
    pub is_streaming: bool,
    /// Partial assistant message for the current streamed response.
    pub streaming_message: Option<AgentMessage>,
    /// Tool call IDs currently executing.
    pub pending_tool_calls: HashSet<String>,
    /// Error message from the most recent failed or aborted turn.
    pub error_message: Option<String>,
}

impl fmt::Debug for AgentState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentState")
            .field("system_prompt", &self.system_prompt)
            .field("model", &self.model)
            .field("thinking_level", &self.thinking_level)
            .field("tools", &self.tools.len())
            .field("messages", &self.messages.len())
            .field("is_streaming", &self.is_streaming)
            .field("streaming_message", &self.streaming_message)
            .field("pending_tool_calls", &self.pending_tool_calls)
            .field("error_message", &self.error_message)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ameli_ai::types::{Cost, InputType};
    use serde_json::json;

    /// Helper: a minimal model for testing.
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

    // -- CustomAgentMessage --

    #[derive(Clone)]
    struct ArtifactMessage {
        content: String,
        timestamp: u64,
    }

    impl CustomAgentMessage for ArtifactMessage {
        fn message_type(&self) -> &str {
            "artifact"
        }
        fn clone_boxed(&self) -> Box<dyn CustomAgentMessage> {
            Box::new(self.clone())
        }
        fn to_json(&self) -> serde_json::Value {
            json!({
                "content": self.content,
                "timestamp": self.timestamp
            })
        }
        fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("ArtifactMessage")
                .field("content", &self.content)
                .field("timestamp", &self.timestamp)
                .finish()
        }
    }

    #[test]
    fn custom_message_role() {
        let msg = ArtifactMessage {
            content: "test".into(),
            timestamp: 1000,
        };
        let agent_msg = AgentMessage::Custom(Box::new(msg));
        assert_eq!(agent_msg.role(), "artifact");
        assert!(!agent_msg.is_standard());
        assert!(agent_msg.as_message().is_none());
    }

    #[test]
    fn standard_message_roles() {
        let user = AgentMessage::User(ameli_ai::types::UserMessage::text("hello"));
        assert_eq!(user.role(), "user");
        assert!(user.is_standard());
        assert!(user.as_message().is_some());

        let tool_result =
            AgentMessage::ToolResult(ToolResultMessage::error("tc_1", "bash", "fail"));
        assert_eq!(tool_result.role(), "toolResult");
        assert!(tool_result.is_standard());
    }

    #[test]
    fn agent_message_from_message() {
        let user_msg = ameli_ai::types::UserMessage::text("hi");
        let msg = Message::User(user_msg.clone());
        let agent_msg: AgentMessage = msg.into();
        assert_eq!(agent_msg.role(), "user");
        assert_eq!(agent_msg.timestamp(), Some(user_msg.timestamp));
    }

    // -- AgentToolResult --

    #[test]
    fn tool_result_text() {
        let result: AgentToolResult<serde_json::Value> =
            AgentToolResult::text("done", json!({"key": "val"}));
        assert_eq!(result.content.len(), 1);
        assert!(!result.terminate);
        assert_eq!(result.details["key"], "val");
    }

    #[test]
    fn tool_result_error() {
        let result = AgentToolResult::<serde_json::Value>::error("something failed");
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.details, json!({}));
    }

    // -- AgentTool --

    struct EchoTool;

    impl AgentTool for EchoTool {
        fn label(&self) -> &str {
            "Echo"
        }
        fn tool_definition(&self) -> Tool {
            Tool {
                name: "echo".into(),
                description: "Echoes input".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "message": { "type": "string" }
                    },
                    "required": ["message"]
                }),
            }
        }
        fn prepare_arguments(&self, args: serde_json::Value) -> serde_json::Value {
            args
        }
        fn execution_mode(&self) -> Option<ToolExecutionMode> {
            None
        }
        fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("EchoTool").finish()
        }
        fn execute(
            &self,
            _tool_call_id: &str,
            params: serde_json::Value,
            _cancel: Option<CancellationToken>,
        ) -> Pin<Box<dyn Future<Output = AgentToolResult<serde_json::Value>> + Send + '_>> {
            Box::pin(async move {
                let msg = params["message"].as_str().unwrap_or("").to_string();
                AgentToolResult::text(msg, json!({}))
            })
        }
    }

    #[tokio::test]
    async fn echo_tool_execute() {
        let tool = EchoTool;
        assert_eq!(tool.label(), "Echo");
        assert_eq!(tool.name(), "echo");

        let result = tool
            .execute("tc_1", json!({"message": "hello"}), None)
            .await;
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.details, json!({}));
    }

    // -- Enums --

    #[test]
    fn tool_execution_mode_default() {
        assert_eq!(ToolExecutionMode::default(), ToolExecutionMode::Parallel);
    }

    #[test]
    fn queue_mode_default() {
        assert_eq!(QueueMode::default(), QueueMode::OneAtATime);
    }

    #[test]
    fn queue_mode_serialize() {
        assert_eq!(
            serde_json::to_string(&QueueMode::OneAtATime).unwrap(),
            r#""one-at-a-time""#
        );
        assert_eq!(serde_json::to_string(&QueueMode::All).unwrap(), r#""all""#);
    }

    // -- BeforeToolCallResult --

    #[test]
    fn before_tool_call_block() {
        let result = BeforeToolCallResult::block("not allowed");
        assert!(result.block);
        assert_eq!(result.reason.as_deref(), Some("not allowed"));
    }

    // -- AgentContext --

    #[test]
    fn agent_context_construction() {
        let ctx = AgentContext {
            system_prompt: "You are helpful.".into(),
            messages: vec![AgentMessage::User(ameli_ai::types::UserMessage::text("hi"))],
            tools: vec![Arc::new(EchoTool)],
        };
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(ctx.tools.len(), 1);
        assert_eq!(ctx.tools[0].name(), "echo");
    }

    // -- AgentState --

    #[test]
    fn agent_state_default_fields() {
        let state = AgentState {
            system_prompt: String::new(),
            model: test_model(),
            thinking_level: ThinkingLevel::Off,
            tools: vec![],
            messages: vec![],
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        };
        assert!(!state.is_streaming);
        assert!(state.messages.is_empty());
        assert!(state.pending_tool_calls.is_empty());
    }
}
