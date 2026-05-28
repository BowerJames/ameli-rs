//! Extension event types and result types.
//!
//! Defines the typed events that extensions can subscribe to, split into three
//! categories based on dispatch semantics:
//!
//! - **Fire-and-forget** — extensions observe but cannot modify. Errors are
//!   caught and reported to error listeners; dispatch continues.
//! - **Sequential chain** — handlers run in order, each seeing accumulated
//!   state from prior handlers. All handlers run (no short-circuit).
//! - **First-to-return** — handlers run in order; the first `Some` result
//!   wins and stops dispatch.
//!
//! | Event | Dispatch | Short-circuit |
//! |-------|----------|---------------|
//! | `agent_start` | Fire-and-forget | No |
//! | `agent_end` | Fire-and-forget | No |
//! | `turn_start` | Fire-and-forget | No |
//! | `turn_end` | Fire-and-forget | No |
//! | `message_start` | Fire-and-forget | No |
//! | `message_update` | Fire-and-forget | No |
//! | `message_end` | Sequential chain | No |
//! | `tool_execution_start` | Fire-and-forget | No |
//! | `tool_execution_update` | Fire-and-forget | No |
//! | `tool_execution_end` | Fire-and-forget | No |
//! | `session_start` | Fire-and-forget | No |
//! | `session_shutdown` | Fire-and-forget | No |
//! | `tool_call` | First-to-block | Yes (on `block: true`) |
//! | `tool_result` | Sequential chain | No |
//! | `context` | Sequential chain | No |
//! | `before_agent_start` | Sequential accumulate | No |
//! | `format_compaction_summary` | First-to-return | Yes (on first `Some`) |
//! | `format_branch_summary` | First-to-return | Yes (on first `Some`) |

use ameli_agent_core::types::AgentMessage;
use ameli_ai::types::{AssistantMessageEvent, ImageContent, MediaContentBlock, ToolResultMessage};
use serde_json::Value;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Type alias
// ---------------------------------------------------------------------------

type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

// ---------------------------------------------------------------------------
// Session lifecycle reasons
// ---------------------------------------------------------------------------

/// Why a session start event was emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionStartReason {
    /// First startup of the agent session.
    Startup,
    /// Extensions and session reloaded.
    Reload,
    /// New session created.
    New,
    /// Existing session resumed.
    Resume,
}

/// Why a session shutdown event was emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionShutdownReason {
    /// Agent is quitting.
    Quit,
    /// Extensions and session are being reloaded.
    Reload,
}

// ---------------------------------------------------------------------------
// Notification events (fire-and-forget)
// ---------------------------------------------------------------------------

/// Emitted once when the agent loop starts.
#[derive(Debug, Clone)]
pub struct AgentStartEvent;

/// Emitted once when the agent loop ends.
#[derive(Debug, Clone)]
pub struct AgentEndEvent {
    /// All messages produced by this agent run.
    pub messages: Vec<AgentMessage>,
}

/// Emitted at the start of each turn (one LLM response + tool execution).
#[derive(Debug, Clone)]
pub struct TurnStartEvent {
    /// Zero-based turn index within the current agent run.
    pub turn_index: u32,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

/// Emitted at the end of each turn.
#[derive(Debug, Clone)]
pub struct TurnEndEvent {
    /// Zero-based turn index within the current agent run.
    pub turn_index: u32,
    /// The assistant message that completed the turn.
    pub message: AgentMessage,
    /// Tool result messages from tool calls in this turn.
    pub tool_results: Vec<ToolResultMessage>,
}

/// Emitted when a message starts (user, assistant, or tool result).
#[derive(Debug, Clone)]
pub struct MessageStartEvent {
    /// The message that started.
    pub message: AgentMessage,
}

/// Emitted during assistant message streaming with token-by-token updates.
#[derive(Debug, Clone)]
pub struct MessageUpdateEvent {
    /// Current partial message snapshot.
    pub message: AgentMessage,
    /// The underlying streaming event from the LLM provider.
    pub assistant_message_event: Box<AssistantMessageEvent>,
}

/// Emitted when a message ends (user, assistant, or tool result).
///
/// This is a **sequential chain hook**: handlers can return a replacement
/// message that preserves the original role. Each handler sees the result
/// of prior handlers.
#[derive(Debug, Clone)]
pub struct MessageEndEvent {
    /// The finalized message.
    pub message: AgentMessage,
}

/// Emitted when a tool starts executing.
#[derive(Debug, Clone)]
pub struct ToolExecutionStartEvent {
    /// Unique ID for this tool call.
    pub tool_call_id: String,
    /// Name of the tool being executed.
    pub tool_name: String,
    /// Validated arguments for the tool call.
    pub args: Value,
}

/// Emitted during tool execution with partial/streaming output.
#[derive(Debug, Clone)]
pub struct ToolExecutionUpdateEvent {
    /// Unique ID for this tool call.
    pub tool_call_id: String,
    /// Name of the tool being executed.
    pub tool_name: String,
    /// Arguments for the tool call.
    pub args: Value,
    /// Partial result produced so far.
    pub partial_result: ameli_agent_core::types::AgentToolResult<Value>,
}

/// Emitted when a tool finishes executing.
#[derive(Debug, Clone)]
pub struct ToolExecutionEndEvent {
    /// Unique ID for this tool call.
    pub tool_call_id: String,
    /// Name of the tool that executed.
    pub tool_name: String,
    /// The tool execution result.
    pub result: ameli_agent_core::types::AgentToolResult<Value>,
    /// Whether the result represents an error.
    pub is_error: bool,
}

/// Emitted when a session starts, loads, or reloads.
#[derive(Debug, Clone)]
pub struct SessionStartEvent {
    /// Why this session start happened.
    pub reason: SessionStartReason,
}

/// Emitted before an extension runtime is torn down due to quit or reload.
#[derive(Debug, Clone)]
pub struct SessionShutdownEvent {
    /// Why the session is shutting down.
    pub reason: SessionShutdownReason,
}

// ---------------------------------------------------------------------------
// Hook events
// ---------------------------------------------------------------------------

/// Emitted before a tool executes. Handlers can block execution.
///
/// **First-to-block wins**: dispatch stops immediately when any handler
/// returns `Some(ToolCallResult { block: true, .. })`.
#[derive(Debug, Clone)]
pub struct ToolCallEvent {
    /// Unique ID for this tool call.
    pub tool_call_id: String,
    /// Name of the tool about to execute.
    pub tool_name: String,
    /// Validated arguments for the tool call.
    pub args: Value,
}

/// Result returned from a `ToolCallEvent` handler.
///
/// Return `Some(...)` to influence execution; return `None` to allow it.
#[derive(Debug, Clone)]
pub struct ToolCallResult {
    /// Whether to block the tool from executing.
    pub block: bool,
    /// Human-readable reason shown if blocked.
    pub reason: Option<String>,
}

impl ToolCallResult {
    /// Create a result that blocks execution with a reason.
    pub fn block(reason: impl Into<String>) -> Self {
        Self {
            block: true,
            reason: Some(reason.into()),
        }
    }
}

/// Emitted after a tool finishes executing. Handlers can modify the result.
///
/// **Sequential chain**: each handler sees accumulated state from prior
/// handlers. Patches are merged field-by-field.
#[derive(Debug, Clone)]
pub struct ToolResultEvent {
    /// Unique ID for this tool call.
    pub tool_call_id: String,
    /// Name of the tool that executed.
    pub tool_name: String,
    /// Validated arguments that were passed to the tool.
    pub args: Value,
    /// The content blocks returned by the tool.
    pub content: Vec<MediaContentBlock>,
    /// The structured details returned by the tool.
    pub details: Value,
    /// Whether the result is currently treated as an error.
    pub is_error: bool,
}

/// Partial override returned from a `ToolResultEvent` handler.
///
/// Omitted fields (`None`) keep their original values. No deep merge.
#[derive(Debug, Clone, Default)]
pub struct ToolResultPatch {
    /// Replacement content blocks, if provided.
    pub content: Option<Vec<MediaContentBlock>>,
    /// Replacement details payload, if provided.
    pub details: Option<Value>,
    /// Replacement error flag, if provided.
    pub is_error: Option<bool>,
    /// Hint that the agent should stop after this tool batch.
    pub terminate: Option<bool>,
}

/// Emitted before each LLM call. Handlers can modify the message list.
///
/// **Sequential chain**: each handler's replacement messages become input to
/// the next handler.
#[derive(Debug, Clone)]
pub struct ContextEvent {
    /// Current messages that will be sent to the LLM.
    pub messages: Vec<AgentMessage>,
}

/// Result returned from a `ContextEvent` handler.
///
/// Return `Some(...)` with replacement messages; return `None` to keep as-is.
#[derive(Debug, Clone)]
pub struct ContextResult {
    /// Replacement message list for the LLM call.
    pub messages: Vec<AgentMessage>,
}

/// Emitted before the agent loop starts processing a prompt.
///
/// **Sequential accumulate**: all handler results are collected. Custom
/// messages are accumulated in order. The last non-`None` `system_prompt`
/// wins.
#[derive(Debug, Clone)]
pub struct BeforeAgentStartEvent {
    /// The raw user prompt text.
    pub prompt: String,
    /// Images attached to the user prompt, if any.
    pub images: Vec<ImageContent>,
    /// The fully assembled system prompt string.
    pub system_prompt: String,
}

/// Result from a single `before_agent_start` handler.
///
/// Each handler may return a custom message to inject alongside the user
/// prompt and/or override the system prompt for this turn.
#[derive(Debug, Clone, Default)]
pub struct BeforeAgentStartResult {
    /// Optional custom message to inject alongside the user message.
    pub message: Option<BeforeAgentStartMessage>,
    /// Replacement system prompt for this turn. Last handler's value wins.
    pub system_prompt: Option<String>,
}

/// A custom message produced by a `before_agent_start` handler.
#[derive(Debug, Clone)]
pub struct BeforeAgentStartMessage {
    /// Custom type discriminator (e.g. `"context"`, `"rules"`).
    pub custom_type: String,
    /// Message content (plain text or rich media blocks).
    pub content: crate::types::CustomMessageContent,
    /// Whether this message should be visible in the UI.
    pub display: bool,
    /// Optional structured details for downstream consumption.
    pub details: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Summary formatting hooks (first-to-return wins)
// ---------------------------------------------------------------------------

/// Emitted when a compaction summary needs formatting into an [`AgentMessage`].
///
/// Handlers return `Some` to replace the default formatting. If no handler
/// returns `Some`, the default conversion wraps the summary in a synthetic
/// user message.
#[derive(Debug, Clone)]
pub struct FormatCompactionSummaryEvent {
    /// The compaction summary text.
    pub summary: String,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

/// Result from a compaction summary formatting hook.
///
/// Return `Some(...)` to replace the default conversion; return `None` to
/// use the default.
#[derive(Debug)]
pub struct FormatCompactionSummaryResult {
    /// Replacement agent message for the compaction summary.
    pub message: AgentMessage,
}

/// Emitted when a branch summary needs formatting into an [`AgentMessage`].
///
/// Handlers return `Some` to replace the default formatting. If no handler
/// returns `Some`, the default conversion wraps the summary in a synthetic
/// user message.
#[derive(Debug, Clone)]
pub struct FormatBranchSummaryEvent {
    /// The branch summary text.
    pub summary: String,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

/// Result from a branch summary formatting hook.
///
/// Return `Some(...)` to replace the default conversion; return `None` to
/// use the default.
#[derive(Debug)]
pub struct FormatBranchSummaryResult {
    /// Replacement agent message for the branch summary.
    pub message: AgentMessage,
}

// ---------------------------------------------------------------------------
// Message end hook (sequential chain)
// ---------------------------------------------------------------------------

/// Result returned from a `MessageEndEvent` handler.
///
/// Handlers can replace the finalized message. The replacement **must**
/// preserve the original message role — a mismatch is logged as an error
/// and the replacement is skipped.
#[derive(Debug, Clone)]
pub struct MessageEndResult {
    /// Replacement message. Must keep the same role as the original.
    pub message: AgentMessage,
}

// ---------------------------------------------------------------------------
// Command system
// ---------------------------------------------------------------------------

/// Context passed to command handlers.
///
/// Wraps an [`ExtensionContext`](super::ExtensionContext) with the context
/// available at the time the command is invoked. Downstream applications can
/// extend command context with their own fields by wrapping this struct.
#[derive(Clone)]
pub struct CommandContext {
    /// Extension context with agent/session/interface access.
    pub extension_context: super::ExtensionContext,
}

impl fmt::Debug for CommandContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandContext")
            .field("extension_context", &self.extension_context)
            .finish()
    }
}

/// Handler function type for extension commands.
pub type CommandHandlerFn =
    dyn Fn(String, CommandContext) -> BoxFuture<anyhow::Result<()>> + Send + Sync;

/// A command registered by an extension.
#[derive(Clone)]
pub struct RegisteredCommand {
    /// Command name (used for dispatch).
    pub name: String,
    /// Optional description for documentation/discovery.
    pub description: Option<String>,
    /// Name of the extension that registered this command.
    pub extension_name: String,
    /// Handler function.
    pub handler: Arc<CommandHandlerFn>,
}

impl fmt::Debug for RegisteredCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredCommand")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("extension_name", &self.extension_name)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Enum for internal dispatch
// ---------------------------------------------------------------------------

/// Union of all extension event types for internal dispatch.
///
/// The runtime maps `AgentEvent` variants to these. Not used directly by
/// extension handlers — they receive the concrete event structs.
#[derive(Debug, Clone)]
pub enum ExtensionEvent {
    // Notification (fire-and-forget)
    AgentStart(AgentStartEvent),
    AgentEnd(AgentEndEvent),
    TurnStart(TurnStartEvent),
    TurnEnd(TurnEndEvent),
    MessageStart(MessageStartEvent),
    MessageUpdate(MessageUpdateEvent),
    MessageEnd(MessageEndEvent),
    ToolExecutionStart(ToolExecutionStartEvent),
    ToolExecutionUpdate(ToolExecutionUpdateEvent),
    ToolExecutionEnd(ToolExecutionEndEvent),
    SessionStart(SessionStartEvent),
    SessionShutdown(SessionShutdownEvent),

    // Hooks
    ToolCall(ToolCallEvent),
    ToolResult(ToolResultEvent),
    Context(ContextEvent),
    BeforeAgentStart(BeforeAgentStartEvent),
    FormatCompactionSummary(FormatCompactionSummaryEvent),
    FormatBranchSummary(FormatBranchSummaryEvent),
}

impl fmt::Display for ExtensionEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AgentStart(_) => write!(f, "agent_start"),
            Self::AgentEnd(_) => write!(f, "agent_end"),
            Self::TurnStart(_) => write!(f, "turn_start"),
            Self::TurnEnd(_) => write!(f, "turn_end"),
            Self::MessageStart(_) => write!(f, "message_start"),
            Self::MessageUpdate(_) => write!(f, "message_update"),
            Self::MessageEnd(_) => write!(f, "message_end"),
            Self::ToolExecutionStart(_) => write!(f, "tool_execution_start"),
            Self::ToolExecutionUpdate(_) => write!(f, "tool_execution_update"),
            Self::ToolExecutionEnd(_) => write!(f, "tool_execution_end"),
            Self::SessionStart(_) => write!(f, "session_start"),
            Self::SessionShutdown(_) => write!(f, "session_shutdown"),
            Self::ToolCall(_) => write!(f, "tool_call"),
            Self::ToolResult(_) => write!(f, "tool_result"),
            Self::Context(_) => write!(f, "context"),
            Self::BeforeAgentStart(_) => write!(f, "before_agent_start"),
            Self::FormatCompactionSummary(_) => write!(f, "format_compaction_summary"),
            Self::FormatBranchSummary(_) => write!(f, "format_branch_summary"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_result_block() {
        let result = ToolCallResult::block("dangerous");
        assert!(result.block);
        assert_eq!(result.reason.as_deref(), Some("dangerous"));
    }

    #[test]
    fn tool_result_patch_default_is_no_override() {
        let patch = ToolResultPatch::default();
        assert!(patch.content.is_none());
        assert!(patch.details.is_none());
        assert!(patch.is_error.is_none());
        assert!(patch.terminate.is_none());
    }

    #[test]
    fn extension_event_display() {
        assert_eq!(
            ExtensionEvent::AgentStart(AgentStartEvent).to_string(),
            "agent_start"
        );
        assert_eq!(
            ExtensionEvent::AgentEnd(AgentEndEvent { messages: vec![] }).to_string(),
            "agent_end"
        );
        assert_eq!(
            ExtensionEvent::TurnStart(TurnStartEvent {
                turn_index: 0,
                timestamp: 0
            })
            .to_string(),
            "turn_start"
        );
        assert_eq!(
            ExtensionEvent::SessionStart(SessionStartEvent {
                reason: SessionStartReason::Startup
            })
            .to_string(),
            "session_start"
        );
        assert_eq!(
            ExtensionEvent::SessionShutdown(SessionShutdownEvent {
                reason: SessionShutdownReason::Quit
            })
            .to_string(),
            "session_shutdown"
        );
        assert_eq!(
            ExtensionEvent::BeforeAgentStart(BeforeAgentStartEvent {
                prompt: "hello".into(),
                images: vec![],
                system_prompt: String::new(),
            })
            .to_string(),
            "before_agent_start"
        );
    }

    #[test]
    fn session_start_reason_copy() {
        let reason = SessionStartReason::Reload;
        let copied = reason;
        assert_eq!(reason, copied);
    }

    #[test]
    fn session_shutdown_reason_copy() {
        let reason = SessionShutdownReason::Quit;
        let copied = reason;
        assert_eq!(reason, copied);
    }

    #[test]
    fn before_agent_start_result_default() {
        let result = BeforeAgentStartResult::default();
        assert!(result.message.is_none());
        assert!(result.system_prompt.is_none());
    }

    #[test]
    fn turn_events_have_index() {
        let start = TurnStartEvent {
            turn_index: 3,
            timestamp: 1000,
        };
        assert_eq!(start.turn_index, 3);

        let end = TurnEndEvent {
            turn_index: 3,
            message: AgentMessage::User(ameli_ai::types::UserMessage::text("hi")),
            tool_results: vec![],
        };
        assert_eq!(end.turn_index, 3);
    }

    #[test]
    fn registered_command_debug() {
        let cmd = RegisteredCommand {
            name: "my-command".into(),
            description: Some("Does a thing".into()),
            extension_name: "test-ext".into(),
            handler: Arc::new(|_args, _ctx| Box::pin(async { Ok(()) })),
        };
        let debug = format!("{cmd:?}");
        assert!(debug.contains("my-command"));
        assert!(debug.contains("test-ext"));
    }

    #[test]
    fn command_context_debug() {
        let ctx = CommandContext {
            extension_context: super::super::ExtensionContext::for_testing(),
        };
        let debug = format!("{ctx:?}");
        assert!(debug.contains("CommandContext"));
    }

    #[test]
    fn message_end_result_preserves_message() {
        let msg = AgentMessage::User(ameli_ai::types::UserMessage::text("hello"));
        let result = MessageEndResult {
            message: msg.clone(),
        };
        assert_eq!(result.message.role(), "user");
    }
}
