//! Extension event types and result types.
//!
//! Defines the typed events that extensions can subscribe to, split into two
//! categories:
//!
//! - **Notification events** — extensions observe but cannot modify (e.g.,
//!   `AgentStartEvent`, `TurnEndEvent`).
//! - **Hook events** — extensions can return a result to influence agent
//!   behaviour (e.g., `ToolCallEvent` → `ToolCallResult` to block execution).

use ameli_agent_core::types::AgentMessage;
use ameli_ai::types::{AssistantMessageEvent, MediaContentBlock, ToolResultMessage};
use serde_json::Value;
use std::fmt;

// ---------------------------------------------------------------------------
// Notification events
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
pub struct TurnStartEvent;

/// Emitted at the end of each turn.
#[derive(Debug, Clone)]
pub struct TurnEndEvent {
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

/// Emitted when a tool finishes executing.
#[derive(Debug, Clone)]
pub struct ToolExecutionEndEvent {
    /// Unique ID for this tool call.
    pub tool_call_id: String,
    /// Name of the tool that executed.
    pub tool_name: String,
    /// The tool execution result.
    pub result: ameli_agent_core::types::AgentToolResult,
    /// Whether the result represents an error.
    pub is_error: bool,
}

// ---------------------------------------------------------------------------
// Hook events
// ---------------------------------------------------------------------------

/// Emitted before a tool executes. Handlers can block execution.
#[derive(Debug, Clone)]
pub struct ToolCallEvent {
    /// Unique ID for this tool call.
    pub tool_call_id: String,
    /// Name of the tool about to execute.
    pub tool_name: String,
    /// Validated arguments for the tool call.
    ///
    /// This field is immutable in the current API — handlers receive an owned
    /// copy but mutations do not propagate to subsequent handlers or the agent
    /// loop. Future work may add argument patching via [`ToolCallResult`].
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

// ---------------------------------------------------------------------------
// Summary formatting hooks
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
// Enum for internal dispatch
// ---------------------------------------------------------------------------

/// Union of all extension event types for internal dispatch.
///
/// The runtime maps `AgentEvent` variants to these. Not used directly by
/// extension handlers — they receive the concrete event structs.
#[derive(Debug, Clone)]
pub enum ExtensionEvent {
    // Notification
    AgentStart(AgentStartEvent),
    AgentEnd(AgentEndEvent),
    TurnStart(TurnStartEvent),
    TurnEnd(TurnEndEvent),
    MessageStart(MessageStartEvent),
    MessageUpdate(MessageUpdateEvent),
    MessageEnd(MessageEndEvent),
    ToolExecutionStart(ToolExecutionStartEvent),
    ToolExecutionEnd(ToolExecutionEndEvent),

    // Hooks
    ToolCall(ToolCallEvent),
    ToolResult(ToolResultEvent),
    Context(ContextEvent),
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
            Self::ToolExecutionEnd(_) => write!(f, "tool_execution_end"),
            Self::ToolCall(_) => write!(f, "tool_call"),
            Self::ToolResult(_) => write!(f, "tool_result"),
            Self::Context(_) => write!(f, "context"),
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
            ExtensionEvent::TurnStart(TurnStartEvent).to_string(),
            "turn_start"
        );
    }
}
