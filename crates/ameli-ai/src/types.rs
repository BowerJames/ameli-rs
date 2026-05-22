//! Core types for the ameli-ai unified LLM API.
//!
//! This module defines the shared vocabulary used across the ameli ecosystem:
//! content blocks, messages, models, streaming events, and stream options.
//!
//! These types are a restricted, lightweight port of the pi-ai TypeScript
//! package, focused on what ameli-agent actually needs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// Reason why an assistant message stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
}

/// Thinking/reasoning level for models that support extended thinking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

/// Thinking level including the explicit "off" state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ModelThinkingLevel {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

impl From<ThinkingLevel> for ModelThinkingLevel {
    fn from(level: ThinkingLevel) -> Self {
        match level {
            ThinkingLevel::Minimal => ModelThinkingLevel::Minimal,
            ThinkingLevel::Low => ModelThinkingLevel::Low,
            ThinkingLevel::Medium => ModelThinkingLevel::Medium,
            ThinkingLevel::High => ModelThinkingLevel::High,
            ThinkingLevel::XHigh => ModelThinkingLevel::XHigh,
        }
    }
}

/// Prompt cache retention preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheRetention {
    None,
    Short,
    Long,
}

/// Preferred transport for providers that support multiple transports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Transport {
    Sse,
    #[serde(rename = "websocket")]
    WebSocket,
    #[serde(rename = "websocket-cached")]
    WebSocketCached,
    Auto,
}

/// Input modality a model supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputType {
    Text,
    Image,
}

// ---------------------------------------------------------------------------
// Content Blocks
// ---------------------------------------------------------------------------

/// Text content produced by or provided to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    pub text: String,
    /// Opaque signature for multi-turn text content continuity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

impl TextContent {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            text_signature: None,
        }
    }
}

/// Extended thinking / reasoning content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    pub thinking: String,
    /// Opaque signature for multi-turn thinking continuity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    /// True when the thinking content was redacted by safety filters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted: Option<bool>,
}

impl ThinkingContent {
    pub fn new(thinking: impl Into<String>) -> Self {
        Self {
            thinking: thinking.into(),
            thinking_signature: None,
            redacted: None,
        }
    }
}

/// Base64-encoded image content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    /// Base64-encoded image data.
    pub data: String,
    /// MIME type (e.g., `"image/png"`).
    pub mime_type: String,
}

/// A tool call requested by the assistant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
    /// Provider-specific opaque signature for thought context reuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

// ---------------------------------------------------------------------------
// Content Block Enums
// ---------------------------------------------------------------------------

/// Content blocks that appear in an assistant message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AssistantContentBlock {
    Text(TextContent),
    Thinking(ThinkingContent),
    ToolCall(ToolCall),
}

/// Content blocks shared by user messages and tool results (text + image).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum MediaContentBlock {
    Text(TextContent),
    Image(ImageContent),
}

/// Content of a user message: either a plain string or structured blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    /// A plain text string.
    Text(String),
    /// Structured content blocks (text and/or images).
    Blocks(Vec<MediaContentBlock>),
}

// ---------------------------------------------------------------------------
// Usage & Cost
// ---------------------------------------------------------------------------

/// Token cost breakdown in USD.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

/// Token usage statistics from an LLM response.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub total_tokens: u64,
    pub cost: UsageCost,
}

impl Usage {
    /// Calculate USD costs from token counts and per-million-token pricing.
    pub fn calculate_cost(&mut self, cost: &Cost) {
        self.cost.input = (self.input as f64 * cost.input) / 1_000_000.0;
        self.cost.output = (self.output as f64 * cost.output) / 1_000_000.0;
        self.cost.cache_read = (self.cache_read as f64 * cost.cache_read) / 1_000_000.0;
        self.cost.cache_write = (self.cache_write as f64 * cost.cache_write) / 1_000_000.0;
        self.cost.total =
            self.cost.input + self.cost.output + self.cost.cache_read + self.cost.cache_write;
    }
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// A message from the user.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessage {
    pub content: UserContent,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

impl UserMessage {
    /// Create a text user message with the current timestamp.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: UserContent::Text(text.into()),
            timestamp: now_ms(),
        }
    }
}

/// A message from the assistant (the LLM response).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub content: Vec<AssistantContentBlock>,
    /// API protocol used (e.g., `"openai-responses"`, `"anthropic-messages"`).
    pub api: String,
    /// Provider name (e.g., `"openai"`, `"anthropic"`).
    pub provider: String,
    /// Requested model identifier.
    pub model: String,
    /// Concrete model that served the request, when different from requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_model: Option<String>,
    /// Provider-specific response/message identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    pub usage: Usage,
    pub stop_reason: StopReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

/// A tool result returned to the model after executing a tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<MediaContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    pub is_error: bool,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

impl ToolResultMessage {
    /// Create an error tool result for a given tool call.
    pub fn error(tool_call_id: impl Into<String>, tool_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            content: vec![MediaContentBlock::Text(TextContent::new(message))],
            details: None,
            is_error: true,
            timestamp: now_ms(),
        }
    }
}

/// A conversation message (user, assistant, or tool result).
///
/// Serialized with a `"role"` tag matching the TypeScript discriminated union.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
    #[serde(rename = "toolResult")]
    ToolResult(ToolResultMessage),
}

// ---------------------------------------------------------------------------
// Model & Tool
// ---------------------------------------------------------------------------

/// Token cost per million tokens (USD).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// Maps thinking levels to provider-specific values.
///
/// A missing key means "use provider default". A key present with `None`
/// means "this level is unsupported". A key present with `Some(value)`
/// maps to the given provider string.
pub type ThinkingLevelMap = HashMap<ModelThinkingLevel, Option<String>>;

/// Descriptor for an LLM model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    /// API protocol (e.g., `"openai-responses"`, `"anthropic-messages"`).
    pub api: String,
    /// Provider name (e.g., `"openai"`, `"anthropic"`).
    pub provider: String,
    pub base_url: String,
    /// Whether the model supports extended thinking/reasoning.
    pub reasoning: bool,
    /// Maps thinking levels to provider-specific values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    /// Supported input modalities.
    pub input: Vec<InputType>,
    pub cost: Cost,
    /// Context window size in tokens.
    pub context_window: u64,
    /// Maximum output tokens.
    pub max_tokens: u64,
    /// Provider-specific compatibility settings.
    ///
    /// Providers deserialize this to their expected compat struct (e.g.,
    /// `OpenAICompletionsCompat`). When `None`, providers use default settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<serde_json::Value>,
}

/// A tool definition with a JSON Schema for its parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the tool's parameters.
    pub parameters: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// The full context sent to an LLM for a completion request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

// ---------------------------------------------------------------------------
// Stream Options
// ---------------------------------------------------------------------------

/// Token budgets for each thinking level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingBudgets {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimal: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub medium: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high: Option<u64>,
}

/// Options passed to a stream function when requesting an LLM completion.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Preferred transport for providers that support multiple transports.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transport: Option<Transport>,
    /// Prompt cache retention preference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_retention: Option<CacheRetention>,
    /// Session identifier for cache-aware backends.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Custom HTTP headers merged with provider defaults.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// HTTP request timeout in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Maximum retry attempts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    /// Maximum delay in milliseconds for provider-requested retries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retry_delay_ms: Option<u64>,
    /// Provider metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// Thinking level for models that support reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ThinkingLevel>,
    /// Custom token budgets for thinking levels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budgets: Option<ThinkingBudgets>,
}

/// HTTP response metadata from a provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Streaming Events
// ---------------------------------------------------------------------------

/// Events emitted during an assistant message stream.
///
/// The stream always terminates with either [`Done`](AssistantMessageEvent::Done)
/// or [`Error`](AssistantMessageEvent::Error). Every non-terminal variant
/// carries a `partial` snapshot of the message being built, so consumers can
/// observe incremental progress.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AssistantMessageEvent {
    #[serde(rename = "start")]
    Start { partial: AssistantMessage },

    #[serde(rename = "text_start")]
    TextStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    #[serde(rename = "text_delta")]
    TextDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    #[serde(rename = "text_end")]
    TextEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },

    #[serde(rename = "thinking_start")]
    ThinkingStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    #[serde(rename = "thinking_end")]
    ThinkingEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },

    #[serde(rename = "toolcall_start")]
    ToolCallStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    #[serde(rename = "toolcall_delta")]
    ToolCallDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    #[serde(rename = "toolcall_end")]
    ToolCallEnd {
        content_index: usize,
        tool_call: ToolCall,
        partial: AssistantMessage,
    },

    #[serde(rename = "done")]
    Done {
        reason: StopReason,
        message: AssistantMessage,
    },
    #[serde(rename = "error")]
    Error {
        reason: StopReason,
        error: AssistantMessage,
    },
}

impl AssistantMessageEvent {
    /// Returns `true` for terminal events ([`Done`] or [`Error`]).
    ///
    /// [`Done`]: AssistantMessageEvent::Done
    /// [`Error`]: AssistantMessageEvent::Error
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done { .. } | Self::Error { .. })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the current Unix timestamp in milliseconds.
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

    #[test]
    fn message_roundtrip_json() {
        // Verify that Message serializes with the correct "role" tag
        let user = Message::User(UserMessage::text("hello"));
        let json = serde_json::to_string(&user).unwrap();
        assert!(json.contains(r#""role":"user""#));
        assert!(json.contains(r#""content":"hello""#));

        let roundtrip: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(user, roundtrip);
    }

    #[test]
    fn tool_result_role_serializes() {
        let msg = Message::ToolResult(ToolResultMessage::error("tc_1", "bash", "failed"));
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""role":"toolResult""#));
    }

    #[test]
    fn stop_reason_serializes() {
        assert_eq!(
            serde_json::to_string(&StopReason::ToolUse).unwrap(),
            r#""toolUse""#
        );
    }

    #[test]
    fn transport_serializes() {
        assert_eq!(
            serde_json::to_string(&Transport::WebSocketCached).unwrap(),
            r#""websocket-cached""#
        );
    }

    #[test]
    fn content_block_roundtrip() {
        let block = AssistantContentBlock::Text(TextContent::new("hi"));
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""type":"text""#));
        assert!(json.contains(r#""text":"hi""#));

        let rt: AssistantContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, rt);
    }

    #[test]
    fn user_content_untagged_text() {
        let content = UserContent::Text("plain".to_string());
        let json = serde_json::to_string(&content).unwrap();
        assert_eq!(json, r#""plain""#);

        let rt: UserContent = serde_json::from_str(&json).unwrap();
        assert_eq!(content, rt);
    }

    #[test]
    fn user_content_untagged_blocks() {
        let content = UserContent::Blocks(vec![MediaContentBlock::Text(TextContent::new("rich"))]);
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.starts_with('['));

        let rt: UserContent = serde_json::from_str(&json).unwrap();
        assert_eq!(content, rt);
    }

    #[test]
    fn event_roundtrip() {
        let partial = AssistantMessage {
            content: vec![],
            api: "test".into(),
            provider: "test".into(),
            model: "test".into(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 1000,
        };
        let event = AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "hello".into(),
            partial,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"text_delta""#));
        assert!(json.contains(r#""delta":"hello""#));

        let rt: AssistantMessageEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, rt);
    }

    #[test]
    fn event_is_terminal() {
        let msg = AssistantMessage {
            content: vec![],
            api: "test".into(),
            provider: "test".into(),
            model: "test".into(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        };
        assert!(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: msg.clone()
        }
        .is_terminal());
        assert!(!AssistantMessageEvent::Start { partial: msg }.is_terminal());
    }
}
