//! Compatibility settings for OpenAI-compatible completions APIs.
//!
//! The [`OpenAICompletionsCompat`] struct controls provider-specific behavior
//! for message formatting, parameter names, and feature support. It is
//! deserialized from the [`Model::compat`](ameli_ai::types::Model::compat)
//! JSON field with sensible defaults for standard OpenAI.

use crate::types::Model;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ThinkingFormat
// ---------------------------------------------------------------------------

/// How thinking/reasoning is expressed in the request body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ThinkingFormat {
    /// OpenAI standard: `reasoning_effort: "low" | "medium" | "high"`.
    #[default]
    OpenAi,
    /// ZAI / Qwen style: top-level `enable_thinking: true/false`.
    Zai,
    /// DeepSeek style: uses `reasoning_effort` like OpenAI but has different
    /// response-side handling for reasoning content.
    Deepseek,
}

// ---------------------------------------------------------------------------
// MaxTokensField
// ---------------------------------------------------------------------------

/// Which JSON field name to use for maximum output tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum MaxTokensField {
    /// Modern OpenAI field: `max_completion_tokens`.
    #[default]
    MaxCompletionTokens,
    /// Legacy / compatible field: `max_tokens`.
    MaxTokens,
}

// ---------------------------------------------------------------------------
// OpenAICompletionsCompat
// ---------------------------------------------------------------------------

/// Compatibility settings for OpenAI-compatible completions APIs.
///
/// Controls provider-specific behavior. Settings are resolved from the model's
/// `compat` JSON field (deserialized into this struct) with sensible defaults
/// that match standard OpenAI behavior.
///
/// # ZAI configuration
///
/// For ZAI models, set:
/// ```json
/// {
///   "thinkingFormat": "zai",
///   "zaiToolStream": true,
///   "supportsDeveloperRole": false,
///   "supportsStore": false,
///   "supportsReasoningEffort": false
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAICompletionsCompat {
    /// Whether the provider supports the `store` field. Default: `true`.
    #[serde(default = "default_true")]
    pub supports_store: bool,

    /// Whether the `developer` role is supported (vs `system` only).
    /// Default: `true`.
    #[serde(default = "default_true")]
    pub supports_developer_role: bool,

    /// Whether the provider supports the `reasoning_effort` parameter.
    /// Default: `true`. Set to `false` for ZAI.
    #[serde(default = "default_true")]
    pub supports_reasoning_effort: bool,

    /// Whether the provider supports `stream_options: { include_usage: true }`
    /// for token usage in streaming responses. Default: `true`.
    #[serde(default = "default_true")]
    pub supports_usage_in_streaming: bool,

    /// Which JSON field name to use for max output tokens.
    #[serde(default)]
    pub max_tokens_field: MaxTokensField,

    /// How thinking/reasoning is expressed in the request body.
    #[serde(default)]
    pub thinking_format: ThinkingFormat,

    /// Whether to send `tool_stream: true` for incremental tool call argument
    /// streaming. ZAI-specific. Default: `false`.
    #[serde(default)]
    pub zai_tool_stream: bool,

    /// Whether the provider supports the `strict` field in tool definitions.
    /// Default: `true`.
    #[serde(default = "default_true")]
    pub supports_strict_mode: bool,

    /// Whether the provider requires `reasoning_content` on assistant messages
    /// in the conversation history. DeepSeek/Xiaomi need this. Default: `false`.
    #[serde(default)]
    pub requires_reasoning_content_on_assistant_messages: bool,
}

fn default_true() -> bool {
    true
}

impl Default for OpenAICompletionsCompat {
    fn default() -> Self {
        Self {
            supports_store: true,
            supports_developer_role: true,
            supports_reasoning_effort: true,
            supports_usage_in_streaming: true,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            thinking_format: ThinkingFormat::OpenAi,
            zai_tool_stream: false,
            supports_strict_mode: true,
            requires_reasoning_content_on_assistant_messages: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Resolve compat from model
// ---------------------------------------------------------------------------

/// Resolve compatibility settings from a model.
///
/// Deserializes the model's `compat` field into [`OpenAICompletionsCompat`].
/// Returns defaults if the field is absent or cannot be deserialized.
pub fn get_compat(model: &Model) -> OpenAICompletionsCompat {
    model
        .compat
        .as_ref()
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_compat_is_openai() {
        let compat = OpenAICompletionsCompat::default();
        assert!(compat.supports_store);
        assert!(compat.supports_developer_role);
        assert!(compat.supports_reasoning_effort);
        assert!(compat.supports_usage_in_streaming);
        assert_eq!(compat.max_tokens_field, MaxTokensField::MaxCompletionTokens);
        assert_eq!(compat.thinking_format, ThinkingFormat::OpenAi);
        assert!(!compat.zai_tool_stream);
        assert!(compat.supports_strict_mode);
    }

    #[test]
    fn zai_compat_deserializes() {
        let json = serde_json::json!({
            "thinkingFormat": "zai",
            "zaiToolStream": true,
            "supportsDeveloperRole": false,
            "supportsStore": false,
            "supportsReasoningEffort": false
        });
        let compat: OpenAICompletionsCompat = serde_json::from_value(json).unwrap();
        assert_eq!(compat.thinking_format, ThinkingFormat::Zai);
        assert!(compat.zai_tool_stream);
        assert!(!compat.supports_developer_role);
        assert!(!compat.supports_store);
        assert!(!compat.supports_reasoning_effort);
    }

    #[test]
    fn partial_compat_uses_defaults() {
        let json = serde_json::json!({
            "thinkingFormat": "zai"
        });
        let compat: OpenAICompletionsCompat = serde_json::from_value(json).unwrap();
        assert_eq!(compat.thinking_format, ThinkingFormat::Zai);
        assert!(compat.supports_store); // default
        assert!(compat.supports_developer_role); // default
    }

    #[test]
    fn deepseek_compat_deserializes() {
        let json = serde_json::json!({
            "thinkingFormat": "deepseek",
            "requiresReasoningContentOnAssistantMessages": true
        });
        let compat: OpenAICompletionsCompat = serde_json::from_value(json).unwrap();
        assert_eq!(compat.thinking_format, ThinkingFormat::Deepseek);
        assert!(compat.requires_reasoning_content_on_assistant_messages);
        // Other fields should be defaults
        assert!(compat.supports_store);
        assert!(compat.supports_developer_role);
    }

    #[test]
    fn compat_roundtrip() {
        let compat = OpenAICompletionsCompat::default();
        let json = serde_json::to_string(&compat).unwrap();
        let back: OpenAICompletionsCompat = serde_json::from_str(&json).unwrap();
        assert_eq!(compat, back);
    }
}
