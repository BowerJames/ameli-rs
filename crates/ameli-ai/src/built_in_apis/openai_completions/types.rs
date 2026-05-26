//! OpenAI Chat Completions SSE wire types for deserializing streaming chunks.

use serde::Deserialize;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Chat Completion Chunk (SSE response)
// ---------------------------------------------------------------------------

/// A single SSE chunk from the OpenAI Chat Completions streaming API.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: Option<String>,
    pub model: Option<String>,
    pub choices: Option<Vec<ChunkChoice>>,
    pub usage: Option<ChunkUsage>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkChoice {
    pub finish_reason: Option<String>,
    pub delta: Option<ChunkDelta>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ChunkToolCall>>,
    /// Extra fields captured for reasoning detection
    /// (`reasoning_content`, `reasoning`, `reasoning_text`, etc.).
    #[serde(flatten)]
    pub other: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkToolCall {
    pub index: Option<u32>,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub function: Option<ChunkToolCallFunction>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkToolCallFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub prompt_tokens_details: Option<ChunkPromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkPromptTokensDetails {
    pub cached_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
}

// ---------------------------------------------------------------------------
// Reasoning helpers
// ---------------------------------------------------------------------------

/// Extract reasoning content from the first matching field in the delta.
///
/// Providers use different field names: `reasoning_content`, `reasoning`,
/// `reasoning_text`. Returns the first non-empty string found.
pub fn get_reasoning_content(delta: &ChunkDelta) -> Option<&str> {
    for field in &["reasoning_content", "reasoning", "reasoning_text"] {
        if let Some(serde_json::Value::String(s)) = delta.other.get(*field) {
            if !s.is_empty() {
                return Some(s);
            }
        }
    }
    None
}

/// Get the field name that contained reasoning content (used as the
/// `thinking_signature` for multi-turn replay).
pub fn get_reasoning_field_name(delta: &ChunkDelta) -> String {
    for field in &["reasoning_content", "reasoning", "reasoning_text"] {
        if let Some(serde_json::Value::String(s)) = delta.other.get(*field) {
            if !s.is_empty() {
                return (*field).to_string();
            }
        }
    }
    "reasoning_content".to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_chunk_with_text() {
        let json = r#"{
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "finish_reason": null,
                "delta": {"role": "assistant", "content": "Hello"}
            }]
        }"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.id.as_deref(), Some("chatcmpl-123"));
        let choice = chunk.choices.unwrap().into_iter().next().unwrap();
        assert!(choice.finish_reason.is_none());
        let delta = choice.delta.unwrap();
        assert_eq!(delta.content.as_deref(), Some("Hello"));
    }

    #[test]
    fn deserialize_chunk_with_tool_call() {
        let json = r#"{
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{\"co"}
                    }]
                }
            }]
        }"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        let choice = chunk.choices.unwrap().into_iter().next().unwrap();
        let delta = choice.delta.unwrap();
        let tc = &delta.tool_calls.unwrap()[0];
        assert_eq!(tc.id.as_deref(), Some("call_abc"));
        assert_eq!(tc.function.as_ref().unwrap().name.as_deref(), Some("bash"));
    }

    #[test]
    fn deserialize_chunk_with_reasoning() {
        let json = r#"{
            "choices": [{
                "delta": {
                    "content": null,
                    "reasoning_content": "Let me think..."
                }
            }]
        }"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        let choice = chunk.choices.unwrap().into_iter().next().unwrap();
        let delta = choice.delta.unwrap();
        assert_eq!(get_reasoning_content(&delta), Some("Let me think..."));
        assert_eq!(get_reasoning_field_name(&delta), "reasoning_content");
    }

    #[test]
    fn deserialize_usage_chunk() {
        let json = r#"{
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {
                    "cached_tokens": 80
                }
            }
        }"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(100));
        assert_eq!(usage.completion_tokens, Some(50));
        assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, Some(80));
    }

    #[test]
    fn deserialize_finish_reason() {
        let json = r#"{"choices": [{"finish_reason": "tool_calls", "delta": {}}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        let choice = chunk.choices.unwrap().into_iter().next().unwrap();
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    }
}
