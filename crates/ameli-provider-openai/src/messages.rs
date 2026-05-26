//! Message and parameter conversion: ameli types → OpenAI Chat Completions format.
//!
//! Converts [`Context`] into OpenAI-compatible request parameters including
//! message history, tool definitions, and streaming options.

use crate::compat::{MaxTokensField, OpenAICompletionsCompat, ThinkingFormat};
use ameli_ai::types::{
    AssistantContentBlock, Context, InputType, MediaContentBlock, Message, Model,
    ModelThinkingLevel, StopReason, StreamOptions, ThinkingContent, Tool, ToolCall,
    ToolResultMessage, UserContent,
};
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build the full request body for the OpenAI Chat Completions API.
pub fn build_request_params(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
    compat: &OpenAICompletionsCompat,
) -> Value {
    let mut map = Map::new();
    map.insert("model".into(), json!(model.id));
    map.insert("stream".into(), json!(true));

    // --- Messages ---
    let mut messages = Vec::new();

    if let Some(system_prompt) = &context.system_prompt {
        if !system_prompt.is_empty() {
            let role = if model.reasoning && compat.supports_developer_role {
                "developer"
            } else {
                "system"
            };
            messages.push(json!({ "role": role, "content": system_prompt }));
        }
    }

    let supports_images = model.input.contains(&InputType::Image);
    let converted = convert_messages(context, model, supports_images);
    messages.extend(converted);
    map.insert("messages".into(), json!(messages));

    // --- Tools ---
    if let Some(tools) = &context.tools {
        if !tools.is_empty() {
            let converted_tools = convert_tools(tools, compat);
            map.insert("tools".into(), json!(converted_tools));
            if compat.zai_tool_stream {
                map.insert("tool_stream".into(), json!(true));
            }
        }
    }

    // --- Streaming options ---
    if compat.supports_usage_in_streaming {
        map.insert("stream_options".into(), json!({ "include_usage": true }));
    }

    if compat.supports_store {
        map.insert("store".into(), json!(false));
    }

    // --- Token limits ---
    if let Some(max_tokens) = options.max_tokens {
        match compat.max_tokens_field {
            MaxTokensField::MaxCompletionTokens => {
                map.insert("max_completion_tokens".into(), json!(max_tokens));
            }
            MaxTokensField::MaxTokens => {
                map.insert("max_tokens".into(), json!(max_tokens));
            }
        }
    }

    if let Some(temp) = options.temperature {
        map.insert("temperature".into(), json!(temp));
    }

    // --- Thinking / reasoning ---
    if let Some(reasoning) = &options.reasoning {
        if model.reasoning {
            match compat.thinking_format {
                ThinkingFormat::OpenAi => {
                    if compat.supports_reasoning_effort {
                        let effort = resolve_reasoning_effort(reasoning, model);
                        map.insert("reasoning_effort".into(), json!(effort));
                    }
                }
                ThinkingFormat::Zai => {
                    map.insert("enable_thinking".into(), json!(true));
                }
            }
        }
    }

    Value::Object(map)
}

// ---------------------------------------------------------------------------
// Message conversion
// ---------------------------------------------------------------------------

/// Convert ameli [`Context`] messages into OpenAI message params.
fn convert_messages(context: &Context, model: &Model, supports_images: bool) -> Vec<Value> {
    let mut params: Vec<Value> = Vec::new();

    for msg in &context.messages {
        match msg {
            Message::User(user_msg) => {
                params.push(convert_user_message(user_msg, supports_images));
            }
            Message::Assistant(assistant_msg) => {
                // Skip error/aborted messages (incomplete turns)
                if assistant_msg.stop_reason == StopReason::Error
                    || assistant_msg.stop_reason == StopReason::Aborted
                {
                    continue;
                }
                if let Some(openai_msg) = convert_assistant_message(assistant_msg, model) {
                    params.push(openai_msg);
                }
            }
            Message::ToolResult(tool_result) => {
                params.push(convert_tool_result(tool_result));
            }
        }
    }

    params
}

/// Convert a user message.
fn convert_user_message(msg: &ameli_ai::types::UserMessage, supports_images: bool) -> Value {
    match &msg.content {
        UserContent::Text(text) => json!({
            "role": "user",
            "content": text,
        }),
        UserContent::Blocks(blocks) => {
            let parts: Vec<Value> = blocks
                .iter()
                .map(|block| match block {
                    MediaContentBlock::Text(tc) => json!({
                        "type": "text",
                        "text": tc.text,
                    }),
                    MediaContentBlock::Image(img) => {
                        if supports_images {
                            json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": format!("data:{};base64,{}", img.mime_type, img.data),
                                },
                            })
                        } else {
                            json!({
                                "type": "text",
                                "text": "(image omitted: model does not support images)",
                            })
                        }
                    }
                })
                .collect();

            if parts.is_empty() {
                json!({ "role": "user", "content": "" })
            } else {
                json!({ "role": "user", "content": parts })
            }
        }
    }
}

/// Convert an assistant message. Returns `None` if the message has no
/// content and no tool calls (empty/aborted responses are skipped).
fn convert_assistant_message(
    msg: &ameli_ai::types::AssistantMessage,
    model: &Model,
) -> Option<Value> {
    let is_same_model =
        msg.provider == model.provider && msg.api == model.api && msg.model == model.id;

    // Collect text content
    let text_parts: Vec<&str> = msg
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantContentBlock::Text(tc) if !tc.text.trim().is_empty() => Some(tc.text.as_str()),
            _ => None,
        })
        .collect();
    let text = text_parts.join("");

    // Collect thinking content
    let thinking_blocks: Vec<&ThinkingContent> = msg
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantContentBlock::Thinking(tc) if !tc.thinking.trim().is_empty() => Some(tc),
            _ => None,
        })
        .collect();

    // Tool calls
    let tool_calls: Vec<Value> = msg
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantContentBlock::ToolCall(tc) => Some(convert_tool_call(tc, is_same_model)),
            _ => None,
        })
        .collect();

    let mut map = Map::new();
    map.insert("role".into(), json!("assistant"));

    // Set content
    if !thinking_blocks.is_empty() {
        // Use the signature from the first block as the field name for replay
        if let Some(thinking_block) = thinking_blocks.first() {
            if let Some(sig) = &thinking_block.thinking_signature {
                if !sig.is_empty() {
                    let thinking_text: String = thinking_blocks
                        .iter()
                        .map(|b| b.thinking.as_str())
                        .collect::<Vec<_>>()
                        .join("\n");
                    map.insert(sig.clone(), json!(thinking_text));
                }
            }
        }
        map.insert(
            "content".into(),
            if text.is_empty() {
                Value::Null
            } else {
                json!(text)
            },
        );
    } else if !text.is_empty() {
        map.insert("content".into(), json!(text));
    } else {
        map.insert("content".into(), Value::Null);
    }

    if !tool_calls.is_empty() {
        map.insert("tool_calls".into(), json!(tool_calls));
    }

    let result = Value::Object(map);

    // Skip empty assistant messages
    let has_content = result
        .get("content")
        .is_some_and(|c| c != &Value::Null && c != &json!(""));
    let has_tool_calls = result.get("tool_calls").is_some();

    if !has_content && !has_tool_calls {
        return None;
    }

    Some(result)
}

/// Convert a tool call for assistant message replay.
fn convert_tool_call(tc: &ToolCall, is_same_model: bool) -> Value {
    let mut map = Map::new();
    map.insert("id".into(), json!(tc.id));
    map.insert("type".into(), json!("function"));
    map.insert(
        "function".into(),
        json!({
            "name": tc.name,
            "arguments": serde_json::to_string(&tc.arguments).unwrap_or_default(),
        }),
    );

    // Include reasoning_details for same-model replay if present
    if is_same_model {
        if let Some(sig) = &tc.thought_signature {
            if let Ok(parsed) = serde_json::from_str::<Value>(sig) {
                // Wrap single detail in array if needed
                let details = match parsed {
                    Value::Array(arr) => arr,
                    v => vec![v],
                };
                if !details.is_empty() {
                    map.insert("reasoning_details".into(), json!(details));
                }
            }
        }
    }

    Value::Object(map)
}

/// Convert a tool result message.
fn convert_tool_result(msg: &ToolResultMessage) -> Value {
    let text: String = msg
        .content
        .iter()
        .filter_map(|block| match block {
            MediaContentBlock::Text(tc) => Some(tc.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    let content = if text.is_empty() {
        "(no text content)".to_string()
    } else {
        text
    };

    json!({
        "role": "tool",
        "content": content,
        "tool_call_id": msg.tool_call_id,
    })
}

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

/// Convert ameli tool definitions to OpenAI tool format.
fn convert_tools(tools: &[Tool], compat: &OpenAICompletionsCompat) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            let mut func_map = Map::new();
            func_map.insert("name".into(), json!(tool.name));
            func_map.insert("description".into(), json!(tool.description));
            func_map.insert("parameters".into(), tool.parameters.clone());
            if compat.supports_strict_mode {
                func_map.insert("strict".into(), json!(false));
            }
            json!({
                "type": "function",
                "function": Value::Object(func_map),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve the `reasoning_effort` value from the thinking level map or
/// fall back to the level name.
fn resolve_reasoning_effort(reasoning: &ameli_ai::types::ThinkingLevel, model: &Model) -> String {
    let model_level: ModelThinkingLevel = (*reasoning).into();
    model
        .thinking_level_map
        .as_ref()
        .and_then(|m| m.get(&model_level))
        .and_then(|v| v.as_ref())
        .cloned()
        .unwrap_or_else(|| thinking_level_name(reasoning))
}

/// Convert a [`ThinkingLevel`] to its lowercase string name.
fn thinking_level_name(level: &ameli_ai::types::ThinkingLevel) -> String {
    match level {
        ameli_ai::types::ThinkingLevel::Minimal => "minimal",
        ameli_ai::types::ThinkingLevel::Low => "low",
        ameli_ai::types::ThinkingLevel::Medium => "medium",
        ameli_ai::types::ThinkingLevel::High => "high",
        ameli_ai::types::ThinkingLevel::XHigh => "xhigh",
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ameli_ai::types::{Cost, InputType, TextContent};

    fn test_model() -> Model {
        Model {
            id: "gpt-4o".into(),
            name: "GPT-4o".into(),
            api: "openai-completions".into(),
            provider: "openai".into(),
            base_url: "https://api.openai.com/v1".into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![InputType::Text],
            cost: Cost::default(),
            context_window: 128_000,
            max_tokens: 16_384,
            compat: None,
        }
    }

    fn test_model_with_reasoning() -> Model {
        Model {
            id: "o3".into(),
            name: "o3".into(),
            api: "openai-completions".into(),
            provider: "openai".into(),
            base_url: "https://api.openai.com/v1".into(),
            reasoning: true,
            thinking_level_map: None,
            input: vec![InputType::Text],
            cost: Cost::default(),
            context_window: 200_000,
            max_tokens: 100_000,
            compat: None,
        }
    }

    #[test]
    fn basic_params_structure() {
        let model = test_model();
        let context = Context::default();
        let options = StreamOptions::default();
        let compat = OpenAICompletionsCompat::default();

        let params = build_request_params(&model, &context, &options, &compat);

        assert_eq!(params["model"], "gpt-4o");
        assert_eq!(params["stream"], true);
        assert_eq!(params["store"], false);
    }

    #[test]
    fn system_prompt_uses_system_role() {
        let model = test_model();
        let context = Context {
            system_prompt: Some("You are helpful.".into()),
            ..Default::default()
        };
        let options = StreamOptions::default();
        let compat = OpenAICompletionsCompat::default();

        let params = build_request_params(&model, &context, &options, &compat);
        let messages = params["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
    }

    #[test]
    fn reasoning_model_uses_developer_role() {
        let model = test_model_with_reasoning();
        let context = Context {
            system_prompt: Some("You are helpful.".into()),
            ..Default::default()
        };
        let options = StreamOptions::default();
        let compat = OpenAICompletionsCompat::default();

        let params = build_request_params(&model, &context, &options, &compat);
        let messages = params["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "developer");
    }

    #[test]
    fn zai_compat_no_developer_role() {
        let model = test_model_with_reasoning();
        let context = Context {
            system_prompt: Some("You are helpful.".into()),
            ..Default::default()
        };
        let options = StreamOptions::default();
        let compat = OpenAICompletionsCompat {
            supports_developer_role: false,
            supports_store: false,
            ..Default::default()
        };

        let params = build_request_params(&model, &context, &options, &compat);
        let messages = params["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert!(params.get("store").is_none());
    }

    #[test]
    fn zai_thinking_format() {
        let model = test_model_with_reasoning();
        let context = Context::default();
        let options = StreamOptions {
            reasoning: Some(ameli_ai::types::ThinkingLevel::Medium),
            ..Default::default()
        };
        let compat = OpenAICompletionsCompat {
            thinking_format: ThinkingFormat::Zai,
            supports_reasoning_effort: false,
            ..Default::default()
        };

        let params = build_request_params(&model, &context, &options, &compat);
        assert_eq!(params["enable_thinking"], true);
        assert!(params.get("reasoning_effort").is_none());
    }

    #[test]
    fn openai_reasoning_effort() {
        let model = test_model_with_reasoning();
        let context = Context::default();
        let options = StreamOptions {
            reasoning: Some(ameli_ai::types::ThinkingLevel::High),
            ..Default::default()
        };
        let compat = OpenAICompletionsCompat::default();

        let params = build_request_params(&model, &context, &options, &compat);
        assert_eq!(params["reasoning_effort"], "high");
    }

    #[test]
    fn reasoning_effort_from_thinking_level_map() {
        let model = Model {
            thinking_level_map: Some(
                [(ModelThinkingLevel::High, Some("custom-high".to_string()))]
                    .into_iter()
                    .collect(),
            ),
            ..test_model_with_reasoning()
        };
        let context = Context::default();
        let options = StreamOptions {
            reasoning: Some(ameli_ai::types::ThinkingLevel::High),
            ..Default::default()
        };
        let compat = OpenAICompletionsCompat::default();

        let params = build_request_params(&model, &context, &options, &compat);
        assert_eq!(params["reasoning_effort"], "custom-high");
    }

    #[test]
    fn zai_tool_stream_flag() {
        let model = test_model();
        let context = Context {
            tools: Some(vec![Tool {
                name: "bash".into(),
                description: "Run a command".into(),
                parameters: json!({"type": "object", "properties": {}}),
            }]),
            ..Default::default()
        };
        let options = StreamOptions::default();
        let compat = OpenAICompletionsCompat {
            zai_tool_stream: true,
            ..Default::default()
        };

        let params = build_request_params(&model, &context, &options, &compat);
        assert_eq!(params["tool_stream"], true);
    }

    #[test]
    fn convert_user_text_message() {
        let model = test_model();
        let context = Context {
            messages: vec![Message::User(ameli_ai::types::UserMessage::text("hello"))],
            ..Default::default()
        };
        let params = convert_messages(&context, &model, true);
        assert_eq!(params[0]["role"], "user");
        assert_eq!(params[0]["content"], "hello");
    }

    #[test]
    fn convert_assistant_with_tool_calls() {
        let model = test_model();
        let assistant = ameli_ai::types::AssistantMessage {
            content: vec![
                AssistantContentBlock::Text(TextContent::new("Let me help.")),
                AssistantContentBlock::ToolCall(ToolCall {
                    id: "call_1".into(),
                    name: "bash".into(),
                    arguments: json!({"command": "ls"}),
                    thought_signature: None,
                }),
            ],
            api: "openai-completions".into(),
            provider: "openai".into(),
            model: "gpt-4o".into(),
            response_model: None,
            response_id: None,
            usage: ameli_ai::types::Usage::default(),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 1000,
        };

        let result = convert_assistant_message(&assistant, &model).unwrap();
        assert_eq!(result["role"], "assistant");
        assert_eq!(result["content"], "Let me help.");
        let tool_calls = result["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "call_1");
    }

    #[test]
    fn skip_error_assistant_messages() {
        let model = test_model();
        let context = Context {
            messages: vec![
                Message::User(ameli_ai::types::UserMessage::text("hello")),
                Message::Assistant(ameli_ai::types::AssistantMessage {
                    content: vec![],
                    api: "openai-completions".into(),
                    provider: "openai".into(),
                    model: "gpt-4o".into(),
                    response_model: None,
                    response_id: None,
                    usage: ameli_ai::types::Usage::default(),
                    stop_reason: StopReason::Error,
                    error_message: Some("failed".into()),
                    timestamp: 1000,
                }),
            ],
            ..Default::default()
        };
        let params = convert_messages(&context, &model, true);
        // Only the user message should remain
        assert_eq!(params.len(), 1);
        assert_eq!(params[0]["role"], "user");
    }

    #[test]
    fn convert_tool_result_message() {
        let msg = ToolResultMessage {
            tool_call_id: "call_1".into(),
            tool_name: "bash".into(),
            content: vec![MediaContentBlock::Text(TextContent::new("file.txt"))],
            details: None,
            is_error: false,
            timestamp: 1000,
        };

        let result = convert_tool_result(&msg);
        assert_eq!(result["role"], "tool");
        assert_eq!(result["tool_call_id"], "call_1");
        assert_eq!(result["content"], "file.txt");
    }

    #[test]
    fn max_tokens_field_names() {
        let model = test_model();
        let context = Context::default();
        let options = StreamOptions {
            max_tokens: Some(4096),
            ..Default::default()
        };

        let compat_max_completion = OpenAICompletionsCompat::default();
        let params = build_request_params(&model, &context, &options, &compat_max_completion);
        assert_eq!(params["max_completion_tokens"], 4096);
        assert!(params.get("max_tokens").is_none());

        let compat_max_tokens = OpenAICompletionsCompat {
            max_tokens_field: MaxTokensField::MaxTokens,
            ..Default::default()
        };
        let params = build_request_params(&model, &context, &options, &compat_max_tokens);
        assert_eq!(params["max_tokens"], 4096);
        assert!(params.get("max_completion_tokens").is_none());
    }
}
