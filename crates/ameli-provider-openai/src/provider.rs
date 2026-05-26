//! OpenAI Completions streaming provider.
//!
//! Implements [`StreamFn`] for the `"openai-completions"` API protocol.
//! Streams SSE chunks from the provider, emitting [`AssistantMessageEvent`]s
//! for text, thinking, and tool call content.

use crate::compat::get_compat;
use crate::json::parse_streaming_json;
use crate::messages::build_request_params;
use crate::types::{ChatCompletionChunk, ChunkUsage};

use ameli_ai::provider::StreamFn;
use ameli_ai::stream::{create_assistant_message_event_stream, AssistantMessageEventProducer};
use ameli_ai::types::{
    AssistantContentBlock, AssistantMessage, AssistantMessageEvent, Context, Cost, Model,
    StopReason, StreamOptions, TextContent, ThinkingContent, ToolCall, Usage,
};

use futures::StreamExt;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// OpenAICompletionsProvider
// ---------------------------------------------------------------------------

/// Streaming provider for OpenAI Chat Completions and compatible APIs.
///
/// Handles SSE streaming, incremental text/thinking/tool-call parsing,
/// usage tracking, and cost calculation. Compatible with OpenAI, ZAI,
/// and other providers via [`OpenAICompletionsCompat`](crate::OpenAICompletionsCompat).
pub struct OpenAICompletionsProvider {
    client: reqwest::Client,
}

impl OpenAICompletionsProvider {
    /// Create a new provider with a default HTTP client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for OpenAICompletionsProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamFn for OpenAICompletionsProvider {
    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: StreamOptions,
    ) -> ameli_ai::stream::AssistantMessageEventStream {
        let (producer, stream) = create_assistant_message_event_stream();
        let client = self.client.clone();
        let model = model.clone();

        tokio::spawn(async move {
            // run_stream takes ownership of producer and guarantees it
            // ends (with either a Done or Error terminal event).
            run_stream(client, &model, context, options, producer).await;
        });

        stream
    }
}

// ---------------------------------------------------------------------------
// Internal: streaming engine
// ---------------------------------------------------------------------------

/// Run the full streaming cycle: HTTP request → SSE → events.
async fn run_stream(
    client: reqwest::Client,
    model: &Model,
    context: Context,
    options: StreamOptions,
    producer: AssistantMessageEventProducer,
) {
    if let Err(e) = run_stream_inner(client, model, context, options, &producer).await {
        // Push error as terminal event. If the inner function already
        // pushed a terminal event, this push will fail (consumer dropped
        // after the stream was ended by a previous event) — that's fine.
        let error_msg = make_error_message(model, &e.to_string());
        producer.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: error_msg,
        });
        producer.end();
    }
}

/// Inner streaming logic. Returns `Ok(())` if a terminal event was pushed,
/// or `Err` if we failed before streaming started.
async fn run_stream_inner(
    client: reqwest::Client,
    model: &Model,
    context: Context,
    options: StreamOptions,
    producer: &AssistantMessageEventProducer,
) -> anyhow::Result<()> {
    let compat = get_compat(model);
    let api_key = resolve_api_key(&options, &model.provider)
        .ok_or_else(|| anyhow::anyhow!("No API key for provider: {}", model.provider))?;

    // Build request
    let params = build_request_params(model, &context, &options, &compat);
    let base_url = model.base_url.trim_end_matches('/');
    let url = format!("{}/chat/completions", base_url);

    let mut request = client.post(&url).bearer_auth(&api_key).json(&params);

    if let Some(timeout) = options.timeout_ms {
        request = request.timeout(std::time::Duration::from_millis(timeout));
    }

    if let Some(headers) = &options.headers {
        for (key, value) in headers {
            request = request.header(key.as_str(), value.as_str());
        }
    }

    // Send request
    let response = request
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to connect to {}: {}", url, e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("HTTP {} from {}: {}", status, url, body);
    }

    // Initialize output message
    let mut output = AssistantMessage {
        content: vec![],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: now_ms(),
    };

    // Emit start
    producer.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });

    // Streaming state
    let mut line_buffer = SseLineBuffer::new();
    let mut byte_stream = response.bytes_stream();
    let mut text_block_index: Option<usize> = None;
    let mut thinking_block_index: Option<usize> = None;
    let mut tool_call_blocks: HashMap<u32, usize> = HashMap::new();
    let mut partial_tool_args: HashMap<u32, String> = HashMap::new();
    let mut has_finish_reason = false;

    // Main SSE loop
    while let Some(chunk_result) = byte_stream.next().await {
        let bytes = chunk_result.map_err(|e| anyhow::anyhow!("Stream read error: {}", e))?;
        let lines = line_buffer.push_bytes(&bytes);

        for line in lines {
            let data = match extract_sse_data(&line) {
                Some(d) => d,
                None => continue,
            };

            if data == "[DONE]" {
                finish_all_blocks(
                    &mut output,
                    &text_block_index,
                    &thinking_block_index,
                    &tool_call_blocks,
                    &partial_tool_args,
                    producer,
                );

                producer.push(if has_finish_reason {
                    match output.stop_reason {
                        StopReason::Error | StopReason::Aborted => AssistantMessageEvent::Error {
                            reason: output.stop_reason,
                            error: output,
                        },
                        _ => AssistantMessageEvent::Done {
                            reason: output.stop_reason,
                            message: output,
                        },
                    }
                } else {
                    AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        error: make_error_message(model, "Stream ended without finish_reason"),
                    }
                });
                return Ok(());
            }

            let chunk: ChatCompletionChunk = match serde_json::from_str(data) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Update response metadata
            if output.response_id.is_none() {
                output.response_id = chunk.id;
            }
            if let Some(m) = &chunk.model {
                if !m.is_empty() && *m != model.id && output.response_model.is_none() {
                    output.response_model = Some(m.clone());
                }
            }

            // Update usage
            if let Some(usage) = &chunk.usage {
                output.usage = parse_chunk_usage(usage, &model.cost);
            }

            // Process choices
            let choice = match chunk.choices.and_then(|mut c| c.pop()) {
                Some(c) => c,
                None => continue,
            };

            if let Some(reason) = &choice.finish_reason {
                output.stop_reason = map_stop_reason(reason);
                if let StopReason::Error = output.stop_reason {
                    output.error_message = Some(format!("Provider finish_reason: {}", reason));
                }
                has_finish_reason = true;
            }

            let delta = match choice.delta {
                Some(d) => d,
                None => continue,
            };

            // --- Text content ---
            if let Some(content) = &delta.content {
                if !content.is_empty() {
                    if text_block_index.is_none() {
                        output
                            .content
                            .push(AssistantContentBlock::Text(TextContent::new("")));
                        text_block_index = Some(output.content.len() - 1);
                        let idx = text_block_index.unwrap_or(0);
                        producer.push(AssistantMessageEvent::TextStart {
                            content_index: idx,
                            partial: output.clone(),
                        });
                    }
                    let idx = text_block_index.unwrap_or(0);
                    if let Some(AssistantContentBlock::Text(tc)) = output.content.get_mut(idx) {
                        tc.text.push_str(content);
                    }
                    producer.push(AssistantMessageEvent::TextDelta {
                        content_index: idx,
                        delta: content.clone(),
                        partial: output.clone(),
                    });
                }
            }

            // --- Thinking / reasoning ---
            if let Some(reasoning) = crate::types::get_reasoning_content(&delta) {
                if !reasoning.is_empty() {
                    if thinking_block_index.is_none() {
                        let sig = crate::types::get_reasoning_field_name(&delta);
                        let block = ThinkingContent {
                            thinking: String::new(),
                            thinking_signature: Some(sig),
                            redacted: None,
                        };
                        output.content.push(AssistantContentBlock::Thinking(block));
                        thinking_block_index = Some(output.content.len() - 1);
                        let idx = thinking_block_index.unwrap_or(0);
                        producer.push(AssistantMessageEvent::ThinkingStart {
                            content_index: idx,
                            partial: output.clone(),
                        });
                    }
                    let idx = thinking_block_index.unwrap_or(0);
                    if let Some(AssistantContentBlock::Thinking(tc)) = output.content.get_mut(idx) {
                        tc.thinking.push_str(reasoning);
                    }
                    producer.push(AssistantMessageEvent::ThinkingDelta {
                        content_index: idx,
                        delta: reasoning.to_string(),
                        partial: output.clone(),
                    });
                }
            }

            // --- Tool calls ---
            if let Some(tool_calls) = &delta.tool_calls {
                for tc in tool_calls {
                    let stream_idx = tc.index.unwrap_or(0);

                    let content_idx = if let Some(&idx) = tool_call_blocks.get(&stream_idx) {
                        idx
                    } else {
                        let block = ToolCall {
                            id: String::new(),
                            name: String::new(),
                            arguments: serde_json::json!({}),
                            thought_signature: None,
                        };
                        output.content.push(AssistantContentBlock::ToolCall(block));
                        let idx = output.content.len() - 1;
                        tool_call_blocks.insert(stream_idx, idx);
                        producer.push(AssistantMessageEvent::ToolCallStart {
                            content_index: idx,
                            partial: output.clone(),
                        });
                        idx
                    };

                    if let Some(id) = &tc.id {
                        if let Some(AssistantContentBlock::ToolCall(block)) =
                            output.content.get_mut(content_idx)
                        {
                            block.id = id.clone();
                        }
                    }
                    if let Some(func) = &tc.function {
                        if let Some(name) = &func.name {
                            if let Some(AssistantContentBlock::ToolCall(block)) =
                                output.content.get_mut(content_idx)
                            {
                                block.name = name.clone();
                            }
                        }
                        if let Some(args_delta) = &func.arguments {
                            let partial = partial_tool_args.entry(stream_idx).or_default();
                            partial.push_str(args_delta);

                            let parsed = parse_streaming_json(partial);
                            if let Some(AssistantContentBlock::ToolCall(block)) =
                                output.content.get_mut(content_idx)
                            {
                                block.arguments = parsed;
                            }

                            producer.push(AssistantMessageEvent::ToolCallDelta {
                                content_index: content_idx,
                                delta: args_delta.clone(),
                                partial: output.clone(),
                            });
                        }
                    }
                }
            }
        }
    }

    // Stream ended without [DONE]
    finish_all_blocks(
        &mut output,
        &text_block_index,
        &thinking_block_index,
        &tool_call_blocks,
        &partial_tool_args,
        producer,
    );

    producer.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: make_error_message(model, "Stream ended unexpectedly without [DONE]"),
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Finish open blocks
// ---------------------------------------------------------------------------

/// Emit end events for all open content blocks (text, thinking, tool calls).
fn finish_all_blocks(
    output: &mut AssistantMessage,
    text_block_index: &Option<usize>,
    thinking_block_index: &Option<usize>,
    tool_call_blocks: &HashMap<u32, usize>,
    partial_tool_args: &HashMap<u32, String>,
    producer: &AssistantMessageEventProducer,
) {
    if let Some(idx) = text_block_index {
        if let Some(AssistantContentBlock::Text(tc)) = output.content.get(*idx) {
            producer.push(AssistantMessageEvent::TextEnd {
                content_index: *idx,
                content: tc.text.clone(),
                partial: output.clone(),
            });
        }
    }

    if let Some(idx) = thinking_block_index {
        if let Some(AssistantContentBlock::Thinking(tc)) = output.content.get(*idx) {
            producer.push(AssistantMessageEvent::ThinkingEnd {
                content_index: *idx,
                content: tc.thinking.clone(),
                partial: output.clone(),
            });
        }
    }

    // Finalize tool call arguments from partial strings
    for (&stream_idx, &content_idx) in tool_call_blocks {
        if let Some(partial) = partial_tool_args.get(&stream_idx) {
            let parsed = parse_streaming_json(partial);
            if let Some(AssistantContentBlock::ToolCall(block)) =
                output.content.get_mut(content_idx)
            {
                block.arguments = parsed;
            }
        }

        if let Some(AssistantContentBlock::ToolCall(block)) = output.content.get(content_idx) {
            producer.push(AssistantMessageEvent::ToolCallEnd {
                content_index: content_idx,
                tool_call: block.clone(),
                partial: output.clone(),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// SSE line buffer
// ---------------------------------------------------------------------------

/// Buffers raw bytes and yields complete lines for SSE parsing.
struct SseLineBuffer {
    buffer: String,
}

impl SseLineBuffer {
    fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Append bytes and return any complete lines.
    fn push_bytes(&mut self, bytes: &[u8]) -> Vec<String> {
        let text = String::from_utf8_lossy(bytes);
        self.buffer.push_str(&text);

        let mut lines = Vec::new();
        while let Some(pos) = self.buffer.find('\n') {
            let line = self.buffer[..pos].trim_end_matches('\r').to_string();
            self.buffer = self.buffer[pos + 1..].to_string();
            lines.push(line);
        }
        lines
    }
}

/// Extract the data payload from an SSE line.
///
/// Returns `None` for empty lines, comments (`: ...`), and non-data lines.
fn extract_sse_data(line: &str) -> Option<&str> {
    let line = line.trim();
    if line.is_empty() || line.starts_with(':') {
        return None;
    }
    line.strip_prefix("data: ")
        .or_else(|| line.strip_prefix("data:"))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve API key from options or environment variables.
fn resolve_api_key(options: &StreamOptions, provider: &str) -> Option<String> {
    if let Some(key) = &options.api_key {
        return Some(key.clone());
    }
    // Try PROVIDER_API_KEY first, then OPENAI_API_KEY as fallback
    let provider_key = format!("{}_API_KEY", provider.to_uppercase().replace('-', "_"));
    std::env::var(&provider_key)
        .ok()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
}

/// Map OpenAI `finish_reason` to our [`StopReason`].
fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "stop" | "end" => StopReason::Stop,
        "length" => StopReason::Length,
        "function_call" | "tool_calls" => StopReason::ToolUse,
        "content_filter" | "network_error" => StopReason::Error,
        _ => StopReason::Error,
    }
}

/// Parse chunk usage into our [`Usage`] type with cost calculation.
fn parse_chunk_usage(raw: &ChunkUsage, cost: &Cost) -> Usage {
    let prompt_tokens = raw.prompt_tokens.unwrap_or(0);
    let cache_read = raw
        .prompt_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens)
        .unwrap_or(0);
    let cache_write = raw
        .prompt_tokens_details
        .as_ref()
        .and_then(|d| d.cache_write_tokens)
        .unwrap_or(0);
    let input = prompt_tokens
        .saturating_sub(cache_read)
        .saturating_sub(cache_write);
    let output_tokens = raw.completion_tokens.unwrap_or(0);

    let mut usage = Usage {
        input,
        output: output_tokens,
        cache_read,
        cache_write,
        total_tokens: input + output_tokens + cache_read + cache_write,
        cost: Default::default(),
    };
    usage.calculate_cost(cost);
    usage
}

/// Construct a minimal error `AssistantMessage` for provider-level failures.
fn make_error_message(model: &Model, error_message: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        usage: Usage::default(),
        stop_reason: StopReason::Error,
        error_message: Some(error_message.to_string()),
        timestamp: now_ms(),
    }
}

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
    use crate::types::ChunkPromptTokensDetails;

    #[test]
    fn map_stop_reason_known_values() {
        assert_eq!(map_stop_reason("stop"), StopReason::Stop);
        assert_eq!(map_stop_reason("end"), StopReason::Stop);
        assert_eq!(map_stop_reason("length"), StopReason::Length);
        assert_eq!(map_stop_reason("tool_calls"), StopReason::ToolUse);
        assert_eq!(map_stop_reason("function_call"), StopReason::ToolUse);
        assert_eq!(map_stop_reason("content_filter"), StopReason::Error);
    }

    #[test]
    fn map_stop_reason_unknown_defaults_to_error() {
        assert_eq!(map_stop_reason("something_odd"), StopReason::Error);
    }

    #[test]
    fn parse_usage_with_cache() {
        let raw = ChunkUsage {
            prompt_tokens: Some(100),
            completion_tokens: Some(50),
            prompt_tokens_details: Some(ChunkPromptTokensDetails {
                cached_tokens: Some(30),
                cache_write_tokens: Some(10),
            }),
        };
        let cost = Cost {
            input: 5.0,
            output: 15.0,
            cache_read: 1.0,
            cache_write: 2.0,
        };
        let usage = parse_chunk_usage(&raw, &cost);
        // input = 100 - 30 - 10 = 60
        assert_eq!(usage.input, 60);
        assert_eq!(usage.output, 50);
        assert_eq!(usage.cache_read, 30);
        assert_eq!(usage.cache_write, 10);
        assert_eq!(usage.total_tokens, 150);
        // cost: input = 60 * 5 / 1M = 0.0003
        assert!((usage.cost.input - 0.0003).abs() < 1e-10);
        assert!((usage.cost.output - 0.00075).abs() < 1e-10);
    }

    #[test]
    fn parse_usage_minimal() {
        let raw = ChunkUsage {
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            prompt_tokens_details: None,
        };
        let usage = parse_chunk_usage(&raw, &Cost::default());
        assert_eq!(usage.input, 10);
        assert_eq!(usage.output, 5);
        assert_eq!(usage.cache_read, 0);
        assert_eq!(usage.cache_write, 0);
    }

    #[test]
    fn sse_line_buffer_complete_lines() {
        let mut buf = SseLineBuffer::new();
        let lines = buf.push_bytes(b"data: hello\n\ndata: world\n\n");
        assert_eq!(lines, vec!["data: hello", "", "data: world", ""]);
    }

    #[test]
    fn sse_line_buffer_partial() {
        let mut buf = SseLineBuffer::new();
        let lines1 = buf.push_bytes(b"data: hel");
        assert!(lines1.is_empty());
        let lines2 = buf.push_bytes(b"lo\n");
        assert_eq!(lines2, vec!["data: hello"]);
    }

    #[test]
    fn extract_sse_data_valid() {
        assert_eq!(extract_sse_data("data: {\"key\":1}"), Some("{\"key\":1}"));
        assert_eq!(extract_sse_data("data:[DONE]"), Some("[DONE]"));
    }

    #[test]
    fn extract_sse_data_ignores_non_data() {
        assert_eq!(extract_sse_data(""), None);
        assert_eq!(extract_sse_data(": comment"), None);
        assert_eq!(extract_sse_data("event: ping"), None);
    }

    #[test]
    fn resolve_api_key_from_options() {
        let options = StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        };
        assert_eq!(resolve_api_key(&options, "openai"), Some("test-key".into()));
    }
}
