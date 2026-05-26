//! Agent loop: the core orchestration logic that drives LLM ↔ tool cycles.
//!
//! Public entry points:
//!
//! - [`agent_loop`] / [`agent_loop_continue`] — streaming, return `EventStream<AgentEvent>`
//! - [`run_agent_loop`] / [`run_agent_loop_continue`] — async, return `Vec<AgentMessage>`
//!
//! The loop streams an LLM response, extracts tool calls, validates and executes
//! them, feeds results back, and repeats until the model stops, the caller
//! requests stop, or cancellation fires.

use crate::types::*;
use ameli_ai::api::{stream_simple, ApiRegistry};
use ameli_ai::stream::{create_event_stream, EventStream};
use ameli_ai::types::{
    AssistantContentBlock, AssistantMessage, AssistantMessageEvent, Context as LlmContext, Model,
    StopReason, ThinkingLevel as StreamThinkingLevel, Tool as LlmTool, ToolCall, ToolResultMessage,
};
use ameli_ai::validation::validate_tool_arguments;
use futures::future::join_all;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Event sink
// ---------------------------------------------------------------------------

/// Callback invoked by the agent loop to emit lifecycle events.
///
/// The TS version is async; in Rust the push to an unbounded channel is
/// synchronous so the sink is a plain `Fn`.
pub type AgentEventSink = Arc<dyn Fn(AgentEvent) + Send + Sync>;

// ---------------------------------------------------------------------------
// Public API: streaming entry points
// ---------------------------------------------------------------------------

/// Start an agent loop with new prompt messages.
///
/// Returns an [`EventStream`] that emits [`AgentEvent`]s. The stream ends
/// after `AgentEnd` (the producer is dropped).
pub fn agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    cancel: Option<CancellationToken>,
    registry: Arc<ApiRegistry>,
) -> EventStream<AgentEvent> {
    let (producer, stream) = create_event_stream::<AgentEvent>();

    tokio::spawn(async move {
        let _ = run_agent_loop(
            prompts,
            context,
            config,
            Arc::new(move |event: AgentEvent| {
                let _ = producer.push(event);
            }),
            cancel,
            registry,
        )
        .await;
    });

    stream
}

/// Continue an agent loop from the current context without adding new prompts.
///
/// Validates that the last message is not an assistant message. If validation
/// fails, returns an empty stream (consumer sees `None` immediately).
pub fn agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel: Option<CancellationToken>,
    registry: Arc<ApiRegistry>,
) -> EventStream<AgentEvent> {
    let (producer, stream) = create_event_stream::<AgentEvent>();

    tokio::spawn(async move {
        let _ = run_agent_loop_continue(
            context,
            config,
            Arc::new(move |event: AgentEvent| {
                let _ = producer.push(event);
            }),
            cancel,
            registry,
        )
        .await;
    });

    stream
}

// ---------------------------------------------------------------------------
// Public API: async entry points
// ---------------------------------------------------------------------------

/// Run an agent loop to completion, returning the new messages added.
///
/// Emits `AgentStart`, `TurnStart`, and message events for the prompts,
/// then delegates to [`run_loop`].
pub async fn run_agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancel: Option<CancellationToken>,
    registry: Arc<ApiRegistry>,
) -> Vec<AgentMessage> {
    let mut new_messages: Vec<AgentMessage> = prompts.clone();
    let mut current_context = context;
    current_context.messages.extend(prompts.iter().cloned());

    emit(AgentEvent::AgentStart);
    emit(AgentEvent::TurnStart);
    for prompt in &new_messages {
        emit(AgentEvent::MessageStart {
            message: prompt.clone(),
        });
        emit(AgentEvent::MessageEnd {
            message: prompt.clone(),
        });
    }

    run_loop(
        current_context,
        &mut new_messages,
        &config,
        &cancel,
        &emit,
        &registry,
    )
    .await;

    new_messages
}

/// Continue an agent loop from the current context.
///
/// The last message must be a user or tool-result message (i.e., not assistant).
/// Returns an error if validation fails.
pub async fn run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancel: Option<CancellationToken>,
    registry: Arc<ApiRegistry>,
) -> anyhow::Result<Vec<AgentMessage>> {
    let last = context
        .messages
        .last()
        .ok_or_else(|| anyhow::anyhow!("Cannot continue: no messages in context"))?;
    let role = last.role();
    if role == "assistant" {
        anyhow::bail!("Cannot continue from message role: assistant");
    }

    let mut new_messages: Vec<AgentMessage> = Vec::new();

    emit(AgentEvent::AgentStart);
    emit(AgentEvent::TurnStart);

    run_loop(
        context,
        &mut new_messages,
        &config,
        &cancel,
        &emit,
        &registry,
    )
    .await;

    Ok(new_messages)
}

// ---------------------------------------------------------------------------
// Private types
// ---------------------------------------------------------------------------

/// Result of preparing a tool call: either ready to execute or an immediate
/// error/blocked result.
enum ToolCallPreparation {
    Prepared {
        tool: Arc<dyn AgentTool>,
        args: Value,
    },
    Immediate {
        result: AgentToolResult<Value>,
        is_error: bool,
    },
}

/// Outcome of executing a tool.
struct ExecutedToolCallOutcome {
    result: AgentToolResult<Value>,
    is_error: bool,
}

/// Fully resolved tool call outcome (after optional `after_tool_call` hook).
#[derive(Clone)]
struct FinalizedToolCallOutcome {
    tool_call: ToolCall,
    result: AgentToolResult<Value>,
    is_error: bool,
}

/// Batch of tool results from executing all tool calls in one assistant message.
struct ExecutedToolCallBatch {
    messages: Vec<ToolResultMessage>,
    terminate: bool,
}

// ---------------------------------------------------------------------------
// Core loop
// ---------------------------------------------------------------------------

/// Main loop logic shared by the four public entry points.
///
/// Runs LLM → tool → LLM cycles until the model stops without tool calls,
/// `should_stop_after_turn` returns `true`, cancellation fires, or follow-up
/// messages are exhausted.
async fn run_loop(
    mut current_context: AgentContext,
    new_messages: &mut Vec<AgentMessage>,
    config: &AgentLoopConfig,
    cancel: &Option<CancellationToken>,
    emit: &AgentEventSink,
    registry: &ApiRegistry,
) {
    let mut current_model: Model = config.model.clone();
    let mut current_reasoning: Option<StreamThinkingLevel> = config.stream_options.reasoning;
    let mut first_turn = true;
    let mut pending_messages = poll_steering(config).await;

    // Outer loop: continues when follow-up messages arrive
    loop {
        let mut has_more_tool_calls = true;

        // Inner loop: process tool calls and steering messages
        while has_more_tool_calls || !pending_messages.is_empty() {
            if !first_turn {
                emit(AgentEvent::TurnStart);
            } else {
                first_turn = false;
            }

            // Inject pending steering messages
            if !pending_messages.is_empty() {
                for message in pending_messages.drain(..) {
                    emit(AgentEvent::MessageStart {
                        message: message.clone(),
                    });
                    emit(AgentEvent::MessageEnd {
                        message: message.clone(),
                    });
                    current_context.messages.push(message.clone());
                    new_messages.push(message);
                }
            }

            // Stream assistant response
            let message = stream_assistant_response(
                &mut current_context,
                &current_model,
                current_reasoning,
                config,
                cancel,
                emit,
                registry,
            )
            .await;
            new_messages.push(AgentMessage::Assistant(message.clone()));

            if message.stop_reason == StopReason::Error
                || message.stop_reason == StopReason::Aborted
            {
                emit(AgentEvent::TurnEnd {
                    message: AgentMessage::Assistant(message),
                    tool_results: vec![],
                });
                emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                });
                return;
            }

            // Check for tool calls
            let tool_calls: Vec<ToolCall> = message
                .content
                .iter()
                .filter_map(|block| match block {
                    AssistantContentBlock::ToolCall(tc) => Some(tc.clone()),
                    _ => None,
                })
                .collect();

            let mut tool_results: Vec<ToolResultMessage> = Vec::new();
            has_more_tool_calls = false;

            if !tool_calls.is_empty() {
                let batch = execute_tool_calls(
                    &current_context,
                    &message,
                    &tool_calls,
                    config,
                    cancel,
                    emit,
                )
                .await;
                tool_results = batch.messages;
                has_more_tool_calls = !batch.terminate;

                for result in &tool_results {
                    let agent_msg = AgentMessage::ToolResult(result.clone());
                    current_context.messages.push(agent_msg.clone());
                    new_messages.push(agent_msg);
                }
            }

            emit(AgentEvent::TurnEnd {
                message: AgentMessage::Assistant(message.clone()),
                tool_results: tool_results.clone(),
            });

            // prepareNextTurn
            if let Some(prepare) = &config.prepare_next_turn {
                let ctx = ShouldStopAfterTurnContext {
                    message: message.clone(),
                    tool_results: tool_results.clone(),
                    context: current_context.clone(),
                    new_messages: new_messages.clone(),
                };
                if let Some(update) = prepare(&ctx).await {
                    if let Some(ctx) = update.context {
                        current_context = ctx;
                    }
                    if let Some(model) = update.model {
                        current_model = model;
                    }
                    if let Some(tl) = update.thinking_level {
                        current_reasoning = match tl {
                            ThinkingLevel::Off => None,
                            ThinkingLevel::Minimal => Some(StreamThinkingLevel::Minimal),
                            ThinkingLevel::Low => Some(StreamThinkingLevel::Low),
                            ThinkingLevel::Medium => Some(StreamThinkingLevel::Medium),
                            ThinkingLevel::High => Some(StreamThinkingLevel::High),
                            ThinkingLevel::XHigh => Some(StreamThinkingLevel::XHigh),
                        };
                    }
                }
            }

            // shouldStopAfterTurn
            if let Some(should_stop) = &config.should_stop_after_turn {
                let ctx = ShouldStopAfterTurnContext {
                    message: message.clone(),
                    tool_results: tool_results.clone(),
                    context: current_context.clone(),
                    new_messages: new_messages.clone(),
                };
                if should_stop(&ctx).await {
                    emit(AgentEvent::AgentEnd {
                        messages: new_messages.clone(),
                    });
                    return;
                }
            }

            if is_cancelled(cancel) {
                return;
            }

            pending_messages = poll_steering(config).await;
        }

        // Agent would stop here. Check for follow-up messages.
        let follow_ups = poll_follow_ups(config).await;
        if !follow_ups.is_empty() {
            pending_messages = follow_ups;
            continue;
        }

        break;
    }

    emit(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    });
}

// ---------------------------------------------------------------------------
// Stream assistant response
// ---------------------------------------------------------------------------

/// Stream an LLM response, converting `AgentMessage[]` → `Message[]` at the
/// boundary. Emits `MessageStart`/`MessageUpdate`/`MessageEnd` events and
/// mutates `context.messages` in-place.
async fn stream_assistant_response(
    context: &mut AgentContext,
    model: &Model,
    reasoning: Option<StreamThinkingLevel>,
    config: &AgentLoopConfig,
    cancel: &Option<CancellationToken>,
    emit: &AgentEventSink,
    registry: &ApiRegistry,
) -> AssistantMessage {
    // 1. Transform context (AgentMessage[] → AgentMessage[])
    let transformed = if let Some(transform) = &config.transform_context {
        transform(&context.messages, cancel.clone()).await
    } else {
        context.messages.clone()
    };

    // 2. Convert to LLM-compatible messages (AgentMessage[] → Message[])
    let llm_messages = (config.convert_to_llm)(&transformed).await;

    // 3. Build tool definitions
    let tools: Vec<LlmTool> = context.tools.iter().map(|t| t.tool_definition()).collect();

    // 4. Build LLM context
    let llm_context = LlmContext {
        system_prompt: if context.system_prompt.is_empty() {
            None
        } else {
            Some(context.system_prompt.clone())
        },
        messages: llm_messages,
        tools: if tools.is_empty() { None } else { Some(tools) },
    };

    // 5. Resolve API key and build stream options
    let mut options = config.stream_options.clone();
    options.reasoning = reasoning;
    if let Some(get_key) = &config.get_api_key {
        if let Some(key) = get_key(&model.provider).await {
            options.api_key = Some(key);
        }
    }

    // 6. Stream from provider
    let mut response = stream_simple(registry, model, llm_context, options);

    // 7. Iterate events
    let mut partial_message: Option<AssistantMessage> = None;
    let mut added_partial = false;

    while let Some(event) = response.recv().await {
        if event.is_terminal() {
            let final_msg = match event {
                AssistantMessageEvent::Done { message, .. } => message,
                AssistantMessageEvent::Error { error, .. } => error,
                _ => unreachable!(), // guarded by is_terminal()
            };
            return finalize_assistant_message(context, final_msg, added_partial, emit);
        }

        // Extract partial snapshot from non-terminal events
        let partial = extract_partial(&event);

        match event {
            AssistantMessageEvent::Start { .. } => {
                partial_message = Some(partial.clone());
                context
                    .messages
                    .push(AgentMessage::Assistant(partial.clone()));
                added_partial = true;
                emit(AgentEvent::MessageStart {
                    message: AgentMessage::Assistant(partial),
                });
            }
            _ => {
                if partial_message.is_some() {
                    partial_message = Some(partial.clone());
                    if let Some(last) = context.messages.last_mut() {
                        *last = AgentMessage::Assistant(partial.clone());
                    }
                    emit(AgentEvent::MessageUpdate {
                        message: AgentMessage::Assistant(partial),
                        assistant_message_event: Box::new(event),
                    });
                }
            }
        }
    }

    // Stream ended without a terminal event — use result() as fallback
    drop(response);
    // If we get here, no terminal event was received. Build a synthetic error.
    let error_msg = AssistantMessage {
        content: vec![],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        usage: Default::default(),
        stop_reason: StopReason::Error,
        error_message: Some("stream ended without a terminal event".into()),
        timestamp: now_ms(),
    };
    finalize_assistant_message(context, error_msg, added_partial, emit)
}

/// Finalize an assistant message: update context, emit end events.
fn finalize_assistant_message(
    context: &mut AgentContext,
    message: AssistantMessage,
    added_partial: bool,
    emit: &AgentEventSink,
) -> AssistantMessage {
    if added_partial {
        if let Some(last) = context.messages.last_mut() {
            *last = AgentMessage::Assistant(message.clone());
        }
    } else {
        context
            .messages
            .push(AgentMessage::Assistant(message.clone()));
        emit(AgentEvent::MessageStart {
            message: AgentMessage::Assistant(message.clone()),
        });
    }
    emit(AgentEvent::MessageEnd {
        message: AgentMessage::Assistant(message.clone()),
    });
    message
}

/// Extract the partial `AssistantMessage` snapshot from a non-terminal event.
fn extract_partial(event: &AssistantMessageEvent) -> AssistantMessage {
    match event {
        AssistantMessageEvent::Start { partial } => partial.clone(),
        AssistantMessageEvent::TextStart { partial, .. } => partial.clone(),
        AssistantMessageEvent::TextDelta { partial, .. } => partial.clone(),
        AssistantMessageEvent::TextEnd { partial, .. } => partial.clone(),
        AssistantMessageEvent::ThinkingStart { partial, .. } => partial.clone(),
        AssistantMessageEvent::ThinkingDelta { partial, .. } => partial.clone(),
        AssistantMessageEvent::ThinkingEnd { partial, .. } => partial.clone(),
        AssistantMessageEvent::ToolCallStart { partial, .. } => partial.clone(),
        AssistantMessageEvent::ToolCallDelta { partial, .. } => partial.clone(),
        AssistantMessageEvent::ToolCallEnd { partial, .. } => partial.clone(),
        // Terminal events should not reach here
        AssistantMessageEvent::Done { message, .. } => message.clone(),
        AssistantMessageEvent::Error { error, .. } => error.clone(),
    }
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

/// Execute tool calls from an assistant message, dispatching to sequential or
/// parallel based on config and per-tool execution mode.
async fn execute_tool_calls(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    cancel: &Option<CancellationToken>,
    emit: &AgentEventSink,
) -> ExecutedToolCallBatch {
    let has_sequential = tool_calls.iter().any(|tc| {
        context.tools.iter().any(|t| {
            t.name() == tc.name && t.execution_mode() == Some(ToolExecutionMode::Sequential)
        })
    });

    if config.tool_execution == ToolExecutionMode::Sequential || has_sequential {
        execute_tool_calls_sequential(context, assistant_message, tool_calls, config, cancel, emit)
            .await
    } else {
        execute_tool_calls_parallel(context, assistant_message, tool_calls, config, cancel, emit)
            .await
    }
}

/// Execute tool calls one by one.
async fn execute_tool_calls_sequential(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    cancel: &Option<CancellationToken>,
    emit: &AgentEventSink,
) -> ExecutedToolCallBatch {
    let mut finalized_calls: Vec<FinalizedToolCallOutcome> = Vec::new();
    let mut messages: Vec<ToolResultMessage> = Vec::new();

    for tool_call in tool_calls {
        emit(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: tool_call.arguments.clone(),
        });

        let preparation =
            prepare_tool_call(context, assistant_message, tool_call, config, cancel).await;

        let finalized = match preparation {
            ToolCallPreparation::Immediate { result, is_error } => FinalizedToolCallOutcome {
                tool_call: tool_call.clone(),
                result,
                is_error,
            },
            ToolCallPreparation::Prepared { tool, args } => {
                let executed =
                    execute_prepared_tool_call(&tool, tool_call, args.clone(), cancel).await;
                finalize_executed_tool_call(
                    context,
                    assistant_message,
                    tool_call,
                    &args,
                    &executed,
                    config.after_tool_call.as_ref(),
                    cancel,
                )
                .await
            }
        };

        emit_tool_execution_end(&finalized, emit);
        let tool_result_msg = create_tool_result_message(&finalized);
        emit_tool_result_message(&tool_result_msg, emit);

        finalized_calls.push(finalized);
        messages.push(tool_result_msg);

        if is_cancelled(cancel) {
            break;
        }
    }

    ExecutedToolCallBatch {
        messages,
        terminate: should_terminate_tool_batch(&finalized_calls),
    }
}

/// Execute tool calls concurrently (preflight sequentially, execute in parallel).
async fn execute_tool_calls_parallel(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    cancel: &Option<CancellationToken>,
    emit: &AgentEventSink,
) -> ExecutedToolCallBatch {
    // Phase 1: Prepare all tool calls sequentially
    let mut pending_futures: Vec<Pin<Box<dyn Future<Output = FinalizedToolCallOutcome> + Send>>> =
        Vec::new();
    // Track which indices have immediate results vs pending futures.
    // We'll resolve everything in order.
    let mut immediate_results: Vec<(usize, FinalizedToolCallOutcome)> = Vec::new();
    let mut future_indices: Vec<usize> = Vec::new();

    for (idx, tool_call) in tool_calls.iter().enumerate() {
        emit(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: tool_call.arguments.clone(),
        });

        let preparation =
            prepare_tool_call(context, assistant_message, tool_call, config, cancel).await;

        match preparation {
            ToolCallPreparation::Immediate { result, is_error } => {
                let finalized = FinalizedToolCallOutcome {
                    tool_call: tool_call.clone(),
                    result,
                    is_error,
                };
                emit_tool_execution_end(&finalized, emit);
                immediate_results.push((idx, finalized));
            }
            ToolCallPreparation::Prepared { tool, args } => {
                let tool_call_clone = tool_call.clone();
                let args_clone = args.clone();
                let emit_clone = emit.clone();
                let context_clone = context.clone();
                let assistant_clone = assistant_message.clone();
                let after_hook = config.after_tool_call.clone();
                let cancel_clone = cancel.clone();

                let fut = Box::pin(async move {
                    let executed = execute_prepared_tool_call(
                        &tool,
                        &tool_call_clone,
                        args_clone,
                        &cancel_clone,
                    )
                    .await;
                    let finalized = finalize_executed_tool_call(
                        &context_clone,
                        &assistant_clone,
                        &tool_call_clone,
                        &tool_call_clone.arguments, // use original args for context
                        &executed,
                        after_hook.as_ref(),
                        &cancel_clone,
                    )
                    .await;
                    emit_tool_execution_end(&finalized, &emit_clone);
                    finalized
                })
                    as Pin<Box<dyn Future<Output = FinalizedToolCallOutcome> + Send>>;

                future_indices.push(idx);
                pending_futures.push(fut);
            }
        }

        if is_cancelled(cancel) {
            break;
        }
    }

    // Phase 2: Execute all pending futures concurrently
    let future_results = join_all(pending_futures).await;

    // Phase 3: Merge results in original order
    let mut ordered: Vec<Option<FinalizedToolCallOutcome>> =
        (0..tool_calls.len()).map(|_| None).collect();
    for (idx, result) in immediate_results {
        if let Some(slot) = ordered.get_mut(idx) {
            *slot = Some(result);
        }
    }
    for (future_idx, result) in future_results.into_iter().enumerate() {
        if let Some(&original_idx) = future_indices.get(future_idx) {
            if let Some(slot) = ordered.get_mut(original_idx) {
                *slot = Some(result);
            }
        }
    }

    let finalized_calls: Vec<FinalizedToolCallOutcome> = ordered.into_iter().flatten().collect();

    // Phase 4: Emit tool result messages in order
    let mut messages: Vec<ToolResultMessage> = Vec::new();
    for finalized in &finalized_calls {
        let msg = create_tool_result_message(finalized);
        emit_tool_result_message(&msg, emit);
        messages.push(msg);
    }

    ExecutedToolCallBatch {
        messages,
        terminate: should_terminate_tool_batch(&finalized_calls),
    }
}

// ---------------------------------------------------------------------------
// Tool call lifecycle
// ---------------------------------------------------------------------------

/// Prepare a tool call: find the tool, validate arguments, run `before_tool_call` hook.
async fn prepare_tool_call(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_call: &ToolCall,
    config: &AgentLoopConfig,
    cancel: &Option<CancellationToken>,
) -> ToolCallPreparation {
    // Find tool
    let tool = match context.tools.iter().find(|t| t.name() == tool_call.name) {
        Some(t) => t.clone(),
        None => {
            return ToolCallPreparation::Immediate {
                result: AgentToolResult::<Value>::error(format!(
                    "Tool {} not found",
                    tool_call.name
                )),
                is_error: true,
            };
        }
    };

    // Prepare and validate arguments
    let prepared_args = tool.prepare_arguments(tool_call.arguments.clone());
    let tool_def = tool.tool_definition();
    let prepared_call = ToolCall {
        id: tool_call.id.clone(),
        name: tool_call.name.clone(),
        arguments: prepared_args,
        thought_signature: tool_call.thought_signature.clone(),
    };

    let validated_args = match validate_tool_arguments(&tool_def, &prepared_call) {
        Ok(args) => args,
        Err(e) => {
            return ToolCallPreparation::Immediate {
                result: AgentToolResult::<Value>::error(e.to_string()),
                is_error: true,
            };
        }
    };

    // beforeToolCall hook
    if let Some(before) = &config.before_tool_call {
        let ctx = BeforeToolCallContext {
            assistant_message: assistant_message.clone(),
            tool_call: tool_call.clone(),
            args: validated_args.clone(),
            context: context.clone(),
        };
        if let Some(result) = before(&ctx, cancel.clone()).await {
            if result.block {
                return ToolCallPreparation::Immediate {
                    result: AgentToolResult::<Value>::error(
                        result
                            .reason
                            .unwrap_or_else(|| "Tool execution was blocked".into()),
                    ),
                    is_error: true,
                };
            }
        }
    }

    if is_cancelled(cancel) {
        return ToolCallPreparation::Immediate {
            result: AgentToolResult::<Value>::error("Operation aborted"),
            is_error: true,
        };
    }

    ToolCallPreparation::Prepared {
        tool,
        args: validated_args,
    }
}

/// Execute a prepared tool call. Returns the raw execution outcome.
async fn execute_prepared_tool_call(
    tool: &Arc<dyn AgentTool>,
    tool_call: &ToolCall,
    args: Value,
    cancel: &Option<CancellationToken>,
) -> ExecutedToolCallOutcome {
    let result = tool.execute(&tool_call.id, args, cancel.clone()).await;
    ExecutedToolCallOutcome {
        result,
        is_error: false,
    }
}

/// Callback type for `after_tool_call` hooks.
type AfterToolCallHook = dyn Fn(
        &AfterToolCallContext,
        Option<CancellationToken>,
    ) -> Pin<Box<dyn Future<Output = Option<AfterToolCallResult>> + Send>>
    + Send
    + Sync;

/// Finalize an executed tool call by applying the optional `after_tool_call` hook.
async fn finalize_executed_tool_call(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_call: &ToolCall,
    args: &Value,
    executed: &ExecutedToolCallOutcome,
    after_hook: Option<&Arc<AfterToolCallHook>>,
    cancel: &Option<CancellationToken>,
) -> FinalizedToolCallOutcome {
    let mut result = executed.result.clone();
    let mut is_error = executed.is_error;

    if let Some(after) = after_hook {
        let ctx = AfterToolCallContext {
            assistant_message: assistant_message.clone(),
            tool_call: tool_call.clone(),
            args: args.clone(),
            result: result.clone(),
            is_error,
            context: context.clone(),
        };
        if let Some(after_result) = after(&ctx, cancel.clone()).await {
            if let Some(content) = after_result.content {
                result.content = content;
            }
            if let Some(details) = after_result.details {
                result.details = details;
            }
            if let Some(terminate) = after_result.terminate {
                result.terminate = terminate;
            }
            if let Some(err) = after_result.is_error {
                is_error = err;
            }
        }
    }

    FinalizedToolCallOutcome {
        tool_call: tool_call.clone(),
        result,
        is_error,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn create_tool_result_message(finalized: &FinalizedToolCallOutcome) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        content: finalized.result.content.clone(),
        details: if finalized.result.details.is_null() {
            None
        } else {
            Some(finalized.result.details.clone())
        },
        is_error: finalized.is_error,
        timestamp: now_ms(),
    }
}

fn should_terminate_tool_batch(finalized_calls: &[FinalizedToolCallOutcome]) -> bool {
    !finalized_calls.is_empty() && finalized_calls.iter().all(|f| f.result.terminate)
}

fn emit_tool_execution_end(finalized: &FinalizedToolCallOutcome, emit: &AgentEventSink) {
    emit(AgentEvent::ToolExecutionEnd {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        result: finalized.result.clone(),
        is_error: finalized.is_error,
    });
}

fn emit_tool_result_message(msg: &ToolResultMessage, emit: &AgentEventSink) {
    emit(AgentEvent::MessageStart {
        message: AgentMessage::ToolResult(msg.clone()),
    });
    emit(AgentEvent::MessageEnd {
        message: AgentMessage::ToolResult(msg.clone()),
    });
}

async fn poll_steering(config: &AgentLoopConfig) -> Vec<AgentMessage> {
    if let Some(get) = &config.get_steering_messages {
        get().await
    } else {
        Vec::new()
    }
}

async fn poll_follow_ups(config: &AgentLoopConfig) -> Vec<AgentMessage> {
    if let Some(get) = &config.get_follow_up_messages {
        get().await
    } else {
        Vec::new()
    }
}

fn is_cancelled(cancel: &Option<CancellationToken>) -> bool {
    cancel.as_ref().is_some_and(|ct| ct.is_cancelled())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
