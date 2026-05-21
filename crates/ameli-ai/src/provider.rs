//! LLM provider trait, registry, and top-level streaming entry points.
//!
//! This module defines the [`StreamFn`] trait — the single abstraction boundary
//! between agent orchestration code and concrete LLM providers. Providers are
//! registered by API protocol name (e.g., `"openai-responses"`, `"anthropic-messages"`)
//! and looked up at request time via the model's [`Model::api`] field.
//!
//! # Entry points
//!
//! - [`stream_simple`] — stream an LLM response, looking up the provider by model API
//! - [`complete_simple`] — await a full LLM response (convenience over `stream_simple`)

use crate::stream::AssistantMessageEventStream;
use crate::types::{AssistantMessage, AssistantMessageEvent, Context, Model, StreamOptions};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

// ---------------------------------------------------------------------------
// StreamFn trait
// ---------------------------------------------------------------------------

/// Trait for LLM streaming providers.
///
/// # Contract
///
/// - [`stream`](StreamFn::stream) must return an [`AssistantMessageEventStream`]
///   immediately (synchronously). Network I/O happens asynchronously inside the
///   returned stream.
/// - Request, model, or runtime failures must be encoded in the returned stream
///   as an [`Error`](AssistantMessageEvent::Error) terminal event with
///   [`StopReason::Error`](crate::types::StopReason::Error) — they must **not**
///   panic.
/// - The stream must always terminate with either
///   [`Done`](AssistantMessageEvent::Done) or
///   [`Error`](AssistantMessageEvent::Error).
pub trait StreamFn: Send + Sync {
    /// Stream an LLM response for the given model, context, and options.
    ///
    /// The `model` reference is borrowed — the caller retains ownership (e.g.,
    /// for reuse across agent loop turns). Implementations should clone
    /// anything they need to move into async tasks.
    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: StreamOptions,
    ) -> AssistantMessageEventStream;
}

// ---------------------------------------------------------------------------
// Provider registry
// ---------------------------------------------------------------------------

static PROVIDER_REGISTRY: LazyLock<Mutex<HashMap<String, Box<dyn StreamFn>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Register a streaming provider for an API protocol.
///
/// Overwrites any previously registered provider for the same API name.
pub fn register_api_provider(api: impl Into<String>, provider: Box<dyn StreamFn>) {
    let mut registry = PROVIDER_REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    registry.insert(api.into(), provider);
}

/// Remove all registered providers.
pub fn clear_api_providers() {
    let mut registry = PROVIDER_REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    registry.clear();
}

/// Stream via the registered provider, holding the lock only for dispatch.
///
/// This is the internal path shared by [`stream_simple`] and other
/// top-level helpers. The lock is held just long enough to call
/// `provider.stream(...)`, which returns immediately per the StreamFn
/// contract.
fn stream_via_registry(
    model: &Model,
    context: Context,
    options: StreamOptions,
) -> Result<AssistantMessageEventStream, String> {
    let registry = PROVIDER_REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    match registry.get(&model.api) {
        Some(provider) => Ok(provider.stream(model, context, options)),
        None => Err(format!("no API provider registered for api: {}", model.api)),
    }
}

// ---------------------------------------------------------------------------
// Clone helper for Box<dyn StreamFn>
// ---------------------------------------------------------------------------



// ---------------------------------------------------------------------------
// Top-level entry points
// ---------------------------------------------------------------------------

/// Stream an LLM response by looking up the registered provider for the
/// model's API protocol.
///
/// # Errors
///
/// If no provider is registered for `model.api`, returns a stream whose sole
/// event is an [`Error`](AssistantMessageEvent::Error) with a descriptive
/// message. This follows the StreamFn contract: failures are encoded in the
/// stream, never thrown.
pub fn stream_simple(
    model: &Model,
    context: Context,
    options: StreamOptions,
) -> AssistantMessageEventStream {
    match stream_via_registry(model, context, options) {
        Ok(stream) => stream,
        Err(error_message) => {
            let (producer, stream) = crate::stream::create_assistant_message_event_stream();
            let error_msg = make_error_message(model, &error_message);
            let _ = producer.push(AssistantMessageEvent::Error {
                reason: crate::types::StopReason::Error,
                error: error_msg,
            });
            producer.end();
            stream
        }
    }
}

/// Await a full LLM response by streaming and collecting the final message.
///
/// Convenience wrapper around [`stream_simple`] for callers that don't need
/// incremental streaming events (e.g., compaction summarization).
pub async fn complete_simple(
    model: &Model,
    context: Context,
    options: StreamOptions,
) -> AssistantMessage {
    stream_simple(model, context, options).result().await
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Construct a minimal error `AssistantMessage` for provider-level failures.
fn make_error_message(model: &Model, error_message: &str) -> AssistantMessage {
    use crate::types::{StopReason, Usage};
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
    use crate::stream::create_assistant_message_event_stream;
    use crate::types::{
        AssistantContentBlock, Cost, InputType, StopReason, TextContent, Usage,
    };

    /// A minimal model for testing.
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
        }
    }

    /// A trivial StreamFn that immediately returns a done message.
    #[derive(Clone)]
    struct ImmediateProvider;

    impl StreamFn for ImmediateProvider {
        fn stream(
            &self,
            model: &Model,
            _context: Context,
            _options: StreamOptions,
        ) -> AssistantMessageEventStream {
            let (producer, stream) = create_assistant_message_event_stream();
            let msg = AssistantMessage {
                content: vec![AssistantContentBlock::Text(TextContent::new("hello"))],
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                response_model: None,
                response_id: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            };
            let _ = producer.push(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message: msg,
            });
            producer.end();
            stream
        }
    }

    #[test]
    fn register_and_clear() {
        clear_api_providers();
        // Verify clearing works without panic
        register_api_provider("test-api", Box::new(ImmediateProvider));
        clear_api_providers();
    }

    #[tokio::test]
    async fn stream_simple_returns_error_for_unregistered_api() {
        clear_api_providers();
        let model = test_model();
        let context = Context::default();
        let result = stream_simple(&model, context, StreamOptions::default())
            .result()
            .await;

        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(result.error_message.unwrap().contains("no API provider"));
    }

    #[tokio::test]
    async fn stream_simple_delegates_to_registered_provider() {
        clear_api_providers();
        register_api_provider("test-api", Box::new(ImmediateProvider));

        let model = test_model();
        let context = Context::default();
        let result = stream_simple(&model, context, StreamOptions::default())
            .result()
            .await;

        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(result.content.len(), 1);

        clear_api_providers();
    }

    #[tokio::test]
    async fn complete_simple_returns_final_message() {
        clear_api_providers();
        register_api_provider("test-api", Box::new(ImmediateProvider));

        let model = test_model();
        let context = Context::default();
        let result = complete_simple(&model, context, StreamOptions::default()).await;

        assert_eq!(result.stop_reason, StopReason::Stop);

        clear_api_providers();
    }
}
