//! LLM provider trait, API registry, and top-level streaming entry points.
//!
//! This module defines the [`StreamFn`] trait — the single abstraction boundary
//! between agent orchestration code and concrete LLM providers. Implementations
//! are registered by API protocol name (e.g., `"openai-completions"`,
//! `"anthropic-messages"`) and looked up at request time via the model's
//! [`Model::api`] field.
//!
//! # Registry
//!
//! [`ApiRegistry`] holds registered [`StreamFn`] implementations keyed by
//! API protocol name. This is an **API-protocol** registry, not a provider
//! registry — multiple providers (OpenAI, ZAI, etc.) may share the same API
//! protocol and are all handled by a single registered [`StreamFn`], with
//! provider-specific behavior configured per-model via [`Model::base_url`] and
//! [`Model::compat`]. It uses interior mutability so callers share a single
//! instance by reference while still allowing registration at runtime.
//!
//! A global default ([`DEFAULT_API_REGISTRY`]) is provided for convenience, accessed
//! via [`stream_simple_global`] and [`complete_simple_global`].
//!
//! # Entry points
//!
//! - [`stream_simple`] — stream an LLM response using an explicit registry
//! - [`stream_simple_global`] — stream using the global default registry
//! - [`complete_simple`] — await a full LLM response (explicit registry)
//! - [`complete_simple_global`] — await a full response (global registry)

use crate::stream::AssistantMessageEventStream;
use crate::types::{AssistantMessage, AssistantMessageEvent, Context, Model, StreamOptions};
use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

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
// ApiRegistry
// ---------------------------------------------------------------------------

/// Registry of [`StreamFn`] implementations keyed by API protocol name.
///
/// This is an **API-protocol** registry, not a provider registry. The key is
/// the API protocol (e.g., `"openai-completions"`, `"anthropic-messages"`),
/// not the provider name. Multiple providers (OpenAI, ZAI, etc.) may share
/// the same API protocol — they are all handled by a single registered
/// [`StreamFn`] implementation, with provider-specific behavior configured
/// per-model via [`Model::base_url`] and [`Model::compat`].
///
/// Uses interior mutability (`RwLock`) so callers can share a single instance
/// by reference while still registering at runtime. Concurrent reads
/// (lookups during streaming) are not blocked by each other; only registration
/// takes a write lock.
///
/// # Examples
///
/// ```
/// use ameli_ai::api::{ApiRegistry, StreamFn};
/// # use ameli_ai::types::{Model, Context, StreamOptions};
/// # use ameli_ai::stream::AssistantMessageEventStream;
///
/// let registry = ApiRegistry::new();
/// // registry.register("openai-completions", Box::new(MyProvider));
/// // ameli_ai::api::stream_simple(&registry, &model, context, options);
/// ```
pub struct ApiRegistry {
    apis: RwLock<HashMap<String, Box<dyn StreamFn>>>,
}

impl ApiRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            apis: RwLock::new(HashMap::new()),
        }
    }

    /// Register a streaming [`StreamFn`] for an API protocol.
    ///
    /// Overwrites any previously registered implementation for the same API name.
    pub fn register(&self, api: impl Into<String>, provider: Box<dyn StreamFn>) {
        let mut apis = self.apis.write().unwrap_or_else(|e| e.into_inner());
        apis.insert(api.into(), provider);
    }

    /// Remove all registered API protocol implementations.
    pub fn clear(&self) {
        let mut apis = self.apis.write().unwrap_or_else(|e| e.into_inner());
        apis.clear();
    }

    /// Look up the [`StreamFn`] for an API protocol and dispatch a stream request.
    ///
    /// The read lock is held only long enough to call `provider.stream(...)`,
    /// which returns immediately per the [`StreamFn`] contract.
    fn stream_via_registry(
        &self,
        model: &Model,
        context: Context,
        options: StreamOptions,
    ) -> Result<AssistantMessageEventStream, String> {
        let apis = self.apis.read().unwrap_or_else(|e| e.into_inner());
        match apis.get(&model.api) {
            Some(provider) => Ok(provider.stream(model, context, options)),
            None => Err(format!("no API provider registered for api: {}", model.api)),
        }
    }
}

impl Default for ApiRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Global default API registry
// ---------------------------------------------------------------------------

/// Global default API registry for convenience.
///
/// Use [`stream_simple_global`] or [`complete_simple_global`] to stream via
/// this registry without passing it explicitly.
pub static DEFAULT_API_REGISTRY: LazyLock<ApiRegistry> = LazyLock::new(|| {
    let registry = ApiRegistry::new();
    crate::built_in_apis::openai_completions::register(&registry);
    registry
});

// ---------------------------------------------------------------------------
// Top-level entry points
// ---------------------------------------------------------------------------

/// Stream an LLM response by looking up the [`StreamFn`] in the given API registry.
///
/// # Errors
///
/// If no [`StreamFn`] is registered for `model.api`, returns a stream whose sole
/// event is an [`Error`](AssistantMessageEvent::Error) with a descriptive
/// message. This follows the StreamFn contract: failures are encoded in the
/// stream, never thrown.
pub fn stream_simple(
    registry: &ApiRegistry,
    model: &Model,
    context: Context,
    options: StreamOptions,
) -> AssistantMessageEventStream {
    match registry.stream_via_registry(model, context, options) {
        Ok(stream) => stream,
        Err(error_message) => {
            let (producer, stream) = crate::stream::create_assistant_message_event_stream();
            let error_msg = make_error_message(model, &error_message);
            producer.push(AssistantMessageEvent::Error {
                reason: crate::types::StopReason::Error,
                error: error_msg,
            });
            producer.end();
            stream
        }
    }
}

/// Stream an LLM response using the [`DEFAULT_API_REGISTRY`].
///
/// Convenience wrapper around [`stream_simple`] for callers that don't need
/// an explicit registry.
pub fn stream_simple_global(
    model: &Model,
    context: Context,
    options: StreamOptions,
) -> AssistantMessageEventStream {
    stream_simple(&DEFAULT_API_REGISTRY, model, context, options)
}

/// Await a full LLM response by streaming and collecting the final message.
///
/// Convenience wrapper around [`stream_simple`] for callers that don't need
/// incremental streaming events (e.g., compaction summarization).
pub async fn complete_simple(
    registry: &ApiRegistry,
    model: &Model,
    context: Context,
    options: StreamOptions,
) -> AssistantMessage {
    stream_simple(registry, model, context, options)
        .result()
        .await
}

/// Await a full LLM response using the [`DEFAULT_API_REGISTRY`].
///
/// Convenience wrapper around [`complete_simple`] for callers that don't need
/// an explicit registry.
pub async fn complete_simple_global(
    model: &Model,
    context: Context,
    options: StreamOptions,
) -> AssistantMessage {
    complete_simple(&DEFAULT_API_REGISTRY, model, context, options).await
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
    use crate::types::{AssistantContentBlock, Cost, InputType, StopReason, TextContent, Usage};

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
            compat: None,
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
            producer.push(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message: msg,
            });
            producer.end();
            stream
        }
    }

    #[test]
    fn registry_register_and_clear() {
        let registry = ApiRegistry::new();
        registry.register("test-api", Box::new(ImmediateProvider));
        registry.clear();
        // Verify clearing works without panic
    }

    #[tokio::test]
    async fn stream_simple_returns_error_for_unregistered_api() {
        let registry = ApiRegistry::new();
        let model = test_model();
        let context = Context::default();
        let result = stream_simple(&registry, &model, context, StreamOptions::default())
            .result()
            .await;

        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(result.error_message.unwrap().contains("no API provider"));
    }

    #[tokio::test]
    async fn stream_simple_delegates_to_registered_provider() {
        let registry = ApiRegistry::new();
        registry.register("test-api", Box::new(ImmediateProvider));

        let model = test_model();
        let context = Context::default();
        let result = stream_simple(&registry, &model, context, StreamOptions::default())
            .result()
            .await;

        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(result.content.len(), 1);
    }

    #[tokio::test]
    async fn complete_simple_returns_final_message() {
        let registry = ApiRegistry::new();
        registry.register("test-api", Box::new(ImmediateProvider));

        let model = test_model();
        let context = Context::default();
        let result = complete_simple(&registry, &model, context, StreamOptions::default()).await;

        assert_eq!(result.stop_reason, StopReason::Stop);
    }
}
