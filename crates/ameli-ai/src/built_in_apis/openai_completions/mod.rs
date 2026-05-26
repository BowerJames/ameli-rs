//! OpenAI Chat Completions built-in API for the ameli ecosystem.
//!
//! Implements the [`StreamFn`](crate::api::StreamFn) trait for the
//! `"openai-completions"` API protocol, supporting OpenAI, ZAI, and other
//! compatible providers via [`OpenAICompletionsCompat`].
//!
//! This API is automatically registered in the
//! [`DEFAULT_API_REGISTRY`](crate::api::DEFAULT_API_REGISTRY). Use
//! [`register`] to register it with a custom `ApiRegistry`.
//!
//! # Usage
//!
//! ```no_run
//! use ameli_ai::api::ApiRegistry;
//!
//! // Custom registry:
//! let registry = ApiRegistry::new();
//! ameli_ai::built_in_apis::openai_completions::register(&registry);
//! ```

pub mod api;
pub mod compat;
pub mod json;
pub mod messages;
pub mod types;

pub use api::OpenAICompletionsProvider;
pub use compat::OpenAICompletionsCompat;

use crate::api::ApiRegistry;

/// Register the OpenAI Completions provider for the `"openai-completions"` API.
pub fn register(registry: &ApiRegistry) {
    registry.register(
        "openai-completions",
        Box::new(OpenAICompletionsProvider::new()),
    );
}
