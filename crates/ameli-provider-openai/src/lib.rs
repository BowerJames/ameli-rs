//! OpenAI Chat Completions provider for the ameli ecosystem.
//!
//! Implements the [`StreamFn`] trait for the `"openai-completions"` API protocol,
//! supporting OpenAI, ZAI, and other compatible providers via the
//! [`OpenAICompletionsCompat`] configuration struct.
//!
//! # Usage
//!
//! ```no_run
//! use ameli_ai::api::ApiRegistry;
//! use ameli_provider_openai::OpenAICompletionsProvider;
//!
//! let registry = ApiRegistry::new();
//! ameli_provider_openai::register(&registry);
//! ```

pub mod compat;
pub mod json;
pub mod messages;
pub mod provider;
pub mod types;

pub use compat::OpenAICompletionsCompat;
pub use provider::OpenAICompletionsProvider;

use ameli_ai::api::ApiRegistry;

/// Register the OpenAI Completions provider for the `"openai-completions"` API.
pub fn register(registry: &ApiRegistry) {
    registry.register(
        "openai-completions",
        Box::new(OpenAICompletionsProvider::new()),
    );
}
