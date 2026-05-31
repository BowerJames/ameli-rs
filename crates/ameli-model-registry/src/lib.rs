//! Model registry for looking up [`Model`] descriptors by provider and model ID.
//!
//! This crate defines the [`ModelRegistry`] trait — a minimal lookup contract —
//! and a concrete [`DefaultModelRegistry`] backed by an interior-mutable
//! `HashMap`. A global default instance ([`DEFAULT_MODEL_REGISTRY`]) is
//! provided for convenience.
//!
//! # Design
//!
//! - **Thin trait** — [`ModelRegistry`] has a single method:
//!   [`get_model`](ModelRegistry::get_model). Richer query methods are
//!   inherent on [`DefaultModelRegistry`], not part of the trait.
//! - **Dedicated error type** — [`ModelNotFoundError`] distinguishes "unknown
//!   provider" from "unknown model for a known provider".
//! - **Global starts with built-in models** — [`DEFAULT_MODEL_REGISTRY`] initialises
//!   with models from [models.dev](https://models.dev). Downstream applications
//!   can add more at runtime.
//!
//! # Example
//!
//! ```
//! use ameli_model_registry::{DefaultModelRegistry, ModelRegistry, ModelNotFoundError};
//! use ameli_ai::types::{Model, Cost, InputType};
//!
//! # fn main() -> Result<(), ModelNotFoundError> {
//! let registry = DefaultModelRegistry::new();
//!
//! let model = Model {
//!     id: "gpt-4o".into(),
//!     name: "GPT-4o".into(),
//!     api: "openai-completions".into(),
//!     provider: "openai".into(),
//!     base_url: "https://api.openai.com/v1".into(),
//!     reasoning: false,
//!     thinking_level_map: None,
//!     input: vec![InputType::Text],
//!     cost: Cost::default(),
//!     context_window: 128_000,
//!     max_tokens: 16_384,
//!     compat: None,
//! };
//!
//! registry.register(model);
//!
//! let found = registry.get_model("openai", "gpt-4o")?;
//! assert_eq!(found.id, "gpt-4o");
//! # Ok(())
//! # }
//! ```

use ameli_ai::types::Model;
use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

mod built_in;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned when a model lookup fails.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelNotFoundError {
    /// No models have been registered for the given provider.
    #[error("unknown provider: {provider}")]
    UnknownProvider {
        /// The provider name that was looked up.
        provider: String,
    },
    /// The provider is known but the specific model ID was not found.
    #[error("unknown model for provider {provider}: {model_id}")]
    UnknownModel {
        /// The provider name.
        provider: String,
        /// The model ID that was looked up.
        model_id: String,
    },
}

// ---------------------------------------------------------------------------
// ModelRegistry trait
// ---------------------------------------------------------------------------

/// Trait for looking up [`Model`] descriptors by provider and model ID.
///
/// This is the single lookup contract. Richer query methods live on
/// [`DefaultModelRegistry`] as inherent methods.
pub trait ModelRegistry: Send + Sync {
    /// Look up a model by provider name and model ID.
    ///
    /// # Errors
    ///
    /// Returns [`ModelNotFoundError::UnknownProvider`] if no models exist for
    /// the given provider, or [`ModelNotFoundError::UnknownModel`] if the
    /// provider is known but the specific model ID was not registered.
    fn get_model(&self, provider: &str, model_id: &str) -> Result<Model, ModelNotFoundError>;
}

// ---------------------------------------------------------------------------
// DefaultModelRegistry
// ---------------------------------------------------------------------------

/// Concrete model registry backed by an interior-mutable `HashMap`.
///
/// Uses `RwLock<HashMap>` so concurrent reads are not blocked by each other;
/// only registration takes a write lock.
///
/// # Examples
///
/// ```
/// use ameli_model_registry::{DefaultModelRegistry, ModelRegistry};
/// use ameli_ai::types::{Model, Cost, InputType};
///
/// let registry = DefaultModelRegistry::new();
///
/// let model = Model {
///     id: "claude-3".into(),
///     name: "Claude 3".into(),
///     api: "anthropic-messages".into(),
///     provider: "anthropic".into(),
///     base_url: "https://api.anthropic.com".into(),
///     reasoning: false,
///     thinking_level_map: None,
///     input: vec![InputType::Text],
///     cost: Cost::default(),
///     context_window: 200_000,
///     max_tokens: 8_192,
///     compat: None,
/// };
///
/// registry.register(model);
/// assert_eq!(registry.all_models().len(), 1);
/// assert_eq!(registry.providers().len(), 1);
/// ```
pub struct DefaultModelRegistry {
    models: RwLock<HashMap<(String, String), Model>>,
}

impl DefaultModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
        }
    }

    /// Register a model.
    ///
    /// If a model with the same `(provider, model_id)` key already exists, it
    /// is overwritten.
    pub fn register(&self, model: Model) {
        let key = (model.provider.clone(), model.id.clone());
        let mut models = self.models.write().unwrap_or_else(|e| e.into_inner());
        models.insert(key, model);
    }

    /// Remove a model by provider and model ID.
    ///
    /// Returns `true` if a model was removed, `false` if no such model existed.
    pub fn unregister(&self, provider: &str, model_id: &str) -> bool {
        let mut models = self.models.write().unwrap_or_else(|e| e.into_inner());
        models
            .remove(&(provider.to_string(), model_id.to_string()))
            .is_some()
    }

    /// Remove all registered models.
    pub fn clear(&self) {
        let mut models = self.models.write().unwrap_or_else(|e| e.into_inner());
        models.clear();
    }

    /// Return a snapshot of all registered models.
    pub fn all_models(&self) -> Vec<Model> {
        let models = self.models.read().unwrap_or_else(|e| e.into_inner());
        models.values().cloned().collect()
    }

    /// Return a snapshot of all models registered for a given provider.
    ///
    /// Returns an empty `Vec` if the provider is unknown.
    pub fn models_for_provider(&self, provider: &str) -> Vec<Model> {
        let models = self.models.read().unwrap_or_else(|e| e.into_inner());
        models
            .values()
            .filter(|m| m.provider == provider)
            .cloned()
            .collect()
    }

    /// Return the distinct set of provider names that have at least one model
    /// registered.
    pub fn providers(&self) -> Vec<String> {
        let models = self.models.read().unwrap_or_else(|e| e.into_inner());
        let mut provider_set: Vec<String> = models.values().map(|m| m.provider.clone()).collect();
        provider_set.sort();
        provider_set.dedup();
        provider_set
    }
}

impl Default for DefaultModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry for DefaultModelRegistry {
    fn get_model(&self, provider: &str, model_id: &str) -> Result<Model, ModelNotFoundError> {
        let models = self.models.read().unwrap_or_else(|e| e.into_inner());

        // Check whether the provider has any models at all.
        let provider_exists = models.keys().any(|(p, _)| p == provider);
        if !provider_exists {
            return Err(ModelNotFoundError::UnknownProvider {
                provider: provider.to_string(),
            });
        }

        models
            .get(&(provider.to_string(), model_id.to_string()))
            .cloned()
            .ok_or_else(|| ModelNotFoundError::UnknownModel {
                provider: provider.to_string(),
                model_id: model_id.to_string(),
            })
    }
}

// ---------------------------------------------------------------------------
// Global default registry
// ---------------------------------------------------------------------------

/// Global default model registry.
///
/// Starts with built-in models from models.dev pre-loaded. Use
/// [`register_global`] or access the registry directly to add more at runtime.
pub static DEFAULT_MODEL_REGISTRY: LazyLock<DefaultModelRegistry> = LazyLock::new(|| {
    let registry = DefaultModelRegistry::new();
    built_in::register_built_in_models(&registry);
    registry
});

/// Convenience function that looks up a model in the [`DEFAULT_MODEL_REGISTRY`].
///
/// # Errors
///
/// See [`ModelRegistry::get_model`].
pub fn get_model(provider: &str, model_id: &str) -> Result<Model, ModelNotFoundError> {
    DEFAULT_MODEL_REGISTRY.get_model(provider, model_id)
}

/// Convenience function that registers a model in the [`DEFAULT_MODEL_REGISTRY`].
pub fn register_global(model: Model) {
    DEFAULT_MODEL_REGISTRY.register(model);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ameli_ai::types::{Cost, InputType};

    /// Helper: create a minimal model for testing.
    fn test_model(provider: &str, id: &str) -> Model {
        Model {
            id: id.into(),
            name: format!("{provider}/{id}"),
            api: "test-api".into(),
            provider: provider.into(),
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

    // -- Empty registry --

    #[test]
    fn empty_registry_returns_unknown_provider() {
        let registry = DefaultModelRegistry::new();
        let err = registry.get_model("openai", "gpt-4o").unwrap_err();
        assert_eq!(
            err,
            ModelNotFoundError::UnknownProvider {
                provider: "openai".into()
            }
        );
    }

    // -- Register + lookup --

    #[test]
    fn register_and_lookup_succeeds() {
        let registry = DefaultModelRegistry::new();
        let model = test_model("openai", "gpt-4o");
        registry.register(model);

        let found = registry.get_model("openai", "gpt-4o").unwrap();
        assert_eq!(found.id, "gpt-4o");
        assert_eq!(found.provider, "openai");
    }

    // -- Overwrite on re-register --

    #[test]
    fn re_register_overwrites() {
        let registry = DefaultModelRegistry::new();
        registry.register(test_model("openai", "gpt-4o"));

        let mut updated = test_model("openai", "gpt-4o");
        updated.context_window = 200_000;
        registry.register(updated);

        let found = registry.get_model("openai", "gpt-4o").unwrap();
        assert_eq!(found.context_window, 200_000);
    }

    // -- Unknown model for known provider --

    #[test]
    fn unknown_model_for_known_provider() {
        let registry = DefaultModelRegistry::new();
        registry.register(test_model("openai", "gpt-4o"));

        let err = registry.get_model("openai", "gpt-3.5-turbo").unwrap_err();
        assert_eq!(
            err,
            ModelNotFoundError::UnknownModel {
                provider: "openai".into(),
                model_id: "gpt-3.5-turbo".into()
            }
        );
    }

    // -- all_models --

    #[test]
    fn all_models_returns_all_entries() {
        let registry = DefaultModelRegistry::new();
        registry.register(test_model("openai", "gpt-4o"));
        registry.register(test_model("anthropic", "claude-3"));

        let mut models = registry.all_models();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "claude-3");
        assert_eq!(models[1].id, "gpt-4o");
    }

    // -- models_for_provider --

    #[test]
    fn models_for_provider_filters_correctly() {
        let registry = DefaultModelRegistry::new();
        registry.register(test_model("openai", "gpt-4o"));
        registry.register(test_model("openai", "gpt-4o-mini"));
        registry.register(test_model("anthropic", "claude-3"));

        let openai = registry.models_for_provider("openai");
        assert_eq!(openai.len(), 2);

        let anthropic = registry.models_for_provider("anthropic");
        assert_eq!(anthropic.len(), 1);
    }

    #[test]
    fn models_for_provider_empty_for_unknown() {
        let registry = DefaultModelRegistry::new();
        registry.register(test_model("openai", "gpt-4o"));

        let unknown = registry.models_for_provider("groq");
        assert!(unknown.is_empty());
    }

    // -- providers --

    #[test]
    fn providers_returns_distinct_names() {
        let registry = DefaultModelRegistry::new();
        registry.register(test_model("openai", "gpt-4o"));
        registry.register(test_model("openai", "gpt-4o-mini"));
        registry.register(test_model("anthropic", "claude-3"));

        let providers = registry.providers();
        assert_eq!(providers, vec!["anthropic", "openai"]);
    }

    // -- unregister --

    #[test]
    fn unregister_removes_model() {
        let registry = DefaultModelRegistry::new();
        registry.register(test_model("openai", "gpt-4o"));

        assert!(registry.unregister("openai", "gpt-4o"));
        let err = registry.get_model("openai", "gpt-4o").unwrap_err();
        assert!(matches!(err, ModelNotFoundError::UnknownProvider { .. }));
    }

    #[test]
    fn unregister_returns_false_for_missing() {
        let registry = DefaultModelRegistry::new();
        assert!(!registry.unregister("openai", "gpt-4o"));
    }

    // -- clear --

    #[test]
    fn clear_empties_registry() {
        let registry = DefaultModelRegistry::new();
        registry.register(test_model("openai", "gpt-4o"));
        registry.register(test_model("anthropic", "claude-3"));
        assert_eq!(registry.all_models().len(), 2);

        registry.clear();
        assert!(registry.all_models().is_empty());
        assert!(registry.providers().is_empty());
    }

    // -- Convenience function delegates to global --

    #[test]
    fn convenience_get_model_delegates_to_global() {
        // The global registry starts empty for each test process, but other
        // tests may have registered models. We test with a unique provider.
        let provider = "test-convenience-get-model";
        DEFAULT_MODEL_REGISTRY.register(test_model(provider, "model-a"));

        let found = get_model(provider, "model-a").unwrap();
        assert_eq!(found.id, "model-a");

        // Clean up
        DEFAULT_MODEL_REGISTRY.unregister(provider, "model-a");
    }

    // -- Error display --

    #[test]
    fn error_display_unknown_provider() {
        let err = ModelNotFoundError::UnknownProvider {
            provider: "groq".into(),
        };
        assert_eq!(format!("{err}"), "unknown provider: groq");
    }

    #[test]
    fn error_display_unknown_model() {
        let err = ModelNotFoundError::UnknownModel {
            provider: "openai".into(),
            model_id: "gpt-5".into(),
        };
        assert_eq!(format!("{err}"), "unknown model for provider openai: gpt-5");
    }

    // -- Default trait --

    #[test]
    fn default_is_empty() {
        let registry = DefaultModelRegistry::default();
        assert!(registry.all_models().is_empty());
    }
}
