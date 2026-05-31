//! API key storage trait and in-memory implementation.
//!
//! This crate defines [`AuthStorage`] — a trait for looking up API keys by
//! provider name — and [`InMemoryAuthStorage`], a concrete implementation
//! backed by an interior-mutable `HashMap` with environment variable fallback.
//!
//! # Design
//!
//! - **Thin trait** — [`AuthStorage`] has a single method:
//!   [`get_api_key`](AuthStorage::get_api_key).
//! - **Object-safe** — uses `Pin<Box<dyn Future>>` return type so
//!   `Arc<dyn AuthStorage>` works.
//! - **Environment fallback** — [`InMemoryAuthStorage`] checks its in-memory
//!   map first, then falls back to `<PROVIDER>_API_KEY` env vars.
//!
//! # Example
//!
//! ```
//! use ameli_auth_storage::{AuthStorage, InMemoryAuthStorage};
//!
//! # fn main() -> Result<(), ameli_auth_storage::ApiKeyNotFoundError> {
//! let storage = InMemoryAuthStorage::new();
//! storage.set_api_key("openai", "sk-...".to_string());
//!
//! // In-memory lookup
//! let key = storage.get_api_key_sync("openai")?;
//! assert_eq!(key, "sk-...");
//!
//! // Provider not found
//! let err = storage.get_api_key_sync("unknown").unwrap_err();
//! assert!(err.is_not_found());
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::RwLock;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned when an API key lookup fails.
///
/// No API key was found in storage or in the environment for the requested
/// provider.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ApiKeyNotFoundError {
    /// No API key found for the given provider.
    #[error("no API key found for provider: {provider}")]
    NotFound {
        /// The provider name that was looked up.
        provider: String,
    },
}

impl ApiKeyNotFoundError {
    /// Create a [`NotFound`](ApiKeyNotFoundError::NotFound) error.
    pub fn not_found(provider: impl Into<String>) -> Self {
        Self::NotFound {
            provider: provider.into(),
        }
    }

    /// Returns `true` if this is a [`NotFound`](ApiKeyNotFoundError::NotFound) error.
    pub fn is_not_found(&self) -> bool {
        matches!(self, Self::NotFound { .. })
    }
}

// ---------------------------------------------------------------------------
// AsyncResult alias
// ---------------------------------------------------------------------------

/// Boxed, sendable async result used by [`AuthStorage`] trait methods.
///
/// Using `Pin<Box<dyn Future>>` ensures the trait is dyn-compatible
/// (object-safe), so `Arc<dyn AuthStorage>` works.
pub type AsyncResult<T> = Pin<Box<dyn Future<Output = Result<T, ApiKeyNotFoundError>> + Send>>;

// ---------------------------------------------------------------------------
// AuthStorage trait
// ---------------------------------------------------------------------------

/// Trait for looking up API keys by provider name.
///
/// Implementations decide their own storage strategy (in-memory, keychain,
/// encrypted file, etc.). The trait is object-safe so it can be used behind
/// `Arc<dyn AuthStorage>`.
///
/// # Examples
///
/// ```
/// use ameli_auth_storage::{AuthStorage, ApiKeyNotFoundError};
/// use std::sync::Arc;
///
/// struct EnvOnlyStorage;
///
/// impl AuthStorage for EnvOnlyStorage {
///     fn get_api_key(&self, provider: &str) -> ameli_auth_storage::AsyncResult<String> {
///         let provider = provider.to_string();
///         Box::pin(async move {
///             let env_name = format!("{}_API_KEY", provider.to_uppercase());
///             std::env::var(&env_name).map_err(|_| {
///                 ApiKeyNotFoundError::not_found(&provider)
///             })
///         })
///     }
/// }
///
/// // Works as Arc<dyn AuthStorage>
/// let _storage: Arc<dyn AuthStorage> = Arc::new(EnvOnlyStorage);
/// ```
pub trait AuthStorage: Send + Sync {
    /// Look up an API key for the given provider.
    ///
    /// # Errors
    ///
    /// Returns [`ApiKeyNotFoundError::NotFound`] if no key is available for
    /// the requested provider.
    fn get_api_key(&self, provider: &str) -> AsyncResult<String>;
}

// ---------------------------------------------------------------------------
// InMemoryAuthStorage
// ---------------------------------------------------------------------------

/// In-memory API key storage with environment variable fallback.
///
/// Stores API keys in an interior-mutable `HashMap`. On lookup, checks the
/// in-memory map first; if not found, falls back to the
/// `<PROVIDER>_API_KEY` environment variable (uppercased). If neither source
/// has a key, returns [`ApiKeyNotFoundError::NotFound`].
///
/// # Thread safety
///
/// Uses `RwLock<HashMap>` so concurrent reads are not blocked by each other;
/// only writes take a write lock. `Send + Sync` is guaranteed.
///
/// # Examples
///
/// ```
/// use ameli_auth_storage::{AuthStorage, InMemoryAuthStorage};
///
/// let storage = InMemoryAuthStorage::new();
/// storage.set_api_key("openai", "sk-test".to_string());
///
/// // In-memory lookup succeeds
/// let key = storage.get_api_key_sync("openai").unwrap();
/// assert_eq!(key, "sk-test");
///
/// // Unknown provider fails
/// assert!(storage.get_api_key_sync("unknown").is_err());
/// ```
pub struct InMemoryAuthStorage {
    keys: RwLock<HashMap<String, String>>,
}

impl InMemoryAuthStorage {
    /// Create an empty in-memory auth storage.
    pub fn new() -> Self {
        Self {
            keys: RwLock::new(HashMap::new()),
        }
    }

    /// Register or overwrite an API key for a provider.
    pub fn set_api_key(&self, provider: &str, key: String) {
        let mut keys = self.keys.write().unwrap_or_else(|e| e.into_inner());
        keys.insert(provider.to_string(), key);
    }

    /// Remove the API key for a provider from in-memory storage.
    ///
    /// Does not affect environment variables. After removal, lookups for this
    /// provider will fall back to the `<PROVIDER>_API_KEY` env var.
    ///
    /// Returns `true` if a key was removed, `false` if no in-memory key
    /// existed.
    pub fn remove_api_key(&self, provider: &str) -> bool {
        let mut keys = self.keys.write().unwrap_or_else(|e| e.into_inner());
        keys.remove(provider).is_some()
    }

    /// Remove all in-memory API keys.
    ///
    /// Does not affect environment variables.
    pub fn clear(&self) {
        let mut keys = self.keys.write().unwrap_or_else(|e| e.into_inner());
        keys.clear();
    }

    /// Synchronous convenience method for looking up an API key.
    ///
    /// Useful in non-async contexts (e.g., CLI startup). Follows the same
    /// lookup order as [`get_api_key`](AuthStorage::get_api_key): in-memory
    /// first, then environment variable.
    pub fn get_api_key_sync(&self, provider: &str) -> Result<String, ApiKeyNotFoundError> {
        // 1. Check in-memory map
        {
            let keys = self.keys.read().unwrap_or_else(|e| e.into_inner());
            if let Some(key) = keys.get(provider) {
                return Ok(key.clone());
            }
        }

        // 2. Fall back to environment variable
        let env_name = format!("{}_API_KEY", provider.to_uppercase());
        if let Ok(key) = std::env::var(&env_name) {
            if !key.is_empty() {
                return Ok(key);
            }
        }

        // 3. Not found
        Err(ApiKeyNotFoundError::not_found(provider))
    }
}

impl Default for InMemoryAuthStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl AuthStorage for InMemoryAuthStorage {
    fn get_api_key(&self, provider: &str) -> AsyncResult<String> {
        let provider = provider.to_string();
        let result = self.get_api_key_sync(&provider);
        Box::pin(async move { result })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // -- ApiKeyNotFoundError --

    #[test]
    fn error_display_message() {
        let err = ApiKeyNotFoundError::not_found("openai");
        assert_eq!(format!("{err}"), "no API key found for provider: openai");
    }

    #[test]
    fn error_is_not_found() {
        let err = ApiKeyNotFoundError::not_found("anthropic");
        assert!(err.is_not_found());
    }

    #[test]
    fn error_clone_eq() {
        let a = ApiKeyNotFoundError::not_found("openai");
        let b = a.clone();
        assert_eq!(a, b);
    }

    // -- InMemoryAuthStorage basic operations --

    #[test]
    fn new_is_empty() {
        let storage = InMemoryAuthStorage::new();
        let keys = storage.keys.read().unwrap_or_else(|e| e.into_inner());
        assert!(keys.is_empty());
    }

    #[test]
    fn default_is_new() {
        let storage = InMemoryAuthStorage::default();
        let keys = storage.keys.read().unwrap_or_else(|e| e.into_inner());
        assert!(keys.is_empty());
    }

    #[test]
    fn set_and_get_sync() {
        let storage = InMemoryAuthStorage::new();
        storage.set_api_key("openai", "sk-test".to_string());

        let key = storage.get_api_key_sync("openai").unwrap();
        assert_eq!(key, "sk-test");
    }

    #[test]
    fn get_sync_unknown_provider_returns_error() {
        let storage = InMemoryAuthStorage::new();
        let err = storage.get_api_key_sync("unknown").unwrap_err();
        assert!(err.is_not_found());
        assert_eq!(err, ApiKeyNotFoundError::not_found("unknown"));
    }

    #[test]
    fn overwrite_key() {
        let storage = InMemoryAuthStorage::new();
        storage.set_api_key("openai", "sk-old".to_string());
        storage.set_api_key("openai", "sk-new".to_string());

        let key = storage.get_api_key_sync("openai").unwrap();
        assert_eq!(key, "sk-new");
    }

    #[test]
    fn remove_key() {
        let storage = InMemoryAuthStorage::new();
        storage.set_api_key("openai", "sk-test".to_string());
        assert!(storage.remove_api_key("openai"));
        assert!(storage.get_api_key_sync("openai").is_err());
    }

    #[test]
    fn remove_missing_key_returns_false() {
        let storage = InMemoryAuthStorage::new();
        assert!(!storage.remove_api_key("openai"));
    }

    #[test]
    fn clear_removes_all_keys() {
        let storage = InMemoryAuthStorage::new();
        storage.set_api_key("openai", "sk-a".to_string());
        storage.set_api_key("anthropic", "sk-b".to_string());
        storage.clear();

        assert!(storage.get_api_key_sync("openai").is_err());
        assert!(storage.get_api_key_sync("anthropic").is_err());
    }

    // -- Async get_api_key via trait --

    #[tokio::test]
    async fn trait_get_api_key_returns_stored_key() {
        let storage = InMemoryAuthStorage::new();
        storage.set_api_key("openai", "sk-async".to_string());

        let key = storage.get_api_key("openai").await.unwrap();
        assert_eq!(key, "sk-async");
    }

    #[tokio::test]
    async fn trait_get_api_key_returns_error_for_unknown() {
        let storage = InMemoryAuthStorage::new();
        let err = storage.get_api_key("unknown").await.unwrap_err();
        assert!(err.is_not_found());
    }

    // -- Trait object safety --

    #[test]
    fn trait_is_object_safe() {
        let storage: Arc<dyn AuthStorage> = Arc::new(InMemoryAuthStorage::new());
        // Verify the trait object compiles and can be used
        let provider = "openai";
        drop(storage.get_api_key(provider));
    }

    // -- Concurrent access --

    #[tokio::test]
    async fn concurrent_read_write() {
        let storage = Arc::new(InMemoryAuthStorage::new());

        let mut handles = Vec::new();
        for i in 0..10 {
            let s = storage.clone();
            handles.push(tokio::spawn(async move {
                s.set_api_key(&format!("provider-{i}"), format!("key-{i}"));
                let key = s.get_api_key(&format!("provider-{i}")).await.unwrap();
                assert_eq!(key, format!("key-{i}"));
            }));
        }

        for handle in handles {
            assert!(handle.await.is_ok());
        }

        // All 10 keys should be present
        for i in 0..10 {
            let key = storage.get_api_key(&format!("provider-{i}")).await.unwrap();
            assert_eq!(key, format!("key-{i}"));
        }
    }

    // -- Environment variable fallback --

    #[test]
    fn env_fallback_when_no_in_memory_key() {
        let storage = InMemoryAuthStorage::new();

        // Set an env var for this test
        let env_key = "AMELI_TEST_AUTH_STORAGE_PROVIDER_API_KEY";
        let test_value = "test-api-key-from-env";
        std::env::set_var(env_key, test_value);

        let result = storage.get_api_key_sync("ameli_test_auth_storage_provider");
        std::env::remove_var(env_key);

        assert_eq!(result.unwrap(), test_value);
    }

    #[test]
    fn in_memory_takes_precedence_over_env() {
        let storage = InMemoryAuthStorage::new();

        let env_key = "AMELI_TEST_PRECEDENCE_PROVIDER_API_KEY";
        std::env::set_var(env_key, "from-env");
        storage.set_api_key("ameli_test_precedence_provider", "from-memory".to_string());

        let result = storage.get_api_key_sync("ameli_test_precedence_provider");
        std::env::remove_var(env_key);

        assert_eq!(result.unwrap(), "from-memory");
    }

    #[test]
    fn empty_env_var_is_ignored() {
        let storage = InMemoryAuthStorage::new();

        let env_key = "AMELI_TEST_EMPTY_ENV_PROVIDER_API_KEY";
        std::env::set_var(env_key, "");

        let result = storage.get_api_key_sync("ameli_test_empty_env_provider");
        std::env::remove_var(env_key);

        assert!(result.is_err());
    }

    #[test]
    fn remove_key_falls_back_to_env() {
        let storage = InMemoryAuthStorage::new();

        let env_key = "AMELI_TEST_FALLBACK_PROVIDER_API_KEY";
        std::env::set_var(env_key, "env-value");
        storage.set_api_key("ameli_test_fallback_provider", "mem-value".to_string());

        // Before removal, in-memory wins
        assert_eq!(
            storage
                .get_api_key_sync("ameli_test_fallback_provider")
                .unwrap(),
            "mem-value"
        );

        // After removal, falls back to env
        storage.remove_api_key("ameli_test_fallback_provider");
        assert_eq!(
            storage
                .get_api_key_sync("ameli_test_fallback_provider")
                .unwrap(),
            "env-value"
        );

        std::env::remove_var(env_key);
    }
}
