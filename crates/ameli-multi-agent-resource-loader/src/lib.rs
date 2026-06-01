//! Multi-agent resource loader — trait for loading agent session resources.
//!
//! This crate defines [`MultiAgentResourceLoader<M>`] — a trait that
//! implementations use to create sessions and load the resources needed to
//! construct an [`AgentSession`](ameli_agent::AgentSession). The consumer
//! combines the returned [`AgentSessionResources`] with their own
//! [`Interface`](ameli_agent::Interface), [`ModelRegistry`](ameli_model_registry::ModelRegistry),
//! and system prompt to call [`create_agent_session()`](ameli_agent::create_agent_session).
//!
//! # Architecture
//!
//! ```text
//! MultiAgentResourceLoader<M>     ← trait (this crate)
//!     ├── create_session()       → session_id
//!     └── load_resources()       → AgentSessionResources<M>
//!
//! AgentSessionResources<M>        ← data bundle
//!     ├── SessionManager<M>      ← session persistence
//!     ├── AuthStorage            ← API key resolution
//!     ├── Extensions             ← extension instances
//!     ├── ModelRef               ← model selection
//!     └── ThinkingLevel          ← reasoning level
//!
//! Consumer                        ← downstream app
//!     ├── AgentSessionResources  ← from this crate
//!     ├── Interface              ← consumer provides
//!     ├── ModelRegistry          ← consumer provides
//!     └── system_prompt          ← consumer provides
//!         → create_agent_session()
//! ```
//!
//! # Design Decisions
//!
//! - **No `Interface`** — the consumer provides the UI mode (TUI, RPC,
//!   headless).
//! - **No `ModelRegistry`** — the consumer resolves models from their own
//!   registry instance.
//! - **No `system_prompt`** — the consumer determines the system prompt per
//!   session.
//! - **Generic over `M: SessionMetadata`** — different storage backends carry
//!   different metadata types.
//! - **No concrete implementations** — implementations live in downstream
//!   crates (e.g., a future `ameli-multi-agent-postgres`).

pub mod error;

// Imports that serve double duty: available internally and re-exported.
pub use ameli_agent::auth_storage::AuthStorage;
pub use ameli_agent::extension::Extension;
pub use ameli_agent::session_manager::{ModelRef, SessionManager, SessionMetadata};
pub use ameli_agent_core::types::ThinkingLevel;
pub use error::{CreateSessionError, LoadResourcesError};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// AsyncResult type alias
// ---------------------------------------------------------------------------

/// Boxed, sendable async result.
///
/// Using `Pin<Box<dyn Future>>` ensures the trait is dyn-compatible
/// (object-safe), so `Arc<dyn MultiAgentResourceLoader<M>>` works.
pub type AsyncResult<T, E> = Pin<Box<dyn Future<Output = Result<T, E>> + Send>>;

// ---------------------------------------------------------------------------
// AgentSessionResources
// ---------------------------------------------------------------------------

/// Resources needed to construct an [`AgentSession`](ameli_agent::AgentSession).
///
/// Returned by [`MultiAgentResourceLoader::load_resources`]. The consumer
/// combines these with their own `Interface`, `ModelRegistry`, and system
/// prompt to call [`create_agent_session()`](ameli_agent::create_agent_session).
///
/// # Type Parameter
///
/// `M` is the session metadata type defined by the storage backend. See
/// [`SessionMetadata`].
pub struct AgentSessionResources<M: SessionMetadata> {
    /// Session storage backend.
    pub session_manager: Arc<dyn SessionManager<M>>,
    /// API key resolution for the model's provider.
    pub auth_storage: Arc<dyn AuthStorage>,
    /// Extension instances to register with the agent.
    pub extensions: Vec<Arc<dyn Extension>>,
    /// Model selection for the session.
    pub model: ModelRef,
    /// Reasoning/thinking level for the session.
    pub thinking_level: ThinkingLevel,
}

impl<M: SessionMetadata> fmt::Debug for AgentSessionResources<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentSessionResources")
            .field(
                "session_manager",
                &format_args!("Arc<dyn SessionManager<{}>>", std::any::type_name::<M>()),
            )
            .field("auth_storage", &format_args!("Arc<dyn AuthStorage>"))
            .field("extensions", &self.extensions.len())
            .field("model", &self.model)
            .field("thinking_level", &self.thinking_level)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// MultiAgentResourceLoader trait
// ---------------------------------------------------------------------------

/// Trait for creating sessions and loading agent session resources.
///
/// Implementations map an `agent_id` to concrete `SessionManager`,
/// `AuthStorage`, `Extensions`, `ModelRef`, and `ThinkingLevel`. The consumer
/// combines these with their own `Interface`, `ModelRegistry`, and system
/// prompt.
///
/// # Type Parameter
///
/// `M` is the session metadata type defined by the storage backend.
///
/// # Object Safety
///
/// The trait is dyn-compatible so that `Arc<dyn MultiAgentResourceLoader<M>>`
/// works. All async methods return [`AsyncResult`] (boxed, pinned futures).
///
/// # Examples
///
/// ```
/// use ameli_multi_agent_resource_loader::{
///     MultiAgentResourceLoader, AgentSessionResources, AsyncResult,
/// };
/// use ameli_agent::session_manager::{SessionMetadata, SessionManager, ModelRef};
/// use std::sync::Arc;
///
/// struct MyLoader;
///
/// impl<M: SessionMetadata> MultiAgentResourceLoader<M> for MyLoader {
///     fn create_session(&self, agent_id: &str) -> AsyncResult<String, ameli_multi_agent_resource_loader::CreateSessionError> {
///         let agent_id = agent_id.to_string();
///         Box::pin(async move {
///             // Create session in storage and return session ID
///             Ok(format!("session-for-{agent_id}"))
///         })
///     }
///
///     fn load_resources(&self, session_id: &str) -> AsyncResult<AgentSessionResources<M>, ameli_multi_agent_resource_loader::LoadResourcesError> {
///         // Load resources for the session
///         todo!()
///     }
/// }
/// ```
pub trait MultiAgentResourceLoader<M: SessionMetadata>: Send + Sync {
    /// Create a new session for the given agent and return its session ID.
    ///
    /// The implementation handles all storage-level session creation (e.g.,
    /// creating a database row, initializing a file). The returned session ID
    /// can later be passed to [`load_resources`](Self::load_resources).
    ///
    /// # Errors
    ///
    /// Returns [`CreateSessionError::AgentNotFound`] if no configuration
    /// exists for the given `agent_id`, or
    /// [`CreateSessionError::CreationFailed`] / [`CreateSessionError::Storage`]
    /// for infrastructure failures.
    fn create_session(&self, agent_id: &str) -> AsyncResult<String, CreateSessionError>;

    /// Load the resources needed to construct an agent session.
    ///
    /// Returns an [`AgentSessionResources`] bundle containing the session
    /// manager, auth storage, extensions, model selection, and thinking level
    /// for the given session.
    ///
    /// # Errors
    ///
    /// Returns [`LoadResourcesError::SessionNotFound`] if the session does not
    /// exist, [`LoadResourcesError::AgentNotFound`] if the session references
    /// an unknown agent, or [`LoadResourcesError::Storage`] for infrastructure
    /// failures.
    fn load_resources(
        &self,
        session_id: &str,
    ) -> AsyncResult<AgentSessionResources<M>, LoadResourcesError>;
}

// Re-exports are handled by the `pub use` imports at the top of the file.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ameli_agent::auth_storage::InMemoryAuthStorage;
    use ameli_agent::extension::ExtensionApi;
    use ameli_agent::session_manager::InMemorySessionManager;

    // -- AgentSessionResources construction --

    #[test]
    fn resources_construction() {
        let resources = AgentSessionResources {
            session_manager: Arc::new(InMemorySessionManager::new()),
            auth_storage: Arc::new(InMemoryAuthStorage::new()),
            extensions: vec![],
            model: ModelRef {
                provider: "openai".into(),
                model_id: "gpt-4o".into(),
            },
            thinking_level: ThinkingLevel::Off,
        };

        assert_eq!(resources.model.provider, "openai");
        assert_eq!(resources.model.model_id, "gpt-4o");
        assert_eq!(resources.extensions.len(), 0);
        assert_eq!(resources.thinking_level, ThinkingLevel::Off);
    }

    // -- AgentSessionResources Debug --

    #[test]
    fn resources_debug_output() {
        let resources = AgentSessionResources {
            session_manager: Arc::new(InMemorySessionManager::new()),
            auth_storage: Arc::new(InMemoryAuthStorage::new()),
            extensions: vec![],
            model: ModelRef {
                provider: "openai".into(),
                model_id: "gpt-4o".into(),
            },
            thinking_level: ThinkingLevel::High,
        };

        let debug = format!("{resources:?}");
        assert!(debug.contains("AgentSessionResources"));
        assert!(debug.contains("openai"));
        assert!(
            debug.contains("extensions"),
            "debug should mention extensions"
        );
    }

    // -- Trait object safety --

    #[test]
    fn trait_is_object_safe() {
        let _loader: Arc<
            dyn MultiAgentResourceLoader<ameli_agent::session_manager::InMemoryMetadata>,
        > = Arc::new(MockResourceLoader);
    }

    // -- Mock implementation round-trip --

    struct NoOpExtension;

    impl ameli_agent::extension::Extension for NoOpExtension {
        fn name(&self) -> &str {
            "no-op"
        }
        fn init(&self, _api: &mut ExtensionApi) {}
    }

    struct MockResourceLoader;

    impl MultiAgentResourceLoader<ameli_agent::session_manager::InMemoryMetadata>
        for MockResourceLoader
    {
        fn create_session(&self, agent_id: &str) -> AsyncResult<String, CreateSessionError> {
            let agent_id = agent_id.to_string();
            Box::pin(async move { Ok(format!("session-{agent_id}")) })
        }

        fn load_resources(
            &self,
            session_id: &str,
        ) -> AsyncResult<
            AgentSessionResources<ameli_agent::session_manager::InMemoryMetadata>,
            LoadResourcesError,
        > {
            let session_id = session_id.to_string();
            Box::pin(async move {
                if session_id == "session-agent-1" {
                    Ok(AgentSessionResources {
                        session_manager: Arc::new(InMemorySessionManager::new()),
                        auth_storage: Arc::new(InMemoryAuthStorage::new()),
                        extensions: vec![Arc::new(NoOpExtension)],
                        model: ModelRef {
                            provider: "openai".into(),
                            model_id: "gpt-4o".into(),
                        },
                        thinking_level: ThinkingLevel::Medium,
                    })
                } else {
                    Err(LoadResourcesError::session_not_found(&session_id))
                }
            })
        }
    }

    #[tokio::test]
    async fn mock_create_session_returns_id() {
        let loader = MockResourceLoader;
        let session_id = loader.create_session("agent-1").await.unwrap();
        assert_eq!(session_id, "session-agent-1");
    }

    #[tokio::test]
    async fn mock_load_resources_round_trip() {
        let loader = MockResourceLoader;

        let session_id = loader.create_session("agent-1").await.unwrap();
        let resources = loader.load_resources(&session_id).await.unwrap();

        assert_eq!(resources.model.provider, "openai");
        assert_eq!(resources.model.model_id, "gpt-4o");
        assert_eq!(resources.extensions.len(), 1);
        assert_eq!(resources.thinking_level, ThinkingLevel::Medium);
    }

    #[tokio::test]
    async fn mock_load_resources_not_found() {
        let loader = MockResourceLoader;
        let err = loader.load_resources("nonexistent").await.unwrap_err();
        assert!(err.to_string().contains("nonexistent"));
    }

    // -- AsyncResult type alias compiles --

    #[test]
    fn async_result_compiles() {
        let _f: AsyncResult<String, CreateSessionError> =
            Box::pin(async { Ok("test".to_string()) });
    }
}
