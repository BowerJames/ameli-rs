//! Agent session creation errors.
//!
//! Defines [`CreateAgentSessionError`] for failures during
//! [`create_agent_session`](crate::create_agent_session).

use crate::session_manager::SessionError;

// ---------------------------------------------------------------------------
// CreateAgentSessionError
// ---------------------------------------------------------------------------

/// Errors produced by [`create_agent_session`](crate::create_agent_session).
///
/// Covers model resolution failures, missing API keys, and session storage
/// errors encountered while constructing a fully loaded [`AgentSession`](crate::AgentSession).
#[derive(Debug, thiserror::Error)]
pub enum CreateAgentSessionError {
    /// The requested model was not found in the model registry.
    #[error("model not found: {0}")]
    ModelNotFound(#[from] ameli_model_registry::ModelNotFoundError),

    /// No API key was available for the requested provider.
    #[error("no API key found for provider: {provider}")]
    ApiKeyNotFound {
        /// Provider name that had no key.
        provider: String,
    },

    /// A session storage error occurred during context restoration or
    /// initial state persistence.
    #[error("session error: {0}")]
    Session(#[from] SessionError),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_agent_session_error_display() {
        let err = CreateAgentSessionError::ApiKeyNotFound {
            provider: "openai".to_string(),
        };
        assert_eq!(err.to_string(), "no API key found for provider: openai");
    }

    #[test]
    fn session_error_source_preserved() {
        let session_err = SessionError::not_found("abc");
        let create_err = CreateAgentSessionError::Session(session_err);
        assert_eq!(
            create_err.to_string(),
            "session error: session not found: abc"
        );
    }
}
