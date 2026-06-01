//! Error types for session creation and resource loading.

// ---------------------------------------------------------------------------
// CreateSessionError
// ---------------------------------------------------------------------------

/// Errors produced by [`MultiAgentResourceLoader::create_session`](crate::MultiAgentResourceLoader::create_session).
#[derive(Debug, thiserror::Error)]
pub enum CreateSessionError {
    /// No agent configuration found for the given agent ID.
    #[error("agent not found: {agent_id}")]
    AgentNotFound {
        /// The agent ID that was looked up.
        agent_id: String,
    },

    /// Session creation failed for the given agent.
    #[error("session creation failed for agent {agent_id}: {reason}")]
    CreationFailed {
        /// The agent ID for which session creation was attempted.
        agent_id: String,
        /// Human-readable reason for the failure.
        reason: String,
    },

    /// An underlying storage I/O or infrastructure error.
    #[error("storage error: {0}")]
    Storage(#[source] Box<dyn std::error::Error + Send + Sync>),
}

impl CreateSessionError {
    /// Create an [`AgentNotFound`](CreateSessionError::AgentNotFound) error.
    pub fn agent_not_found(agent_id: impl Into<String>) -> Self {
        Self::AgentNotFound {
            agent_id: agent_id.into(),
        }
    }

    /// Create a [`CreationFailed`](CreateSessionError::CreationFailed) error.
    pub fn creation_failed(agent_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::CreationFailed {
            agent_id: agent_id.into(),
            reason: reason.into(),
        }
    }

    /// Create a [`Storage`](CreateSessionError::Storage) error wrapping an
    /// underlying failure.
    pub fn storage(err: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Storage(Box::new(err))
    }
}

// ---------------------------------------------------------------------------
// LoadResourcesError
// ---------------------------------------------------------------------------

/// Errors produced by [`MultiAgentResourceLoader::load_resources`](crate::MultiAgentResourceLoader::load_resources).
#[derive(Debug, thiserror::Error)]
pub enum LoadResourcesError {
    /// No session found for the given session ID.
    #[error("session not found: {session_id}")]
    SessionNotFound {
        /// The session ID that was looked up.
        session_id: String,
    },

    /// No agent configuration found for the agent associated with the session.
    #[error("agent not found: {agent_id}")]
    AgentNotFound {
        /// The agent ID that was resolved from the session.
        agent_id: String,
    },

    /// Resource loading failed for the given session.
    #[error("failed to load resources for session {session_id}: {reason}")]
    LoadFailed {
        /// The session ID for which loading was attempted.
        session_id: String,
        /// Human-readable reason for the failure.
        reason: String,
    },

    /// An underlying storage I/O or infrastructure error.
    #[error("storage error: {0}")]
    Storage(#[source] Box<dyn std::error::Error + Send + Sync>),
}

impl LoadResourcesError {
    /// Create a [`SessionNotFound`](LoadResourcesError::SessionNotFound) error.
    pub fn session_not_found(session_id: impl Into<String>) -> Self {
        Self::SessionNotFound {
            session_id: session_id.into(),
        }
    }

    /// Create an [`AgentNotFound`](LoadResourcesError::AgentNotFound) error.
    pub fn agent_not_found(agent_id: impl Into<String>) -> Self {
        Self::AgentNotFound {
            agent_id: agent_id.into(),
        }
    }

    /// Create a [`LoadFailed`](LoadResourcesError::LoadFailed) error.
    pub fn load_failed(session_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::LoadFailed {
            session_id: session_id.into(),
            reason: reason.into(),
        }
    }

    /// Create a [`Storage`](LoadResourcesError::Storage) error wrapping an
    /// underlying failure.
    pub fn storage(err: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Storage(Box::new(err))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- CreateSessionError display --

    #[test]
    fn create_agent_not_found_display() {
        let err = CreateSessionError::agent_not_found("agent-1");
        assert_eq!(err.to_string(), "agent not found: agent-1");
    }

    #[test]
    fn create_creation_failed_display() {
        let err = CreateSessionError::creation_failed("agent-2", "db connection refused");
        assert_eq!(
            err.to_string(),
            "session creation failed for agent agent-2: db connection refused"
        );
    }

    #[test]
    fn create_storage_display() {
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broke");
        let err = CreateSessionError::storage(io_err);
        assert!(err.to_string().contains("pipe broke"));
    }

    // -- CreateSessionError helpers --

    #[test]
    fn create_agent_not_found_fields() {
        let err = CreateSessionError::agent_not_found("my-agent");
        match err {
            CreateSessionError::AgentNotFound { agent_id } => assert_eq!(agent_id, "my-agent"),
            _ => panic!("expected AgentNotFound variant"),
        }
    }

    #[test]
    fn create_creation_failed_fields() {
        let err = CreateSessionError::creation_failed("a1", "timeout");
        match err {
            CreateSessionError::CreationFailed { agent_id, reason } => {
                assert_eq!(agent_id, "a1");
                assert_eq!(reason, "timeout");
            }
            _ => panic!("expected CreationFailed variant"),
        }
    }

    #[test]
    fn create_storage_source_preserved() {
        use std::error::Error;
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broke");
        let err = CreateSessionError::storage(io_err);
        assert!(err.source().is_some());
    }

    // -- LoadResourcesError display --

    #[test]
    fn load_session_not_found_display() {
        let err = LoadResourcesError::session_not_found("sess-1");
        assert_eq!(err.to_string(), "session not found: sess-1");
    }

    #[test]
    fn load_agent_not_found_display() {
        let err = LoadResourcesError::agent_not_found("agent-x");
        assert_eq!(err.to_string(), "agent not found: agent-x");
    }

    #[test]
    fn load_failed_display() {
        let err = LoadResourcesError::load_failed("sess-3", "config missing");
        assert_eq!(
            err.to_string(),
            "failed to load resources for session sess-3: config missing"
        );
    }

    #[test]
    fn load_storage_display() {
        let io_err = std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout");
        let err = LoadResourcesError::storage(io_err);
        assert!(err.to_string().contains("timeout"));
    }

    // -- LoadResourcesError helpers --

    #[test]
    fn load_session_not_found_fields() {
        let err = LoadResourcesError::session_not_found("s1");
        match err {
            LoadResourcesError::SessionNotFound { session_id } => {
                assert_eq!(session_id, "s1");
            }
            _ => panic!("expected SessionNotFound variant"),
        }
    }

    #[test]
    fn load_agent_not_found_fields() {
        let err = LoadResourcesError::agent_not_found("a1");
        match err {
            LoadResourcesError::AgentNotFound { agent_id } => {
                assert_eq!(agent_id, "a1");
            }
            _ => panic!("expected AgentNotFound variant"),
        }
    }

    #[test]
    fn load_failed_fields() {
        let err = LoadResourcesError::load_failed("s2", "corrupt");
        match err {
            LoadResourcesError::LoadFailed { session_id, reason } => {
                assert_eq!(session_id, "s2");
                assert_eq!(reason, "corrupt");
            }
            _ => panic!("expected LoadFailed variant"),
        }
    }

    #[test]
    fn load_storage_source_preserved() {
        use std::error::Error;
        let io_err = std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout");
        let err = LoadResourcesError::storage(io_err);
        assert!(err.source().is_some());
    }
}
