//! Session error types.
//!
//! Defines [`SessionError`] — the domain-specific error enum for session
//! operations (storage, context building, tree manipulation).

/// Errors produced by session storage, context building, and tree operations.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// A requested session or entry could not be found.
    #[error("session not found: {0}")]
    NotFound(String),

    /// A session file or data structure is malformed or corrupted.
    #[error("invalid session: {0}")]
    InvalidSession(String),

    /// A session entry is malformed or cannot be processed.
    #[error("invalid entry: {0}")]
    InvalidEntry(String),

    /// A fork operation targets an entry that cannot be forked.
    #[error("invalid fork target: {0}")]
    InvalidForkTarget(String),

    /// An underlying storage I/O or serialization error.
    #[error("storage error: {0}")]
    Storage(#[source] Box<dyn std::error::Error + Send + Sync>),
}

impl SessionError {
    /// Create a [`NotFound`](SessionError::NotFound) error.
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }

    /// Create an [`InvalidSession`](SessionError::InvalidSession) error.
    pub fn invalid_session(msg: impl Into<String>) -> Self {
        Self::InvalidSession(msg.into())
    }

    /// Create an [`InvalidEntry`](SessionError::InvalidEntry) error.
    pub fn invalid_entry(msg: impl Into<String>) -> Self {
        Self::InvalidEntry(msg.into())
    }

    /// Create an [`InvalidForkTarget`](SessionError::InvalidForkTarget) error.
    pub fn invalid_fork_target(msg: impl Into<String>) -> Self {
        Self::InvalidForkTarget(msg.into())
    }

    /// Create a [`Storage`](SessionError::Storage) error wrapping an underlying failure.
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

    #[test]
    fn error_display_messages() {
        assert_eq!(
            SessionError::not_found("abc").to_string(),
            "session not found: abc"
        );
        assert_eq!(
            SessionError::invalid_session("bad header").to_string(),
            "invalid session: bad header"
        );
        assert_eq!(
            SessionError::invalid_entry("line 5").to_string(),
            "invalid entry: line 5"
        );
        assert_eq!(
            SessionError::invalid_fork_target("root").to_string(),
            "invalid fork target: root"
        );
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broke");
        assert!(SessionError::storage(io_err)
            .to_string()
            .contains("pipe broke"));
    }

    #[test]
    fn storage_source_is_preserved() {
        use std::error::Error;
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broke");
        let session_err = SessionError::storage(io_err);
        assert!(session_err.source().is_some());
    }
}
