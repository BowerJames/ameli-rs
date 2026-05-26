//! Extension context passed to event handlers at runtime.
//!
//! [`ExtensionContext`] provides handlers with access to the current agent
//! state. It is created by the extension runtime for each event dispatch and
//! is deliberately lightweight — fields are cloned from the runtime's current
//! state.

use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// ExtensionContext
// ---------------------------------------------------------------------------

/// Context passed to extension event handlers.
///
/// Created by the extension runtime per event dispatch. Lightweight and
/// cheaply cloneable. Will expand as more infrastructure is added (session
/// access, model info, etc.).
#[derive(Debug, Clone)]
pub struct ExtensionContext {
    /// Whether the agent is currently idle (not streaming/processing).
    pub is_idle: bool,
    /// Cancellation token for the current agent run, if active.
    pub cancel_token: Option<CancellationToken>,
}

impl ExtensionContext {
    /// Create a minimal context for testing.
    pub fn for_testing() -> Self {
        Self {
            is_idle: true,
            cancel_token: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_testing_is_idle() {
        let ctx = ExtensionContext::for_testing();
        assert!(ctx.is_idle);
        assert!(ctx.cancel_token.is_none());
    }
}
