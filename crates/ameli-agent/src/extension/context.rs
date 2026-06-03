//! Extension context passed to event handlers at runtime.
//!
//! [`ExtensionContext`] provides handlers with access to the cancellation token
//! and the UI interface. It is created by the extension runtime for each event
//! dispatch and is cheaply cloneable.
//!
//! Extensions that need runtime actions (send messages, query model, etc.)
//! should capture the `Arc<ExtensionApi>` received in [`Extension::init`]
//! within their handler closures.

use std::fmt;
use std::sync::Arc;

use crate::interface::Interface;

// ---------------------------------------------------------------------------
// ExtensionContext
// ---------------------------------------------------------------------------

/// Context passed to extension event handlers.
///
/// Created by the extension runtime per event dispatch. Lightweight and
/// cheaply cloneable. Provides the cancellation token and UI interface.
///
/// For runtime actions (send messages, query/set model, tools, etc.),
/// capture the `Arc<ExtensionApi>` from `init()` in your handler closures.
pub struct ExtensionContext {
    /// Cancellation token for the current agent run, if active.
    pub cancel_token: Option<tokio_util::sync::CancellationToken>,
    /// UI interface for output/rendering.
    interface: Arc<dyn Interface>,
}

impl ExtensionContext {
    /// Create a minimal context for testing (no-op interface, no cancel token).
    pub fn for_testing() -> Self {
        Self {
            cancel_token: None,
            interface: Arc::new(crate::interface::NoopInterface),
        }
    }

    /// Create a context with the given interface and cancellation token.
    pub(crate) fn new(
        interface: Arc<dyn Interface>,
        cancel_token: Option<tokio_util::sync::CancellationToken>,
    ) -> Self {
        Self {
            cancel_token,
            interface,
        }
    }

    /// Get the current interface.
    pub fn interface(&self) -> Arc<dyn Interface> {
        self.interface.clone()
    }
}

impl Clone for ExtensionContext {
    fn clone(&self) -> Self {
        Self {
            cancel_token: self.cancel_token.clone(),
            interface: self.interface.clone(),
        }
    }
}

impl fmt::Debug for ExtensionContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionContext")
            .field("cancel_token", &self.cancel_token)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_testing_defaults() {
        let ctx = ExtensionContext::for_testing();
        assert!(ctx.cancel_token.is_none());
    }

    #[test]
    fn clone_copies_fields() {
        let ctx = ExtensionContext::for_testing();
        let cloned = ctx.clone();
        assert_eq!(ctx.cancel_token.is_some(), cloned.cancel_token.is_some());
    }

    #[test]
    fn debug_skips_interface() {
        let ctx = ExtensionContext::for_testing();
        let debug = format!("{ctx:?}");
        assert!(debug.contains("ExtensionContext"));
    }
}
