//! Extension context passed to event handlers at runtime.
//!
//! [`ExtensionContext`] provides handlers with access to the current agent
//! state and user interface. It is created by the extension runtime for each
//! event dispatch and is deliberately lightweight — fields are cloned from
//! the runtime's current state.

use crate::interface::{Interface, NoopInterface};
use std::fmt;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// ExtensionContext
// ---------------------------------------------------------------------------

/// Context passed to extension event handlers.
///
/// Created by the extension runtime per event dispatch. Lightweight and
/// cheaply cloneable. Will expand as more infrastructure is added (session
/// access, model info, etc.).
pub struct ExtensionContext {
    /// Whether the agent is currently idle (not streaming/processing).
    pub is_idle: bool,
    /// Cancellation token for the current agent run, if active.
    pub cancel_token: Option<CancellationToken>,
    /// UI interface for user interaction.
    pub interface: Arc<dyn Interface>,
}

impl ExtensionContext {
    /// Create a minimal context for testing (no-op interface).
    pub fn for_testing() -> Self {
        Self {
            is_idle: true,
            cancel_token: None,
            interface: Arc::new(NoopInterface),
        }
    }
}

impl Clone for ExtensionContext {
    fn clone(&self) -> Self {
        Self {
            is_idle: self.is_idle,
            cancel_token: self.cancel_token.clone(),
            interface: self.interface.clone(),
        }
    }
}

impl fmt::Debug for ExtensionContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionContext")
            .field("is_idle", &self.is_idle)
            .field("cancel_token", &self.cancel_token)
            .field("interface", &"<Interface>")
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
        assert!(ctx.is_idle);
        assert!(ctx.cancel_token.is_none());
        // Interface is present (NoopInterface)
        ctx.interface
            .notify(crate::interface::NotifyMessage::info("test"));
    }

    #[test]
    fn clone_copies_fields() {
        let ctx = ExtensionContext::for_testing();
        let cloned = ctx.clone();
        assert_eq!(cloned.is_idle, ctx.is_idle);
        // Both share the same Arc
        assert!(Arc::ptr_eq(&cloned.interface, &ctx.interface));
    }

    #[test]
    fn debug_skips_interface() {
        let ctx = ExtensionContext::for_testing();
        let debug = format!("{ctx:?}");
        assert!(debug.contains("ExtensionContext"));
        assert!(debug.contains("<Interface>"));
        assert!(!debug.contains("NoopInterface"));
    }
}
