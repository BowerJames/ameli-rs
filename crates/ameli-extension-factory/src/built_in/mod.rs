//! Built-in extension templates.
//!
//! This module provides [`register_built_in_templates`] which registers all
//! built-in templates on an [`ExtensionFactory`](crate::ExtensionFactory).
//!
//! Currently empty — built-in templates will be added here as the framework
//! grows.

use crate::ExtensionFactory;

/// Register built-in extension templates on the given factory.
///
/// Currently a no-op. Future built-in templates (e.g., `append_system_message`,
/// `tool_blocker`, `context_injection`) will be registered here.
pub fn register_built_in_templates(_factory: &ExtensionFactory) {
    // Future: register built-in templates here.
    // Example:
    // factory.register(Box::new(AppendSystemMessageTemplate));
}
