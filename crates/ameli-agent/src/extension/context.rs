//! Extension context passed to event handlers at runtime.
//!
//! [`ExtensionContext`] provides handlers with access to the extension API
//! (for runtime actions), the cancellation token, and the UI interface. It is
//! created by the extension runtime for each event dispatch and is cheaply
//! cloneable.

use crate::extension::actions::{AsyncResult, ExtensionActionError, MessageDelivery, ToolInfo};
use crate::extension::ExtensionApi;
use crate::interface::Interface;
use ameli_agent_core::types::AgentMessage;
use ameli_ai::types::{ImageContent, Model};
use std::fmt;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// ExtensionContext
// ---------------------------------------------------------------------------

/// Context passed to extension event handlers.
///
/// Created by the extension runtime per event dispatch. Lightweight and
/// cheaply cloneable. Provides access to the shared [`ExtensionApi`] for
/// runtime actions and convenience delegating methods.
pub struct ExtensionContext {
    /// Cancellation token for the current agent run, if active.
    pub cancel_token: Option<tokio_util::sync::CancellationToken>,
    /// Shared extension API for runtime actions.
    api: Arc<ExtensionApi>,
}

impl ExtensionContext {
    /// Create a minimal context for testing (no-op interface, no-op actions).
    pub fn for_testing() -> Self {
        Self {
            cancel_token: None,
            api: Arc::new(ExtensionApi::new(Arc::new(
                crate::extension::NoopExtensionActions,
            ))),
        }
    }

    /// Create a context with the given API and cancellation token.
    pub(crate) fn new(
        api: Arc<ExtensionApi>,
        cancel_token: Option<tokio_util::sync::CancellationToken>,
    ) -> Self {
        Self { cancel_token, api }
    }

    /// Get a reference to the shared extension API.
    pub fn api(&self) -> &Arc<ExtensionApi> {
        &self.api
    }

    /// Get the current interface.
    pub fn interface(&self) -> Arc<dyn Interface> {
        self.api
            .get_interface()
            .unwrap_or_else(|| Arc::new(crate::interface::NoopInterface))
    }

    // -------------------------------------------------------------------
    // Convenience delegating methods
    // -------------------------------------------------------------------

    /// Get the current model.
    pub fn model(&self) -> Option<Model> {
        self.api.model()
    }

    /// Inject a message into the agent.
    pub fn send_message(
        &self,
        msg: AgentMessage,
        delivery: MessageDelivery,
    ) -> AsyncResult<(), ExtensionActionError> {
        self.api.send_message(msg, delivery)
    }

    /// Inject a user message into the agent.
    pub fn send_user_message(
        &self,
        text: String,
        images: Vec<ImageContent>,
        delivery: MessageDelivery,
    ) -> AsyncResult<(), ExtensionActionError> {
        self.api.send_user_message(text, images, delivery)
    }

    /// Persist a custom entry to the session.
    pub fn append_entry(
        &self,
        custom_type: &str,
        data: Option<serde_json::Value>,
    ) -> AsyncResult<(), ExtensionActionError> {
        self.api.append_entry(custom_type, data)
    }

    /// Get the names of currently active tools.
    pub fn get_active_tools(&self) -> Vec<String> {
        self.api.get_active_tools()
    }

    /// Get metadata for all registered tools.
    pub fn get_all_tools(&self) -> Vec<ToolInfo> {
        self.api.get_all_tools()
    }

    /// Dynamically change which tools are active.
    pub fn set_active_tools(&self, names: Vec<String>) {
        self.api.set_active_tools(names);
    }

    /// Switch the model at runtime.
    pub fn set_model(&self, model: Model) -> AsyncResult<bool, ExtensionActionError> {
        self.api.set_model(model)
    }

    /// Get the current thinking level.
    pub fn get_thinking_level(&self) -> ameli_agent_core::types::ThinkingLevel {
        self.api.get_thinking_level()
    }

    /// Set the thinking level.
    pub fn set_thinking_level(&self, level: ameli_agent_core::types::ThinkingLevel) {
        self.api.set_thinking_level(level);
    }

    /// Get the current system prompt.
    pub fn get_system_prompt(&self) -> String {
        self.api.get_system_prompt()
    }

    /// Whether there are queued messages waiting.
    pub fn has_pending_messages(&self) -> bool {
        self.api.has_pending_messages()
    }

    /// Abort the current agent operation.
    pub fn abort(&self) {
        self.api.abort();
    }

    /// Whether the agent is currently idle.
    pub fn is_idle(&self) -> bool {
        self.api.is_idle()
    }
}

impl Clone for ExtensionContext {
    fn clone(&self) -> Self {
        Self {
            cancel_token: self.cancel_token.clone(),
            api: self.api.clone(),
        }
    }
}

impl fmt::Debug for ExtensionContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionContext")
            .field("cancel_token", &self.cancel_token)
            .field("api", &"<ExtensionApi>")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ameli_agent_core::types::ThinkingLevel;

    #[test]
    fn for_testing_defaults() {
        let ctx = ExtensionContext::for_testing();
        assert!(ctx.cancel_token.is_none());
        assert!(ctx.is_idle());
        assert!(!ctx.has_pending_messages());
    }

    #[test]
    fn clone_copies_fields() {
        let ctx = ExtensionContext::for_testing();
        let cloned = ctx.clone();
        assert_eq!(cloned.is_idle(), ctx.is_idle());
        // Both share the same Arc
        assert!(Arc::ptr_eq(&ctx.api, &cloned.api));
    }

    #[test]
    fn debug_skips_api() {
        let ctx = ExtensionContext::for_testing();
        let debug = format!("{ctx:?}");
        assert!(debug.contains("ExtensionContext"));
        assert!(debug.contains("<ExtensionApi>"));
    }

    #[test]
    fn convenience_methods_delegate() {
        let ctx = ExtensionContext::for_testing();
        assert!(ctx.model().is_none());
        assert!(ctx.get_active_tools().is_empty());
        assert!(ctx.get_all_tools().is_empty());
        assert_eq!(ctx.get_thinking_level(), ThinkingLevel::Off);
        assert!(ctx.get_system_prompt().is_empty());
    }
}
