//! Extension action trait and supporting types.
//!
//! [`ExtensionActions`] defines the runtime operations that the
//! [`ExtensionApi`](super::ExtensionApi) delegates to. A concrete implementation
//! (e.g. `SessionActions` in `agent_session.rs`) wires these to the real agent,
//! session manager, and auth storage.
//!
//! Extensions never interact with this trait directly — they call methods on
//! [`ExtensionApi`](super::ExtensionApi), which delegates here.

use ameli_agent_core::types::{AgentMessage, ThinkingLevel};
use ameli_ai::types::{ImageContent, Model};
use std::fmt;
use std::future::Future;
use std::pin::Pin;

// ---------------------------------------------------------------------------
// Type alias
// ---------------------------------------------------------------------------

/// Boxed, sendable async result.
pub type AsyncResult<T, E> = Pin<Box<dyn Future<Output = Result<T, E>> + Send>>;

// ---------------------------------------------------------------------------
// MessageDelivery
// ---------------------------------------------------------------------------

/// How a message should be delivered to the agent.
///
/// - [`Steer`](MessageDelivery::Steer): injected during the current agent run,
///   after the current assistant turn finishes.
/// - [`FollowUp`](MessageDelivery::FollowUp): injected when the agent would
///   otherwise stop, keeping the run going.
/// - [`NextTurn`](MessageDelivery::NextTurn): held until the next explicit user
///   prompt, then injected alongside it before `before_agent_start` runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageDelivery {
    Steer,
    FollowUp,
    NextTurn,
}

impl fmt::Display for MessageDelivery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Steer => write!(f, "steer"),
            Self::FollowUp => write!(f, "followUp"),
            Self::NextTurn => write!(f, "nextTurn"),
        }
    }
}

// ---------------------------------------------------------------------------
// ToolInfo
// ---------------------------------------------------------------------------

/// Metadata about a registered tool, for runtime introspection by extensions.
#[derive(Debug, Clone)]
pub struct ToolInfo {
    /// Tool name (used in LLM tool calls).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema for the tool's parameters.
    pub parameters: serde_json::Value,
}

// ---------------------------------------------------------------------------
// ExtensionActionError
// ---------------------------------------------------------------------------

/// Error returned by extension action methods.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ExtensionActionError {
    /// The agent is currently busy and cannot perform the action immediately.
    #[error("agent is busy")]
    AgentBusy,
    /// A storage operation failed.
    #[error("{0}")]
    StorageError(String),
    /// The action backend has not been fully initialized yet.
    #[error("action backend not initialized")]
    NotInitialized,
}

// ---------------------------------------------------------------------------
// ExtensionActions trait
// ---------------------------------------------------------------------------

/// Runtime actions that the [`ExtensionApi`](super::ExtensionApi) delegates to.
///
/// A concrete implementation is created by the session layer and wired into
/// the API before any extensions are initialized. Extensions never see this
/// trait directly — they call methods on `ExtensionApi`.
///
/// # Object safety
///
/// The trait is object-safe so it can be stored as `Arc<dyn ExtensionActions>`.
/// Async methods return [`AsyncResult`] (boxed, pinned futures).
pub trait ExtensionActions: Send + Sync {
    /// Inject a message into the agent with the specified delivery mode.
    fn send_message(
        &self,
        msg: AgentMessage,
        delivery: MessageDelivery,
    ) -> AsyncResult<(), ExtensionActionError>;

    /// Inject a user message (text + optional images) with the specified
    /// delivery mode.
    fn send_user_message(
        &self,
        text: String,
        images: Vec<ImageContent>,
        delivery: MessageDelivery,
    ) -> AsyncResult<(), ExtensionActionError>;

    /// Persist a custom entry to the session for state persistence.
    /// Not sent to the LLM.
    fn append_entry(
        &self,
        custom_type: &str,
        data: Option<serde_json::Value>,
    ) -> AsyncResult<(), ExtensionActionError>;

    /// Get the names of currently active tools.
    fn get_active_tools(&self) -> AsyncResult<Vec<String>, ExtensionActionError>;

    /// Get metadata for all registered tools (active or not).
    fn get_all_tools(&self) -> Vec<ToolInfo>;

    /// Set which tools are active by name. Takes immediate effect.
    fn set_active_tools(&self, names: Vec<String>) -> AsyncResult<(), ExtensionActionError>;

    /// Get the current model, if set.
    fn model(&self) -> AsyncResult<Option<Model>, ExtensionActionError>;

    /// Switch the model. Returns `Ok(false)` if no API key is available.
    fn set_model(&self, model: Model) -> AsyncResult<bool, ExtensionActionError>;

    /// Get the current thinking level.
    fn get_thinking_level(&self) -> AsyncResult<ThinkingLevel, ExtensionActionError>;

    /// Set the thinking level.
    fn set_thinking_level(&self, level: ThinkingLevel) -> AsyncResult<(), ExtensionActionError>;

    /// Get the current system prompt.
    fn get_system_prompt(&self) -> AsyncResult<String, ExtensionActionError>;

    /// Whether there are queued messages waiting.
    fn has_pending_messages(&self) -> AsyncResult<bool, ExtensionActionError>;

    /// Abort the current agent operation.
    ///
    /// This is fire-and-forget: the abort request is spawned as a detached
    /// tokio task and there is no way to confirm whether the abort succeeded.
    /// This is intentional — abort should be non-blocking and immediately
    /// return control to the caller.
    fn abort(&self);

    /// Whether the agent is currently idle (not streaming).
    fn is_idle(&self) -> AsyncResult<bool, ExtensionActionError>;
}

/// No-op implementation of [`ExtensionActions`] for testing and defaults.
pub struct NoopExtensionActions;

impl ExtensionActions for NoopExtensionActions {
    fn send_message(
        &self,
        _msg: AgentMessage,
        _delivery: MessageDelivery,
    ) -> AsyncResult<(), ExtensionActionError> {
        Box::pin(async { Ok(()) })
    }

    fn send_user_message(
        &self,
        _text: String,
        _images: Vec<ImageContent>,
        _delivery: MessageDelivery,
    ) -> AsyncResult<(), ExtensionActionError> {
        Box::pin(async { Ok(()) })
    }

    fn append_entry(
        &self,
        _custom_type: &str,
        _data: Option<serde_json::Value>,
    ) -> AsyncResult<(), ExtensionActionError> {
        Box::pin(async { Ok(()) })
    }

    fn get_active_tools(&self) -> AsyncResult<Vec<String>, ExtensionActionError> {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn get_all_tools(&self) -> Vec<ToolInfo> {
        Vec::new()
    }

    fn set_active_tools(&self, _names: Vec<String>) -> AsyncResult<(), ExtensionActionError> {
        Box::pin(async { Ok(()) })
    }

    fn model(&self) -> AsyncResult<Option<Model>, ExtensionActionError> {
        Box::pin(async { Ok(None) })
    }

    fn set_model(&self, _model: Model) -> AsyncResult<bool, ExtensionActionError> {
        Box::pin(async { Ok(true) })
    }

    fn get_thinking_level(&self) -> AsyncResult<ThinkingLevel, ExtensionActionError> {
        Box::pin(async { Ok(ThinkingLevel::Off) })
    }

    fn set_thinking_level(&self, _level: ThinkingLevel) -> AsyncResult<(), ExtensionActionError> {
        Box::pin(async { Ok(()) })
    }

    fn get_system_prompt(&self) -> AsyncResult<String, ExtensionActionError> {
        Box::pin(async { Ok(String::new()) })
    }

    fn has_pending_messages(&self) -> AsyncResult<bool, ExtensionActionError> {
        Box::pin(async { Ok(false) })
    }

    fn abort(&self) {}

    fn is_idle(&self) -> AsyncResult<bool, ExtensionActionError> {
        Box::pin(async { Ok(true) })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_delivery_display() {
        assert_eq!(MessageDelivery::Steer.to_string(), "steer");
        assert_eq!(MessageDelivery::FollowUp.to_string(), "followUp");
        assert_eq!(MessageDelivery::NextTurn.to_string(), "nextTurn");
    }

    #[test]
    fn message_delivery_copy() {
        let a = MessageDelivery::Steer;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn tool_info_construction() {
        let info = ToolInfo {
            name: "echo".into(),
            description: "Echoes input".into(),
            parameters: serde_json::json!({"type": "object"}),
        };
        assert_eq!(info.name, "echo");
        assert_eq!(info.description, "Echoes input");
    }

    #[test]
    fn extension_action_error_display() {
        let err = ExtensionActionError::AgentBusy;
        assert_eq!(format!("{err}"), "agent is busy");

        let err = ExtensionActionError::StorageError("disk full".into());
        assert_eq!(format!("{err}"), "disk full");

        let err = ExtensionActionError::NotInitialized;
        assert_eq!(format!("{err}"), "action backend not initialized");
    }

    #[test]
    fn async_result_compiles() {
        let _f: AsyncResult<(), ExtensionActionError> = Box::pin(async { Ok(()) });
    }
}
