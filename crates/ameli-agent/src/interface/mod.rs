//! Interface trait — abstraction for the user interface exposed to extensions.
//!
//! Provides a minimal surface for extensions to communicate with the user.
//! Implementations vary by mode:
//! - [`NoopInterface`] — headless/print mode (silently discarded)
//! - Future: `RpcInterface` — forwards to RPC client
//! - Future: `TuiInterface` — interactive terminal
//!
//! # Extensibility
//!
//! [`NotifyMessage`] mirrors the [`AgentMessage`](ameli_agent_core::types::AgentMessage)
//! pattern: a [`Text`](NotifyMessage::Text) variant for built-in use and a
//! [`Custom`](NotifyMessage::Custom) variant via the [`CustomNotifyMessage`] trait
//! so downstream packages can define structured notification types.

use serde_json::Value;
use std::fmt;

// ---------------------------------------------------------------------------
// NotifyKind
// ---------------------------------------------------------------------------

/// Kind of notification for the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotifyKind {
    Info,
    Warning,
    Error,
}

// ---------------------------------------------------------------------------
// CustomNotifyMessage
// ---------------------------------------------------------------------------

/// Trait for custom notification types that extend [`NotifyMessage`].
///
/// Downstream packages implement this trait to define structured notification
/// types, mirroring how [`CustomAgentMessage`] extends [`AgentMessage`].
///
/// [`CustomAgentMessage`]: ameli_agent_core::types::CustomAgentMessage
/// [`AgentMessage`]: ameli_agent_core::types::AgentMessage
///
/// # Examples
///
/// ```
/// use ameli_agent::interface::CustomNotifyMessage;
/// use serde_json::json;
/// use std::fmt;
///
/// #[derive(Clone)]
/// struct ProgressNotify {
///     percent: u8,
///     label: String,
/// }
///
/// impl CustomNotifyMessage for ProgressNotify {
///     fn message_type(&self) -> &str { "progress" }
///     fn clone_boxed(&self) -> Box<dyn CustomNotifyMessage> {
///         Box::new(self.clone())
///     }
///     fn to_json(&self) -> serde_json::Value {
///         json!({ "percent": self.percent, "label": self.label })
///     }
///     fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         f.debug_struct("ProgressNotify")
///             .field("percent", &self.percent)
///             .field("label", &self.label)
///             .finish()
///     }
/// }
/// ```
pub trait CustomNotifyMessage: Send + Sync {
    /// Discriminant for the custom notification type (display/logging).
    fn message_type(&self) -> &str;

    /// Clone into a boxed trait object.
    fn clone_boxed(&self) -> Box<dyn CustomNotifyMessage>;

    /// Serialize the custom notification for persistence/debugging.
    fn to_json(&self) -> Value;

    /// Format the custom notification for debugging.
    fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl fmt::Debug for dyn CustomNotifyMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_debug(f)
    }
}

// ---------------------------------------------------------------------------
// NotifyMessage
// ---------------------------------------------------------------------------

/// A notification to show the user.
///
/// Union of a built-in text notification and extensible custom notifications
/// via [`CustomNotifyMessage`]. Mirrors the
/// [`AgentMessage`](ameli_agent_core::types::AgentMessage) pattern.
pub enum NotifyMessage {
    /// Plain text notification with a severity kind.
    Text { message: String, kind: NotifyKind },
    /// Custom notification from a downstream package.
    Custom(Box<dyn CustomNotifyMessage>),
}

impl NotifyMessage {
    /// Create an info-level text notification.
    pub fn info(message: impl Into<String>) -> Self {
        Self::Text {
            message: message.into(),
            kind: NotifyKind::Info,
        }
    }

    /// Create a warning-level text notification.
    pub fn warning(message: impl Into<String>) -> Self {
        Self::Text {
            message: message.into(),
            kind: NotifyKind::Warning,
        }
    }

    /// Create an error-level text notification.
    pub fn error(message: impl Into<String>) -> Self {
        Self::Text {
            message: message.into(),
            kind: NotifyKind::Error,
        }
    }
}

impl Clone for NotifyMessage {
    fn clone(&self) -> Self {
        match self {
            Self::Text { message, kind } => Self::Text {
                message: message.clone(),
                kind: *kind,
            },
            Self::Custom(m) => Self::Custom(m.clone_boxed()),
        }
    }
}

impl fmt::Debug for NotifyMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text { message, kind } => f
                .debug_struct("Text")
                .field("message", message)
                .field("kind", kind)
                .finish(),
            Self::Custom(m) => f.debug_tuple("Custom").field(m).finish(),
        }
    }
}

// ---------------------------------------------------------------------------
// Interface trait
// ---------------------------------------------------------------------------

/// Trait abstracting the user interface for ameli-agent extensions.
///
/// Implementations provide mode-specific behavior. The extension runner holds
/// an `Arc<dyn Interface>` and injects it into [`ExtensionContext`](crate::ExtensionContext).
///
/// # Fire-and-forget
///
/// [`notify`](Interface::notify) is synchronous. Implementations that need
/// to perform async work (e.g., rendering) should spawn an internal task.
/// The caller is never blocked waiting for the notification to be displayed.
pub trait Interface: Send + Sync + 'static {
    /// Show a notification to the user. Fire-and-forget.
    fn notify(&self, message: NotifyMessage);
}

// ---------------------------------------------------------------------------
// NoopInterface
// ---------------------------------------------------------------------------

/// No-op implementation for headless/print mode.
///
/// All notifications are silently discarded.
pub struct NoopInterface;

impl Interface for NoopInterface {
    fn notify(&self, _message: NotifyMessage) {
        // Silently discarded
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    // -- NotifyKind --

    #[test]
    fn notify_kind_traits() {
        let kind = NotifyKind::Warning;
        let cloned = kind;
        assert_eq!(kind, cloned);
        assert_eq!(format!("{kind:?}"), "Warning");
    }

    // -- NotifyMessage constructors --

    #[test]
    fn info_constructor() {
        let msg = NotifyMessage::info("hello");
        match &msg {
            NotifyMessage::Text { message, kind } => {
                assert_eq!(message, "hello");
                assert_eq!(*kind, NotifyKind::Info);
            }
            NotifyMessage::Custom(_) => panic!("expected Text variant"),
        }
    }

    #[test]
    fn warning_constructor() {
        let msg = NotifyMessage::warning("careful");
        match &msg {
            NotifyMessage::Text { message, kind } => {
                assert_eq!(message, "careful");
                assert_eq!(*kind, NotifyKind::Warning);
            }
            NotifyMessage::Custom(_) => panic!("expected Text variant"),
        }
    }

    #[test]
    fn error_constructor() {
        let msg = NotifyMessage::error("broken");
        match &msg {
            NotifyMessage::Text { message, kind } => {
                assert_eq!(message, "broken");
                assert_eq!(*kind, NotifyKind::Error);
            }
            NotifyMessage::Custom(_) => panic!("expected Text variant"),
        }
    }

    // -- NotifyMessage Clone/Debug --

    #[test]
    fn text_clone() {
        let msg = NotifyMessage::info("original");
        let cloned = msg.clone();
        match (&msg, &cloned) {
            (
                NotifyMessage::Text {
                    message: a,
                    kind: ka,
                },
                NotifyMessage::Text {
                    message: b,
                    kind: kb,
                },
            ) => {
                assert_eq!(a, b);
                assert_eq!(ka, kb);
            }
            _ => panic!("both should be Text"),
        }
    }

    #[test]
    fn text_debug_format() {
        let msg = NotifyMessage::error("fail");
        let debug = format!("{msg:?}");
        assert!(debug.contains("Text"));
        assert!(debug.contains("fail"));
        assert!(debug.contains("Error"));
    }

    // -- CustomNotifyMessage --

    #[derive(Clone)]
    struct ProgressNotify {
        percent: u8,
        label: String,
    }

    impl CustomNotifyMessage for ProgressNotify {
        fn message_type(&self) -> &str {
            "progress"
        }
        fn clone_boxed(&self) -> Box<dyn CustomNotifyMessage> {
            Box::new(self.clone())
        }
        fn to_json(&self) -> Value {
            serde_json::json!({
                "percent": self.percent,
                "label": self.label,
            })
        }
        fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("ProgressNotify")
                .field("percent", &self.percent)
                .field("label", &self.label)
                .finish()
        }
    }

    #[test]
    fn custom_message_type() {
        let notify = ProgressNotify {
            percent: 75,
            label: "Uploading".into(),
        };
        assert_eq!(notify.message_type(), "progress");
    }

    #[test]
    fn custom_to_json() {
        let notify = ProgressNotify {
            percent: 50,
            label: "Loading".into(),
        };
        let json = notify.to_json();
        assert_eq!(json["percent"], 50);
        assert_eq!(json["label"], "Loading");
    }

    #[test]
    fn custom_clone_boxed() {
        let notify = ProgressNotify {
            percent: 10,
            label: "test".into(),
        };
        let boxed: Box<dyn CustomNotifyMessage> = notify.clone_boxed();
        assert_eq!(boxed.message_type(), "progress");
    }

    #[test]
    fn custom_debug() {
        let notify = ProgressNotify {
            percent: 42,
            label: "work".into(),
        };
        // Test via the dyn CustomNotifyMessage Debug impl (uses fmt_debug)
        let dyn_ref: &dyn CustomNotifyMessage = &notify;
        let debug = format!("{dyn_ref:?}");
        assert!(debug.contains("ProgressNotify"));
        assert!(debug.contains("42"));
        assert!(debug.contains("work"));
    }

    #[test]
    fn custom_notify_message_in_enum() {
        let msg = NotifyMessage::Custom(Box::new(ProgressNotify {
            percent: 90,
            label: "Almost done".into(),
        }));
        match &msg {
            NotifyMessage::Custom(c) => assert_eq!(c.message_type(), "progress"),
            NotifyMessage::Text { .. } => panic!("expected Custom variant"),
        }
    }

    #[test]
    fn custom_notify_clone() {
        let msg = NotifyMessage::Custom(Box::new(ProgressNotify {
            percent: 30,
            label: "Cloning".into(),
        }));
        let cloned = msg.clone();
        match (&msg, &cloned) {
            (NotifyMessage::Custom(a), NotifyMessage::Custom(b)) => {
                assert_eq!(a.message_type(), b.message_type());
            }
            _ => panic!("both should be Custom"),
        }
    }

    #[test]
    fn custom_notify_debug() {
        let msg = NotifyMessage::Custom(Box::new(ProgressNotify {
            percent: 99,
            label: "debug".into(),
        }));
        let debug = format!("{msg:?}");
        assert!(debug.contains("Custom"));
        assert!(debug.contains("ProgressNotify"));
    }

    // -- NoopInterface --

    #[test]
    fn noop_interface_accepts_text() {
        let iface = NoopInterface;
        iface.notify(NotifyMessage::info("discarded"));
    }

    #[test]
    fn noop_interface_accepts_custom() {
        let iface = NoopInterface;
        iface.notify(NotifyMessage::Custom(Box::new(ProgressNotify {
            percent: 0,
            label: String::new(),
        })));
    }

    // -- Compile-time check: NoopInterface implements Interface --

    fn assert_interface<T: Interface>() {}

    #[test]
    fn noop_interface_implements_interface() {
        assert_interface::<NoopInterface>();
    }

    // -- Mock Interface for recording calls --

    #[derive(Debug, Clone)]
    struct RecordedNotify {
        message: String,
        kind: Option<NotifyKind>,
        custom_type: Option<String>,
    }

    struct MockInterface {
        notifications: Arc<Mutex<Vec<RecordedNotify>>>,
    }

    impl MockInterface {
        fn new() -> Self {
            Self {
                notifications: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn recorded(&self) -> Vec<RecordedNotify> {
            self.notifications
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        }
    }

    impl Interface for MockInterface {
        fn notify(&self, message: NotifyMessage) {
            let record = match &message {
                NotifyMessage::Text { message, kind } => RecordedNotify {
                    message: message.clone(),
                    kind: Some(*kind),
                    custom_type: None,
                },
                NotifyMessage::Custom(c) => RecordedNotify {
                    message: String::new(),
                    kind: None,
                    custom_type: Some(c.message_type().to_string()),
                },
            };
            self.notifications
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(record);
        }
    }

    #[test]
    fn mock_interface_records_text() {
        let mock = MockInterface::new();
        mock.notify(NotifyMessage::error("oops"));
        mock.notify(NotifyMessage::info("done"));

        let recorded = mock.recorded();
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].message, "oops");
        assert_eq!(recorded[0].kind, Some(NotifyKind::Error));
        assert_eq!(recorded[1].message, "done");
        assert_eq!(recorded[1].kind, Some(NotifyKind::Info));
    }

    #[test]
    fn mock_interface_records_custom() {
        let mock = MockInterface::new();
        mock.notify(NotifyMessage::Custom(Box::new(ProgressNotify {
            percent: 50,
            label: "halfway".into(),
        })));

        let recorded = mock.recorded();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].custom_type, Some("progress".to_string()));
        assert!(recorded[0].kind.is_none());
    }
}
