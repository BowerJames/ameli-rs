//! Session manager trait and shared context-building helpers.
//!
//! Defines [`SessionManager<M>`] — the single trait that session backends
//! implement. Replaces the previous two-layer `SessionStorage<M>` +
//! `Session<M>` design. Each implementation decides its own ID generation,
//! persistence strategy, and internal data structures.
//!
//! # Shared helper
//!
//! [`build_session_context_from_path`] is a `pub(crate)` function that all
//! implementations call from their [`build_context`](SessionManager::build_context)
//! method. It produces [`SessionContext`] with [`SessionMessage`] values that
//! preserve type identity — compaction and branch summary entries are **not**
//! converted to [`AgentMessage`] here. That conversion happens in the future
//! `AgentSession`, which consults extension formatting hooks.
//!
//! # Extension formatting hooks
//!
//! The extension system defines `on_format_compaction_summary` and
//! `on_format_branch_summary` hooks. When `AgentSession` converts
//! `Vec<SessionMessage>` to `Vec<AgentMessage>`, it calls these hooks for
//! [`SessionMessage::Compaction`] and [`SessionMessage::BranchSummary`]
//! variants, falling back to the default conversion helpers if no extension
//! overrides the formatting.

use crate::error::SessionError;
use crate::types::{
    CompactionEntry, CustomMessageContent, CustomMessageEntry, ModelRef, SessionContext,
    SessionEntry, SessionMessage,
};
use ameli_agent_core::types::{AgentMessage, CustomAgentMessage};
use ameli_ai::types::{MediaContentBlock, TextContent};
use chrono::{DateTime, Utc};
use std::fmt;
use std::future::Future;
use std::pin::Pin;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Boxed, sendable async result used by [`SessionManager`] trait methods.
///
/// Using `Pin<Box<dyn Future>>` ensures the trait is dyn-compatible
/// (object-safe), so `Arc<dyn SessionManager<M>>` works.
type AsyncResult<T> = Pin<Box<dyn Future<Output = Result<T, SessionError>> + Send>>;

// ---------------------------------------------------------------------------
// SessionMetadata
// ---------------------------------------------------------------------------

/// Metadata identifying and describing a session.
///
/// Different storage backends carry different metadata — for example,
/// a file-backed session includes the file path and working directory,
/// while an in-memory session only needs an ID and creation timestamp.
/// This trait captures the common denominator.
///
/// # Examples
///
/// ```
/// use ameli_agent::session_manager::SessionMetadata;
///
/// struct InMemoryMetadata {
///     id: String,
///     created_at: String,
/// }
///
/// impl SessionMetadata for InMemoryMetadata {
///     fn id(&self) -> &str { &self.id }
///     fn created_at(&self) -> &str { &self.created_at }
/// }
/// ```
pub trait SessionMetadata: Send + Sync + 'static {
    /// Unique session identifier.
    fn id(&self) -> &str;

    /// ISO 8601 timestamp of when the session was created.
    fn created_at(&self) -> &str;
}

// ---------------------------------------------------------------------------
// SessionManager trait
// ---------------------------------------------------------------------------

/// Storage backend and domain operations for a session tree.
///
/// Implementations manage the append-only tree of [`SessionEntry`] values.
/// All methods take `&self` — implementations are expected to use interior
/// mutability (e.g., `RwLock`) for async-safe mutation.
///
/// # Tree Model
///
/// Entries form a tree via `parent_id`. The "leaf" tracks the current
/// position in the tree. Appending creates a child of the current leaf.
/// Branching moves the leaf to an earlier entry, allowing new branches
/// without modifying history.
///
/// # Concurrency
///
/// Implementations must be `Send + Sync` so that the session can be shared
/// across async tasks. Interior mutability ensures concurrent reads are
/// not blocked by each other.
///
/// # Type Parameter
///
/// `M` is the metadata type for this session. Different backends carry
/// different metadata — see [`SessionMetadata`].
pub trait SessionManager<M: SessionMetadata>: Send + Sync {
    // -----------------------------------------------------------------------
    // Read operations
    // -----------------------------------------------------------------------

    /// Returns the session metadata.
    fn metadata(&self) -> AsyncResult<M>;

    /// Returns the current leaf entry ID, or `None` if the session is empty.
    fn leaf_id(&self) -> AsyncResult<Option<String>>;

    /// Look up a single entry by ID.
    fn entry(&self, id: &str) -> AsyncResult<Option<SessionEntry>>;

    /// Return all entries in the session.
    fn entries(&self) -> AsyncResult<Vec<SessionEntry>>;

    /// Walk from the given leaf to the root, returning entries in root-to-leaf
    /// order. If `from_id` is `None`, uses the current leaf.
    fn branch(&self, from_id: Option<&str>) -> AsyncResult<Vec<SessionEntry>>;

    /// Build the resolved session context from the current tree position.
    ///
    /// Implementations should call [`build_session_context_from_path`] with
    /// the result of [`branch(None)`](Self::branch).
    fn build_context(&self) -> AsyncResult<SessionContext>;

    /// Return the resolved label for an entry, if any.
    fn label(&self, id: &str) -> AsyncResult<Option<String>>;

    // -----------------------------------------------------------------------
    // Write operations
    // -----------------------------------------------------------------------

    /// Append a conversation message as a child of the current leaf.
    ///
    /// Returns the new entry ID.
    fn append_message(&self, message: AgentMessage) -> AsyncResult<String>;

    /// Append a thinking level change as a child of the current leaf.
    ///
    /// Returns the new entry ID.
    fn append_thinking_level_change(&self, thinking_level: &str) -> AsyncResult<String>;

    /// Append a model change as a child of the current leaf.
    ///
    /// Returns the new entry ID.
    fn append_model_change(&self, provider: &str, model_id: &str) -> AsyncResult<String>;

    /// Append a compaction summary as a child of the current leaf.
    ///
    /// Returns the new entry ID.
    fn append_compaction(
        &self,
        summary: &str,
        first_kept_entry_id: &str,
        tokens_before: u64,
        details: Option<serde_json::Value>,
        from_hook: bool,
    ) -> AsyncResult<String>;

    /// Append a generic custom entry (for extensions). Does NOT participate
    /// in LLM context.
    ///
    /// Returns the new entry ID.
    fn append_custom_entry(
        &self,
        custom_type: &str,
        data: Option<serde_json::Value>,
    ) -> AsyncResult<String>;

    /// Append an extension-injected message that participates in LLM context.
    ///
    /// Returns the new entry ID.
    fn append_custom_message_entry(
        &self,
        custom_type: &str,
        content: CustomMessageContent,
        display: bool,
        details: Option<serde_json::Value>,
    ) -> AsyncResult<String>;

    /// Move the active leaf to a different position in the tree.
    ///
    /// After this call, the next append will create a child of the new leaf.
    /// Optionally appends a branch summary entry capturing context from the
    /// abandoned path.
    ///
    /// Returns the branch summary entry ID if a summary was provided, or
    /// `None` otherwise.
    fn move_to(
        &self,
        entry_id: Option<&str>,
        summary: Option<BranchSummaryData>,
    ) -> AsyncResult<Option<String>>;
}

// ---------------------------------------------------------------------------
// BranchSummaryData
// ---------------------------------------------------------------------------

/// Data for an optional branch summary when moving the leaf pointer.
#[derive(Debug, Clone)]
pub struct BranchSummaryData {
    /// LLM-readable summary of the abandoned branch.
    pub summary: String,
    /// Extension-specific data (not sent to LLM).
    pub details: Option<serde_json::Value>,
    /// `true` if generated by an extension hook.
    pub from_hook: bool,
}

// ---------------------------------------------------------------------------
// Shared context-building helper
// ---------------------------------------------------------------------------

/// Build a [`SessionContext`] from a root-to-leaf entry path.
///
/// All [`SessionManager`] implementations use this function —
/// entry-to-message conversion is universal. Compaction and branch summary
/// entries become [`SessionMessage::Compaction`] and
/// [`SessionMessage::BranchSummary`] respectively, preserving raw data for
/// later formatting by `AgentSession` with extension hooks.
///
/// # Compaction handling
///
/// When a [`CompactionEntry`] is present, only the summary + kept messages
/// (from `first_kept_entry_id`) + post-compaction messages are emitted.
/// Pre-compaction messages before `first_kept_entry_id` are dropped.
///
/// # Panics
///
/// Does not panic. Returns a default context for empty paths.
pub(crate) fn build_session_context_from_path(path: &[SessionEntry]) -> SessionContext {
    if path.is_empty() {
        return SessionContext {
            messages: Vec::new(),
            thinking_level: "off".to_string(),
            model: None,
        };
    }

    // Walk path to extract settings and find the last compaction.
    // Model resolution: last-wins, with assistant messages overwriting
    // explicit ModelChange entries (matches the TS reference behavior).
    let mut thinking_level = "off".to_string();
    let mut model: Option<ModelRef> = None;
    let mut compaction: Option<(&CompactionEntry, usize)> = None;

    for (idx, entry) in path.iter().enumerate() {
        match entry {
            SessionEntry::ThinkingLevelChange(e) => {
                thinking_level = e.thinking_level.clone();
            }
            SessionEntry::ModelChange(e) => {
                model = Some(ModelRef {
                    provider: e.provider.clone(),
                    model_id: e.model_id.clone(),
                });
            }
            SessionEntry::Message(e) => {
                if let AgentMessage::Assistant(am) = &e.message {
                    model = Some(ModelRef {
                        provider: am.provider.clone(),
                        model_id: am.model.clone(),
                    });
                }
            }
            // Only the most recent compaction applies.
            SessionEntry::Compaction(e) => {
                compaction = Some((e, idx));
            }
            _ => {}
        }
    }

    let mut messages: Vec<SessionMessage> = Vec::new();

    let append_message = |entry: &SessionEntry, msgs: &mut Vec<SessionMessage>| match entry {
        SessionEntry::Message(e) => {
            msgs.push(SessionMessage::Agent(Box::new(e.message.clone())));
        }
        SessionEntry::CustomMessage(e) => {
            msgs.push(SessionMessage::Agent(Box::new(
                custom_message_entry_to_agent_message(e),
            )));
        }
        SessionEntry::BranchSummary(e) => {
            msgs.push(SessionMessage::BranchSummary {
                summary: e.summary.clone(),
                timestamp: now_ms_from_iso(&e.timestamp),
            });
        }
        _ => {}
    };

    if let Some((comp, compaction_idx)) = compaction {
        // Emit compaction summary first.
        messages.push(SessionMessage::Compaction {
            summary: comp.summary.clone(),
            timestamp: now_ms_from_iso(&comp.timestamp),
        });

        // Emit kept messages (from firstKeptEntryId up to compaction).
        let mut found_first_kept = false;
        if let Some(pre_compaction) = path.get(..compaction_idx) {
            for entry in pre_compaction {
                if entry.id() == comp.first_kept_entry_id {
                    found_first_kept = true;
                }
                if found_first_kept {
                    append_message(entry, &mut messages);
                }
            }
        }

        // Emit messages after compaction.
        if let Some(post_compaction) = path.get(compaction_idx + 1..) {
            for entry in post_compaction {
                append_message(entry, &mut messages);
            }
        }
    } else {
        // No compaction — emit all messages.
        for entry in path {
            append_message(entry, &mut messages);
        }
    }

    SessionContext {
        messages,
        thinking_level,
        model,
    }
}

// ---------------------------------------------------------------------------
// Internal CustomAgentMessage implementation
// ---------------------------------------------------------------------------

/// Extension-injected message content wrapped as a custom agent message.
#[derive(Clone)]
struct ExtensionCustomMessage {
    custom_type: String,
    content: CustomMessageContent,
    display: bool,
}

impl CustomAgentMessage for ExtensionCustomMessage {
    fn message_type(&self) -> &str {
        &self.custom_type
    }
    fn clone_boxed(&self) -> Box<dyn CustomAgentMessage> {
        Box::new(self.clone())
    }
    fn to_json(&self) -> serde_json::Value {
        match &self.content {
            CustomMessageContent::Text(t) => serde_json::json!({
                "customType": self.custom_type,
                "content": t,
                "display": self.display,
            }),
            CustomMessageContent::Rich(blocks) => serde_json::json!({
                "customType": self.custom_type,
                "content": blocks,
                "display": self.display,
            }),
        }
    }
    fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionCustomMessage")
            .field("custom_type", &self.custom_type)
            .field("display", &self.display)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Default conversion helpers (pub(crate) for use by AgentSession)
// ---------------------------------------------------------------------------

/// Convert a [`CustomMessageEntry`] to an [`AgentMessage::Custom`].
///
/// Called inside [`build_session_context_from_path`] and available for
/// the future `AgentSession`.
pub(crate) fn custom_message_entry_to_agent_message(entry: &CustomMessageEntry) -> AgentMessage {
    let ext_msg = ExtensionCustomMessage {
        custom_type: entry.custom_type.clone(),
        content: entry.content.clone(),
        display: entry.display,
    };
    AgentMessage::Custom(Box::new(ext_msg))
}

/// Default formatting for a compaction summary as a synthetic user message.
///
/// The future `AgentSession` uses this as the fallback when no extension
/// overrides `on_format_compaction_summary`.
pub(crate) fn compaction_summary_to_agent_message(summary: &str, timestamp: u64) -> AgentMessage {
    let text = format!(
        "The conversation history before this point was compacted into the following summary:\n\n\
         <summary>\n{summary}\n</summary>",
    );
    let content = vec![MediaContentBlock::Text(TextContent::new(text))];
    AgentMessage::User(ameli_ai::types::UserMessage {
        content: ameli_ai::types::UserContent::Blocks(content),
        timestamp,
    })
}

/// Default formatting for a branch summary as a synthetic user message.
///
/// The future `AgentSession` uses this as the fallback when no extension
/// overrides `on_format_branch_summary`.
pub(crate) fn branch_summary_to_agent_message(summary: &str, timestamp: u64) -> AgentMessage {
    let text = format!(
        "The following is a summary of a branch that this conversation came back from:\n\n\
         <summary>\n{summary}\n</summary>",
    );
    let content = vec![MediaContentBlock::Text(TextContent::new(text))];
    AgentMessage::User(ameli_ai::types::UserMessage {
        content: ameli_ai::types::UserContent::Blocks(content),
        timestamp,
    })
}

/// Best-effort parse of ISO 8601 timestamp to milliseconds.
/// Falls back to 0 if parsing fails.
pub(crate) fn now_ms_from_iso(iso: &str) -> u64 {
    DateTime::parse_from_rfc3339(iso)
        .map(|dt| dt.timestamp_millis().unsigned_abs())
        .unwrap_or(0)
}

/// Current time as ISO 8601 string (seconds precision, UTC).
pub(crate) fn now_iso8601() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use ameli_ai::types::Usage;

    fn test_user_message(text: &str) -> AgentMessage {
        AgentMessage::User(ameli_ai::types::UserMessage::text(text))
    }

    fn test_assistant_message(text: &str) -> AgentMessage {
        AgentMessage::Assistant(ameli_ai::types::AssistantMessage {
            content: vec![ameli_ai::types::AssistantContentBlock::Text(
                TextContent::new(text),
            )],
            api: "test".to_string(),
            provider: "openai".to_string(),
            model: "gpt-4o".to_string(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason: ameli_ai::types::StopReason::Stop,
            error_message: None,
            timestamp: 1000,
        })
    }

    fn make_message_entry(id: &str, parent_id: Option<&str>, msg: AgentMessage) -> SessionEntry {
        SessionEntry::Message(MessageEntry {
            id: id.to_string(),
            parent_id: parent_id.map(String::from),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            message: msg,
        })
    }

    fn make_thinking_entry(id: &str, parent_id: Option<&str>, level: &str) -> SessionEntry {
        SessionEntry::ThinkingLevelChange(ThinkingLevelChangeEntry {
            id: id.to_string(),
            parent_id: parent_id.map(String::from),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            thinking_level: level.to_string(),
        })
    }

    fn make_model_entry(
        id: &str,
        parent_id: Option<&str>,
        provider: &str,
        model_id: &str,
    ) -> SessionEntry {
        SessionEntry::ModelChange(ModelChangeEntry {
            id: id.to_string(),
            parent_id: parent_id.map(String::from),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            provider: provider.to_string(),
            model_id: model_id.to_string(),
        })
    }

    fn make_compaction_entry(
        id: &str,
        parent_id: Option<&str>,
        summary: &str,
        first_kept: &str,
    ) -> SessionEntry {
        SessionEntry::Compaction(CompactionEntry {
            id: id.to_string(),
            parent_id: parent_id.map(String::from),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            summary: summary.to_string(),
            first_kept_entry_id: first_kept.to_string(),
            tokens_before: 10000,
            details: None,
            from_hook: false,
        })
    }

    fn make_branch_summary_entry(
        id: &str,
        parent_id: Option<&str>,
        from_id: &str,
        summary: &str,
    ) -> SessionEntry {
        SessionEntry::BranchSummary(BranchSummaryEntry {
            id: id.to_string(),
            parent_id: parent_id.map(String::from),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            from_id: from_id.to_string(),
            summary: summary.to_string(),
            details: None,
            from_hook: false,
        })
    }

    fn make_custom_message_entry(
        id: &str,
        parent_id: Option<&str>,
        custom_type: &str,
        text: &str,
    ) -> SessionEntry {
        SessionEntry::CustomMessage(CustomMessageEntry {
            id: id.to_string(),
            parent_id: parent_id.map(String::from),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            custom_type: custom_type.to_string(),
            content: CustomMessageContent::Text(text.to_string()),
            display: true,
            details: None,
        })
    }

    /// Helper: get message at index, panicking with context if out of bounds.
    fn get_msg(ctx: &SessionContext, index: usize) -> &SessionMessage {
        ctx.messages.get(index).unwrap_or_else(|| {
            panic!(
                "expected at least {} messages, got {}",
                index + 1,
                ctx.messages.len()
            )
        })
    }

    #[test]
    fn empty_path_returns_defaults() {
        let ctx = build_session_context_from_path(&[]);
        assert!(ctx.messages.is_empty());
        assert_eq!(ctx.thinking_level, "off");
        assert!(ctx.model.is_none());
    }

    #[test]
    fn messages_only_are_in_order() {
        let path = vec![
            make_message_entry("1", None, test_user_message("hello")),
            make_message_entry("2", Some("1"), test_assistant_message("hi")),
            make_message_entry("3", Some("2"), test_user_message("bye")),
        ];
        let ctx = build_session_context_from_path(&path);
        assert_eq!(ctx.messages.len(), 3);
        assert!(matches!(&ctx.messages[0], SessionMessage::Agent(m) if m.role() == "user"));
        assert!(matches!(&ctx.messages[1], SessionMessage::Agent(m) if m.role() == "assistant"));
        assert!(matches!(&ctx.messages[2], SessionMessage::Agent(m) if m.role() == "user"));
    }

    #[test]
    fn thinking_level_last_wins() {
        let path = vec![
            make_thinking_entry("1", None, "off"),
            make_thinking_entry("2", Some("1"), "medium"),
            make_thinking_entry("3", Some("2"), "high"),
        ];
        let ctx = build_session_context_from_path(&path);
        assert_eq!(ctx.thinking_level, "high");
        assert!(ctx.messages.is_empty());
    }

    #[test]
    fn model_change_last_wins() {
        let path = vec![
            make_model_entry("1", None, "openai", "gpt-4o"),
            make_model_entry("2", Some("1"), "anthropic", "claude-3"),
        ];
        let ctx = build_session_context_from_path(&path);
        assert_eq!(
            ctx.model.as_ref().map(|m| m.model_id.as_str()),
            Some("claude-3")
        );
    }

    #[test]
    fn assistant_message_model_wins() {
        let path = vec![
            make_model_entry("1", None, "openai", "gpt-4o"),
            make_message_entry("2", Some("1"), test_assistant_message("hi")),
        ];
        let ctx = build_session_context_from_path(&path);
        // Assistant message model takes precedence
        assert_eq!(
            ctx.model.as_ref().map(|m| m.model_id.as_str()),
            Some("gpt-4o")
        );
    }

    #[test]
    fn compaction_emits_summary_then_kept_then_post() {
        let path = vec![
            make_message_entry("1", None, test_user_message("old1")),
            make_message_entry("2", Some("1"), test_assistant_message("old2")),
            make_message_entry("3", Some("2"), test_user_message("kept1")),
            make_compaction_entry("4", Some("3"), "summary of old", "3"),
            make_message_entry("5", Some("4"), test_user_message("new1")),
        ];
        let ctx = build_session_context_from_path(&path);

        // Expect: Compaction, kept1 (Agent), new1 (Agent)
        assert_eq!(ctx.messages.len(), 3);
        // First: compaction summary
        assert!(
            matches!(get_msg(&ctx, 0), SessionMessage::Compaction { summary, .. } if summary == "summary of old")
        );
        // Second: kept1
        assert!(matches!(get_msg(&ctx, 1), SessionMessage::Agent(m) if m.role() == "user"));
        // Third: new1
        assert!(matches!(get_msg(&ctx, 2), SessionMessage::Agent(m) if m.role() == "user"));
    }

    #[test]
    fn branch_summary_produces_branch_summary_variant() {
        let path = vec![
            make_message_entry("1", None, test_user_message("hello")),
            make_branch_summary_entry("2", Some("1"), "1", "branch was about X"),
        ];
        let ctx = build_session_context_from_path(&path);
        assert_eq!(ctx.messages.len(), 2);
        assert!(matches!(get_msg(&ctx, 0), SessionMessage::Agent(_)));
        assert!(
            matches!(get_msg(&ctx, 1), SessionMessage::BranchSummary { summary, .. } if summary == "branch was about X")
        );
    }

    #[test]
    fn custom_message_becomes_agent_variant() {
        let path = vec![make_custom_message_entry(
            "1",
            None,
            "my_ext",
            "extension data",
        )];
        let ctx = build_session_context_from_path(&path);
        assert_eq!(ctx.messages.len(), 1);
        assert!(matches!(get_msg(&ctx, 0), SessionMessage::Agent(m) if m.role() == "my_ext"));
    }

    #[test]
    fn mixed_path_with_all_types() {
        let path = vec![
            make_thinking_entry("1", None, "off"),
            make_model_entry("2", Some("1"), "openai", "gpt-4o"),
            make_message_entry("3", Some("2"), test_user_message("hello")),
            make_message_entry("4", Some("3"), test_assistant_message("hi")),
            make_thinking_entry("5", Some("4"), "high"),
        ];
        let ctx = build_session_context_from_path(&path);
        assert_eq!(ctx.messages.len(), 2);
        assert_eq!(ctx.thinking_level, "high");
        assert_eq!(
            ctx.model.as_ref().map(|m| m.model_id.as_str()),
            Some("gpt-4o")
        );
    }

    #[test]
    fn compaction_with_no_post_messages() {
        let path = vec![
            make_message_entry("1", None, test_user_message("old")),
            make_message_entry("2", Some("1"), test_assistant_message("reply")),
            make_compaction_entry("3", Some("2"), "compacted", "2"),
        ];
        let ctx = build_session_context_from_path(&path);
        // Compaction summary + kept message "2" (reply) onward = summary + kept
        assert_eq!(ctx.messages.len(), 2);
    }

    #[test]
    fn now_iso8601_is_valid_format() {
        let ts = now_iso8601();
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[7..8], "-");
        assert_eq!(&ts[10..11], "T");
        assert_eq!(&ts[13..14], ":");
        assert_eq!(&ts[16..17], ":");
        assert!(ts.ends_with('Z'));
    }

    #[test]
    fn now_ms_from_iso_parses_correctly() {
        let ms = now_ms_from_iso("2026-01-01T00:00:00Z");
        // 2026-01-01 should be a reasonable non-zero value
        assert!(ms > 0);
    }

    #[test]
    fn now_ms_from_iso_handles_short_input() {
        assert_eq!(now_ms_from_iso(""), 0);
        assert_eq!(now_ms_from_iso("short"), 0);
    }
}
