//! High-level session API and context builder.
//!
//! [`Session`] wraps a [`SessionStorage`] backend and provides typed methods
//! for appending entries, navigating the tree, and building the resolved
//! [`SessionContext`] for the LLM.
//!
//! [`build_session_context`] is a pure function that reconstructs the message
//! list, thinking level, and model selection from a root-to-leaf path of
//! entries, handling compaction and branch summaries.

use crate::error::SessionError;
use crate::storage::{SessionMetadata, SessionStorage};
use crate::types::{
    BranchSummaryEntry, CompactionEntry, CustomMessageContent, CustomMessageEntry, ModelRef,
    SessionContext, SessionEntry,
};
use ameli_agent_core::types::{AgentMessage, CustomAgentMessage};
use ameli_ai::types::{MediaContentBlock, TextContent};
use chrono::{DateTime, Utc};
use std::fmt;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Internal CustomAgentMessage implementations
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
// build_session_context
// ---------------------------------------------------------------------------

/// Reconstruct the [`SessionContext`] from a root-to-leaf path of entries.
///
/// This is a pure function with no storage dependency. It handles:
/// - Extracting the last thinking level and model from the path
/// - Compaction: emitting the summary, then kept messages, then post-compaction messages
/// - Branch summaries: converted to custom agent messages
/// - Custom messages: converted to custom agent messages
///
/// # Compaction assumption
///
/// When a [`CompactionEntry`] is present, its `first_kept_entry_id` is assumed to
/// reference an entry that exists on the active branch path (i.e., appears among the
/// entries preceding the compaction). If it does not — which would indicate a bug
/// in the compaction implementation — all pre-compaction messages are silently
/// dropped and only the compaction summary and post-compaction messages are emitted.
///
/// # Panics
///
/// Does not panic. Returns a default context for empty paths.
pub fn build_session_context(path: &[SessionEntry]) -> SessionContext {
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

    let mut messages: Vec<AgentMessage> = Vec::new();

    let append_message = |entry: &SessionEntry, msgs: &mut Vec<AgentMessage>| match entry {
        SessionEntry::Message(e) => {
            msgs.push(e.message.clone());
        }
        SessionEntry::CustomMessage(e) => {
            msgs.push(custom_message_entry_to_agent_message(e));
        }
        SessionEntry::BranchSummary(e) => {
            msgs.push(branch_summary_to_agent_message(e));
        }
        _ => {}
    };

    if let Some((comp, compaction_idx)) = compaction {
        // Emit compaction summary first.
        messages.push(compaction_summary_to_agent_message(comp));

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
// Conversion helpers
// ---------------------------------------------------------------------------

fn now_iso8601() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn compaction_summary_to_agent_message(entry: &CompactionEntry) -> AgentMessage {
    let summary = format!(
        "The conversation history before this point was compacted into the following summary:\n\n\
         <summary>\n{}\n</summary>",
        entry.summary
    );
    let content = vec![MediaContentBlock::Text(TextContent::new(summary))];
    AgentMessage::User(ameli_ai::types::UserMessage {
        content: ameli_ai::types::UserContent::Blocks(content),
        timestamp: now_ms_from_iso(&entry.timestamp),
    })
}

fn branch_summary_to_agent_message(entry: &BranchSummaryEntry) -> AgentMessage {
    let summary = format!(
        "The following is a summary of a branch that this conversation came back from:\n\n\
         <summary>\n{}\n</summary>",
        entry.summary
    );
    let content = vec![MediaContentBlock::Text(TextContent::new(summary))];
    AgentMessage::User(ameli_ai::types::UserMessage {
        content: ameli_ai::types::UserContent::Blocks(content),
        timestamp: now_ms_from_iso(&entry.timestamp),
    })
}

fn custom_message_entry_to_agent_message(entry: &CustomMessageEntry) -> AgentMessage {
    let ext_msg = ExtensionCustomMessage {
        custom_type: entry.custom_type.clone(),
        content: entry.content.clone(),
        display: entry.display,
    };
    AgentMessage::Custom(Box::new(ext_msg))
}

/// Best-effort parse of ISO 8601 timestamp to milliseconds.
/// Falls back to 0 if parsing fails.
fn now_ms_from_iso(iso: &str) -> u64 {
    DateTime::parse_from_rfc3339(iso)
        .map(|dt| dt.timestamp_millis().unsigned_abs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// High-level session API wrapping a [`SessionStorage`] backend.
///
/// Provides typed methods for appending entries, navigating the tree, and
/// building the resolved [`SessionContext`] for the LLM. Construct with
/// [`Session::new`].
///
/// # Type Parameter
///
/// `M` is the metadata type for this session. Different storage backends
/// carry different metadata — e.g., file-backed sessions include the file
/// path, while in-memory sessions only need an ID.
///
/// # Examples
///
/// ```no_run
/// use ameli_agent::Session;
/// use ameli_agent::storage::{SessionMetadata, SessionStorage};
/// use ameli_agent::types::SessionEntry;
/// use ameli_agent::error::SessionError;
///
/// // Implement SessionStorage for your backend, then:
/// // let storage: Arc<dyn SessionStorage<MyMetadata>> = ...;
/// // let session = Session::new(storage);
/// // let entry_id = session.append_message(my_message).await?;
/// ```
pub struct Session<M: SessionMetadata + 'static> {
    storage: Arc<dyn SessionStorage<M>>,
}

impl<M: SessionMetadata + 'static> Session<M> {
    /// Create a new session wrapping the given storage backend.
    pub fn new(storage: Arc<dyn SessionStorage<M>>) -> Self {
        Self { storage }
    }

    // -----------------------------------------------------------------------
    // Read operations
    // -----------------------------------------------------------------------

    /// Returns the session metadata.
    pub async fn metadata(&self) -> Result<M, SessionError> {
        self.storage.metadata().await
    }

    /// Returns the current leaf entry ID, or `None` if the session is empty.
    pub async fn leaf_id(&self) -> Result<Option<String>, SessionError> {
        self.storage.leaf_id().await
    }

    /// Look up a single entry by ID.
    pub async fn entry(&self, id: &str) -> Result<Option<SessionEntry>, SessionError> {
        self.storage.get_entry(id).await
    }

    /// Return all entries in the session.
    pub async fn entries(&self) -> Result<Vec<SessionEntry>, SessionError> {
        self.storage.entries().await
    }

    /// Walk from a given entry to the root, returning entries in root-to-leaf
    /// order. If `from_id` is `None`, uses the current leaf.
    pub async fn branch(&self, from_id: Option<&str>) -> Result<Vec<SessionEntry>, SessionError> {
        self.storage.path_to_root(from_id).await
    }

    /// Build the resolved session context from the current tree position.
    pub async fn build_context(&self) -> Result<SessionContext, SessionError> {
        let path = self.storage.path_to_root(None).await?;
        Ok(build_session_context(&path))
    }

    /// Return the resolved label for an entry, if any.
    pub async fn label(&self, id: &str) -> Result<Option<String>, SessionError> {
        self.storage.get_label(id).await
    }

    // -----------------------------------------------------------------------
    // Write operations
    // -----------------------------------------------------------------------

    /// Append a conversation message as a child of the current leaf.
    ///
    /// Returns the new entry ID.
    pub async fn append_message(&self, message: AgentMessage) -> Result<String, SessionError> {
        let id = self.storage.create_entry_id().await?;
        let parent_id = self.storage.leaf_id().await?;
        let timestamp = now_iso8601();
        let entry = SessionEntry::Message(crate::types::MessageEntry {
            id: id.clone(),
            parent_id,
            timestamp,
            message,
        });
        self.storage.append_entry(entry).await?;
        Ok(id)
    }

    /// Append a thinking level change as a child of the current leaf.
    ///
    /// Returns the new entry ID.
    pub async fn append_thinking_level_change(
        &self,
        thinking_level: &str,
    ) -> Result<String, SessionError> {
        let id = self.storage.create_entry_id().await?;
        let parent_id = self.storage.leaf_id().await?;
        let timestamp = now_iso8601();
        let entry = SessionEntry::ThinkingLevelChange(crate::types::ThinkingLevelChangeEntry {
            id: id.clone(),
            parent_id,
            timestamp,
            thinking_level: thinking_level.to_string(),
        });
        self.storage.append_entry(entry).await?;
        Ok(id)
    }

    /// Append a model change as a child of the current leaf.
    ///
    /// Returns the new entry ID.
    pub async fn append_model_change(
        &self,
        provider: &str,
        model_id: &str,
    ) -> Result<String, SessionError> {
        let id = self.storage.create_entry_id().await?;
        let parent_id = self.storage.leaf_id().await?;
        let timestamp = now_iso8601();
        let entry = SessionEntry::ModelChange(crate::types::ModelChangeEntry {
            id: id.clone(),
            parent_id,
            timestamp,
            provider: provider.to_string(),
            model_id: model_id.to_string(),
        });
        self.storage.append_entry(entry).await?;
        Ok(id)
    }

    /// Append a compaction summary as a child of the current leaf.
    ///
    /// Returns the new entry ID.
    pub async fn append_compaction(
        &self,
        summary: &str,
        first_kept_entry_id: &str,
        tokens_before: u64,
        details: Option<serde_json::Value>,
        from_hook: bool,
    ) -> Result<String, SessionError> {
        let id = self.storage.create_entry_id().await?;
        let parent_id = self.storage.leaf_id().await?;
        let timestamp = now_iso8601();
        let entry = SessionEntry::Compaction(crate::types::CompactionEntry {
            id: id.clone(),
            parent_id,
            timestamp,
            summary: summary.to_string(),
            first_kept_entry_id: first_kept_entry_id.to_string(),
            tokens_before,
            details,
            from_hook,
        });
        self.storage.append_entry(entry).await?;
        Ok(id)
    }

    /// Append a generic custom entry (for extensions). Does NOT participate
    /// in LLM context.
    ///
    /// Returns the new entry ID.
    pub async fn append_custom_entry(
        &self,
        custom_type: &str,
        data: Option<serde_json::Value>,
    ) -> Result<String, SessionError> {
        let id = self.storage.create_entry_id().await?;
        let parent_id = self.storage.leaf_id().await?;
        let timestamp = now_iso8601();
        let entry = SessionEntry::Custom(crate::types::CustomEntry {
            id: id.clone(),
            parent_id,
            timestamp,
            custom_type: custom_type.to_string(),
            data,
        });
        self.storage.append_entry(entry).await?;
        Ok(id)
    }

    /// Append an extension-injected message that participates in LLM context.
    ///
    /// Returns the new entry ID.
    pub async fn append_custom_message_entry(
        &self,
        custom_type: &str,
        content: CustomMessageContent,
        display: bool,
        details: Option<serde_json::Value>,
    ) -> Result<String, SessionError> {
        let id = self.storage.create_entry_id().await?;
        let parent_id = self.storage.leaf_id().await?;
        let timestamp = now_iso8601();
        let entry = SessionEntry::CustomMessage(crate::types::CustomMessageEntry {
            id: id.clone(),
            parent_id,
            timestamp,
            custom_type: custom_type.to_string(),
            content,
            display,
            details,
        });
        self.storage.append_entry(entry).await?;
        Ok(id)
    }

    // -----------------------------------------------------------------------
    // Tree navigation
    // -----------------------------------------------------------------------

    /// Move the active leaf to a different position in the tree.
    ///
    /// After this call, the next append will create a child of the new leaf.
    /// Optionally appends a branch summary entry capturing context from the
    /// abandoned path.
    ///
    /// Returns the branch summary entry ID if a summary was provided, or
    /// `None` otherwise.
    pub async fn move_to(
        &self,
        entry_id: Option<&str>,
        summary: Option<BranchSummaryData>,
    ) -> Result<Option<String>, SessionError> {
        if let Some(id) = entry_id {
            let exists = self.storage.get_entry(id).await?;
            if exists.is_none() {
                return Err(SessionError::not_found(format!("Entry {id} not found")));
            }
        }

        self.storage.set_leaf_id(entry_id).await?;

        let Some(summary_data) = summary else {
            return Ok(None);
        };

        let id = self.storage.create_entry_id().await?;
        let parent_id = entry_id.map(String::from);
        let timestamp = now_iso8601();
        let entry = SessionEntry::BranchSummary(crate::types::BranchSummaryEntry {
            id: id.clone(),
            parent_id,
            timestamp,
            from_id: entry_id.unwrap_or("root").to_string(),
            summary: summary_data.summary,
            details: summary_data.details,
            from_hook: summary_data.from_hook,
        });
        self.storage.append_entry(entry).await?;
        Ok(Some(id))
    }
}

impl<M: SessionMetadata + 'static> fmt::Debug for Session<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session").finish_non_exhaustive()
    }
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

    // -- build_session_context tests --

    #[test]
    fn empty_path_returns_defaults() {
        let ctx = build_session_context(&[]);
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
        let ctx = build_session_context(&path);
        assert_eq!(ctx.messages.len(), 3);
        assert_eq!(ctx.messages[0].role(), "user");
        assert_eq!(ctx.messages[1].role(), "assistant");
        assert_eq!(ctx.messages[2].role(), "user");
    }

    #[test]
    fn thinking_level_last_wins() {
        let path = vec![
            make_thinking_entry("1", None, "off"),
            make_thinking_entry("2", Some("1"), "medium"),
            make_thinking_entry("3", Some("2"), "high"),
        ];
        let ctx = build_session_context(&path);
        assert_eq!(ctx.thinking_level, "high");
        assert!(ctx.messages.is_empty());
    }

    #[test]
    fn model_change_last_wins() {
        let path = vec![
            make_model_entry("1", None, "openai", "gpt-4o"),
            make_model_entry("2", Some("1"), "anthropic", "claude-3"),
        ];
        let ctx = build_session_context(&path);
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
        let ctx = build_session_context(&path);
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
        let ctx = build_session_context(&path);

        // Expect: compaction summary, kept1, new1
        assert_eq!(ctx.messages.len(), 3);
        // First message is the compaction summary (user message)
        assert_eq!(ctx.messages[0].role(), "user");
        // Second is kept1
        assert_eq!(ctx.messages[1].role(), "user");
        // Third is new1
        assert_eq!(ctx.messages[2].role(), "user");
    }

    #[test]
    fn branch_summary_becomes_user_message() {
        let path = vec![
            make_message_entry("1", None, test_user_message("hello")),
            make_branch_summary_entry("2", Some("1"), "1", "branch was about X"),
        ];
        let ctx = build_session_context(&path);
        // Message + branch summary converted to user message
        assert_eq!(ctx.messages.len(), 2);
        assert_eq!(ctx.messages[0].role(), "user");
        assert_eq!(ctx.messages[1].role(), "user");
    }

    #[test]
    fn custom_message_becomes_custom_agent_message() {
        let path = vec![make_custom_message_entry(
            "1",
            None,
            "my_ext",
            "extension data",
        )];
        let ctx = build_session_context(&path);
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(ctx.messages[0].role(), "my_ext");
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
        let ctx = build_session_context(&path);
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
        let ctx = build_session_context(&path);
        // Compaction summary + kept message "2" (reply) onward = summary + kept
        assert_eq!(ctx.messages.len(), 2);
    }

    // -- ISO 8601 timestamp tests --

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
