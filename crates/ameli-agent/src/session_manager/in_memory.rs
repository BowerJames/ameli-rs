//! In-memory session manager implementation.
//!
//! Provides [`InMemorySessionManager`] — a concrete, tree-based session
//! backend backed by interior-mutable `HashMap` storage. Suitable for
//! testing, prototyping, and short-lived sessions.
//!
//! # Thread safety
//!
//! Uses `RwLock` for both the entry store and the leaf pointer so concurrent
//! reads are not blocked by each other; only writes take a write lock.

use super::manager::{AsyncResult, BranchSummaryData, SessionManager, SessionMetadata};
use super::types::{
    BranchSummaryEntry, CompactionEntry, CustomEntry, CustomMessageContent, CustomMessageEntry,
    MessageEntry, ModelChangeEntry, ModelRef, SessionContext, SessionEntry, SessionMessage,
    ThinkingLevelChangeEntry,
};
use ameli_agent_core::types::AgentMessage;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// ---------------------------------------------------------------------------
// InMemoryMetadata
// ---------------------------------------------------------------------------

/// Metadata for [`InMemorySessionManager`].
///
/// Auto-generated with a unique ID and creation timestamp when using
/// [`InMemorySessionManager::new()`].
#[derive(Debug, Clone)]
pub struct InMemoryMetadata {
    id: String,
    created_at: String,
}

impl InMemoryMetadata {
    /// Create metadata with explicit values.
    pub fn new(id: impl Into<String>, created_at: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            created_at: created_at.into(),
        }
    }
}

impl SessionMetadata for InMemoryMetadata {
    fn id(&self) -> &str {
        &self.id
    }

    fn created_at(&self) -> &str {
        &self.created_at
    }
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

/// Mutable state behind a single `RwLock`.
struct State {
    /// All entries indexed by ID.
    entries: HashMap<String, SessionEntry>,
    /// Current leaf entry ID (where the next append goes).
    leaf_id: Option<String>,
    /// Monotonic counter for generating entry IDs.
    next_id: u64,
}

impl State {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            leaf_id: None,
            next_id: 0,
        }
    }

    /// Generate a unique entry ID and advance the counter.
    fn alloc_id(&mut self) -> String {
        let id = format!("entry-{}", self.next_id);
        self.next_id += 1;
        id
    }

    /// Current ISO 8601 timestamp.
    fn now_timestamp() -> String {
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
    }

    /// Parse an ISO 8601 timestamp string to unix milliseconds.
    ///
    /// Falls back to `0` if parsing fails, matching the graceful degradation
    /// pattern used elsewhere in the crate.
    fn parse_timestamp_ms(iso: &str) -> u64 {
        chrono::DateTime::parse_from_rfc3339(iso)
            .map(|dt| dt.timestamp_millis() as u64)
            .unwrap_or(0)
    }

    /// Insert an entry and update the leaf pointer.
    fn insert_entry(&mut self, entry: SessionEntry) -> String {
        let id = entry.id().to_string();
        self.leaf_id = Some(id.clone());
        self.entries.insert(id.clone(), entry);
        id
    }
}

// ---------------------------------------------------------------------------
// InMemorySessionManager
// ---------------------------------------------------------------------------

/// In-memory, tree-based session manager.
///
/// Stores entries in a `HashMap<String, SessionEntry>` behind an `RwLock`.
/// The tree structure is maintained via `parent_id` links on each entry.
/// The current leaf position is tracked separately for O(1) append
/// operations.
///
/// # Examples
///
/// ```
/// use ameli_agent::session_manager::{InMemorySessionManager, SessionManager};
/// use ameli_agent_core::types::AgentMessage;
///
/// # #[tokio::main]
/// # async fn main() {
/// let sm = InMemorySessionManager::new();
///
/// let id = sm.append_message(AgentMessage::User(
///     ameli_ai::types::UserMessage::text("hello"),
/// )).await.unwrap();
///
/// let entry = sm.entry(&id).await.unwrap().unwrap();
/// assert_eq!(entry.entry_type(), "message");
/// # }
/// ```
pub struct InMemorySessionManager {
    metadata: InMemoryMetadata,
    state: Arc<RwLock<State>>,
}

impl InMemorySessionManager {
    /// Create a new empty session with auto-generated metadata.
    ///
    /// The session ID is a counter-based string and the creation timestamp
    /// is the current time in ISO 8601 format.
    pub fn new() -> Self {
        let created_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = format!(
            "session-{}",
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        Self {
            metadata: InMemoryMetadata::new(id, created_at),
            state: Arc::new(RwLock::new(State::new())),
        }
    }

    /// Create a new empty session with custom metadata.
    pub fn with_metadata(metadata: InMemoryMetadata) -> Self {
        Self {
            metadata,
            state: Arc::new(RwLock::new(State::new())),
        }
    }
}

impl Default for InMemorySessionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionManager<InMemoryMetadata> for InMemorySessionManager {
    fn metadata(&self) -> AsyncResult<InMemoryMetadata> {
        let metadata = self.metadata.clone();
        Box::pin(async move { Ok(metadata) })
    }

    fn leaf_id(&self) -> AsyncResult<Option<String>> {
        let state = self.state.clone();
        Box::pin(async move {
            let s = state.read().await;
            Ok(s.leaf_id.clone())
        })
    }

    fn entry(&self, id: &str) -> AsyncResult<Option<SessionEntry>> {
        let id = id.to_string();
        let state = self.state.clone();
        Box::pin(async move {
            let s = state.read().await;
            Ok(s.entries.get(&id).cloned())
        })
    }

    fn entries(&self) -> AsyncResult<Vec<SessionEntry>> {
        let state = self.state.clone();
        Box::pin(async move {
            let s = state.read().await;
            Ok(s.entries.values().cloned().collect())
        })
    }

    fn branch(&self, from_id: Option<&str>) -> AsyncResult<Vec<SessionEntry>> {
        let from_id = from_id.map(String::from);
        let state = self.state.clone();
        Box::pin(async move {
            let s = state.read().await;

            let leaf = match &from_id {
                Some(id) => Some(id.clone()),
                None => s.leaf_id.clone(),
            };

            let Some(leaf_id) = leaf else {
                return Ok(Vec::new());
            };

            // Walk from leaf to root, collecting entries.
            let mut path = Vec::new();
            let mut current_id = Some(leaf_id);
            while let Some(cid) = current_id {
                if let Some(entry) = s.entries.get(&cid) {
                    current_id = entry.parent_id().map(String::from);
                    path.push(entry.clone());
                } else {
                    break;
                }
            }

            // Reverse to root-to-leaf order.
            path.reverse();
            Ok(path)
        })
    }

    fn build_context(&self) -> AsyncResult<SessionContext> {
        let state = self.state.clone();
        Box::pin(async move {
            let s = state.read().await;

            // Walk from leaf to root.
            let leaf = match &s.leaf_id {
                Some(id) => id.clone(),
                None => {
                    return Ok(SessionContext {
                        messages: Vec::new(),
                        thinking_level: "off".to_string(),
                        model: None,
                    });
                }
            };

            let mut path = Vec::new();
            let mut current_id = Some(leaf);
            while let Some(cid) = current_id {
                if let Some(entry) = s.entries.get(&cid) {
                    current_id = entry.parent_id().map(String::from);
                    path.push(entry.clone());
                } else {
                    break;
                }
            }
            path.reverse();

            // Convert entries to session context.
            let mut messages = Vec::new();
            let mut thinking_level = "off".to_string();
            let mut model: Option<ModelRef> = None;

            for entry in &path {
                match entry {
                    SessionEntry::Message(e) => {
                        messages.push(SessionMessage::Agent(Box::new(e.message.clone())));
                    }
                    SessionEntry::ThinkingLevelChange(e) => {
                        thinking_level = e.thinking_level.clone();
                    }
                    SessionEntry::ModelChange(e) => {
                        model = Some(ModelRef {
                            provider: e.provider.clone(),
                            model_id: e.model_id.clone(),
                        });
                    }
                    SessionEntry::Compaction(e) => {
                        messages.push(SessionMessage::Compaction {
                            summary: e.summary.clone(),
                            timestamp: State::parse_timestamp_ms(&e.timestamp),
                        });
                    }
                    SessionEntry::BranchSummary(e) => {
                        messages.push(SessionMessage::BranchSummary {
                            summary: e.summary.clone(),
                            timestamp: State::parse_timestamp_ms(&e.timestamp),
                        });
                    }
                    // Custom and CustomMessage entries are not part of LLM context.
                    SessionEntry::Custom(_) | SessionEntry::CustomMessage(_) => {}
                }
            }

            Ok(SessionContext {
                messages,
                thinking_level,
                model,
            })
        })
    }

    fn label(&self, _id: &str) -> AsyncResult<Option<String>> {
        Box::pin(async move { Ok(None) })
    }

    fn append_message(&self, message: AgentMessage) -> AsyncResult<String> {
        let state = self.state.clone();
        Box::pin(async move {
            let mut s = state.write().await;
            let id = s.alloc_id();
            let parent_id = s.leaf_id.clone();
            let entry = SessionEntry::Message(MessageEntry {
                id: id.clone(),
                parent_id,
                timestamp: State::now_timestamp(),
                message,
            });
            s.insert_entry(entry);
            Ok(id)
        })
    }

    fn append_thinking_level_change(&self, thinking_level: &str) -> AsyncResult<String> {
        let thinking_level = thinking_level.to_string();
        let state = self.state.clone();
        Box::pin(async move {
            let mut s = state.write().await;
            let id = s.alloc_id();
            let parent_id = s.leaf_id.clone();
            let entry = SessionEntry::ThinkingLevelChange(ThinkingLevelChangeEntry {
                id: id.clone(),
                parent_id,
                timestamp: State::now_timestamp(),
                thinking_level,
            });
            s.insert_entry(entry);
            Ok(id)
        })
    }

    fn append_model_change(&self, provider: &str, model_id: &str) -> AsyncResult<String> {
        let provider = provider.to_string();
        let model_id = model_id.to_string();
        let state = self.state.clone();
        Box::pin(async move {
            let mut s = state.write().await;
            let id = s.alloc_id();
            let parent_id = s.leaf_id.clone();
            let entry = SessionEntry::ModelChange(ModelChangeEntry {
                id: id.clone(),
                parent_id,
                timestamp: State::now_timestamp(),
                provider,
                model_id,
            });
            s.insert_entry(entry);
            Ok(id)
        })
    }

    fn append_compaction(
        &self,
        summary: &str,
        first_kept_entry_id: &str,
        tokens_before: u64,
        details: Option<serde_json::Value>,
        from_hook: bool,
    ) -> AsyncResult<String> {
        let summary = summary.to_string();
        let first_kept_entry_id = first_kept_entry_id.to_string();
        let state = self.state.clone();
        Box::pin(async move {
            let mut s = state.write().await;
            let id = s.alloc_id();
            let parent_id = s.leaf_id.clone();
            let entry = SessionEntry::Compaction(CompactionEntry {
                id: id.clone(),
                parent_id,
                timestamp: State::now_timestamp(),
                summary,
                first_kept_entry_id,
                tokens_before,
                details,
                from_hook,
            });
            s.insert_entry(entry);
            Ok(id)
        })
    }

    fn append_custom_entry(
        &self,
        custom_type: &str,
        data: Option<serde_json::Value>,
    ) -> AsyncResult<String> {
        let custom_type = custom_type.to_string();
        let state = self.state.clone();
        Box::pin(async move {
            let mut s = state.write().await;
            let id = s.alloc_id();
            let parent_id = s.leaf_id.clone();
            let entry = SessionEntry::Custom(CustomEntry {
                id: id.clone(),
                parent_id,
                timestamp: State::now_timestamp(),
                custom_type,
                data,
            });
            s.insert_entry(entry);
            Ok(id)
        })
    }

    fn append_custom_message_entry(
        &self,
        custom_type: &str,
        content: CustomMessageContent,
        display: bool,
        details: Option<serde_json::Value>,
    ) -> AsyncResult<String> {
        let custom_type = custom_type.to_string();
        let state = self.state.clone();
        Box::pin(async move {
            let mut s = state.write().await;
            let id = s.alloc_id();
            let parent_id = s.leaf_id.clone();
            let entry = SessionEntry::CustomMessage(CustomMessageEntry {
                id: id.clone(),
                parent_id,
                timestamp: State::now_timestamp(),
                custom_type,
                content,
                display,
                details,
            });
            s.insert_entry(entry);
            Ok(id)
        })
    }

    fn move_to(
        &self,
        entry_id: Option<&str>,
        summary: Option<BranchSummaryData>,
    ) -> AsyncResult<Option<String>> {
        let entry_id = entry_id.map(String::from);
        let state = self.state.clone();
        Box::pin(async move {
            let mut s = state.write().await;

            // Update leaf pointer.
            s.leaf_id = entry_id;

            // Optionally append a branch summary.
            let summary_id = match summary {
                Some(data) => {
                    let id = s.alloc_id();
                    let parent_id = s.leaf_id.clone();
                    let entry_id_for_from = parent_id.clone().unwrap_or_default();
                    let entry = SessionEntry::BranchSummary(BranchSummaryEntry {
                        id: id.clone(),
                        parent_id,
                        timestamp: State::now_timestamp(),
                        from_id: entry_id_for_from,
                        summary: data.summary,
                        details: data.details,
                        from_hook: data.from_hook,
                    });
                    s.insert_entry(entry);
                    Some(id)
                }
                None => None,
            };

            Ok(summary_id)
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ameli_ai::types::{AssistantContentBlock, TextContent};

    fn test_user_message(text: &str) -> AgentMessage {
        AgentMessage::User(ameli_ai::types::UserMessage::text(text))
    }

    // -- Empty session --

    #[tokio::test]
    async fn empty_session() {
        let sm = InMemorySessionManager::new();
        assert!(sm.leaf_id().await.unwrap().is_none());
        assert!(sm.entries().await.unwrap().is_empty());
        assert!(sm.branch(None).await.unwrap().is_empty());

        let ctx = sm.build_context().await.unwrap();
        assert!(ctx.messages.is_empty());
        assert_eq!(ctx.thinking_level, "off");
        assert!(ctx.model.is_none());
    }

    // -- Metadata --

    #[tokio::test]
    async fn metadata_auto_generated() {
        let sm = InMemorySessionManager::new();
        let meta = sm.metadata().await.unwrap();
        assert!(meta.id().starts_with("session-"));
        assert!(!meta.created_at().is_empty());
    }

    #[tokio::test]
    async fn metadata_custom() {
        let sm = InMemorySessionManager::with_metadata(InMemoryMetadata::new(
            "custom-id",
            "2026-01-01T00:00:00Z",
        ));
        let meta = sm.metadata().await.unwrap();
        assert_eq!(meta.id(), "custom-id");
        assert_eq!(meta.created_at(), "2026-01-01T00:00:00Z");
    }

    // -- Append + lookup --

    #[tokio::test]
    async fn append_message_and_lookup() {
        let sm = InMemorySessionManager::new();

        let id = sm.append_message(test_user_message("hello")).await.unwrap();

        let entry = sm.entry(&id).await.unwrap().unwrap();
        assert_eq!(entry.entry_type(), "message");

        if let SessionEntry::Message(me) = &entry {
            assert_eq!(me.message.role(), "user");
            assert!(me.parent_id.is_none());
        } else {
            panic!("Expected Message entry");
        }

        assert_eq!(sm.leaf_id().await.unwrap(), Some(id));
    }

    #[tokio::test]
    async fn append_chains_parent_ids() {
        let sm = InMemorySessionManager::new();

        let id1 = sm.append_message(test_user_message("first")).await.unwrap();
        let id2 = sm
            .append_message(test_user_message("second"))
            .await
            .unwrap();

        let e1 = sm.entry(&id1).await.unwrap().unwrap();
        let e2 = sm.entry(&id2).await.unwrap().unwrap();
        assert!(e1.parent_id().is_none());
        assert_eq!(e2.parent_id(), Some(id1.as_str()));
    }

    // -- Branch walking --

    #[tokio::test]
    async fn branch_walks_root_to_leaf() {
        let sm = InMemorySessionManager::new();

        let id1 = sm.append_message(test_user_message("a")).await.unwrap();
        let _id2 = sm.append_message(test_user_message("b")).await.unwrap();
        let id3 = sm.append_message(test_user_message("c")).await.unwrap();

        let branch = sm.branch(None).await.unwrap();
        assert_eq!(branch.len(), 3);
        assert_eq!(branch[0].id(), id1);
        assert_eq!(branch[2].id(), id3);
    }

    #[tokio::test]
    async fn branch_from_specific_id() {
        let sm = InMemorySessionManager::new();

        let id1 = sm.append_message(test_user_message("a")).await.unwrap();
        let id2 = sm.append_message(test_user_message("b")).await.unwrap();

        // Branch from id2 should only include a + b
        let branch = sm.branch(Some(&id2)).await.unwrap();
        assert_eq!(branch.len(), 2);
        assert_eq!(branch[0].id(), id1);
        assert_eq!(branch[1].id(), id2);
    }

    // -- Build context --

    #[tokio::test]
    async fn build_context_with_messages() {
        let sm = InMemorySessionManager::new();

        sm.append_message(test_user_message("hello")).await.unwrap();
        sm.append_message(AgentMessage::Assistant(ameli_ai::types::AssistantMessage {
            content: vec![AssistantContentBlock::Text(TextContent::new("hi"))],
            api: "test".into(),
            provider: "test".into(),
            model: "test".into(),
            response_model: None,
            response_id: None,
            usage: ameli_ai::types::Usage::default(),
            stop_reason: ameli_ai::types::StopReason::Stop,
            error_message: None,
            timestamp: 1000,
        }))
        .await
        .unwrap();

        let ctx = sm.build_context().await.unwrap();
        assert_eq!(ctx.messages.len(), 2);
    }

    #[tokio::test]
    async fn build_context_tracks_thinking_level_and_model() {
        let sm = InMemorySessionManager::new();

        sm.append_message(test_user_message("hello")).await.unwrap();
        sm.append_thinking_level_change("high").await.unwrap();
        sm.append_model_change("openai", "gpt-4o").await.unwrap();

        let ctx = sm.build_context().await.unwrap();
        assert_eq!(ctx.thinking_level, "high");
        assert_eq!(
            ctx.model,
            Some(ModelRef {
                provider: "openai".into(),
                model_id: "gpt-4o".into(),
            })
        );
    }

    // -- Move to / branching --

    #[tokio::test]
    async fn move_to_changes_leaf() {
        let sm = InMemorySessionManager::new();

        let id1 = sm.append_message(test_user_message("a")).await.unwrap();
        let _id2 = sm.append_message(test_user_message("b")).await.unwrap();

        // Move back to id1.
        sm.move_to(Some(&id1), None).await.unwrap();
        assert_eq!(sm.leaf_id().await.unwrap(), Some(id1.clone()));

        // Append creates a child of id1.
        let id3 = sm.append_message(test_user_message("c")).await.unwrap();
        let e3 = sm.entry(&id3).await.unwrap().unwrap();
        assert_eq!(e3.parent_id(), Some(id1.as_str()));
    }

    #[tokio::test]
    async fn move_to_with_summary() {
        let sm = InMemorySessionManager::new();

        let id1 = sm.append_message(test_user_message("a")).await.unwrap();

        let summary_id = sm
            .move_to(
                Some(&id1),
                Some(BranchSummaryData {
                    summary: "abandoned branch".into(),
                    details: None,
                    from_hook: false,
                }),
            )
            .await
            .unwrap();

        assert!(summary_id.is_some());
        let entry = sm
            .entry(summary_id.as_ref().unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(entry.entry_type(), "branch_summary");
    }

    // -- All entry types --

    #[tokio::test]
    async fn append_all_entry_types() {
        let sm = InMemorySessionManager::new();

        let id1 = sm.append_message(test_user_message("hello")).await.unwrap();
        let id2 = sm.append_thinking_level_change("high").await.unwrap();
        let id3 = sm.append_model_change("openai", "gpt-4o").await.unwrap();
        let id4 = sm
            .append_compaction("summary", &id1, 100, None, false)
            .await
            .unwrap();
        let id5 = sm
            .append_custom_entry("my-ext", Some(serde_json::json!({"key": "val"})))
            .await
            .unwrap();
        let id6 = sm
            .append_custom_message_entry(
                "context",
                CustomMessageContent::Text("some context".into()),
                true,
                None,
            )
            .await
            .unwrap();

        let entries = sm.entries().await.unwrap();
        assert_eq!(entries.len(), 6);

        assert_eq!(
            sm.entry(&id1).await.unwrap().unwrap().entry_type(),
            "message"
        );
        assert_eq!(
            sm.entry(&id2).await.unwrap().unwrap().entry_type(),
            "thinking_level_change"
        );
        assert_eq!(
            sm.entry(&id3).await.unwrap().unwrap().entry_type(),
            "model_change"
        );
        assert_eq!(
            sm.entry(&id4).await.unwrap().unwrap().entry_type(),
            "compaction"
        );
        assert_eq!(
            sm.entry(&id5).await.unwrap().unwrap().entry_type(),
            "custom"
        );
        assert_eq!(
            sm.entry(&id6).await.unwrap().unwrap().entry_type(),
            "custom_message"
        );
    }

    // -- Label (not supported) --

    #[tokio::test]
    async fn label_returns_none() {
        let sm = InMemorySessionManager::new();
        let id = sm.append_message(test_user_message("hello")).await.unwrap();
        assert!(sm.label(&id).await.unwrap().is_none());
    }

    // -- Entry not found --

    #[tokio::test]
    async fn entry_not_found() {
        let sm = InMemorySessionManager::new();
        assert!(sm.entry("nonexistent").await.unwrap().is_none());
    }

    // -- Trait object safety --

    #[tokio::test]
    async fn trait_object_works() {
        let sm: Arc<dyn SessionManager<InMemoryMetadata>> = Arc::new(InMemorySessionManager::new());
        sm.append_message(test_user_message("hello")).await.unwrap();
        let ctx = sm.build_context().await.unwrap();
        assert_eq!(ctx.messages.len(), 1);
    }
}
