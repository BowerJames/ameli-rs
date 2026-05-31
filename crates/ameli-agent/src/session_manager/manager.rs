//! Session manager trait and shared context-building helpers.
//!
//! Defines [`SessionManager<M>`] — the single trait that session backends
//! implement. Each implementation decides its own ID generation,
//! persistence strategy, and internal data structures.
//!
//! Implementations build [`SessionContext`] from their stored entries in
//! their [`build_context`](SessionManager::build_context) method, producing
//! [`SessionMessage`] values that preserve type identity — compaction and
//! branch summary entries are **not** converted to [`AgentMessage`] here.
//! That conversion happens in [`AgentSession`], which consults extension
//! formatting hooks.
//!
//! [`AgentSession`]: ameli_agent::AgentSession

use super::error::SessionError;
use super::types::{CustomMessageContent, SessionContext, SessionEntry};
use ameli_agent_core::types::AgentMessage;
use std::future::Future;
use std::pin::Pin;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Boxed, sendable async result used by [`SessionManager`] trait methods.
///
/// Using `Pin<Box<dyn Future>>` ensures the trait is dyn-compatible
/// (object-safe), so `Arc<dyn SessionManager<M>>` works.
pub type AsyncResult<T> = Pin<Box<dyn Future<Output = Result<T, SessionError>> + Send>>;

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
    /// Implementations walk their entry tree and produce [`SessionContext`]
    /// with [`SessionMessage`] values.
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
