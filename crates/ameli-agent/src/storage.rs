//! Session metadata and storage traits.
//!
//! Defines the [`SessionMetadata`] trait for session identity and the
//! [`SessionStorage`] trait that abstracts the storage backend. Concrete
//! implementations (in-memory, JSONL, database, etc.) implement these traits
//! and are injected into [`Session`](crate::Session).

use crate::error::SessionError;
use crate::types::SessionEntry;
use std::future::Future;
use std::pin::Pin;

// ---------------------------------------------------------------------------
// Type alias for boxed async results
// ---------------------------------------------------------------------------

/// Boxed, sendable async result used by [`SessionStorage`] trait methods.
///
/// Using `Pin<Box<dyn Future>>` ensures the trait is dyn-compatible
/// (object-safe), so `Arc<dyn SessionStorage<M>>` works.
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
/// use ameli_agent::storage::SessionMetadata;
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
// SessionStorage
// ---------------------------------------------------------------------------

/// Storage backend for a session tree.
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
/// Implementations must be `Send + Sync` so that `Session` can be shared
/// across async tasks. Interior mutability ensures concurrent reads are
/// not blocked by each other.
pub trait SessionStorage<M: SessionMetadata>: Send + Sync {
    /// Returns the session metadata.
    fn metadata(&self) -> AsyncResult<M>;

    /// Returns the current leaf entry ID, or `None` if the session is empty.
    fn leaf_id(&self) -> AsyncResult<Option<String>>;

    /// Set the active leaf entry ID.
    ///
    /// After this call, the next append will create a child of the new leaf.
    /// Pass `None` to reset to before the first entry (for re-editing the
    /// first user message).
    fn set_leaf_id(&self, leaf_id: Option<&str>) -> AsyncResult<()>;

    /// Generate a unique entry ID for a new entry.
    fn create_entry_id(&self) -> AsyncResult<String>;

    /// Append an entry as a child of the current leaf, then advance the leaf.
    fn append_entry(&self, entry: SessionEntry) -> AsyncResult<()>;

    /// Look up a single entry by ID.
    fn get_entry(&self, id: &str) -> AsyncResult<Option<SessionEntry>>;

    /// Return the resolved label for an entry, if any.
    fn get_label(&self, id: &str) -> AsyncResult<Option<String>>;

    /// Walk from the given leaf to the root, returning entries in root-to-leaf
    /// order. If `leaf_id` is `None`, uses the current leaf.
    fn path_to_root(&self, leaf_id: Option<&str>) -> AsyncResult<Vec<SessionEntry>>;

    /// Return all entries in the session (excluding any internal metadata).
    fn entries(&self) -> AsyncResult<Vec<SessionEntry>>;
}
