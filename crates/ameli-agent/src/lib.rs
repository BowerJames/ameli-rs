//! Higher-level, configurable agent built on top of `ameli-agent-core`.
//!
//! This crate provides a configurable agent with abstracted session
//! management (via a trait so different session backends can be plugged in)
//! and a general agent environment trait for different execution environments.
//!
//! # Session Management
//!
//! The session system is built around three core abstractions:
//!
//! - [`SessionMetadata`] — trait for session identity (ID, creation time).
//!   Different backends extend this with their own fields.
//! - [`SessionStorage`] — async trait for the storage backend (in-memory,
//!   JSONL file, database, etc.).
//! - [`Session`] — high-level API that wraps a storage backend and provides
//!   typed methods for appending entries and building context.
//!
//! # Session Tree
//!
//! Sessions are append-only trees of [`SessionEntry`] values. Each entry has
//! an `id` and `parent_id` forming the tree. The active "leaf" tracks the
//! current position. Branching moves the leaf to an earlier entry, allowing
//! new branches without modifying history.
//!
//! # Entry Types
//!
//! Seven entry types are supported:
//!
//! - [`MessageEntry`] — conversation messages (user, assistant, tool result)
//! - [`ThinkingLevelChangeEntry`] — records thinking level changes
//! - [`ModelChangeEntry`] — records model switches
//! - [`CompactionEntry`] — summary of compacted conversation history
//! - [`BranchSummaryEntry`] — summary of an abandoned branch
//! - [`CustomEntry`] — extension state persistence (not in LLM context)
//! - [`CustomMessageEntry`] — extension messages (in LLM context)

pub mod error;
pub mod extension;
pub mod session;
pub mod storage;
pub mod types;

// Re-export primary types for convenience.
pub use error::SessionError;
pub use extension::{Extension, ExtensionApi, ExtensionContext};
pub use session::{BranchSummaryData, Session};
pub use storage::{SessionMetadata, SessionStorage};
pub use types::{
    BranchSummaryEntry, CompactionEntry, CustomEntry, CustomMessageContent, CustomMessageEntry,
    MessageEntry, ModelChangeEntry, ModelRef, SessionContext, SessionEntry,
    ThinkingLevelChangeEntry,
};
