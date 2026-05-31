//! Session management trait and in-memory implementation.
//!
//! This crate defines [`SessionManager<M>`] — the single trait that session
//! backends implement — and provides [`InMemorySessionManager`] as a
//! reference implementation backed by interior-mutable `HashMap` storage.
//!
//! # Architecture
//!
//! ```text
//! SessionMetadata        ← trait for session identity (ID, creation time)
//! SessionManager<M>      ← trait for session operations
//! InMemorySessionManager ← reference implementation (tree-based, in-memory)
//! ```
//!
//! # Session Types
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
pub mod in_memory;
pub mod manager;
pub mod types;

// Re-export primary types for convenience.
pub use error::SessionError;
pub use in_memory::{InMemoryMetadata, InMemorySessionManager};
pub use manager::{AsyncResult, BranchSummaryData, SessionManager, SessionMetadata};
pub use types::{
    BranchSummaryEntry, CompactionEntry, CustomEntry, CustomMessageContent, CustomMessageEntry,
    MessageEntry, ModelChangeEntry, ModelRef, SessionContext, SessionEntry, SessionMessage,
    ThinkingLevelChangeEntry,
};
