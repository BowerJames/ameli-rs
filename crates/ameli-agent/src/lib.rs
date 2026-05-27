//! Higher-level, configurable agent built on top of `ameli-agent-core`.
//!
//! This crate provides a configurable agent with abstracted session
//! management (via a trait so different session backends can be plugged in)
//! and a general agent environment trait for different execution environments.
//!
//! # Session Management
//!
//! The session system is built around two core abstractions:
//!
//! - [`SessionMetadata`] — trait for session identity (ID, creation time).
//!   Different backends extend this with their own fields.
//! - [`SessionManager`] — trait for session operations. Implementations
//!   decide their own ID generation, persistence strategy, and internals.
//!
//! # Session Tree
//!
//! Sessions are append-only trees of [`SessionEntry`] values. Each entry has
//! an `id` and `parent_id` forming the tree. The active "leaf" tracks the
//! current position. Branching moves the leaf to an earlier entry, allowing
//! new branches without modifying history.
//!
//! # Session Messages
//!
//! [`SessionMessage`] preserves type identity through context building.
//! Compaction and branch summary entries are **not** converted to
//! [`AgentMessage`] during context building — instead they become
//! [`SessionMessage::Compaction`] and [`SessionMessage::BranchSummary`]
//! variants. The future `AgentSession` converts these to `AgentMessage`,
//! consulting extension formatting hooks, guaranteeing that extensions
//! can customize how summaries appear in LLM context.
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
pub mod interface;
pub mod session_manager;
pub mod types;

// Re-export primary types for convenience.
pub use error::SessionError;
pub use extension::{Extension, ExtensionApi, ExtensionContext};
pub use interface::{CustomNotifyMessage, Interface, NoopInterface, NotifyKind, NotifyMessage};
pub use session_manager::{BranchSummaryData, SessionManager, SessionMetadata};
pub use types::{
    BranchSummaryEntry, CompactionEntry, CustomEntry, CustomMessageContent, CustomMessageEntry,
    MessageEntry, ModelChangeEntry, ModelRef, SessionContext, SessionEntry, SessionMessage,
    ThinkingLevelChangeEntry,
};
