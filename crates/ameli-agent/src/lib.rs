//! Higher-level, configurable agent built on top of `ameli-agent-core`.
//!
//! This crate provides a configurable agent with abstracted session
//! management (via a trait so different session backends can be plugged in),
//! an extension system, and the [`AgentSession`] composition layer.
//!
//! # Architecture
//!
//! ```text
//! AgentSession<M>        ← composition layer (new!)
//!     ├── ArcAgent       ← stateful agent (from ameli-agent-core)
//!     ├── SessionManager<M>  ← session persistence trait
//!     ├── ExtensionRunner    ← extension event dispatch
//!     └── Interface          ← minimal UI abstraction
//! ```
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
//! [`AgentSession`] converts [`SessionMessage`] variants (including
//! `Compaction` and `BranchSummary`) to [`AgentMessage`] using extension
//! formatting hooks, with default fallbacks when no extension overrides.
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

pub mod agent_session;
pub mod error;
pub mod extension;
pub mod interface;
pub mod session_manager;
pub mod types;

// Re-export primary types for convenience.
pub use agent_session::{AgentSession, AgentSessionConfig};
pub use error::SessionError;
pub use extension::{
    BeforeAgentStartEvent, BeforeAgentStartMessage, BeforeAgentStartResult, CommandContext,
    Extension, ExtensionApi, ExtensionContext, ExtensionError, ExtensionRunner, MessageEndResult,
    RegisteredCommand, SessionShutdownEvent, SessionShutdownReason, SessionStartEvent,
    SessionStartReason, ToolExecutionUpdateEvent,
};
pub use interface::{CustomNotifyMessage, Interface, NoopInterface, NotifyKind, NotifyMessage};
pub use session_manager::{BranchSummaryData, SessionManager, SessionMetadata};
pub use types::{
    BranchSummaryEntry, CompactionEntry, CustomEntry, CustomMessageContent, CustomMessageEntry,
    MessageEntry, ModelChangeEntry, ModelRef, SessionContext, SessionEntry, SessionMessage,
    ThinkingLevelChangeEntry,
};
