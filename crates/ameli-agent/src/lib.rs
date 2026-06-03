//! Higher-level, configurable agent built on top of `ameli-agent-core`.
//!
//! This crate provides a configurable agent with abstracted session
//! management (via a trait so different session backends can be plugged in),
//! an extension system, and the [`AgentSession`] composition layer.
//!
//! # Architecture
//!
//! ```text
//! AgentSession<M>              ← composition layer
//!     ├── ArcAgent             ← stateful agent (from ameli-agent-core)
//!     ├── SessionManager<M>    ← session persistence trait (session_manager module)
//!     ├── AuthStorage          ← API key resolution (auth_storage module)
//!     ├── ExtensionRunner      ← extension event dispatch
//!     └── Interface            ← minimal UI abstraction
//! ```
//!
//! # Session Management
//!
//! The session system is built around two core abstractions in the
//! [`session_manager`] module:
//!
//! - [`session_manager::SessionMetadata`] — trait for session identity
//!   (ID, creation time). Different backends extend this with their own fields.
//! - [`session_manager::SessionManager`] — trait for session operations.
//!   Implementations decide their own ID generation, persistence strategy, and internals.
//!
//! [`AgentSession`] converts [`session_manager::SessionMessage`] variants
//! (including `Compaction` and `BranchSummary`) to [`AgentMessage`] using
//! extension formatting hooks, with default fallbacks when no extension overrides.
//!
//! # Auth Storage
//!
//! API key resolution is handled by the [`auth_storage`] module:
//!
//! - [`auth_storage::AuthStorage`] — trait for looking up API keys by provider.
//! - [`auth_storage::InMemoryAuthStorage`] — in-memory implementation with env var fallback.
//!
//! # Entry Types
//!
//! Seven entry types are supported by the session manager:
//!
//! - [`session_manager::MessageEntry`] — conversation messages (user, assistant, tool result)
//! - [`session_manager::ThinkingLevelChangeEntry`] — records thinking level changes
//! - [`session_manager::ModelChangeEntry`] — records model switches
//! - [`session_manager::CompactionEntry`] — summary of compacted conversation history
//! - [`session_manager::BranchSummaryEntry`] — summary of an abandoned branch
//! - [`session_manager::CustomEntry`] — extension state persistence (not in LLM context)
//! - [`session_manager::CustomMessageEntry`] — extension messages (in LLM context)

pub mod agent_session;
pub mod auth_storage;
pub mod error;
pub mod extension;
pub mod interface;
pub mod session_manager;

// Re-export primary types for convenience.
pub use agent_session::{
    create_agent_session, AgentSession, AgentSessionConfig, CreateAgentSessionOptions,
    CreateAgentSessionResult,
};
pub use error::CreateAgentSessionError;
pub use extension::{
    BeforeAgentStartEvent, BeforeAgentStartMessage, BeforeAgentStartResult, CommandContext,
    Extension, ExtensionApi, ExtensionContext, ExtensionError, ExtensionRunner, MessageEndResult,
    RegisteredCommand, SessionShutdownEvent, SessionShutdownReason, SessionStartEvent,
    SessionStartReason, ToolExecutionUpdateEvent,
};
pub use interface::{CustomNotifyMessage, Interface, NoopInterface, NotifyKind, NotifyMessage};
