//! Higher-level, configurable agent built on top of `ameli-agent-core`.
//!
//! This crate provides a configurable agent with abstracted session
//! management (via a trait so different session backends can be plugged in),
//! an extension system, and the [`AgentSession`] composition layer.
//!
//! # Architecture
//!
//! ```text
//! AgentSession                ← composition layer
//!     ├── ArcAgent            ← stateful agent (from ameli-agent-core)
//!     ├── SessionManager      ← session persistence trait (session_manager module)
//!     ├── AuthStorage         ← API key resolution (auth_storage module)
//!     ├── ExtensionRunner     ← extension event dispatch
//!     ├── ExtensionActions    ← extension runtime actions (Weak<Agent>)
//!     └── Interface           ← minimal UI abstraction
//! ```
//!
//! # Construction
//!
//! [`create_agent_session`] is the primary entry point. The construction
//! order is designed so that every structure is fully built at creation:
//!
//! 1. Resolve model + validate API key
//! 2. Create empty `ExtensionRunner`
//! 3. Construct `ArcAgent` (no hooks or tools yet)
//! 4. Construct `ExtensionActions` with `Weak<Agent>` (fully wired)
//! 5. Initialize extensions (register hooks/tools into runner)
//! 6. Install extension hooks on the agent
//! 7. Set tools from extensions on the agent
//! 8. Create `AgentSession` (subscribe + emit session_start)
//! 9. Restore/init session context
//!
//! # Session Management
//!
//! The session system is built around two core abstractions in the
//! [`session_manager`] module:
//!
//! - [`session_manager::SessionMetadata`] — concrete struct for session identity
//!   (ID, creation time).
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
    Extension, ExtensionActions, ExtensionApi, ExtensionContext, ExtensionError, ExtensionRunner,
    MessageEndResult, MessageMode, RegisteredCommand, SessionShutdownEvent, SessionShutdownReason,
    SessionStartEvent, SessionStartReason, ToolExecutionUpdateEvent,
};
pub use interface::{CustomNotifyMessage, Interface, NoopInterface, NotifyKind, NotifyMessage};
