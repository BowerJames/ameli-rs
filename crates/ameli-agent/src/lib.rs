//! Higher-level, configurable agent built on top of `ameli-agent-core`.
//!
//! This crate will provide a configurable agent with abstracted session
//! management (via a trait so different session backends can be plugged in)
//! and a general agent environment trait for different execution environments.
//!
//! It is the Rust equivalent of the TypeScript `pi-coding-agent` package,
//! adapted for greater configurability.
