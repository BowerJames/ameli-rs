//! `ameli run` command handler.
//!
//! Wires up model registry, auth storage, session manager, extensions, and
//! launches the interactive TUI.

pub mod extension_loader;
pub mod tui;

use crate::cli::RunArgs;
use ameli_agent::auth_storage::InMemoryAuthStorage;
use ameli_agent::session_manager::{InMemorySessionManager, ModelRef};
use ameli_agent::{create_agent_session, CreateAgentSessionOptions, NoopInterface};
use ameli_agent_core::types::ThinkingLevel;
use ameli_model_registry::DefaultModelRegistry;
use anyhow::Result;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Execute the `ameli run` command.
pub async fn run_run(args: RunArgs) -> Result<()> {
    // 1. Resolve model from the global registry
    let model = ameli_model_registry::get_model(&args.provider, &args.model)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // 2. Parse thinking level
    let thinking_level = parse_thinking_level(&args.thinking);

    // 3. Auth storage — store API key if provided
    let auth_storage = Arc::new(InMemoryAuthStorage::new());
    if let Some(ref key) = args.api_key {
        auth_storage.set_api_key(&args.provider, key.clone());
    }

    // 4. Model registry — new registry with just the resolved model
    let model_registry = Arc::new(DefaultModelRegistry::new());
    model_registry.register(model.clone());

    // 5. Load extensions from dylib paths
    let ext_set = extension_loader::load_extension_set(&args.extension)?;

    // 6. Create the agent session
    let session = create_agent_session(CreateAgentSessionOptions {
        model: ModelRef {
            provider: args.provider.clone(),
            model_id: args.model.clone(),
        },
        model_registry,
        auth_storage,
        session_manager: Arc::new(InMemorySessionManager::new()),
        interface: Arc::new(NoopInterface),
        extensions: ext_set.extensions,
        thinking_level: Some(thinking_level),
        system_prompt: None,
    })
    .await
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    // 7. Launch TUI — this blocks until the user exits
    tui::run(Arc::new(session.session)).await?;

    // ext_set._libraries are dropped here, unloading extension dylibs
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a thinking level string into a [`ThinkingLevel`].
///
/// Falls back to [`ThinkingLevel::Off`] for unrecognized values.
fn parse_thinking_level(s: &str) -> ThinkingLevel {
    match s.to_lowercase().as_str() {
        "minimal" => ThinkingLevel::Minimal,
        "low" => ThinkingLevel::Low,
        "medium" => ThinkingLevel::Medium,
        "high" => ThinkingLevel::High,
        "xhigh" => ThinkingLevel::XHigh,
        _ => ThinkingLevel::Off,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_thinking_level_known_values() {
        assert_eq!(parse_thinking_level("off"), ThinkingLevel::Off);
        assert_eq!(parse_thinking_level("minimal"), ThinkingLevel::Minimal);
        assert_eq!(parse_thinking_level("low"), ThinkingLevel::Low);
        assert_eq!(parse_thinking_level("medium"), ThinkingLevel::Medium);
        assert_eq!(parse_thinking_level("high"), ThinkingLevel::High);
        assert_eq!(parse_thinking_level("xhigh"), ThinkingLevel::XHigh);
    }

    #[test]
    fn parse_thinking_level_case_insensitive() {
        assert_eq!(parse_thinking_level("Medium"), ThinkingLevel::Medium);
        assert_eq!(parse_thinking_level("HIGH"), ThinkingLevel::High);
    }

    #[test]
    fn parse_thinking_level_unknown_falls_back_to_off() {
        assert_eq!(parse_thinking_level("unknown"), ThinkingLevel::Off);
        assert_eq!(parse_thinking_level(""), ThinkingLevel::Off);
    }
}
