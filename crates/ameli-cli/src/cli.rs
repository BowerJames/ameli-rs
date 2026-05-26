//! Top-level CLI definitions and command dispatch for the `ameli` binary.
//!
//! Uses clap derive to define the command hierarchy:
//!
//! ```text
//! ameli
//!  └── ai
//!       └── complete   — single-shot LLM completion
//! ```

use ameli_ai::types::{
    AssistantContentBlock, Context, Cost, InputType, Message, Model, StreamOptions, TextContent,
    UserMessage,
};
use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use std::io::{self, IsTerminal};

// ---------------------------------------------------------------------------
// Top-level CLI
// ---------------------------------------------------------------------------

/// The ameli command-line tool.
#[derive(Debug, Parser)]
#[command(name = "ameli", version, about = "The ameli command-line tool")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

/// Top-level subcommands.
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Interact with the ameli AI layer.
    Ai {
        #[command(subcommand)]
        command: AiSubcommand,
    },
}

/// Subcommands under `ameli ai`.
#[derive(Debug, Subcommand)]
pub enum AiSubcommand {
    /// Perform a single-shot LLM completion and print the result.
    Complete(CompleteArgs),
}

// ---------------------------------------------------------------------------
// `ameli ai complete` arguments
// ---------------------------------------------------------------------------

/// Arguments for `ameli ai complete`.
#[derive(Debug, Parser)]
pub struct CompleteArgs {
    /// Model ID (e.g. "gpt-4o").
    #[arg(long)]
    pub model: String,

    /// Provider name (e.g. "openai").
    #[arg(long, default_value = "openai")]
    pub provider: String,

    /// Override the API base URL.
    #[arg(long, default_value = "https://api.openai.com/v1")]
    pub base_url: String,

    /// API key. Falls back to <PROVIDER>_API_KEY or OPENAI_API_KEY env vars if not set.
    #[arg(long)]
    pub api_key: Option<String>,

    /// System prompt to include with the request.
    #[arg(long)]
    pub system_prompt: Option<String>,

    /// Output the full AssistantMessage as JSON instead of text only.
    #[arg(long)]
    pub json: bool,

    /// The prompt text. Reads from stdin if not provided and stdin is piped.
    #[arg()]
    pub prompt: Option<String>,
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Parse CLI arguments and dispatch to the appropriate command handler.
pub async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Ai { command } => match command {
            AiSubcommand::Complete(args) => run_complete(args).await,
        },
    }
}

// ---------------------------------------------------------------------------
// `ameli ai complete` implementation
// ---------------------------------------------------------------------------

/// Execute a single-shot completion and print the result.
async fn run_complete(args: CompleteArgs) -> Result<()> {
    let prompt = resolve_prompt(&args.prompt)?;

    let model = Model {
        id: args.model.clone(),
        name: args.model,
        api: "openai-completions".into(),
        provider: args.provider.clone(),
        base_url: args.base_url,
        reasoning: false,
        thinking_level_map: None,
        input: vec![InputType::Text],
        cost: Cost::default(),
        context_window: 128_000,
        max_tokens: 16_384,
        compat: None,
    };

    let context = Context {
        system_prompt: args.system_prompt,
        messages: vec![Message::User(UserMessage::text(prompt))],
        tools: None,
    };

    let options = StreamOptions {
        api_key: args.api_key,
        ..Default::default()
    };

    let message = ameli_ai::complete_simple_global(&model, context, options).await;

    if message.stop_reason == ameli_ai::types::StopReason::Error {
        let error = message
            .error_message
            .unwrap_or_else(|| "unknown error".to_string());
        bail!("{}", error);
    }

    if args.json {
        let json = serde_json::to_string_pretty(&message)?;
        println!("{json}");
    } else {
        let text_blocks: Vec<&str> = message
            .content
            .iter()
            .filter_map(|block| match block {
                AssistantContentBlock::Text(TextContent { text, .. }) if !text.is_empty() => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect();

        if text_blocks.is_empty() {
            bail!("No text content in response (use --json to see full output)");
        }

        for text in &text_blocks {
            print!("{text}");
        }
        println!();
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve prompt text from the positional arg or stdin.
fn resolve_prompt(arg: &Option<String>) -> Result<String> {
    if let Some(text) = arg {
        if !text.is_empty() {
            return Ok(text.clone());
        }
    }

    if !io::stdin().is_terminal() {
        let mut input = String::new();
        io::Read::read_to_string(&mut io::stdin(), &mut input)?;
        let trimmed = input.trim().to_string();
        if !trimmed.is_empty() {
            return Ok(trimmed);
        }
    }

    bail!("No prompt provided. Pass a positional argument or pipe to stdin. Use --help for usage.");
}
