//! Entry point for the `ameli` CLI binary.

use anyhow::Result;
use clap::Parser;

mod cli;
mod run;

#[tokio::main]
async fn main() -> Result<()> {
    let args = cli::Cli::parse();
    cli::run(args).await
}
