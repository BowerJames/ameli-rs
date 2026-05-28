# AGENTS.md
This document provides guidelines for AI coding agents working on this codebase.

# Development

## Development CLI Tools
- `gh` - GitHub CLI. Used for GitHub operations (creating PRs, viewing issues, etc.).
- `tmux` - tmux is a terminal multiplexer: it enables a number of terminals to be created, accessed, and controlled from a single screen. tmux may be detached from a screen and continue running in the background, then later reattached. 


## Development Guidelines
These are development guidelines that should be considered whenever planning, developing or reviewing all feature and bugs. They must be adhered to wherever possible and highlighted when they are not.

- Use explicit error handling by propagating Result and Option via the ? operator or combinators; do not use .unwrap() or .expect().
- `cargo fmt --check` must pass with no errors (code must be formatted).
- `cargo clippy` must pass with no warnings or errors.
- Do not add lint suppressions (e.g. `#[allow(clippy::...)]` or `#[expect(clippy::...)]`) without express permission from the user. If you encounter a situation where suppressing a lint seems warranted, raise it to the user immediately.

# Architecture & Design Decisions

This project is a Rust port of the [pi](https://github.com/earendil-works/pi) TypeScript AI agent framework, but with deliberate architectural differences. Coding agents working on this codebase must understand these decisions and follow them when planning and implementing changes.

## General-Purpose Agent, Not a Coding Agent

`ameli-agent` is inspired by `pi-coding-agent` but is **not** a specialised coding agent. It is a general-purpose, configurable agent framework. This means:

- No coding-specific assumptions are baked into the core crates (`ameli-ai`, `ameli-agent-core`, `ameli-agent`).
- Downstream applications (coding agents, chatbots, automation tools, etc.) specialise the framework by providing their own tools, extensions, session storage, and UI.
- When porting or adapting patterns from the pi reference code, always ask: "Does this belong in a general-purpose agent framework, or is it specific to a coding agent use case?" If the latter, it belongs in a downstream application, not in these crates.

## SessionManager Trait vs Concrete Implementation

pi-coding-agent ships a concrete session implementation (file-backed, with a specific schema and storage format). **ameli-agent defines `SessionManager<M>` as a trait only.** This is an intentional design decision:

- Downstream applications implement the trait against their own storage backend (files, databases, in-memory, etc.).
- The shared `build_session_context_from_path` helper handles universal context-building logic so implementations only need to manage persistence and tree traversal.
- Do not introduce a concrete `SessionManager` implementation into `ameli-agent`. If a default implementation is needed for testing, keep it in tests or behind a feature flag — never as part of the public API.

## Interface Trait vs UI Provider

pi-coding-agent has a broad UI provider that handles rendering, commands, and interactive features. **ameli-agent defines an `Interface` trait with a deliberately reduced scope.** The `Interface` trait is limited to:

- Sending `NotifyMessage` values to the application layer (tool call notifications, compaction events, errors, etc.).
- Providing a minimal abstraction for the agent to communicate state changes outward.

It does **not** handle rendering, keyboard input, commands, or any interactive UI concerns. Those responsibilities belong to the downstream application. Do not expand the `Interface` trait to include UI-specific functionality.

## Tools Are Managed by Extensions, Not Built-In

pi-coding-agent ships with built-in coding tools (file read/write, bash execution, search, etc.). **These tools will not be ported to ameli-agent.** Instead:

- Tools are registered by extensions via `ExtensionApi::register_tool()`.
- The `AgentTool` trait is the contract; any tool an extension registers is available to the model during agent runs.
- The core crates ship with **no default tools**. This is by design — the framework is agnostic to what tools an agent needs.
- Do not add built-in tool implementations to `ameli-agent-core` or `ameli-agent`. If a common tool is needed for testing, keep it in test modules only.

# Code Review
To launch an independent code review run:

```bash
mypi --profile reviewer "/review <issue_number> <target_branch>"
```

# References

This project is a Rust adaptation of the [pi](https://github.com/earendil-works/pi) TypeScript AI agent framework. The reference code lives in `reference/` and should be consulted for architectural context, but the Rust implementation intentionally diverges in the areas described in the **Architecture & Design Decisions** section above.

If any reference projects are missing from `reference/`, ask the user to clone them:

- [pi](https://github.com/earendil-works/pi.git): Contains `pi-ai` and `pi-agent-core` (TypeScript inspirations for `ameli-ai` and `ameli-agent-core`) and `pi-coding-agent` (TypeScript inspiration for the higher-level agent pattern adapted in `ameli-agent`).