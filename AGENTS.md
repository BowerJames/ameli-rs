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

# Code Review
To launch an independent code review run:

```bash
mypi --profile reviewer "/review <issue_number> <target_branch>"
```

# References
This project is heavily inspired by repositories that can be found in the `reference/` folder. Below are the projects that should be cloned into `reference/`. If any are not there you should ask the user to clone them.

- [pi](https://github.com/earendil-works/pi.git): Contains `pi-ai` and `pi-agent-core` which are the typescript inspirations for `ameli-ai` and `ameli-agent`.