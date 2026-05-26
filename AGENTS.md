# AGENTS.md
This document provides guidelines for AI coding agents working on this codebase.

# Development

## Development CLI Tools
- `gh` - GitHub CLI. Used for GitHub operations (creating PRs, viewing issues, etc.).
- `tmux` - tmux is a terminal multiplexer: it enables a number of terminals to be created, accessed, and controlled from a single screen. tmux may be detached from a screen and continue running in the background, then later reattached. 


## Development Guidelines
These are development guidelines that should be considered whenever planning, developing or reviewing all feature and bugs. They must be adhered to wherever possible and highlighted when they are not.

- Use explicit error handling by propagating Result and Option via the ? operator or combinators; do not use .unwrap() or .expect().

# Code Review
To launch an independent code review run:

```bash
mypi --profile reviewer "/review <issue_number> <target_branch>"
```