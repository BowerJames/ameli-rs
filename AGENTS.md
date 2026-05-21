# AGENTS.md
This document provides guidelines for AI coding agents working on this codebase.

# Development Guidelines
These are development guidelines that should be considered whenever planning, developing or reviewing all feature and bugs. They must be adhered to wherever possible and highlighted when they are not.

- Use explicit error handling by propagating Result and Option via the ? operator or combinators; do not use .unwrap() or .expect().