# Todo Extension Example

A standalone extension for `ameli run` that provides a persistent todo list. This example validates the dynamic extension loading feature.

## What It Provides

### LLM Tools

The extension registers three tools that the model can call during conversation:

- **`todo_add`** — Add a new todo item (`text` parameter)
- **`todo_list`** — List all todo items
- **`todo_complete`** — Mark a todo as completed (`id` parameter)

### TUI Commands

The extension registers three slash commands for direct user interaction:

- **`/todos`** — List all todos
- **`/todo-add <text>`** — Add a new todo
- **`/todo-complete <id>`** — Complete a todo by ID

### System Prompt

A `before_agent_start` hook appends todo-related instructions and the current todo list state to the system prompt on each prompt cycle. The injection is idempotent — previous injections are stripped before re-appending fresh state.

## Building

```sh
cd examples/todo-extension
cargo build --release
```

## Usage

```sh
# Build the extension first
cd examples/todo-extension && cargo build --release && cd ../..

# Run ameli with the extension
ameli run -p openai -m gpt-4o -e examples/todo-extension/target/release/libtodo_extension.so
```

On macOS the dylib has a `.dylib` extension; on Windows it has `.dll`.

## Architecture

- Shared state: `Arc<Mutex<Vec<TodoItem>>>` accessible from tools, commands, and hooks
- Three separate tools (not one tool with actions) to validate multiple tool registration through dynamic loading
- Commands use `ctx.extension_context.interface.notify()` for output, which renders in the TUI chat log via `TuiInterface`
- Extension compiles as `cdylib` and exports `ameli_create_extension` FFI entry point
