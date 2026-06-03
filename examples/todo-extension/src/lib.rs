//! Todo extension example for `ameli run`.
//!
//! A standalone extension that demonstrates dynamic extension loading by
//! providing:
//!
//! - Three LLM-callable tools (`todo_add`, `todo_list`, `todo_complete`)
//! - Three TUI commands (`/todos`, `/todo-add`, `/todo-complete`)
//! - A `before_agent_start` hook that injects todo instructions into the
//!   system prompt
//!
//! # Building
//!
//! ```sh
//! cd examples/todo-extension
//! cargo build --release
//! ```
//!
//! # Usage
//!
//! ```sh
//! ameli run -p openai -m gpt-4o -e <path-to-dylib>
//! ```

use ameli_agent::extension::{BeforeAgentStartResult, Extension, ExtensionApi};
use ameli_agent::interface::NotifyMessage;
use ameli_agent_core::types::{AgentTool, AgentToolResult, ToolExecutionMode};
use ameli_ai::types::Tool;
use serde_json::Value;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// TodoItem
// ---------------------------------------------------------------------------

/// A single todo item in the shared list.
#[derive(Clone, Debug)]
struct TodoItem {
    id: usize,
    text: String,
    completed: bool,
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// Shared todo list state accessible from tools, commands, and hooks.
type SharedTodos = Arc<Mutex<Vec<TodoItem>>>;

/// Marker injected into the system prompt so the `before_agent_start` hook
/// can find and strip a previous injection before re-appending fresh state.
const INJECTION_MARKER: &str = "\n\n---\n[Todo Extension]";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Format the todo list as a human-readable string.
fn format_todos(todos: &[TodoItem]) -> String {
    if todos.is_empty() {
        return "No todos yet.".to_string();
    }
    todos
        .iter()
        .map(|t| {
            let check = if t.completed { "x" } else { " " };
            format!("{}. [{}] {}", t.id, check, t.text)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build the system prompt suffix with todo instructions and current state.
fn build_system_prompt_suffix(todos: &[TodoItem]) -> String {
    format!(
        "{marker}\n\
         You have access to a todo list that persists across this conversation.\n\
         You can manage it using the following tools:\n\
         - todo_add: Add a new todo item (param: text)\n\
         - todo_list: List all todo items\n\
         - todo_complete: Mark a todo item as completed (param: id)\n\
         \n\
         Current todo list:\n\
         {list}",
        marker = INJECTION_MARKER,
        list = format_todos(todos),
    )
}

/// Strip a previous todo extension injection from the system prompt.
fn strip_previous_injection(system_prompt: &str) -> String {
    if let Some(pos) = system_prompt.find(INJECTION_MARKER) {
        system_prompt[..pos].to_string()
    } else {
        system_prompt.to_string()
    }
}

// ---------------------------------------------------------------------------
// TodoAddTool
// ---------------------------------------------------------------------------

/// Tool that adds a new todo item.
struct TodoAddTool {
    todos: SharedTodos,
}

impl AgentTool for TodoAddTool {
    fn label(&self) -> &str {
        "TodoAdd"
    }

    fn tool_definition(&self) -> Tool {
        Tool {
            name: "todo_add".into(),
            description: "Add a new item to the todo list.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The todo item text."
                    }
                },
                "required": ["text"]
            }),
        }
    }

    fn prepare_arguments(&self, args: Value) -> Value {
        args
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        None
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TodoAddTool").finish()
    }

    fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: Option<tokio_util::sync::CancellationToken>,
    ) -> Pin<Box<dyn Future<Output = AgentToolResult<Value>> + Send + '_>> {
        let todos = self.todos.clone();
        Box::pin(async move {
            let text = params
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            if text.is_empty() {
                return AgentToolResult::<Value>::error("text parameter must not be empty");
            }

            let mut list = todos.lock().await;
            let id = list.len().saturating_add(1);
            list.push(TodoItem {
                id,
                text,
                completed: false,
            });

            AgentToolResult::text(
                format!("Added todo #{id}."),
                serde_json::json!({ "id": id }),
            )
        })
    }
}

// ---------------------------------------------------------------------------
// TodoListTool
// ---------------------------------------------------------------------------

/// Tool that lists all todo items.
struct TodoListTool {
    todos: SharedTodos,
}

impl AgentTool for TodoListTool {
    fn label(&self) -> &str {
        "TodoList"
    }

    fn tool_definition(&self) -> Tool {
        Tool {
            name: "todo_list".into(),
            description: "List all todo items.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    fn prepare_arguments(&self, args: Value) -> Value {
        args
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        None
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TodoListTool").finish()
    }

    fn execute(
        &self,
        _tool_call_id: &str,
        _params: Value,
        _cancel: Option<tokio_util::sync::CancellationToken>,
    ) -> Pin<Box<dyn Future<Output = AgentToolResult<Value>> + Send + '_>> {
        let todos = self.todos.clone();
        Box::pin(async move {
            let list = todos.lock().await;
            let output = format_todos(&list);
            AgentToolResult::text(output, serde_json::json!({}))
        })
    }
}

// ---------------------------------------------------------------------------
// TodoCompleteTool
// ---------------------------------------------------------------------------

/// Tool that marks a todo item as completed.
struct TodoCompleteTool {
    todos: SharedTodos,
}

impl AgentTool for TodoCompleteTool {
    fn label(&self) -> &str {
        "TodoComplete"
    }

    fn tool_definition(&self) -> Tool {
        Tool {
            name: "todo_complete".into(),
            description: "Mark a todo item as completed by its ID.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The ID of the todo item to mark as completed."
                    }
                },
                "required": ["id"]
            }),
        }
    }

    fn prepare_arguments(&self, args: Value) -> Value {
        args
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        None
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TodoCompleteTool").finish()
    }

    fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: Option<tokio_util::sync::CancellationToken>,
    ) -> Pin<Box<dyn Future<Output = AgentToolResult<Value>> + Send + '_>> {
        let todos = self.todos.clone();
        Box::pin(async move {
            let id = match params.get("id").and_then(|v| v.as_u64()) {
                Some(id) => id as usize,
                None => {
                    return AgentToolResult::<Value>::error(
                        "id parameter must be a positive integer",
                    )
                }
            };

            let mut list = todos.lock().await;
            let item = list.iter_mut().find(|t| t.id == id);
            match item {
                Some(t) => {
                    if t.completed {
                        return AgentToolResult::text(
                            format!("Todo #{id} is already completed."),
                            serde_json::json!({}),
                        );
                    }
                    t.completed = true;
                    AgentToolResult::text(
                        format!("Completed todo #{id}: {}", t.text),
                        serde_json::json!({ "id": id }),
                    )
                }
                None => {
                    let count = list.len();
                    if count == 0 {
                        AgentToolResult::<Value>::error("No todos exist yet. Use todo_add first.")
                    } else {
                        AgentToolResult::<Value>::error(format!(
                            "Todo #{id} not found. Valid IDs: 1-{count}."
                        ))
                    }
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// TodoExtension
// ---------------------------------------------------------------------------

/// Extension that wires up todo tools, commands, and the before_agent_start
/// hook.
struct TodoExtension {
    todos: SharedTodos,
}

impl TodoExtension {
    fn new() -> Self {
        Self {
            todos: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl Extension for TodoExtension {
    fn init(&self, api: &Arc<ExtensionApi>) {
        let todos = self.todos.clone();

        // --- Hook: before_agent_start ---
        let hook_todos = todos.clone();
        api.on_before_agent_start(move |event, _ctx| {
            let todos = hook_todos.clone();
            Box::pin(async move {
                let list = todos.lock().await;
                // Strip any previous injection to avoid accumulation
                let base = strip_previous_injection(&event.system_prompt);
                let suffix = build_system_prompt_suffix(&list);
                let new_prompt = format!("{base}{suffix}");
                Some(BeforeAgentStartResult {
                    system_prompt: Some(new_prompt),
                    message: None,
                })
            })
        });

        // --- Tools ---
        api.register_tool(Arc::new(TodoAddTool {
            todos: todos.clone(),
        }));
        api.register_tool(Arc::new(TodoListTool {
            todos: todos.clone(),
        }));
        api.register_tool(Arc::new(TodoCompleteTool {
            todos: todos.clone(),
        }));

        // --- Commands ---

        // /todos — list all todos
        let cmd_todos = todos.clone();
        api.register_command(
            "todos",
            Some("List all todo items".into()),
            Arc::new(move |_args, ctx| {
                let todos = cmd_todos.clone();
                Box::pin(async move {
                    let list = todos.lock().await;
                    let output = format_todos(&list);
                    ctx.extension_context
                        .interface
                        .notify(NotifyMessage::info(output));
                    Ok(())
                })
            }),
        );

        // /todo-add — add a todo
        let cmd_todos_add = todos.clone();
        api.register_command(
            "todo-add",
            Some("Add a todo item (usage: /todo-add <text>)".into()),
            Arc::new(move |args, ctx| {
                let todos = cmd_todos_add.clone();
                let text = args.trim().to_string();
                Box::pin(async move {
                    if text.is_empty() {
                        ctx.extension_context
                            .interface
                            .notify(NotifyMessage::warning("Usage: /todo-add <text>"));
                        return Ok(());
                    }
                    let mut list = todos.lock().await;
                    let id = list.len().saturating_add(1);
                    list.push(TodoItem {
                        id,
                        text,
                        completed: false,
                    });
                    ctx.extension_context
                        .interface
                        .notify(NotifyMessage::info(format!("Added todo #{id}.")));
                    Ok(())
                })
            }),
        );

        // /todo-complete — complete a todo by id
        let cmd_todos_complete = todos.clone();
        api.register_command(
            "todo-complete",
            Some("Complete a todo item (usage: /todo-complete <id>)".into()),
            Arc::new(move |args, ctx| {
                let todos = cmd_todos_complete.clone();
                let args = args.trim().to_string();
                Box::pin(async move {
                    let id: usize = match args.parse() {
                        Ok(id) => id,
                        Err(_) => {
                            ctx.extension_context
                                .interface
                                .notify(NotifyMessage::warning("Usage: /todo-complete <id>"));
                            return Ok(());
                        }
                    };

                    let mut list = todos.lock().await;
                    let item = list.iter_mut().find(|t| t.id == id);
                    match item {
                        Some(t) => {
                            if t.completed {
                                ctx.extension_context.interface.notify(NotifyMessage::info(
                                    format!("Todo #{id} is already completed."),
                                ));
                            } else {
                                let text = t.text.clone();
                                t.completed = true;
                                ctx.extension_context.interface.notify(NotifyMessage::info(
                                    format!("Completed todo #{id}: {text}"),
                                ));
                            }
                        }
                        None => {
                            let count = list.len();
                            if count == 0 {
                                ctx.extension_context
                                    .interface
                                    .notify(NotifyMessage::warning(
                                        "No todos exist yet. Use /todo-add first.".to_string(),
                                    ));
                            } else {
                                ctx.extension_context
                                    .interface
                                    .notify(NotifyMessage::warning(format!(
                                        "Todo #{id} not found. Valid IDs: 1-{count}."
                                    )));
                            }
                        }
                    }
                    Ok(())
                })
            }),
        );
    }
}

// ---------------------------------------------------------------------------
// FFI entry point
// ---------------------------------------------------------------------------

/// Double-boxed extension factory for the `ameli run` dynamic loader.
///
/// # Safety
///
/// Called by `ameli run` via `libloading`. Must return a valid double-boxed
/// pointer or null.
#[no_mangle]
pub extern "C" fn ameli_create_extension() -> *mut () {
    let ext: Box<dyn Extension> = Box::new(TodoExtension::new());
    // Double-box: `Box<dyn Extension>` is a fat pointer. Boxing it yields
    // a thin pointer that can safely cross the FFI boundary.
    Box::into_raw(Box::new(ext)) as *mut ()
}
