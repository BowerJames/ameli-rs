//! Interactive append-only TUI for `ameli run`.
//!
//! Uses `ratatui` + `crossterm` to render a scrolling chat log with an input
//! bar. Agent events arrive via an `mpsc` channel and are rendered in
//! real-time. User input is forwarded to the agent session as prompts,
//! commands, or steering messages.

use ameli_agent::session_manager::InMemoryMetadata;
use ameli_agent::AgentSession;
use ameli_agent_core::types::{AgentEvent, AgentMessage};
use ameli_ai::types::{AssistantContentBlock, AssistantMessageEvent, TextContent, UserMessage};
use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::backend::CrosstermBackend;
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};
use std::io::Write;
use std::sync::Arc;
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// TerminalGuard — ensures terminal is restored even on panic/error
// ---------------------------------------------------------------------------

/// RAII guard that restores the terminal to its original state on drop.
///
/// Created after entering raw mode and alternate screen. If the TUI loop
/// panics or returns early, the `Drop` impl ensures the user's terminal
/// is not left in a broken state.
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let mut stdout = std::io::stdout();
        let _ = crossterm::execute!(stdout, LeaveAlternateScreen);
        let _ = stdout.flush();
    }
}

// ---------------------------------------------------------------------------
// ChatEntry — one row in the chat log
// ---------------------------------------------------------------------------

/// A single entry in the append-only chat log.
#[derive(Debug, Clone)]
enum ChatEntry {
    User { text: String },
    Assistant { text: String },
    Thinking { text: String },
    ToolStart { name: String, args_summary: String },
    ToolEnd { name: String, is_error: bool },
    Error { message: String },
    Info { message: String },
}

// ---------------------------------------------------------------------------
// TuiState — mutable state for the event loop
// ---------------------------------------------------------------------------

/// State mutated by the event loop.
struct TuiState {
    entries: Vec<ChatEntry>,
    input: String,
    should_quit: bool,
    agent_active: bool,
    /// Index into `entries` for the assistant entry currently being streamed.
    current_assistant_idx: Option<usize>,
    /// Index into `entries` for the thinking entry currently being streamed.
    current_thinking_idx: Option<usize>,
}

impl TuiState {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            input: String::new(),
            should_quit: false,
            agent_active: false,
            current_assistant_idx: None,
            current_thinking_idx: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the interactive TUI.
///
/// Takes ownership of the session and the extension set. The session is
/// shut down when the user exits.
pub async fn run(session: Arc<AgentSession<InMemoryMetadata>>) -> Result<()> {
    // 1. Set up terminal
    terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;

    // Guard ensures terminal is restored even on panic/error
    let _guard = TerminalGuard;

    // 2. Create channels
    let (agent_tx, mut agent_rx) = mpsc::unbounded_channel::<AgentEvent>();
    let (key_tx, mut key_rx) = mpsc::unbounded_channel::<KeyEvent>();
    let (error_tx, mut error_rx) = mpsc::unbounded_channel::<String>();

    // 3. Subscribe to agent events
    let _subscription = session
        .agent()
        .subscribe(Arc::new(move |event, _cancel| {
            let _ = agent_tx.send(event);
            Box::pin(async {})
        }))
        .await;

    // 4. Spawn crossterm polling task
    let crossterm_handle = tokio::spawn(async move {
        loop {
            if event::poll(std::time::Duration::from_millis(33)).is_err() {
                break;
            }
            if let Ok(Event::Key(key)) = event::read() {
                if key_tx.send(key).is_err() {
                    break;
                }
            }
        }
    });

    // 5. Initialize state
    let mut state = TuiState::new();
    let agent_state = session.agent().state().await;
    state.entries.push(ChatEntry::Info {
        message: format!(
            "Model: {} | Provider: {} | Thinking: {:?}",
            agent_state.model.id, agent_state.model.provider, agent_state.thinking_level
        ),
    });

    // 6. Main event loop
    loop {
        tokio::select! {
            key = key_rx.recv() => {
                if let Some(key) = key {
                    handle_key(key, &mut state, &session, &error_tx);
                }
            }
            event = agent_rx.recv() => {
                if let Some(event) = event {
                    handle_agent_event(event, &mut state);
                }
            }
            err = error_rx.recv() => {
                if let Some(err) = err {
                    state.entries.push(ChatEntry::Error { message: err });
                }
            }
        }

        terminal.draw(|f| render(f, &state))?;

        if state.should_quit {
            break;
        }
    }

    // 7. Clean up
    crossterm_handle.abort();
    session.shutdown().await;

    // Terminal is restored by _guard Drop
    Ok(())
}

// ---------------------------------------------------------------------------
// Agent event handling
// ---------------------------------------------------------------------------

/// Map an [`AgentEvent`] to a [`TuiState`] mutation.
fn handle_agent_event(event: AgentEvent, state: &mut TuiState) {
    match event {
        AgentEvent::AgentStart => {
            state.agent_active = true;
        }

        AgentEvent::AgentEnd { .. } => {
            state.agent_active = false;
            state.current_assistant_idx = None;
            state.current_thinking_idx = None;
        }

        AgentEvent::MessageStart { message } => match &message {
            AgentMessage::User(msg) => {
                let text = extract_user_text(msg);
                state.entries.push(ChatEntry::User { text });
            }
            AgentMessage::Assistant(msg) => {
                let text = extract_assistant_text(msg);
                if !text.is_empty() {
                    state.entries.push(ChatEntry::Assistant { text });
                    state.current_assistant_idx = Some(state.entries.len() - 1);
                } else {
                    state.entries.push(ChatEntry::Assistant {
                        text: String::new(),
                    });
                    state.current_assistant_idx = Some(state.entries.len() - 1);
                }
            }
            _ => {}
        },

        AgentEvent::MessageUpdate {
            assistant_message_event,
            ..
        } => match &*assistant_message_event {
            AssistantMessageEvent::TextDelta { delta, .. } => {
                if let Some(idx) = state.current_assistant_idx {
                    if let Some(ChatEntry::Assistant { text }) = state.entries.get_mut(idx) {
                        text.push_str(delta);
                    }
                }
            }
            AssistantMessageEvent::ThinkingStart { .. } => {
                state.entries.push(ChatEntry::Thinking {
                    text: String::new(),
                });
                state.current_thinking_idx = Some(state.entries.len() - 1);
            }
            AssistantMessageEvent::ThinkingDelta { delta, .. } => {
                if let Some(idx) = state.current_thinking_idx {
                    if let Some(ChatEntry::Thinking { text }) = state.entries.get_mut(idx) {
                        text.push_str(delta);
                    }
                }
            }
            AssistantMessageEvent::ThinkingEnd { .. } => {
                state.current_thinking_idx = None;
            }
            _ => {}
        },

        AgentEvent::MessageEnd { .. } => {
            state.current_assistant_idx = None;
            state.current_thinking_idx = None;
        }

        AgentEvent::ToolExecutionStart {
            tool_name, args, ..
        } => {
            let args_summary = truncate_str(&args.to_string(), 120);
            state.entries.push(ChatEntry::ToolStart {
                name: tool_name,
                args_summary,
            });
        }

        AgentEvent::ToolExecutionEnd {
            tool_name,
            result,
            is_error,
            ..
        } => {
            if is_error {
                // Extract error text from the result content
                let text = extract_tool_result_text(&result.content);
                state.entries.push(ChatEntry::Error {
                    message: format!("{tool_name}: {text}"),
                });
            } else {
                state.entries.push(ChatEntry::ToolEnd {
                    name: tool_name,
                    is_error,
                });
            }
        }

        AgentEvent::TurnEnd { .. } => {}

        AgentEvent::TurnStart => {}

        AgentEvent::ToolExecutionUpdate { .. } => {}
    }
}

// ---------------------------------------------------------------------------
// Key input handling
// ---------------------------------------------------------------------------

/// Handle a keyboard event.
fn handle_key(
    key: KeyEvent,
    state: &mut TuiState,
    session: &Arc<AgentSession<InMemoryMetadata>>,
    error_tx: &mpsc::UnboundedSender<String>,
) {
    // Handle Ctrl+C before the key.code match, because Ctrl+C produces
    // KeyCode::Char('c') which would otherwise append 'c' to the input.
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        state.should_quit = true;
        let agent = session.agent().clone();
        tokio::spawn(async move {
            agent.abort().await;
        });
        return;
    }

    match key.code {
        KeyCode::Char(c) => {
            state.input.push(c);
        }
        KeyCode::Backspace => {
            state.input.pop();
        }
        KeyCode::Enter => {
            let text = std::mem::take(&mut state.input);
            if text.is_empty() {
                return;
            }

            if text.starts_with('/') {
                let (name, args) = parse_command(&text);
                if name.is_empty() {
                    // Bare `/` with no command name — ignore
                    return;
                }
                let name = name.to_string();
                let args = args.to_string();
                let session = session.clone();
                let error_tx = error_tx.clone();
                tokio::spawn(async move {
                    if let Err(e) = session.command(&name, &args).await {
                        let _ = error_tx.send(format!("Command '{name}' failed: {e}"));
                    }
                });
            } else if state.agent_active {
                // Steering message — queue for next turn.
                // Do NOT push ChatEntry::User here; the agent event subscriber
                // emits MessageStart(User) which adds it to the chat log.
                let msg = AgentMessage::User(UserMessage::text(&text));
                let agent = session.agent().clone();
                tokio::spawn(async move {
                    agent.steer(msg).await;
                });
            } else {
                // New prompt.
                // Do NOT push ChatEntry::User here; the agent event subscriber
                // emits MessageStart(User) which adds it to the chat log.
                let session = session.clone();
                let error_tx = error_tx.clone();
                tokio::spawn(async move {
                    if let Err(e) = session.prompt(&text, vec![]).await {
                        let _ = error_tx.send(format!("Prompt failed: {e}"));
                    }
                });
            }
        }
        KeyCode::Esc => {
            state.should_quit = true;
            let agent = session.agent().clone();
            tokio::spawn(async move {
                agent.abort().await;
            });
        }
        _ => {}
    }
}

/// Parse a `/command args` string into `(name, args)`.
fn parse_command(input: &str) -> (&str, &str) {
    let stripped = input.strip_prefix('/').unwrap_or(input);
    stripped.split_once(' ').unwrap_or((stripped, ""))
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Render the current state to the terminal.
fn render(frame: &mut Frame, state: &TuiState) {
    let chunks = Layout::vertical([
        Constraint::Min(1),    // chat area
        Constraint::Length(3), // input bar
    ])
    .split(frame.area());

    let chat_area = chunks.first().copied().unwrap_or_default();
    let input_area = chunks.get(1).copied().unwrap_or_default();

    render_chat(frame, chat_area, state);
    render_input(frame, input_area, state);
}

/// Render the scrolling chat area.
fn render_chat(frame: &mut Frame, area: Rect, state: &TuiState) {
    let mut lines: Vec<Line> = Vec::new();

    for entry in &state.entries {
        match entry {
            ChatEntry::User { text } => {
                lines.push(Line::from(vec![
                    Span::styled(
                        "You: ",
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(text, Style::default().fg(Color::Green)),
                ]));
            }
            ChatEntry::Assistant { text } => {
                if text.is_empty() {
                    lines.push(Line::from(Span::styled(
                        "Assistant: ",
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    )));
                } else {
                    lines.push(Line::from(vec![
                        Span::styled(
                            "Assistant: ",
                            Style::default()
                                .fg(Color::White)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(text, Style::default().fg(Color::White)),
                    ]));
                }
            }
            ChatEntry::Thinking { text } => {
                lines.push(Line::from(vec![
                    Span::styled(
                        "[thinking] ",
                        Style::default()
                            .fg(Color::DarkGray)
                            .add_modifier(Modifier::ITALIC),
                    ),
                    Span::styled(text, Style::default().fg(Color::DarkGray)),
                ]));
            }
            ChatEntry::ToolStart { name, args_summary } => {
                lines.push(Line::from(vec![
                    Span::styled("⚙ ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("{name}: "),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(args_summary, Style::default().fg(Color::Yellow)),
                ]));
            }
            ChatEntry::ToolEnd { name, is_error } => {
                let marker = if *is_error { "✗" } else { "✓" };
                let color = if *is_error {
                    Color::Red
                } else {
                    Color::DarkGray
                };
                lines.push(Line::from(Span::styled(
                    format!("  {marker} {name}"),
                    Style::default().fg(color),
                )));
            }
            ChatEntry::Error { message } => {
                lines.push(Line::from(Span::styled(
                    format!("Error: {message}"),
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                )));
            }
            ChatEntry::Info { message } => {
                lines.push(Line::from(Span::styled(
                    message,
                    Style::default().fg(Color::Blue),
                )));
            }
        }
    }

    // Add an empty line at the bottom for breathing room
    lines.push(Line::from(""));

    let paragraph = Paragraph::new(lines)
        .block(Block::default().borders(Borders::NONE))
        .wrap(Wrap { trim: false });

    // Auto-scroll to show the bottom of the chat log.
    //
    // ratatui does NOT clamp scroll offsets — if scroll.y exceeds the
    // number of wrapped lines, the widget skips all lines and renders
    // nothing. We must compute a valid scroll value ourselves.
    //
    // We estimate the total number of wrapped lines by summing each
    // entry's character count divided by the area width. This is an
    // approximation (it doesn't account for word-wrap splitting or ANSI
    // sequences) but is safe because overestimating produces a scroll
    // that is clamped by saturating_sub to 0, showing the top of the
    // log — which is always correct.
    let total_lines: u16 = state
        .entries
        .iter()
        .map(|e| estimate_entry_lines(e, area.width))
        .sum();
    // +1 for the trailing empty breathing-room line
    let total_with_padding = total_lines.saturating_add(1);
    let scroll = total_with_padding.saturating_sub(area.height);

    frame.render_widget(paragraph.scroll((scroll, 0)), area);
}

/// Render the input bar at the bottom.
fn render_input(frame: &mut Frame, area: Rect, state: &TuiState) {
    let status = if state.agent_active {
        " (thinking…)"
    } else {
        ""
    };

    let input = Paragraph::new(format!("> {}{status}", state.input))
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .borders(Borders::TOP)
                .style(Style::default().fg(Color::DarkGray)),
        );

    frame.render_widget(input, area);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract plain text from a [`UserMessage`].
fn extract_user_text(msg: &ameli_ai::types::UserMessage) -> String {
    match &msg.content {
        ameli_ai::types::UserContent::Text(t) => t.clone(),
        ameli_ai::types::UserContent::Blocks(blocks) => {
            let mut parts = Vec::new();
            for block in blocks {
                if let ameli_ai::types::MediaContentBlock::Text(t) = block {
                    parts.push(t.text.clone());
                }
            }
            parts.join(" ")
        }
    }
}

/// Extract accumulated plain text from an [`AssistantMessage`].
fn extract_assistant_text(msg: &ameli_ai::types::AssistantMessage) -> String {
    msg.content
        .iter()
        .filter_map(|block| match block {
            AssistantContentBlock::Text(TextContent { text, .. }) if !text.is_empty() => {
                Some(text.as_str())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Extract text content from tool result content blocks.
fn extract_tool_result_text(content: &[ameli_ai::types::MediaContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ameli_ai::types::MediaContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Estimate the number of wrapped lines an entry will occupy at a given width.
///
/// Returns at least 1 (every entry produces at least one rendered line).
/// The estimate uses Unicode width–aware character counting and accounts for
/// the label prefixes added during rendering.
fn estimate_entry_lines(entry: &ChatEntry, width: u16) -> u16 {
    if width == 0 {
        return 1;
    }
    let text = match entry {
        ChatEntry::User { text } => format!("You: {text}"),
        ChatEntry::Assistant { text } => {
            if text.is_empty() {
                return 1; // "Assistant: " label only
            }
            format!("Assistant: {text}")
        }
        ChatEntry::Thinking { text } => format!("[thinking] {text}"),
        ChatEntry::ToolStart {
            name,
            args_summary,
        } => format!("\u{2699} {name}: {args_summary}"),
        ChatEntry::ToolEnd { name, .. } => format!("  \u{2713} {name}"),
        ChatEntry::Error { message } => format!("Error: {message}"),
        ChatEntry::Info { message } => message.clone(),
    };
    wrap_line_count(&text, width)
}

/// Estimate how many terminal lines `text` occupies when wrapped at `width`.
///
/// Uses `unicode-width` for accurate display-width measurement.
fn wrap_line_count(text: &str, width: u16) -> u16 {
    if width == 0 {
        return 1;
    }
    let width = width as usize;
    let mut lines = 1u16;
    let mut col = 0usize;
    for ch in text.chars() {
        let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if col + cw > width {
            lines = lines.saturating_add(1);
            col = cw;
        } else {
            col += cw;
        }
    }
    lines
}

/// Truncate a string to `max_len` characters with "…" appended.
///
/// Operates on Unicode code points (not bytes) for correct multi-byte handling.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let mut truncated: String = s.chars().take(max_len - 1).collect();
        truncated.push('…');
        truncated
    }
}
