//! Async event streams for LLM streaming responses and agent events.
//!
//! The TS `EventStream` is a push-based async iterable. In Rust, we use
//! `tokio::sync::mpsc` unbounded channels — `push()` never blocks, matching
//! the original's unbounded internal queue semantics.
//!
//! Two stream types:
//! - [`EventStream`] — generic, for agent-level events
//! - [`AssistantMessageEventStream`] — specialised with `result()` capture

use crate::types::{AssistantMessage, AssistantMessageEvent, StopReason, Usage};
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Generic EventStream
// ---------------------------------------------------------------------------

/// Producer end of an [`EventStream`].
///
/// Call [`push`](EventStreamProducer::push) to send events. The stream ends
/// when the producer is dropped or [`end`](EventStreamProducer::end) is called.
pub struct EventStreamProducer<T: Send> {
    tx: mpsc::UnboundedSender<T>,
}

impl<T: Send> EventStreamProducer<T> {
    /// Push an event into the stream.
    ///
    /// Returns `Err` if the consumer has already been dropped.
    pub fn push(&self, event: T) -> Result<(), mpsc::error::SendError<T>> {
        self.tx.send(event)
    }

    /// Signal end-of-stream without sending a final event.
    ///
    /// Equivalent to dropping the producer.
    pub fn end(self) {
        drop(self);
    }
}

/// Consumer end of a generic event stream.
///
/// Iterate with [`recv`](EventStream::recv). The stream ends when the producer
/// is dropped or returns `None` from `recv()`.
pub struct EventStream<T: Send> {
    rx: mpsc::UnboundedReceiver<T>,
}

impl<T: Send> EventStream<T> {
    /// Receive the next event, or `None` if the stream has ended.
    pub async fn recv(&mut self) -> Option<T> {
        self.rx.recv().await
    }
}

/// Create a linked producer/consumer pair with an unbounded buffer.
pub fn create_event_stream<T: Send>() -> (EventStreamProducer<T>, EventStream<T>) {
    let (tx, rx) = mpsc::unbounded_channel();
    (EventStreamProducer { tx }, EventStream { rx })
}

// ---------------------------------------------------------------------------
// AssistantMessageEventStream
// ---------------------------------------------------------------------------

/// Event stream specialised for LLM assistant message streaming.
///
/// Extends [`EventStream`] semantics with automatic capture of the terminal
/// message so that [`result()`](AssistantMessageEventStream::result) can be
/// called after iteration has already consumed the terminal event.
///
/// # Terminal events
///
/// The stream always ends with a [`Done`](AssistantMessageEvent::Done) or
/// [`Error`](AssistantMessageEvent::Error) event. All other events carry a
/// partial message snapshot for incremental UI updates.
pub struct AssistantMessageEventStream {
    rx: mpsc::UnboundedReceiver<AssistantMessageEvent>,
    /// Captured from the terminal event during iteration so `result()` can
    /// return it even after the event was already consumed.
    final_result: Option<AssistantMessage>,
}

/// Producer end for an [`AssistantMessageEventStream`].
pub struct AssistantMessageEventProducer {
    tx: mpsc::UnboundedSender<AssistantMessageEvent>,
}

impl AssistantMessageEventProducer {
    /// Push a streaming event.
    ///
    /// If the consumer has already been dropped, the event is silently discarded.
    /// All callers in the codebase discard the result, and a dropped receiver is
    /// a terminal condition with no meaningful recovery.
    pub fn push(&self, event: AssistantMessageEvent) {
        let _ = self.tx.send(event);
    }

    /// Signal end-of-stream.
    pub fn end(self) {
        drop(self);
    }
}

/// Create a linked producer/consumer pair for assistant message streaming.
pub fn create_assistant_message_event_stream(
) -> (AssistantMessageEventProducer, AssistantMessageEventStream) {
    let (tx, rx) = mpsc::unbounded_channel();
    (
        AssistantMessageEventProducer { tx },
        AssistantMessageEventStream {
            rx,
            final_result: None,
        },
    )
}

impl AssistantMessageEventStream {
    /// Receive the next streaming event.
    ///
    /// When a terminal event ([`Done`](AssistantMessageEvent::Done) or
    /// [`Error`](AssistantMessageEvent::Error)) is received, the final
    /// `AssistantMessage` is captured internally so that [`result()`](Self::result)
    /// can return it later.
    pub async fn recv(&mut self) -> Option<AssistantMessageEvent> {
        let event = self.rx.recv().await?;
        self.capture_terminal(&event);
        Some(event)
    }

    /// Return the final `AssistantMessage`.
    ///
    /// # Two modes
    ///
    /// 1. **After iteration** — if the stream was iterated and a terminal event
    ///    was already received, the captured message is returned immediately.
    /// 2. **Without iteration** — if called on a fresh (or not-yet-terminal)
    ///    stream, this drains remaining events until a terminal one arrives.
    ///
    /// If the stream ends without a terminal event, a synthetic error message
    /// is returned.
    pub async fn result(mut self) -> AssistantMessage {
        // Already captured during iteration.
        if let Some(msg) = self.final_result.take() {
            return msg;
        }

        // Drain to terminal event.
        while let Some(event) = self.rx.recv().await {
            match event {
                AssistantMessageEvent::Done { message, .. } => return message,
                AssistantMessageEvent::Error { error, .. } => return error,
                _ => {}
            }
        }

        // Stream closed without a terminal event.
        make_error_message("event stream ended without a terminal event")
    }
}

impl AssistantMessageEventStream {
    fn capture_terminal(&mut self, event: &AssistantMessageEvent) {
        match event {
            AssistantMessageEvent::Done { message, .. } => {
                self.final_result = Some(message.clone());
            }
            AssistantMessageEvent::Error { error, .. } => {
                self.final_result = Some(error.clone());
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Construct a minimal error `AssistantMessage` for stream-level failures.
fn make_error_message(error_message: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![],
        api: String::new(),
        provider: String::new(),
        model: String::new(),
        response_model: None,
        response_id: None,
        usage: Usage::default(),
        stop_reason: StopReason::Error,
        error_message: Some(error_message.to_string()),
        timestamp: now_ms(),
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AssistantContentBlock, TextContent};

    fn sample_message(text: &str) -> AssistantMessage {
        AssistantMessage {
            content: vec![AssistantContentBlock::Text(TextContent::new(text))],
            api: "test".into(),
            provider: "test".into(),
            model: "test".into(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        }
    }

    #[tokio::test]
    async fn result_without_iteration() {
        let (producer, stream) = create_assistant_message_event_stream();
        let msg = sample_message("done");

        producer.push(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: msg.clone(),
        });
        drop(producer);

        let result = stream.result().await;
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn result_after_iteration() {
        let (producer, mut stream) = create_assistant_message_event_stream();
        let partial = sample_message("");
        let msg = sample_message("final");

        producer.push(AssistantMessageEvent::Start {
            partial: partial.clone(),
        });
        producer.push(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: msg.clone(),
        });
        drop(producer);

        // Iterate through all events
        while let Some(event) = stream.recv().await {
            if event.is_terminal() {
                break;
            }
        }

        // result() should still work — message was captured during recv()
        let result = stream.result().await;
        assert_eq!(result.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn error_event_produces_error_result() {
        let (producer, stream) = create_assistant_message_event_stream();
        let mut err_msg = sample_message("");
        err_msg.stop_reason = StopReason::Error;

        producer.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: err_msg,
        });
        drop(producer);

        let result = stream.result().await;
        assert_eq!(result.stop_reason, StopReason::Error);
    }

    #[tokio::test]
    async fn stream_closed_without_terminal_gives_error() {
        let (producer, stream) = create_assistant_message_event_stream();
        let partial = sample_message("");

        producer.push(AssistantMessageEvent::Start { partial });
        drop(producer); // close without terminal event

        let result = stream.result().await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(result
            .error_message
            .unwrap()
            .contains("without a terminal event"));
    }

    #[tokio::test]
    async fn generic_event_stream_recv() {
        let (producer, mut stream) = create_event_stream::<i32>();
        producer.push(1).unwrap();
        producer.push(2).unwrap();
        drop(producer);

        assert_eq!(stream.recv().await, Some(1));
        assert_eq!(stream.recv().await, Some(2));
        assert_eq!(stream.recv().await, None);
    }

    #[tokio::test]
    async fn recv_returns_none_after_drop() {
        let (producer, mut stream) = create_assistant_message_event_stream();
        drop(producer);
        assert_eq!(stream.recv().await, None);
    }
}
