pub mod api;
pub mod stream;
pub mod types;
pub mod validation;

// Re-export the primary API protocol types and entry points for convenience.
pub use api::{
    complete_simple, complete_simple_global, stream_simple, stream_simple_global, ApiRegistry,
    StreamFn,
};
