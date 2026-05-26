pub mod api;
pub mod built_in_apis;
pub mod stream;
pub mod types;
pub mod validation;

// Re-export the primary API protocol types and entry points for convenience.
pub use api::{
    complete_simple, complete_simple_global, stream_simple, stream_simple_global, ApiRegistry,
    StreamFn,
};

// Re-export built-in API registration helpers.
pub use built_in_apis::register as register_openai_completions;
