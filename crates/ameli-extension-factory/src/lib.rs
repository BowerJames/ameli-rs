//! Extension factory — registry-based construction of extensions from declarative templates.
//!
//! This crate defines [`ExtensionTemplate`] — a trait for declarative extension
//! descriptors that provide a JSON Schema for their builder options and can
//! construct a `Box<dyn ameli_agent::extension::Extension>` from validated
//! configuration.
//!
//! [`ExtensionFactory`] is the registry that holds templates, validates
//! configuration against their schemas, and builds extensions.
//!
//! # Architecture
//!
//! ```text
//! ExtensionTemplate         ← trait: name + schema + build()
//!     ↓
//! ExtensionFactory          ← registry (RwLock<HashMap>)
//!     ├── register()        ← add a template
//!     ├── build()           ← validate + build → Box<dyn Extension>
//!     ├── validate()        ← validate only (no build)
//!     ├── get_template_info() ← query metadata
//!     └── template_names()  ← list registered names
//!
//! DEFAULT_EXTENSION_FACTORY ← global default (LazyLock)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use ameli_extension_factory::{ExtensionFactory, ExtensionTemplate, BuildError};
//!
//! // 1. Create a factory and register templates
//! let factory = ExtensionFactory::new();
//! factory.register(Box::new(MyTemplate));
//!
//! // 2. Build an extension from JSON config
//! let extension = factory.build("my_template", &serde_json::json!({
//!     "message": "You are helpful"
//! }))?;
//!
//! // 3. Pass to create_agent_session()
//! // options.extensions.push(extension);
//! ```
//!
//! # Downstream consumers
//!
//! Resource loaders (e.g., `ameli-multi-agent-resource-loader`) can use the
//! factory to build extensions from stored JSON configuration and pass them to
//! [`create_agent_session`](ameli_agent::create_agent_session).
//!
//! # Built-in templates
//!
//! The global [`DEFAULT_EXTENSION_FACTORY`] starts with built-in templates
//! pre-registered. Currently none are defined — they will be added in the
//! `built_in` module as the framework grows.

pub mod built_in;
pub mod factory;
pub mod template;
pub mod utils;

// Re-export primary types for convenience.
pub use factory::{
    build_default, register_default, ExtensionFactory, FactoryError, DEFAULT_EXTENSION_FACTORY,
};
pub use template::{BuildError, ExtensionTemplate, TemplateInfo};
