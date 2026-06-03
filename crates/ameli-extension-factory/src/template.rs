//! Extension template trait — declarative descriptors for building extensions.
//!
//! An [`ExtensionTemplate`] defines a JSON Schema for its builder options and
//! can construct a `Box<dyn Extension>` from validated configuration. This
//! enables extensions to be configured from stored JSON (files, databases, etc.)
//! and constructed by resource loaders at runtime.

use ameli_agent::extension::Extension;

// ---------------------------------------------------------------------------
// BuildError
// ---------------------------------------------------------------------------

/// Error produced when an extension template fails to build.
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    /// A required configuration field is missing.
    #[error("missing required field: {field}")]
    MissingField {
        /// Name of the missing field.
        field: String,
    },
    /// The configuration is semantically invalid (passed schema validation
    /// but failed template-specific checks).
    #[error("invalid configuration: {message}")]
    InvalidConfig {
        /// Human-readable description of the problem.
        message: String,
    },
    /// An internal or unexpected error during build.
    #[error("{0}")]
    Internal(#[from] anyhow::Error),
}

// ---------------------------------------------------------------------------
// TemplateInfo
// ---------------------------------------------------------------------------

/// Lightweight descriptor for querying template metadata without building.
///
/// Returned by [`ExtensionFactory::get_template_info`](crate::ExtensionFactory::get_template_info).
#[derive(Debug, Clone)]
pub struct TemplateInfo {
    /// Unique template name.
    pub name: String,
    /// JSON Schema describing the configuration this template accepts.
    pub schema: serde_json::Value,
}

// ---------------------------------------------------------------------------
// ExtensionTemplate trait
// ---------------------------------------------------------------------------

/// Trait for extension templates — declarative descriptors that define a
/// JSON Schema for their builder options and can construct a
/// `Box<dyn Extension>` from validated configuration.
///
/// # Lifecycle
///
/// 1. A template is registered with an [`ExtensionFactory`](crate::ExtensionFactory) by name.
/// 2. When a consumer calls [`factory.build(name, options)`](crate::ExtensionFactory::build),
///    the factory validates `options` against the template's schema.
/// 3. If validation passes, the factory calls [`build`](ExtensionTemplate::build)
///    on the template with the validated options.
/// 4. The template constructs and returns a `Box<dyn Extension>`.
///
/// # Example
///
/// ```
/// use ameli_extension_factory::ExtensionTemplate;
/// use ameli_agent::extension::{Extension, ExtensionApi};
/// use ameli_extension_factory::BuildError;
/// use std::sync::LazyLock;
/// use serde_json::json;
///
/// struct AppendSystemMessageTemplate;
///
/// impl ExtensionTemplate for AppendSystemMessageTemplate {
///     fn name(&self) -> &str { "append_system_message" }
///
///     fn schema(&self) -> &serde_json::Value {
///         static SCHEMA: LazyLock<serde_json::Value> = LazyLock::new(|| {
///             json!({
///                 "$schema": "https://json-schema.org/draft/2020-12/schema",
///                 "title": "AppendSystemMessage",
///                 "type": "object",
///                 "required": ["message"],
///                 "properties": {
///                     "message": { "type": "string" }
///                 }
///             })
///         });
///         &SCHEMA
///     }
///
///     fn build(&self, options: &serde_json::Value) -> Result<Box<dyn Extension>, BuildError> {
///         let message = options.get("message")
///             .and_then(|v| v.as_str())
///             .ok_or_else(|| BuildError::MissingField { field: "message".into() })?
///             .to_string();
///
///         Ok(Box::new(AppendSystemMessageExtension { message }))
///     }
/// }
///
/// struct AppendSystemMessageExtension {
///     message: String,
/// }
///
/// impl Extension for AppendSystemMessageExtension {
///     fn name(&self) -> &str { "append_system_message" }
///     fn init(&self, _api: &mut ExtensionApi) {}
/// }
/// ```
pub trait ExtensionTemplate: Send + Sync {
    /// Unique name for this template (e.g., `"append_system_message"`).
    ///
    /// Used as the registry key in the factory. Must be stable across sessions
    /// so stored configurations remain valid.
    fn name(&self) -> &str;

    /// JSON Schema (Draft 2020-12) describing the configuration this template
    /// accepts.
    ///
    /// The factory validates builder options against this schema before calling
    /// [`build`](Self::build). Templates should return a long-lived reference
    /// (e.g., a `static` or leaked `Box`) to avoid re-serialization on every
    /// call.
    fn schema(&self) -> &serde_json::Value;

    /// Build an extension from validated configuration.
    ///
    /// The `options` value has already been validated against
    /// [`schema`](Self::schema) before this method is called, so basic type
    /// correctness is guaranteed. Templates may still perform additional
    /// semantic validation and return [`BuildError`] if the configuration is
    /// logically invalid.
    fn build(&self, options: &serde_json::Value) -> Result<Box<dyn Extension>, BuildError>;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_error_missing_field_display() {
        let err = BuildError::MissingField {
            field: "message".into(),
        };
        assert_eq!(format!("{err}"), "missing required field: message");
    }

    #[test]
    fn build_error_invalid_config_display() {
        let err = BuildError::InvalidConfig {
            message: "must be positive".into(),
        };
        assert_eq!(format!("{err}"), "invalid configuration: must be positive");
    }

    #[test]
    fn build_error_internal_display() {
        let err = BuildError::Internal(anyhow::anyhow!("something broke"));
        assert_eq!(format!("{err}"), "something broke");
    }

    #[test]
    fn template_info_construction() {
        let info = TemplateInfo {
            name: "test".into(),
            schema: serde_json::json!({"type": "object"}),
        };
        assert_eq!(info.name, "test");
        assert_eq!(info.schema["type"], "object");
    }
}
