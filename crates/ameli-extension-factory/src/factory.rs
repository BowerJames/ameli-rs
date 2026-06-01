//! Extension factory — registry of [`ExtensionTemplate`] implementations.
//!
//! [`ExtensionFactory`] holds registered templates keyed by name, validates
//! builder options against each template's JSON Schema, and constructs
//! `Box<dyn Extension>` instances from validated configuration.
//!
//! # Architecture
//!
//! ```text
//! ExtensionFactory            ← registry (RwLock<HashMap>)
//!     ├── register()          ← register a template
//!     ├── build()             ← validate + build → Box<dyn Extension>
//!     ├── validate()          ← validate only (no build)
//!     ├── get_template_info() ← query metadata
//!     └── template_names()    ← list registered names
//!
//! DEFAULT_EXTENSION_FACTORY   ← global default (LazyLock)
//! ```
//!
//! # Usage
//!
//! ```no_run
//! use ameli_extension_factory::{ExtensionFactory, build_default};
//!
//! // Use the global factory
//! let extension = build_default("my_template", &serde_json::json!({"key": "value"})).unwrap();
//!
//! // Or create your own
//! let factory = ExtensionFactory::new();
//! // factory.register(Box::new(MyTemplate));
//! ```

use crate::template::{BuildError, ExtensionTemplate, TemplateInfo};
use ameli_agent::extension::Extension;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::RwLock;

// ---------------------------------------------------------------------------
// FactoryError
// ---------------------------------------------------------------------------

/// Error produced by [`ExtensionFactory`] operations.
#[derive(Debug, thiserror::Error)]
pub enum FactoryError {
    /// No template registered with the requested name.
    #[error("template not found: {name}")]
    TemplateNotFound {
        /// The template name that was looked up.
        name: String,
    },

    /// Builder options failed JSON Schema validation.
    #[error("schema validation failed for template {template}: {errors}")]
    SchemaValidation {
        /// The template name.
        template: String,
        /// Formatted validation errors.
        errors: String,
    },

    /// The template's [`build`](ExtensionTemplate::build) method returned an error.
    #[error("build failed for template {template}: {source}")]
    BuildFailed {
        /// The template name.
        template: String,
        /// The underlying build error.
        source: BuildError,
    },
}

// ---------------------------------------------------------------------------
// ExtensionFactory
// ---------------------------------------------------------------------------

/// Registry of [`ExtensionTemplate`] implementations.
///
/// Uses `RwLock<HashMap>` for interior mutability so concurrent reads are not
/// blocked by each other; only registration takes a write lock. This mirrors
/// the pattern used by `DefaultModelRegistry` and `ApiRegistry`.
///
/// # Examples
///
/// ```
/// use ameli_extension_factory::ExtensionFactory;
///
/// let factory = ExtensionFactory::new();
///
/// // factory.register(Box::new(MyTemplate));
/// // let extension = factory.build("my_template", &serde_json::json!({})).unwrap();
/// ```
pub struct ExtensionFactory {
    templates: RwLock<HashMap<String, Box<dyn ExtensionTemplate>>>,
}

impl ExtensionFactory {
    /// Create an empty factory.
    pub fn new() -> Self {
        Self {
            templates: RwLock::new(HashMap::new()),
        }
    }

    /// Register an extension template.
    ///
    /// If a template with the same name already exists, it is overwritten.
    pub fn register(&self, template: Box<dyn ExtensionTemplate>) {
        let mut templates = self.templates.write().unwrap_or_else(|e| e.into_inner());
        let name = template.name().to_string();
        templates.insert(name, template);
    }

    /// Remove a template by name.
    ///
    /// Returns `true` if a template was removed, `false` if no template was
    /// registered with the given name.
    pub fn unregister(&self, name: &str) -> bool {
        let mut templates = self.templates.write().unwrap_or_else(|e| e.into_inner());
        templates.remove(name).is_some()
    }

    /// Build an extension from a template and builder options.
    ///
    /// 1. Looks up the template by name (returns [`FactoryError::TemplateNotFound`]
    ///    if missing).
    /// 2. Validates `options` against the template's JSON Schema (returns
    ///    [`FactoryError::SchemaValidation`] if invalid).
    /// 3. Calls [`ExtensionTemplate::build`] on the template with the validated
    ///    options (returns [`FactoryError::BuildFailed`] on error).
    pub fn build(
        &self,
        name: &str,
        options: &serde_json::Value,
    ) -> Result<Box<dyn Extension>, FactoryError> {
        let templates = self.templates.read().unwrap_or_else(|e| e.into_inner());

        let template = templates
            .get(name)
            .ok_or_else(|| FactoryError::TemplateNotFound {
                name: name.to_string(),
            })?;

        // Validate options against the template's schema.
        validate_options(template.schema(), name, options)?;

        // Build the extension.
        template
            .build(options)
            .map_err(|source| FactoryError::BuildFailed {
                template: name.to_string(),
                source,
            })
    }

    /// Validate builder options against a template's schema without building.
    ///
    /// Returns `Ok(())` if the options are valid, or a
    /// [`FactoryError::SchemaValidation`] error if they are not.
    pub fn validate(&self, name: &str, options: &serde_json::Value) -> Result<(), FactoryError> {
        let templates = self.templates.read().unwrap_or_else(|e| e.into_inner());

        let template = templates
            .get(name)
            .ok_or_else(|| FactoryError::TemplateNotFound {
                name: name.to_string(),
            })?;

        validate_options(template.schema(), name, options)
    }

    /// Query a registered template's metadata without building.
    ///
    /// Returns `None` if no template is registered with the given name.
    pub fn get_template_info(&self, name: &str) -> Option<TemplateInfo> {
        let templates = self.templates.read().unwrap_or_else(|e| e.into_inner());
        templates.get(name).map(|t| TemplateInfo {
            name: t.name().to_string(),
            schema: t.schema().clone(),
        })
    }

    /// Return the sorted list of all registered template names.
    pub fn template_names(&self) -> Vec<String> {
        let templates = self.templates.read().unwrap_or_else(|e| e.into_inner());
        let mut names: Vec<String> = templates.keys().cloned().collect();
        names.sort();
        names
    }
}

impl Default for ExtensionFactory {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate `options` against a JSON Schema. Returns a
/// [`FactoryError::SchemaValidation`] on failure.
fn validate_options(
    schema: &serde_json::Value,
    template_name: &str,
    options: &serde_json::Value,
) -> Result<(), FactoryError> {
    let validator = match jsonschema::validator_for(schema) {
        Ok(v) => v,
        Err(e) => {
            return Err(FactoryError::SchemaValidation {
                template: template_name.to_string(),
                errors: format!("invalid template schema: {e}"),
            });
        }
    };

    if validator.is_valid(options) {
        return Ok(());
    }

    let errors: Vec<String> = validator
        .iter_errors(options)
        .map(|e| format!("  - {e}"))
        .collect();

    Err(FactoryError::SchemaValidation {
        template: template_name.to_string(),
        errors: errors.join("\n"),
    })
}

// ---------------------------------------------------------------------------
// Global default factory
// ---------------------------------------------------------------------------

/// Global default extension factory.
///
/// Starts with built-in templates pre-registered. Use
/// [`register_default`] to add custom templates at runtime.
pub static DEFAULT_EXTENSION_FACTORY: LazyLock<ExtensionFactory> = LazyLock::new(|| {
    let factory = ExtensionFactory::new();
    crate::built_in::register_built_in_templates(&factory);
    factory
});

/// Register a template on the [`DEFAULT_EXTENSION_FACTORY`].
pub fn register_default(template: Box<dyn ExtensionTemplate>) {
    DEFAULT_EXTENSION_FACTORY.register(template);
}

/// Build an extension from the [`DEFAULT_EXTENSION_FACTORY`].
///
/// Convenience wrapper around [`ExtensionFactory::build`] for callers that
/// don't need an explicit factory instance.
///
/// # Errors
///
/// See [`FactoryError`] for failure modes.
pub fn build_default(
    name: &str,
    options: &serde_json::Value,
) -> Result<Box<dyn Extension>, FactoryError> {
    DEFAULT_EXTENSION_FACTORY.build(name, options)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ameli_agent::extension::ExtensionApi;

    // -- Test fixtures -------------------------------------------------------

    /// A minimal template that builds a no-op extension.
    struct NoOpTemplate;

    impl ExtensionTemplate for NoOpTemplate {
        fn name(&self) -> &str {
            "noop"
        }
        fn schema(&self) -> &serde_json::Value {
            static SCHEMA: LazyLock<serde_json::Value> = LazyLock::new(|| {
                serde_json::json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "enabled": { "type": "boolean" }
                    }
                })
            });
            &SCHEMA
        }
        fn build(&self, _options: &serde_json::Value) -> Result<Box<dyn Extension>, BuildError> {
            Ok(Box::new(NoOpExtension))
        }
    }

    struct NoOpExtension;

    impl Extension for NoOpExtension {
        fn name(&self) -> &str {
            "noop"
        }
        fn init(&self, _api: &mut ExtensionApi) {}
    }

    /// A template that requires a "message" field.
    struct MessageTemplate;

    impl ExtensionTemplate for MessageTemplate {
        fn name(&self) -> &str {
            "message"
        }
        fn schema(&self) -> &serde_json::Value {
            static SCHEMA: LazyLock<serde_json::Value> = LazyLock::new(|| {
                serde_json::json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "required": ["message"],
                    "properties": {
                        "message": { "type": "string" }
                    }
                })
            });
            &SCHEMA
        }
        fn build(&self, options: &serde_json::Value) -> Result<Box<dyn Extension>, BuildError> {
            let _msg = options
                .get("message")
                .and_then(|v| v.as_str())
                .ok_or_else(|| BuildError::MissingField {
                    field: "message".into(),
                })?;
            Ok(Box::new(NoOpExtension))
        }
    }

    /// A template whose build method always fails.
    struct FailingTemplate;

    impl ExtensionTemplate for FailingTemplate {
        fn name(&self) -> &str {
            "failing"
        }
        fn schema(&self) -> &serde_json::Value {
            static SCHEMA: LazyLock<serde_json::Value> = LazyLock::new(|| {
                serde_json::json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object"
                })
            });
            &SCHEMA
        }
        fn build(&self, _options: &serde_json::Value) -> Result<Box<dyn Extension>, BuildError> {
            Err(BuildError::InvalidConfig {
                message: "always fails".into(),
            })
        }
    }

    // -- Construction -------------------------------------------------------

    #[test]
    fn new_is_empty() {
        let factory = ExtensionFactory::new();
        assert!(factory.template_names().is_empty());
    }

    #[test]
    fn default_is_empty() {
        let factory = ExtensionFactory::default();
        assert!(factory.template_names().is_empty());
    }

    // -- Register + build ---------------------------------------------------

    #[test]
    fn register_and_build_succeeds() {
        let factory = ExtensionFactory::new();
        factory.register(Box::new(NoOpTemplate));

        let ext = factory
            .build("noop", &serde_json::json!({"enabled": true}))
            .unwrap();
        assert_eq!(ext.name(), "noop");
    }

    #[test]
    fn register_and_build_with_empty_options() {
        let factory = ExtensionFactory::new();
        factory.register(Box::new(NoOpTemplate));

        let ext = factory.build("noop", &serde_json::json!({})).unwrap();
        assert_eq!(ext.name(), "noop");
    }

    // -- Template not found -------------------------------------------------

    #[test]
    fn build_returns_template_not_found() {
        let factory = ExtensionFactory::new();
        let result = factory.build("missing", &serde_json::json!({}));
        assert!(
            matches!(result, Err(FactoryError::TemplateNotFound { name }) if name == "missing")
        );
    }

    // -- Schema validation --------------------------------------------------

    #[test]
    fn build_rejects_invalid_options() {
        let factory = ExtensionFactory::new();
        factory.register(Box::new(MessageTemplate));

        // "message" is required but missing
        let result = factory.build("message", &serde_json::json!({}));
        assert!(
            matches!(result, Err(FactoryError::SchemaValidation { template, .. }) if template == "message")
        );
    }

    #[test]
    fn build_rejects_wrong_type() {
        let factory = ExtensionFactory::new();
        factory.register(Box::new(MessageTemplate));

        // "message" should be a string, not a number
        let result = factory.build("message", &serde_json::json!({"message": 42}));
        assert!(matches!(result, Err(FactoryError::SchemaValidation { .. })));
    }

    #[test]
    fn validate_succeeds_for_valid_options() {
        let factory = ExtensionFactory::new();
        factory.register(Box::new(MessageTemplate));

        factory
            .validate("message", &serde_json::json!({"message": "hi"}))
            .unwrap();
    }

    #[test]
    fn validate_rejects_invalid_options() {
        let factory = ExtensionFactory::new();
        factory.register(Box::new(MessageTemplate));

        let err = factory
            .validate("message", &serde_json::json!({}))
            .unwrap_err();
        assert!(matches!(err, FactoryError::SchemaValidation { .. }));
    }

    #[test]
    fn validate_returns_template_not_found() {
        let factory = ExtensionFactory::new();
        let err = factory
            .validate("missing", &serde_json::json!({}))
            .unwrap_err();
        assert!(matches!(err, FactoryError::TemplateNotFound { .. }));
    }

    // -- Build failure -------------------------------------------------------

    #[test]
    fn build_returns_build_failed() {
        let factory = ExtensionFactory::new();
        factory.register(Box::new(FailingTemplate));

        let result = factory.build("failing", &serde_json::json!({}));
        assert!(
            matches!(result, Err(FactoryError::BuildFailed { template, .. }) if template == "failing")
        );
    }

    // -- Re-register overwrites ---------------------------------------------

    #[test]
    fn re_register_overwrites() {
        let factory = ExtensionFactory::new();
        factory.register(Box::new(NoOpTemplate));

        struct NoOpV2;
        impl ExtensionTemplate for NoOpV2 {
            fn name(&self) -> &str {
                "noop"
            }
            fn schema(&self) -> &serde_json::Value {
                static SCHEMA: LazyLock<serde_json::Value> =
                    LazyLock::new(|| serde_json::json!({"type": "object"}));
                &SCHEMA
            }
            fn build(
                &self,
                _options: &serde_json::Value,
            ) -> Result<Box<dyn Extension>, BuildError> {
                Ok(Box::new(NoOpExtension))
            }
        }

        factory.register(Box::new(NoOpV2));
        assert_eq!(factory.template_names().len(), 1);
        let ext = factory.build("noop", &serde_json::json!({})).unwrap();
        assert_eq!(ext.name(), "noop");
    }

    // -- Unregister ---------------------------------------------------------

    #[test]
    fn unregister_removes_template() {
        let factory = ExtensionFactory::new();
        factory.register(Box::new(NoOpTemplate));
        assert_eq!(factory.template_names().len(), 1);

        assert!(factory.unregister("noop"));
        assert!(factory.template_names().is_empty());
    }

    #[test]
    fn unregister_returns_false_for_missing() {
        let factory = ExtensionFactory::new();
        assert!(!factory.unregister("missing"));
    }

    // -- Template names -----------------------------------------------------

    #[test]
    fn template_names_returns_sorted() {
        let factory = ExtensionFactory::new();
        factory.register(Box::new(MessageTemplate));
        factory.register(Box::new(NoOpTemplate));

        let names = factory.template_names();
        assert_eq!(names, vec!["message", "noop"]);
    }

    // -- get_template_info --------------------------------------------------

    #[test]
    fn get_template_info_returns_metadata() {
        let factory = ExtensionFactory::new();
        factory.register(Box::new(NoOpTemplate));

        let info = factory.get_template_info("noop").unwrap();
        assert_eq!(info.name, "noop");
        assert_eq!(
            info.schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
    }

    #[test]
    fn get_template_info_returns_none_for_missing() {
        let factory = ExtensionFactory::new();
        assert!(factory.get_template_info("missing").is_none());
    }

    // -- Global factory ------------------------------------------------------

    #[test]
    fn global_factory_is_empty_initially() {
        // The global factory starts with no built-in templates registered.
        // This test only checks that querying a missing template returns None.
        assert!(DEFAULT_EXTENSION_FACTORY
            .get_template_info("missing")
            .is_none());
    }

    // -- FactoryError display ------------------------------------------------

    #[test]
    fn factory_error_template_not_found_display() {
        let err = FactoryError::TemplateNotFound { name: "foo".into() };
        assert_eq!(format!("{err}"), "template not found: foo");
    }

    #[test]
    fn factory_error_schema_validation_display() {
        let err = FactoryError::SchemaValidation {
            template: "bar".into(),
            errors: "  - something wrong".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("bar"));
        assert!(msg.contains("something wrong"));
    }

    #[test]
    fn factory_error_build_failed_display() {
        let err = FactoryError::BuildFailed {
            template: "baz".into(),
            source: BuildError::InvalidConfig {
                message: "bad config".into(),
            },
        };
        let msg = format!("{err}");
        assert!(msg.contains("baz"));
        assert!(msg.contains("bad config"));
    }
}
