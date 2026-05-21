//! Tool argument validation and coercion against JSON Schema.
//!
//! LLMs frequently return arguments that don't strictly match the tool's
//! parameter schema — numbers as strings, booleans as strings, etc. This
//! module coerces primitive types where possible, then validates the result
//! against the full JSON Schema.
//!
//! This is used by the agent loop between raw LLM output and tool execution.

use crate::types::{Tool, ToolCall};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Validate and coerce tool call arguments against the tool's JSON Schema.
///
/// Returns the (possibly coerced) arguments on success, or a
/// [`ToolValidationError`] with a human-readable error report on failure.
///
/// # Coercion
///
/// Before validation, primitive values are coerced to match the expected
/// schema types:
///
/// - `"123"` → `123` (string → number/integer)
/// - `"true"` / `"false"` → `true` / `false` (string → boolean)
/// - `123` → `"123"` (number → string)
/// - `null` → type-appropriate defaults (`0`, `false`, `""`)
///
/// Object properties and array items are coerced recursively.
pub fn validate_tool_arguments(
    tool: &Tool,
    tool_call: &ToolCall,
) -> Result<Value, ToolValidationError> {
    let mut args = tool_call.arguments.clone();

    // Phase 1: coerce against the schema
    coerce_with_schema(&mut args, &tool.parameters);

    // Phase 2: validate against the full schema
    let validator = match jsonschema::validator_for(&tool.parameters) {
        Ok(v) => v,
        Err(e) => {
            return Err(ToolValidationError {
                tool_name: tool_call.name.clone(),
                errors: vec![format!("invalid tool schema: {e}")],
                arguments: tool_call.arguments.clone(),
            });
        }
    };

    if validator.is_valid(&args) {
        return Ok(args);
    }

    let errors: Vec<String> = validator
        .iter_errors(&args)
        .map(|e| format!("  - {}: {}", format_instance_path(&e.instance_path.to_string()), e))
        .collect();

    Err(ToolValidationError {
        tool_name: tool_call.name.clone(),
        errors,
        arguments: tool_call.arguments.clone(),
    })
}

/// Error produced when tool argument validation fails.
#[derive(Debug, Clone)]
pub struct ToolValidationError {
    /// Name of the tool whose arguments failed validation.
    pub tool_name: String,
    /// Formatted validation errors (one per line, with path prefixes).
    pub errors: Vec<String>,
    /// The original arguments as received from the LLM.
    pub arguments: Value,
}

impl std::fmt::Display for ToolValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Validation failed for tool \"{}\":\n{}\n\nReceived arguments:\n{}",
            self.tool_name,
            self.errors.join("\n"),
            serde_json::to_string_pretty(&self.arguments).unwrap_or_default(),
        )
    }
}

impl std::error::Error for ToolValidationError {}

// ---------------------------------------------------------------------------
// Coercion
// ---------------------------------------------------------------------------

/// Coerce a JSON value to match a JSON Schema's type expectations.
///
/// Walks the schema and arguments together, mutating `value` in place.
/// Handles `type`, `properties`, `items`, `allOf`, `anyOf`, `oneOf`.
fn coerce_with_schema(value: &mut Value, schema: &Value) {
    // Apply allOf: each sub-schema contributes constraints
    if let Some(all_of) = schema.get("allOf").and_then(|v| v.as_array()) {
        for sub in all_of {
            coerce_with_schema(value, sub);
        }
    }

    // Apply anyOf: try to find a matching sub-schema and coerce to it
    if let Some(any_of) = schema.get("anyOf").and_then(|v| v.as_array()) {
        coerce_with_union(value, any_of);
    }

    // Apply oneOf: same approach as anyOf
    if let Some(one_of) = schema.get("oneOf").and_then(|v| v.as_array()) {
        coerce_with_union(value, one_of);
    }

    // Coerce based on the "type" field
    if let Some(type_val) = schema.get("type") {
        match type_val {
            Value::String(t) => coerce_to_type(value, t),
            Value::Array(types) => {
                // Union type — try each until one sticks
                for t in types {
                    if let Some(t_str) = t.as_str() {
                        let candidate = coerce_primitive(value.clone(), t_str);
                        if candidate != *value || type_matches(value, t_str) {
                            *value = candidate;
                            break;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Recurse into object properties
    if value.is_object() {
        if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
            let obj = value.as_object_mut().unwrap();
            for (key, prop_schema) in properties {
                if let Some(field) = obj.get_mut(key) {
                    coerce_with_schema(field, prop_schema);
                }
            }
        }
        // Handle additionalProperties as a schema object
        if let Some(additional) = schema.get("additionalProperties") {
            if additional.is_object() {
                let defined_keys: std::collections::HashSet<&str> = schema
                    .get("properties")
                    .and_then(|v| v.as_object())
                    .map(|p| p.keys().map(|k| k.as_str()).collect())
                    .unwrap_or_default();
                let obj = value.as_object_mut().unwrap();
                for (key, field) in obj.iter_mut() {
                    if !defined_keys.contains(key.as_str()) {
                        coerce_with_schema(field, additional);
                    }
                }
            }
        }
    }

    // Recurse into array items
    if value.is_array() {
        if let Some(items) = schema.get("items") {
            let arr = value.as_array_mut().unwrap();
            if items.is_array() {
                // Tuple validation: each position has its own schema
                for (i, item_schema) in items.as_array().unwrap().iter().enumerate() {
                    if let Some(elem) = arr.get_mut(i) {
                        coerce_with_schema(elem, item_schema);
                    }
                }
            } else if items.is_object() {
                // All items share one schema
                for elem in arr.iter_mut() {
                    coerce_with_schema(elem, items);
                }
            }
        }
    }
}

/// Try to coerce `value` to match at least one schema in a union (`anyOf`/`oneOf`).
fn coerce_with_union(value: &mut Value, schemas: &[Value]) {
    for schema in schemas {
        let mut candidate = value.clone();
        coerce_with_schema(&mut candidate, schema);
        if validate_single(&candidate, schema) {
            *value = candidate;
            return;
        }
    }
}

/// Coerce a value to a named JSON Schema type.
fn coerce_to_type(value: &mut Value, type_name: &str) {
    if type_matches(value, type_name) {
        return;
    }
    *value = coerce_primitive(value.clone(), type_name);
}

/// Attempt to coerce a value to a primitive type, returning the result.
fn coerce_primitive(value: Value, type_name: &str) -> Value {
    match type_name {
        "number" => match &value {
            Value::String(s) => s.trim().parse::<f64>().ok().map(Value::from).unwrap_or(value),
            Value::Bool(b) => Value::from(if *b { 1.0 } else { 0.0 }),
            Value::Null => Value::from(0.0),
            _ => value,
        },
        "integer" => match &value {
            Value::String(s) => s
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|f| f.fract() == 0.0)
                .map(|f| Value::from(f as i64))
                .unwrap_or(value),
            Value::Bool(b) => Value::from(if *b { 1_i64 } else { 0_i64 }),
            Value::Number(n) => n.as_f64().filter(|f| f.fract() == 0.0).map(|f| Value::from(f as i64)).unwrap_or(value),
            Value::Null => Value::from(0_i64),
            _ => value,
        },
        "boolean" => match &value {
            Value::String(s) => match s.as_str() {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                _ => value,
            },
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    if i == 1 {
                        return Value::Bool(true);
                    }
                    if i == 0 {
                        return Value::Bool(false);
                    }
                }
                value
            }
            Value::Null => Value::Bool(false),
            _ => value,
        },
        "string" => match &value {
            Value::Number(n) => Value::String(n.to_string()),
            Value::Bool(b) => Value::String(b.to_string()),
            Value::Null => Value::String(String::new()),
            _ => value,
        },
        "null" => match &value {
            Value::String(s) if s.is_empty() => Value::Null,
            Value::Number(n) if n.as_f64() == Some(0.0) => Value::Null,
            Value::Bool(false) => Value::Null,
            _ => value,
        },
        _ => value,
    }
}

/// Check if a value already matches a JSON Schema type name.
fn type_matches(value: &Value, type_name: &str) -> bool {
    match type_name {
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

/// Quick validation of a value against a single schema.
fn validate_single(value: &Value, schema: &Value) -> bool {
    jsonschema::validator_for(schema)
        .map(|v| v.is_valid(value))
        .unwrap_or(false)
}

/// Format a JSON Pointer instance path for display.
fn format_instance_path(path: &str) -> String {
    if path.is_empty() {
        return "root".to_string();
    }
    path.trim_start_matches('/')
        .split('/')
        .enumerate()
        .map(|(i, segment)| {
            if i == 0 {
                segment.to_string()
            } else {
                format!(".{segment}")
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Tool;
    use serde_json::json;

    fn test_tool(schema: Value) -> Tool {
        Tool {
            name: "test_tool".into(),
            description: "A test tool".into(),
            parameters: schema,
        }
    }

    fn test_call(args: Value) -> ToolCall {
        ToolCall {
            id: "tc_1".into(),
            name: "test_tool".into(),
            arguments: args,
            thought_signature: None,
        }
    }

    #[test]
    fn valid_arguments_pass_through() {
        let tool = test_tool(json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "count": { "type": "integer" }
            },
            "required": ["name"]
        }));
        let call = test_call(json!({ "name": "hello", "count": 5 }));
        let result = validate_tool_arguments(&tool, &call).unwrap();
        assert_eq!(result["name"], "hello");
        assert_eq!(result["count"], 5);
    }

    #[test]
    fn string_to_number_coercion() {
        let tool = test_tool(json!({
            "type": "object",
            "properties": {
                "x": { "type": "number" }
            }
        }));
        let call = test_call(json!({ "x": "3.14" }));
        let result = validate_tool_arguments(&tool, &call).unwrap();
        assert_eq!(result["x"], 3.14);
    }

    #[test]
    fn string_to_integer_coercion() {
        let tool = test_tool(json!({
            "type": "object",
            "properties": {
                "n": { "type": "integer" }
            }
        }));
        let call = test_call(json!({ "n": "42" }));
        let result = validate_tool_arguments(&tool, &call).unwrap();
        assert_eq!(result["n"], 42);
    }

    #[test]
    fn string_to_boolean_coercion() {
        let tool = test_tool(json!({
            "type": "object",
            "properties": {
                "flag": { "type": "boolean" }
            }
        }));

        let call_true = test_call(json!({ "flag": "true" }));
        let result = validate_tool_arguments(&tool, &call_true).unwrap();
        assert_eq!(result["flag"], true);

        let call_false = test_call(json!({ "flag": "false" }));
        let result = validate_tool_arguments(&tool, &call_false).unwrap();
        assert_eq!(result["flag"], false);
    }

    #[test]
    fn number_to_string_coercion() {
        let tool = test_tool(json!({
            "type": "object",
            "properties": {
                "label": { "type": "string" }
            }
        }));
        let call = test_call(json!({ "label": 42 }));
        let result = validate_tool_arguments(&tool, &call).unwrap();
        assert_eq!(result["label"], "42");
    }

    #[test]
    fn missing_required_field_is_error() {
        let tool = test_tool(json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "required": ["name"]
        }));
        let call = test_call(json!({}));
        let err = validate_tool_arguments(&tool, &call).unwrap_err();
        assert!(err.to_string().contains("test_tool"));
        assert!(!err.errors.is_empty());
    }

    #[test]
    fn wrong_type_that_cannot_be_coerced_is_error() {
        let tool = test_tool(json!({
            "type": "object",
            "properties": {
                "val": { "type": "integer" }
            }
        }));
        let call = test_call(json!({ "val": "not a number" }));
        let err = validate_tool_arguments(&tool, &call).unwrap_err();
        assert!(!err.errors.is_empty());
    }

    #[test]
    fn nested_object_coercion() {
        let tool = test_tool(json!({
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "timeout": { "type": "number" },
                        "enabled": { "type": "boolean" }
                    }
                }
            }
        }));
        let call = test_call(json!({
            "config": { "timeout": "30", "enabled": "true" }
        }));
        let result = validate_tool_arguments(&tool, &call).unwrap();
        assert_eq!(result["config"]["timeout"], 30.0);
        assert_eq!(result["config"]["enabled"], true);
    }

    #[test]
    fn array_items_coercion() {
        let tool = test_tool(json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": { "type": "integer" }
                }
            }
        }));
        let call = test_call(json!({ "items": ["1", "2", "3"] }));
        let result = validate_tool_arguments(&tool, &call).unwrap();
        assert_eq!(result["items"], json!([1, 2, 3]));
    }

    #[test]
    fn all_of_merges_constraints() {
        let tool = test_tool(json!({
            "type": "object",
            "allOf": [
                {
                    "properties": {
                        "a": { "type": "integer" }
                    }
                },
                {
                    "properties": {
                        "b": { "type": "boolean" }
                    }
                }
            ],
            "required": ["a", "b"]
        }));
        let call = test_call(json!({ "a": "5", "b": "true" }));
        let result = validate_tool_arguments(&tool, &call).unwrap();
        assert_eq!(result["a"], 5);
        assert_eq!(result["b"], true);
    }

    #[test]
    fn error_preserves_original_arguments() {
        let tool = test_tool(json!({
            "type": "object",
            "properties": {
                "x": { "type": "string" }
            },
            "required": ["x"]
        }));
        let call = test_call(json!({ "wrong": true }));
        let err = validate_tool_arguments(&tool, &call).unwrap_err();
        assert_eq!(err.arguments, json!({ "wrong": true }));
        assert_eq!(err.tool_name, "test_tool");
    }

    #[test]
    fn format_instance_path_shows_root() {
        assert_eq!(format_instance_path(""), "root");
    }

    #[test]
    fn format_instance_path_shows_dotted_path() {
        assert_eq!(format_instance_path("/config/timeout"), "config.timeout");
    }

    #[test]
    fn null_to_type_defaults() {
        let tool = test_tool(json!({
            "type": "object",
            "properties": {
                "n": { "type": "number" },
                "s": { "type": "string" },
                "b": { "type": "boolean" }
            }
        }));
        let call = test_call(json!({ "n": null, "s": null, "b": null }));
        let result = validate_tool_arguments(&tool, &call).unwrap();
        assert_eq!(result["n"], 0.0);
        assert_eq!(result["s"], "");
        assert_eq!(result["b"], false);
    }

    #[test]
    fn bool_to_number_coercion() {
        let tool = test_tool(json!({
            "type": "object",
            "properties": {
                "x": { "type": "number" }
            }
        }));
        let call = test_call(json!({ "x": true }));
        let result = validate_tool_arguments(&tool, &call).unwrap();
        assert_eq!(result["x"], 1.0);
    }
}
