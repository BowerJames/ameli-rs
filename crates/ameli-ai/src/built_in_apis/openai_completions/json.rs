//! Streaming JSON repair for incomplete tool call arguments.
//!
//! LLMs stream tool call arguments incrementally. During streaming the
//! accumulated JSON may be incomplete or malformed. This module provides
//! best-effort parsing with a repair fallback.

use serde_json::Value;

/// Parse potentially incomplete JSON from streaming tool call arguments.
///
/// Returns an empty object if the JSON cannot be parsed even after repair.
pub fn parse_streaming_json(input: &str) -> Value {
    if input.trim().is_empty() {
        return Value::Object(serde_json::Map::new());
    }

    // Try direct parse first
    if let Ok(v) = serde_json::from_str(input) {
        return v;
    }

    // Try to repair by closing open strings and containers
    let repaired = repair_json(input);
    if let Ok(v) = serde_json::from_str(&repaired) {
        return v;
    }

    Value::Object(serde_json::Map::new())
}

/// Attempt to repair incomplete JSON by:
/// 1. Closing open strings
/// 2. Handling trailing incomplete key-value pairs
/// 3. Closing open containers
fn repair_json(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut in_string = false;
    let mut escape_next = false;
    let mut stack: Vec<char> = Vec::new();

    // Phase 1: scan to determine state at end of input
    for &ch in &chars {
        if escape_next {
            escape_next = false;
            continue;
        }
        if in_string {
            if ch == '\\' {
                escape_next = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' | '[' => stack.push(ch),
            '}' | ']' => {
                stack.pop();
            }
            _ => {}
        }
    }

    let mut result = input.to_string();

    // Close open string
    if in_string {
        result.push('"');
    }

    // Close open containers in reverse order
    while let Some(open) = stack.pop() {
        result.push(match open {
            '{' => '}',
            '[' => ']',
            _ => '}',
        });
    }

    // Try to parse; if it still fails, strip trailing incomplete tokens
    if serde_json::from_str::<Value>(&result).is_ok() {
        return result;
    }

    // Strip back to last complete value boundary
    strip_trailing_incomplete(&result)
}

/// Strip trailing incomplete tokens and close remaining containers.
///
/// Handles patterns like `{"key": "value", "key2":` by removing
/// everything from the last `,` onward and closing containers.
fn strip_trailing_incomplete(input: &str) -> String {
    let trimmed = input.trim_end_matches('}').trim_end_matches(']');

    // Find last complete value boundary: look for closing quote or digit
    // after a colon, followed by optional whitespace
    let mut cut = trimmed.len();
    let chars: Vec<char> = trimmed.chars().collect();

    // Walk backwards to find a good cut point
    let mut i = chars.len();
    while i > 0 {
        i -= 1;
        let Some(&ch) = chars.get(i) else {
            break;
        };
        if ch == ',' {
            cut = i;
            break;
        }
        if ch == '{' || ch == '[' {
            cut = i + 1;
            break;
        }
    }

    let mut result = trimmed[..cut].to_string();

    // Remove trailing comma
    result = result.trim_end_matches(',').to_string();

    // Close remaining containers
    let mut in_str = false;
    let mut esc = false;
    let mut depth = 0i32;
    for ch in result.chars() {
        if esc {
            esc = false;
            continue;
        }
        if in_str {
            if ch == '\\' {
                esc = true;
            } else if ch == '"' {
                in_str = false;
            }
            continue;
        }
        match ch {
            '"' => in_str = true,
            '{' | '[' => depth += 1,
            '}' | ']' => depth = (depth - 1).max(0),
            _ => {}
        }
    }
    for _ in 0..depth {
        result.push('}');
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_input_returns_empty_object() {
        assert_eq!(parse_streaming_json(""), json!({}));
        assert_eq!(parse_streaming_json("  "), json!({}));
    }

    #[test]
    fn valid_json_passes_through() {
        let input = r#"{"key": "value"}"#;
        assert_eq!(parse_streaming_json(input), json!({"key": "value"}));
    }

    #[test]
    fn incomplete_string_value_repaired() {
        let input = r#"{"key": "val"#;
        assert_eq!(parse_streaming_json(input), json!({"key": "val"}));
    }

    #[test]
    fn open_object_closed() {
        let input = r#"{"key": "value""#;
        assert_eq!(parse_streaming_json(input), json!({"key": "value"}));
    }

    #[test]
    fn bare_open_brace() {
        assert_eq!(parse_streaming_json("{"), json!({}));
    }

    #[test]
    fn incomplete_second_key_stripped() {
        // {"key": "value", "key2":
        let result = parse_streaming_json(r#"{"key": "value", "key2":"#);
        assert_eq!(result, json!({"key": "value"}));
    }

    #[test]
    fn nested_object_closed() {
        let input = r#"{"outer": {"inner": 42"#;
        assert_eq!(parse_streaming_json(input), json!({"outer": {"inner": 42}}));
    }

    #[test]
    fn array_items_closed() {
        let input = r#"{"items": [1, 2, 3"#;
        assert_eq!(parse_streaming_json(input), json!({"items": [1, 2, 3]}));
    }
}
