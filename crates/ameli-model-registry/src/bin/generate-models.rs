//! Binary to fetch models from models.dev and generate `src/models.json`.
//!
//! Run via:
//!   cargo run -p ameli-model-registry --features generate --bin generate-models
//!
//! Reads `providers.json` from the crate root, fetches the models.dev API,
//! filters to tool-call-capable, non-deprecated models, and writes the
//! resulting JSON array to `src/models.json`.

use ameli_ai::types::{Cost, InputType, Model};
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Provider config (from providers.json)
// ---------------------------------------------------------------------------

/// A curated provider entry from `providers.json`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProviderConfig {
    /// Our provider identifier (e.g., "groq", "xiaomi-token-plan-cn").
    id: String,
    /// Key to look up in the models.dev API response (e.g., "groq", "xiaomi").
    models_dev_key: String,
    /// Base URL for the provider's OpenAI-compatible API.
    base_url: String,
    /// API protocol (always "openai-completions" for now).
    #[allow(dead_code)]
    api: String,
    /// Optional provider-level compat overrides (passed through as raw JSON).
    compat: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// models.dev API types
// ---------------------------------------------------------------------------

/// A model entry from the models.dev API.
#[derive(Debug, Deserialize)]
struct ModelsDevModel {
    name: Option<String>,
    tool_call: Option<bool>,
    status: Option<String>,
    reasoning: Option<bool>,
    limit: Option<ModelsDevLimit>,
    cost: Option<ModelsDevCost>,
    modalities: Option<ModelsDevModalities>,
}

#[derive(Debug, Deserialize)]
struct ModelsDevLimit {
    context: Option<u64>,
    output: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ModelsDevCost {
    input: Option<f64>,
    output: Option<f64>,
    #[serde(rename = "cache_read")]
    cache_read: Option<f64>,
    #[serde(rename = "cache_write")]
    cache_write: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ModelsDevModalities {
    input: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));

    // 1. Read providers.json
    let providers_path = manifest_dir.join("providers.json");
    let providers_json = fs::read_to_string(&providers_path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {e}", providers_path.display());
        std::process::exit(1);
    });
    let providers: Vec<ProviderConfig> =
        serde_json::from_str(&providers_json).unwrap_or_else(|e| {
            eprintln!("Failed to parse providers.json: {e}");
            std::process::exit(1);
        });
    eprintln!("Loaded {} provider configs", providers.len());

    // 2. Fetch models.dev API
    eprintln!("Fetching models from https://models.dev/api.json ...");
    let response = reqwest::get("https://models.dev/api.json")
        .await
        .unwrap_or_else(|e| {
            eprintln!("Failed to fetch models.dev API: {e}");
            std::process::exit(1);
        });

    let api_data: serde_json::Value = response.json().await.unwrap_or_else(|e| {
        eprintln!("Failed to parse models.dev response: {e}");
        std::process::exit(1);
    });

    let api_object = api_data.as_object().unwrap_or_else(|| {
        eprintln!("Expected JSON object from models.dev API");
        std::process::exit(1);
    });

    // 3. Process providers
    let mut all_models: Vec<Model> = Vec::new();

    for provider_config in &providers {
        // Try the configured key and common fallbacks
        let provider_data = try_get_provider_data(api_object, &provider_config.models_dev_key);

        let models_data = match provider_data {
            Some(data) => data,
            None => {
                eprintln!(
                    "  Warning: provider key '{}' not found in models.dev API, skipping",
                    provider_config.models_dev_key
                );
                continue;
            }
        };

        let models_object = match models_data.get("models").and_then(|m| m.as_object()) {
            Some(obj) => obj,
            None => {
                eprintln!(
                    "  Warning: no models found for provider key '{}', skipping",
                    provider_config.models_dev_key
                );
                continue;
            }
        };

        let mut provider_count = 0;
        for (model_id, model_value) in models_object {
            let dev_model: ModelsDevModel = match serde_json::from_value(model_value.clone()) {
                Ok(m) => m,
                Err(_) => continue,
            };

            // Filter: must support tool calls
            if dev_model.tool_call != Some(true) {
                continue;
            }

            // Filter: must not be deprecated
            if dev_model.status.as_deref() == Some("deprecated") {
                continue;
            }

            // Build input modalities
            let mut input = vec![InputType::Text];
            if dev_model
                .modalities
                .as_ref()
                .and_then(|m| m.input.as_ref())
                .is_some_and(|inputs| inputs.iter().any(|i| i == "image"))
            {
                input.push(InputType::Image);
            }

            // Build cost
            let cost = Cost {
                input: dev_model.cost.as_ref().and_then(|c| c.input).unwrap_or(0.0),
                output: dev_model
                    .cost
                    .as_ref()
                    .and_then(|c| c.output)
                    .unwrap_or(0.0),
                cache_read: dev_model
                    .cost
                    .as_ref()
                    .and_then(|c| c.cache_read)
                    .unwrap_or(0.0),
                cache_write: dev_model
                    .cost
                    .as_ref()
                    .and_then(|c| c.cache_write)
                    .unwrap_or(0.0),
            };

            let model = Model {
                id: model_id.clone(),
                name: dev_model.name.clone().unwrap_or_else(|| model_id.clone()),
                api: "openai-completions".to_string(),
                provider: provider_config.id.clone(),
                base_url: provider_config.base_url.clone(),
                reasoning: dev_model.reasoning == Some(true),
                thinking_level_map: None,
                input,
                cost,
                context_window: dev_model
                    .limit
                    .as_ref()
                    .and_then(|l| l.context)
                    .unwrap_or(4096),
                max_tokens: dev_model
                    .limit
                    .as_ref()
                    .and_then(|l| l.output)
                    .unwrap_or(4096),
                compat: provider_config.compat.clone(),
            };

            all_models.push(model);
            provider_count += 1;
        }

        eprintln!(
            "  {} (key: {}): {} models",
            provider_config.id, provider_config.models_dev_key, provider_count
        );
    }

    // 4. Sort for deterministic output: by provider, then model id
    all_models.sort_by(|a, b| a.provider.cmp(&b.provider).then_with(|| a.id.cmp(&b.id)));

    // 5. Write models.json
    let output_path = manifest_dir.join("src").join("models.json");
    let output_json = serde_json::to_string_pretty(&all_models).unwrap_or_else(|e| {
        eprintln!("Failed to serialize models: {e}");
        std::process::exit(1);
    });

    fs::write(&output_path, &output_json).unwrap_or_else(|e| {
        eprintln!("Failed to write {}: {e}", output_path.display());
        std::process::exit(1);
    });

    // 6. Print statistics
    eprintln!(
        "\nGenerated {} models -> {}",
        all_models.len(),
        output_path.display()
    );

    let mut provider_counts: HashMap<&str, usize> = HashMap::new();
    for model in &all_models {
        *provider_counts.entry(&model.provider).or_default() += 1;
    }
    let mut providers: Vec<&str> = provider_counts.keys().copied().collect();
    providers.sort();
    for provider in &providers {
        eprintln!("  {}: {} models", provider, provider_counts[*provider]);
    }
}

/// Try to get provider data from the models.dev API response, trying common
/// key variants (e.g., "togetherai" -> "togetherai", "together", "together-ai").
fn try_get_provider_data<'a>(
    api_object: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'a serde_json::Value> {
    // Try exact key first
    if let Some(data) = api_object.get(key) {
        return Some(data);
    }

    // Try common fallbacks for specific providers
    let fallbacks: &[&str] = match key {
        "togetherai" => &["together", "together-ai"],
        _ => &[],
    };

    for fallback in fallbacks {
        if let Some(data) = api_object.get(*fallback) {
            return Some(data);
        }
    }

    None
}
