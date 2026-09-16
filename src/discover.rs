// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model family discovery and search via the `HuggingFace` Hub API.
//!
//! Queries the HF Hub for popular models, extracts `model_type` metadata,
//! compares against locally cached families, and fetches model card metadata.

use std::collections::{BTreeMap, HashMap};
use std::hash::BuildHasher;
use std::sync::Arc;

use serde::Deserialize;

use crate::error::FetchError;

/// A model found by searching the `HuggingFace` Hub.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The repository identifier (e.g., `"RWKV/RWKV7-Goose-World3-1.5B-HF"`).
    pub model_id: String,
    /// Total download count.
    pub downloads: u64,
    /// Library framework (e.g., `"transformers"`, `"peft"`, `"diffusers"`), if reported.
    pub library_name: Option<String>,
    /// Pipeline task tag (e.g., `"text-generation"`), if reported.
    pub pipeline_tag: Option<String>,
    /// Tags from the model's metadata (e.g., `["gguf", "conversational"]`).
    pub tags: Vec<String>,
}

/// A model family discovered from the `HuggingFace` Hub.
#[derive(Debug, Clone)]
pub struct DiscoveredFamily {
    /// The `model_type` identifier (e.g., `"gpt_neox"`, `"llama"`).
    pub model_type: String,
    /// The most-downloaded representative model for this family.
    pub top_model: String,
    /// Download count of the representative model.
    pub downloads: u64,
}

/// JSON response structure for an individual model from the HF API.
#[derive(Debug, Deserialize)]
struct ApiModelEntry {
    #[serde(rename = "modelId")]
    model_id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    config: Option<ApiConfig>,
    #[serde(default)]
    library_name: Option<String>,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

/// The `config` object embedded in a model API response.
#[derive(Debug, Deserialize)]
struct ApiConfig {
    model_type: Option<String>,
}

/// Access control status of a model on the `HuggingFace` Hub.
///
/// Some models require users to accept license terms before downloading.
/// The gating mode determines whether approval is automatic or manual.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GateStatus {
    /// No gate — anyone can download without restrictions.
    Open,
    /// Automatic approval after the user accepts terms on the Hub.
    Auto,
    /// Manual approval by the model author after the user requests access.
    Manual,
}

impl GateStatus {
    /// Returns `true` if the model requires accepting terms before download.
    #[must_use]
    pub const fn is_gated(&self) -> bool {
        matches!(self, Self::Auto | Self::Manual)
    }
}

impl std::fmt::Display for GateStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open => write!(f, "open"),
            Self::Auto => write!(f, "auto"),
            Self::Manual => write!(f, "manual"),
        }
    }
}

/// Metadata from a `HuggingFace` model card.
///
/// Extracted from the single-model API endpoint
/// (`GET /api/models/{owner}/{model}`). All fields are optional
/// because model cards may omit any of them.
#[derive(Debug, Clone)]
pub struct ModelCardMetadata {
    /// SPDX license identifier (e.g., `"apache-2.0"`).
    pub license: Option<String>,
    /// Pipeline tag (e.g., `"text-generation"`).
    pub pipeline_tag: Option<String>,
    /// Tags associated with the model (e.g., `["pytorch", "safetensors"]`).
    pub tags: Vec<String>,
    /// Library name (e.g., `"transformers"`, `"vllm"`).
    pub library_name: Option<String>,
    /// Languages the model supports (e.g., `["en", "fr"]`).
    pub languages: Vec<String>,
    /// Access control status (open, auto-gated, or manually gated).
    pub gated: GateStatus,
}

/// JSON response for a single model from `GET /api/models/{model_id}`.
#[derive(Debug, Deserialize)]
struct ApiModelDetail {
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    library_name: Option<String>,
    #[serde(default)]
    gated: ApiGated,
    #[serde(default, rename = "cardData")]
    card_data: Option<ApiCardData>,
}

/// The `cardData` sub-object (parsed YAML front matter from the model README).
#[derive(Debug, Deserialize)]
struct ApiCardData {
    #[serde(default)]
    license: Option<String>,
    #[serde(default)]
    language: Option<ApiLanguage>,
}

/// Languages in `cardData` can be a single string or a list of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiLanguage {
    Single(String),
    Multiple(Vec<String>),
}

/// The `gated` field can be `false` (boolean) or a string like `"auto"` / `"manual"`.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiGated {
    Bool(bool),
    Mode(String),
}

impl Default for ApiGated {
    fn default() -> Self {
        Self::Bool(false)
    }
}

const PAGE_SIZE: usize = 100;
const HF_API_BASE: &str = "https://huggingface.co/api/models";

/// Queries the `HuggingFace` Hub API for top models by downloads
/// and returns families not present in the local cache.
///
/// # Arguments
///
/// * `local_families` — Set of `model_type` values already cached locally.
/// * `max_models` — Maximum number of models to scan (paginated in batches of 100).
/// * `tag` — Optional tag filter (e.g., `"gguf"`, `"bitsandbytes"`). When set,
///   only models carrying this tag contribute to family discovery.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if any API request fails.
pub async fn discover_new_families<S: BuildHasher>(
    local_families: &std::collections::HashSet<String, S>,
    max_models: usize,
    tag: Option<&str>,
) -> Result<Vec<DiscoveredFamily>, FetchError> {
    let client = reqwest::Client::new();
    let mut remote_families: BTreeMap<String, (String, u64)> = BTreeMap::new();
    let mut offset: usize = 0;

    while offset < max_models {
        let page_limit = PAGE_SIZE.min(max_models.saturating_sub(offset));
        let page_limit_str = page_limit.to_string();
        let offset_str = offset.to_string();

        // BORROW: explicit .as_str() instead of Deref coercion
        let mut query_params: Vec<(&str, &str)> = vec![
            ("config", "true"),
            ("sort", "downloads"),
            ("direction", "-1"),
            ("limit", page_limit_str.as_str()),
            ("offset", offset_str.as_str()),
        ];
        if let Some(t) = tag {
            query_params.push(("filter", t));
        }

        let response = client
            .get(HF_API_BASE)
            .query(&query_params)
            .send()
            .await
            .map_err(|e| FetchError::Http(e.to_string()))?;

        if !response.status().is_success() {
            return Err(FetchError::Http(format!(
                "HF API returned status {}",
                response.status()
            )));
        }

        let models: Vec<ApiModelEntry> = response
            .json()
            .await
            .map_err(|e| FetchError::Http(e.to_string()))?;

        if models.is_empty() {
            break;
        }

        for model in &models {
            // Client-side tag filter: the HF API may ignore the `filter` query
            // parameter when combined with other params, so verify the tag is
            // actually present on each returned model.
            if let Some(t) = tag
                && !model.tags.iter().any(|model_tag| {
                    // BORROW: explicit .as_str() instead of Deref coercion
                    model_tag.as_str().eq_ignore_ascii_case(t)
                })
            {
                continue;
            }

            // BORROW: explicit .as_ref() and .as_deref() for Option<String>
            let model_type = model.config.as_ref().and_then(|c| c.model_type.as_deref());

            if let Some(mt) = model_type {
                remote_families
                    .entry(mt.to_owned())
                    .or_insert_with(|| (model.model_id.clone(), model.downloads));
            }
        }

        offset = offset.saturating_add(models.len());
    }

    // Filter to families not already cached locally.
    // BORROW: explicit .as_str() instead of Deref coercion
    let discovered: Vec<DiscoveredFamily> = remote_families
        .into_iter()
        .filter(|(mt, _)| !local_families.contains(mt.as_str()))
        .map(|(model_type, (top_model, downloads))| DiscoveredFamily {
            model_type,
            top_model,
            downloads,
        })
        .collect();

    Ok(discovered)
}

/// Normalizes common quantization synonyms in a search query so that
/// variant spellings (e.g., `"8bit"`, `"8-bit"`, `"int8"`) produce
/// consistent results.
#[must_use]
fn normalize_quantization_terms(query: &str) -> String {
    /// Synonym groups: all variants map to the first (canonical) form.
    const SYNONYMS: &[(&[&str], &str)] = &[
        (&["8bit", "8-bit", "int8"], "8-bit"),
        (&["4bit", "4-bit", "int4"], "4-bit"),
        (&["fp8", "float8"], "fp8"),
    ];

    query
        .split_whitespace()
        .map(|token| {
            // BORROW: explicit .to_lowercase() for case-insensitive comparison
            let lower = token.to_lowercase();
            for &(variants, canonical) in SYNONYMS {
                // BORROW: explicit .as_str() instead of Deref coercion
                if variants.contains(&lower.as_str()) {
                    // BORROW: explicit .to_owned() for &str → owned String
                    return (*canonical).to_owned();
                }
            }
            // BORROW: explicit .to_owned() for &str → owned String
            token.to_owned()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Searches the `HuggingFace` Hub for models matching a query string.
///
/// Optionally filters by `library` framework (e.g., `"transformers"`, `"peft"`),
/// `pipeline` task tag (e.g., `"text-generation"`), and/or `tag` (e.g., `"gguf"`).
/// Library and pipeline filters are sent as query parameters; tag is sent via the
/// `filter` parameter. All three are also applied client-side for correctness.
///
/// Common quantization synonyms (`"8bit"` / `"8-bit"` / `"int8"`,
/// `"4bit"` / `"4-bit"` / `"int4"`, `"fp8"` / `"float8"`) are normalized
/// before querying the API so that variant spellings return consistent results.
///
/// Results are sorted by download count (most popular first).
///
/// # Arguments
///
/// * `query` — Free-text search string (e.g., `"RWKV-7"`, `"llama 3"`).
/// * `limit` — Maximum number of results to return.
/// * `library` — Optional library filter (e.g., `"peft"`, `"transformers"`).
/// * `pipeline` — Optional pipeline tag filter (e.g., `"text-generation"`).
/// * `tag` — Optional tag filter (e.g., `"gguf"`, `"conversational"`).
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the API request fails.
pub async fn search_models(
    query: &str,
    limit: usize,
    library: Option<&str>,
    pipeline: Option<&str>,
    tag: Option<&str>,
) -> Result<Vec<SearchResult>, FetchError> {
    let normalized = normalize_quantization_terms(query);
    let client = reqwest::Client::new();

    // BORROW: explicit .as_str() instead of Deref coercion
    let mut query_params: Vec<(&str, &str)> = vec![
        ("search", normalized.as_str()),
        ("sort", "downloads"),
        ("direction", "-1"),
    ];
    if let Some(lib) = library {
        query_params.push(("library", lib));
    }
    if let Some(pipe) = pipeline {
        query_params.push(("pipeline_tag", pipe));
    }
    if let Some(t) = tag {
        query_params.push(("filter", t));
    }

    let response = client
        .get(HF_API_BASE)
        .query(&query_params)
        .query(&[("limit", limit)])
        .send()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;

    if !response.status().is_success() {
        return Err(FetchError::Http(format!(
            "HF API returned status {}",
            response.status()
        )));
    }

    let models: Vec<ApiModelEntry> = response
        .json()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;

    // Client-side filtering: the HF search API may ignore library/pipeline_tag/filter
    // query parameters when combined with the `search` parameter, so we filter
    // the results ourselves to guarantee correctness.
    let results = models
        .into_iter()
        .filter(|m| {
            if let Some(lib) = library {
                match m.library_name {
                    // BORROW: explicit .as_str() instead of Deref coercion
                    Some(ref name) if name.as_str().eq_ignore_ascii_case(lib) => {}
                    _ => return false,
                }
            }
            if let Some(pipe) = pipeline {
                match m.pipeline_tag {
                    // BORROW: explicit .as_str() instead of Deref coercion
                    Some(ref t) if t.as_str().eq_ignore_ascii_case(pipe) => {}
                    _ => return false,
                }
            }
            if let Some(t) = tag
                && !m.tags.iter().any(|model_tag| {
                    // BORROW: explicit .as_str() instead of Deref coercion
                    model_tag.as_str().eq_ignore_ascii_case(t)
                })
            {
                return false;
            }
            true
        })
        .map(|m| SearchResult {
            model_id: m.model_id,
            downloads: m.downloads,
            library_name: m.library_name,
            pipeline_tag: m.pipeline_tag,
            tags: m.tags,
        })
        .collect();

    Ok(results)
}

/// Returns the total size in bytes of all files in the given repository's
/// `main` revision, summed across `siblings[].size` from the
/// `/api/models/{repo_id}?blobs=true` endpoint.
///
/// Used by `hf-fm search --show size` to enrich result rows with a total-repo
/// size column. Failures are surfaced to the caller so individual repo
/// lookups can be skipped (the search itself is not aborted on a single
/// 404 / network blip).
///
/// # Arguments
///
/// * `repo_id` — The full model identifier (e.g., `"org/model"`).
/// * `client` — Shared `reqwest::Client` (callers fan out N lookups concurrently).
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the API request
/// fails or returns a non-success status.
/// Returns [`FetchError::RepoNotFound`] if
/// the repository does not exist on the Hub.
pub async fn fetch_repo_total_size(
    repo_id: &str,
    client: &reqwest::Client,
) -> Result<u64, FetchError> {
    let files = crate::repo::list_repo_files_with_metadata(repo_id, None, None, client).await?;
    Ok(files.iter().filter_map(|f| f.size).sum())
}

/// Backlink verification outcome for a [`QuantCandidate`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum QuantVerification {
    /// A `.gguf` file's `general.source.url` / `general.base_model.*.repo_url`
    /// metadata confirms this repo targets the base model.
    Verified,
    /// No verification could run: the candidate has no `.gguf` file, or its
    /// GGUF metadata carries no recognized backlink key. The naming match
    /// stands alone.
    Unverified,
    /// The backlink check itself failed (network error, timeout, a gated
    /// repo) rather than returning a definite answer. The naming match
    /// stands alone; the reason is kept for display.
    CheckFailed(String),
}

/// A quant-sibling repo candidate discovered for a base model, plus its file
/// listing (reused by callers building a size table, so [`discover_quant_siblings`]
/// is the only Hub round-trip needed per candidate beyond the initial search).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct QuantCandidate {
    /// The candidate repository identifier.
    pub repo_id: String,
    /// Backlink verification outcome (see [`QuantVerification`]).
    pub verification: QuantVerification,
    /// The candidate repo's file listing (sizes and SHA256 where known).
    pub files: Vec<crate::repo::RepoFile>,
}

impl QuantCandidate {
    /// Builds a [`QuantCandidate`] directly.
    ///
    /// `#[non_exhaustive]` blocks struct-literal construction from outside
    /// this crate, so this is the canonical way to build one from the `hf-fm`
    /// binary crate — used by its test suite; production code only ever
    /// receives `QuantCandidate` values from [`discover_quant_siblings`].
    #[must_use]
    pub fn new(
        repo_id: String,
        verification: QuantVerification,
        files: Vec<crate::repo::RepoFile>,
    ) -> Self {
        Self {
            repo_id,
            verification,
            files,
        }
    }
}

/// Known `GGUF` metadata keys that point back at a source `HuggingFace` repo,
/// per `llama.cpp`'s `gguf-py` metadata writer conventions.
fn gguf_source_backlinks(metadata: &HashMap<String, String>) -> Vec<&str> {
    metadata
        .iter()
        .filter(|(key, _)| {
            key.as_str() == "general.source.url"
                || key.as_str() == "general.source.huggingface.repository"
                || (key.starts_with("general.base_model.") && key.ends_with(".repo_url"))
        })
        // BORROW: explicit .as_str() instead of Deref coercion
        .map(|(_, value)| value.as_str())
        .collect()
}

/// Lists `candidate_repo_id`'s files and, if it holds at least one `.gguf`
/// file, cross-checks the smallest one's metadata backlink against
/// `base_repo_id`.
///
/// Returns `None` only when a backlink is present and explicitly names a
/// *different* repo — a naming collision, not a real sibling. Every other
/// outcome (no `.gguf` file, no backlink key, or the header fetch itself
/// failing) is returned as a kept candidate with the reason recorded in
/// [`QuantVerification`], per [`discover_quant_siblings`]'s contract that a
/// transient failure must not hide a real candidate. A file-listing failure
/// (the repo cannot be enumerated at all) drops the candidate silently —
/// there would be nothing to render for it regardless of verification status.
async fn build_quant_candidate(
    candidate_repo_id: String,
    base_repo_id: &str,
    token: Option<&str>,
    client: &reqwest::Client,
) -> Option<QuantCandidate> {
    let files = crate::repo::list_repo_files_with_metadata(&candidate_repo_id, token, None, client)
        .await
        .ok()?;

    let mut gguf_files: Vec<_> = files
        .iter()
        .filter(|f| f.filename.to_ascii_lowercase().ends_with(".gguf"))
        .cloned()
        .collect();
    gguf_files.sort_by_key(|f| f.size.unwrap_or(u64::MAX));

    let Some(representative) = gguf_files.first() else {
        return Some(QuantCandidate {
            repo_id: candidate_repo_id,
            verification: QuantVerification::Unverified,
            files,
        });
    };

    match crate::inspect::inspect_gguf(&candidate_repo_id, &representative.filename, token, None)
        .await
    {
        Err(e) => Some(QuantCandidate {
            repo_id: candidate_repo_id,
            verification: QuantVerification::CheckFailed(e.to_string()),
            files,
        }),
        Ok((info, _source, _stats)) => {
            let backlinks = info
                .metadata
                .as_ref()
                .map(|m| gguf_source_backlinks(m))
                .unwrap_or_default();
            if backlinks.is_empty() {
                return Some(QuantCandidate {
                    repo_id: candidate_repo_id,
                    verification: QuantVerification::Unverified,
                    files,
                });
            }
            // BORROW: explicit .to_lowercase() for case-insensitive substring match
            let base_lower = base_repo_id.to_lowercase();
            let matched = backlinks
                .iter()
                .any(|b| b.to_lowercase().contains(&base_lower));
            if matched {
                Some(QuantCandidate {
                    repo_id: candidate_repo_id,
                    verification: QuantVerification::Verified,
                    files,
                })
            } else {
                None // EXPLICIT: backlink present but points elsewhere — not a real sibling
            }
        }
    }
}

/// Discovers quant-sibling repos for a base model.
///
/// Searches the Hub for the base model's short name (the part of
/// `base_repo_id` after `/`), keeps results whose repo ID contains that name
/// as a case-insensitive substring (excluding `base_repo_id` itself), then
/// fans out [`build_quant_candidate`] across the survivors through a bounded
/// `tokio::sync::Semaphore` (8 permits, mirroring [`fetch_repo_sizes_concurrent`]).
///
/// Sibling discovery has no dedicated Hub endpoint — quant repos
/// overwhelmingly either name themselves `<base>-<SCHEME>` or carry a GGUF
/// metadata backlink to the original, so this combines both signals: the
/// naming match decides the candidate *pool*, and the backlink (when present
/// and checkable) raises confidence without ever silently dropping a
/// candidate over a transient network failure.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the initial Hub search fails. Per-candidate
/// failures below that point are absorbed into [`QuantVerification::CheckFailed`]
/// rather than aborting the whole discovery.
pub async fn discover_quant_siblings(
    base_repo_id: &str,
    token: Option<&str>,
    client: &reqwest::Client,
) -> Result<Vec<QuantCandidate>, FetchError> {
    let short_name = base_repo_id.rsplit('/').next().unwrap_or(base_repo_id);
    let results = search_models(short_name, 50, None, None, None).await?;

    // BORROW: explicit .to_lowercase() for case-insensitive substring match
    let short_name_lower = short_name.to_lowercase();
    let semaphore = Arc::new(tokio::sync::Semaphore::new(8));
    let mut set: tokio::task::JoinSet<Option<QuantCandidate>> = tokio::task::JoinSet::new();

    for result in results {
        if result.model_id.eq_ignore_ascii_case(base_repo_id) {
            continue; // EXPLICIT: the base repo itself, not a sibling
        }
        if !result.model_id.to_lowercase().contains(&short_name_lower) {
            continue;
        }

        let limiter = Arc::clone(&semaphore);
        let client = client.clone();
        // BORROW: explicit .to_owned() — the spawned task must be 'static
        let base_repo_owned = base_repo_id.to_owned();
        let token_owned = token.map(str::to_owned);
        set.spawn(async move {
            let _permit = limiter.acquire_owned().await.ok()?;
            // BORROW: explicit .as_deref() for Option<String> → Option<&str>
            build_quant_candidate(
                result.model_id,
                &base_repo_owned,
                token_owned.as_deref(),
                &client,
            )
            .await
        });
    }

    let mut candidates: Vec<QuantCandidate> = Vec::new();
    while let Some(joined) = set.join_next().await {
        if let Ok(Some(candidate)) = joined {
            candidates.push(candidate);
        }
    }
    candidates.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));

    Ok(candidates)
}

/// Fans out [`fetch_repo_total_size`] across the given repository IDs through
/// a bounded `tokio::sync::Semaphore` (8 permits) to stay friendly to the HF
/// Hub on `--limit 100`-style invocations.
///
/// Per-repo failures (network errors, 404s, missing `size` fields) are
/// silently dropped from the returned map; callers should render rows whose
/// `repo_id` is absent from the map with a placeholder (`—`). The search
/// itself is not aborted on a single failure.
///
/// # Arguments
///
/// * `repo_ids` — Owned list of model identifiers. Ownership is moved into
///   the spawned tasks so each future is `'static`.
#[must_use]
pub async fn fetch_repo_sizes_concurrent(repo_ids: Vec<String>) -> HashMap<String, u64> {
    let semaphore = Arc::new(tokio::sync::Semaphore::new(8));
    let client = reqwest::Client::new();
    let mut set: tokio::task::JoinSet<Option<(String, u64)>> = tokio::task::JoinSet::new();

    for repo_id in repo_ids {
        let limiter = Arc::clone(&semaphore);
        let client = client.clone();
        set.spawn(async move {
            let _permit = limiter.acquire_owned().await.ok()?;
            // EXPLICIT: per-repo failure intentionally swallowed — the caller
            // renders the row with "—" rather than aborting the search.
            match fetch_repo_total_size(&repo_id, &client).await {
                Ok(bytes) => Some((repo_id, bytes)),
                Err(_) => None,
            }
        });
    }

    let mut by_repo: HashMap<String, u64> = HashMap::new();
    while let Some(joined) = set.join_next().await {
        if let Ok(Some((repo_id, bytes))) = joined {
            by_repo.insert(repo_id, bytes);
        }
    }

    by_repo
}

/// Fans out [`fetch_model_card`] across the given repository IDs through a
/// bounded `tokio::sync::Semaphore` (8 permits) and returns a map from
/// `repo_id` to the model card's tag list.
///
/// Per-repo failures (network errors, 404s, missing models) are silently
/// dropped from the returned map. Callers that want strict semantics should
/// treat absence as "no tags known". Mirrors the same fan-out pattern used by
/// [`fetch_repo_sizes_concurrent`] for `search --show size`.
///
/// # Arguments
///
/// * `repo_ids` — Owned list of model identifiers. Ownership is moved into
///   the spawned tasks so each future is `'static`.
#[must_use]
pub async fn fetch_tags_concurrent(repo_ids: Vec<String>) -> HashMap<String, Vec<String>> {
    let semaphore = Arc::new(tokio::sync::Semaphore::new(8));
    let mut set: tokio::task::JoinSet<Option<(String, Vec<String>)>> = tokio::task::JoinSet::new();

    for repo_id in repo_ids {
        let limiter = Arc::clone(&semaphore);
        set.spawn(async move {
            let _permit = limiter.acquire_owned().await.ok()?;
            // EXPLICIT: per-repo failure intentionally swallowed — missing
            // tags mean the row simply doesn't match any --tag filter (the
            // user's listing is not aborted on a single 404 / network blip).
            match fetch_model_card(&repo_id).await {
                Ok(card) => Some((repo_id, card.tags)),
                Err(_) => None,
            }
        });
    }

    let mut by_repo: HashMap<String, Vec<String>> = HashMap::new();
    while let Some(joined) = set.join_next().await {
        if let Ok(Some((repo_id, tags))) = joined {
            by_repo.insert(repo_id, tags);
        }
    }

    by_repo
}

/// Fetches model card metadata for a specific model from the `HuggingFace` Hub.
///
/// Queries `GET https://huggingface.co/api/models/{model_id}` and extracts
/// license, pipeline tag, tags, library name, and languages from the response.
///
/// # Arguments
///
/// * `model_id` — The full model identifier (e.g., `"mistralai/Ministral-3-3B-Instruct-2512"`).
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the API request fails or the model is not found.
pub async fn fetch_model_card(model_id: &str) -> Result<ModelCardMetadata, FetchError> {
    let client = reqwest::Client::new();
    let url = format!("{HF_API_BASE}/{model_id}");

    let response = client
        .get(url.as_str()) // BORROW: explicit .as_str()
        .send()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;

    if !response.status().is_success() {
        return Err(FetchError::Http(format!(
            "HF API returned status {} for model {model_id}",
            response.status()
        )));
    }

    let detail: ApiModelDetail = response
        .json()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;

    let (license, languages) = if let Some(card) = detail.card_data {
        let langs = match card.language {
            Some(ApiLanguage::Single(s)) => vec![s],
            Some(ApiLanguage::Multiple(v)) => v,
            None => Vec::new(),
        };
        (card.license, langs)
    } else {
        (None, Vec::new())
    };

    let gated = match detail.gated {
        ApiGated::Bool(false) => GateStatus::Open,
        ApiGated::Mode(ref mode) if mode.eq_ignore_ascii_case("manual") => GateStatus::Manual,
        ApiGated::Bool(true) | ApiGated::Mode(_) => GateStatus::Auto,
    };

    Ok(ModelCardMetadata {
        license,
        pipeline_tag: detail.pipeline_tag,
        tags: detail.tags,
        library_name: detail.library_name,
        languages,
        gated,
    })
}

/// Fetches the raw README text for a `HuggingFace` model repository.
///
/// Downloads `README.md` from the repository at the given revision.
/// Returns `Ok(None)` if the file does not exist (HTTP 404).
///
/// # Arguments
///
/// * `model_id` — The full model identifier (e.g., `"mistralai/Ministral-3-3B-Instruct-2512"`).
/// * `revision` — Git revision to fetch (defaults to `"main"` when `None`).
/// * `token` — Optional authentication token.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the request fails (other than 404).
pub async fn fetch_readme(
    model_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
) -> Result<Option<String>, FetchError> {
    let rev = revision.unwrap_or("main");
    let url = crate::chunked::build_download_url(model_id, rev, "README.md");
    let client = crate::chunked::build_client(token)?;

    let response = client
        .get(url.as_str()) // BORROW: explicit .as_str() instead of Deref coercion
        .send()
        .await
        .map_err(|e| FetchError::Http(format!("failed to fetch README for {model_id}: {e}")))?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }

    if !response.status().is_success() {
        return Err(FetchError::Http(format!(
            "README request for {model_id} returned status {}",
            response.status()
        )));
    }

    let text = response
        .text()
        .await
        .map_err(|e| FetchError::Http(format!("failed to read README for {model_id}: {e}")))?;

    Ok(Some(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_8bit_variants() {
        assert_eq!(normalize_quantization_terms("AWQ 8bit"), "AWQ 8-bit");
        assert_eq!(normalize_quantization_terms("AWQ 8-bit"), "AWQ 8-bit");
        assert_eq!(normalize_quantization_terms("AWQ int8"), "AWQ 8-bit");
        assert_eq!(normalize_quantization_terms("AWQ INT8"), "AWQ 8-bit");
    }

    #[test]
    fn normalize_4bit_variants() {
        assert_eq!(normalize_quantization_terms("GPTQ 4bit"), "GPTQ 4-bit");
        assert_eq!(normalize_quantization_terms("GPTQ INT4"), "GPTQ 4-bit");
        assert_eq!(normalize_quantization_terms("GPTQ 4-bit"), "GPTQ 4-bit");
    }

    #[test]
    fn normalize_fp8_variants() {
        assert_eq!(normalize_quantization_terms("FP8"), "fp8");
        assert_eq!(normalize_quantization_terms("float8"), "fp8");
        assert_eq!(normalize_quantization_terms("fp8"), "fp8");
    }

    #[test]
    fn normalize_passthrough() {
        assert_eq!(normalize_quantization_terms("llama 3"), "llama 3");
        assert_eq!(normalize_quantization_terms("RWKV-7"), "RWKV-7");
    }

    // ---------- quants sibling discovery ----------

    #[test]
    fn gguf_source_backlinks_finds_source_url() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.source.url".to_owned(),
            "https://huggingface.co/poolside/Laguna-XS-2.1".to_owned(),
        );
        metadata.insert("general.architecture".to_owned(), "llama".to_owned());
        assert_eq!(
            gguf_source_backlinks(&metadata),
            vec!["https://huggingface.co/poolside/Laguna-XS-2.1"]
        );
    }

    #[test]
    fn gguf_source_backlinks_finds_base_model_repo_url() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.base_model.0.repo_url".to_owned(),
            "https://huggingface.co/poolside/Laguna-XS-2.1".to_owned(),
        );
        assert_eq!(gguf_source_backlinks(&metadata).len(), 1);
    }

    #[test]
    fn gguf_source_backlinks_finds_huggingface_repository_key() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.source.huggingface.repository".to_owned(),
            "poolside/Laguna-XS-2.1".to_owned(),
        );
        assert_eq!(gguf_source_backlinks(&metadata).len(), 1);
    }

    #[test]
    fn gguf_source_backlinks_ignores_unrelated_keys() {
        let mut metadata = HashMap::new();
        metadata.insert("general.architecture".to_owned(), "llama".to_owned());
        metadata.insert("general.name".to_owned(), "Laguna-XS-2.1-GGUF".to_owned());
        assert!(gguf_source_backlinks(&metadata).is_empty());
    }
}
