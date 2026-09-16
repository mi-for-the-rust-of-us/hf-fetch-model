// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model family discovery and search via the `HuggingFace` Hub API.
//!
//! Queries the HF Hub for popular models, extracts `model_type` metadata,
//! compares against locally cached families, and fetches model card metadata.

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
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
/// * `token` — Authentication token (or `None` for anonymous requests).
///   Gated repos are typically still visible without one (gating restricts
///   content downloads, not search visibility or metadata), but a private
///   repo the token has access to is only ever found with it.
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
    token: Option<&str>,
) -> Result<Vec<SearchResult>, FetchError> {
    let normalized = normalize_quantization_terms(query);
    let client = crate::chunked::build_client(token)?;

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

/// How a repo's (or a filtered listing's) `.gguf` files relate to each other.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GgufFileSetKind {
    /// A single logical file split via `llama.cpp`'s official
    /// `*-NNNNN-of-MMMMM.gguf` convention — summing sizes across the set is
    /// correct (you need every shard). `first_shard` is the filename of the
    /// split whose index is `1` — under `llama.cpp`'s own convention, the
    /// *only* shard carrying the `GGUF` metadata `KV` table (and therefore
    /// any backlink key); later shards' headers hold tensor info alone.
    /// Carried here, rather than re-derived by [`pick_backlink_representative`]
    /// via its own separate scan, so classification and "which file is
    /// shard 1" can never drift apart.
    Sharded {
        /// The filename of the shard whose parsed index is `1`.
        first_shard: String,
    },
    /// Two or more mutually-exclusive quantization alternatives — summing
    /// sizes across the set is a category error, since nobody downloads more
    /// than one.
    QuantAlternatives,
    /// Zero or one `.gguf` file present — nothing to disambiguate; today's
    /// summing behavior is already correct.
    NotApplicable,
}

/// Returns `true` when `filename` has a `.gguf` extension, case-insensitively.
#[must_use]
pub fn is_gguf_filename(filename: &str) -> bool {
    filename.to_ascii_lowercase().ends_with(".gguf")
}

/// Parses `filename` against `llama.cpp`'s split-file convention
/// (`<prefix>-<index>-of-<total>.gguf`, `index`/`total` equal-width,
/// zero-padded digit strings). Returns `(prefix, index, total)` on a match.
fn parse_gguf_split_name(filename: &str) -> Option<(&str, u32, u32)> {
    if !is_gguf_filename(filename) {
        return None;
    }
    // `.gguf` is 5 ASCII bytes, so this length is always a valid boundary.
    let stem = filename.get(..filename.len().checked_sub(5)?)?;
    let (before, total_str) = stem.split_once("-of-")?;
    let (prefix, index_str) = before.rsplit_once('-')?;
    if index_str.is_empty()
        || total_str.is_empty()
        || index_str.len() != total_str.len()
        || !index_str.bytes().all(|b| b.is_ascii_digit())
        || !total_str.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let index: u32 = index_str.parse().ok()?;
    let total: u32 = total_str.parse().ok()?;
    Some((prefix, index, total))
}

/// Classifies a repo's `.gguf` files as a sharded single file or a set of
/// mutually-exclusive quant alternatives.
///
/// `Sharded` requires **every** `.gguf` file in `filenames` to match
/// [`parse_gguf_split_name`] with the same prefix and total, the file count
/// to equal that total, and the indices to cover `1..=total` exactly.
/// Anything else with two or more `.gguf` files — a mismatched prefix, a
/// file outside the split convention entirely, a missing or duplicate index
/// — is `QuantAlternatives`.
#[must_use]
pub fn classify_gguf_files(filenames: &[&str]) -> GgufFileSetKind {
    let gguf: Vec<&str> = filenames
        .iter()
        .copied()
        .filter(|f| is_gguf_filename(f))
        .collect();
    if gguf.len() <= 1 {
        return GgufFileSetKind::NotApplicable;
    }

    let Some(parsed): Option<Vec<(&str, u32, u32)>> =
        gguf.iter().map(|f| parse_gguf_split_name(f)).collect()
    else {
        return GgufFileSetKind::QuantAlternatives;
    };
    // INDEX: gguf.len() > 1 checked above, so parsed (same length) is non-empty
    let Some(&(first_prefix, _, first_total)) = parsed.first() else {
        return GgufFileSetKind::QuantAlternatives;
    };
    let same_group = parsed
        .iter()
        .all(|&(prefix, _, total)| prefix == first_prefix && total == first_total);
    if !same_group {
        return GgufFileSetKind::QuantAlternatives;
    }
    // CAST: u32 -> usize, total is a small shard count from a zero-padded filename token
    #[allow(clippy::as_conversions)]
    let total_usize = first_total as usize;
    if gguf.len() != total_usize {
        return GgufFileSetKind::QuantAlternatives;
    }

    let mut indices: Vec<u32> = parsed.iter().map(|&(_, index, _)| index).collect();
    indices.sort_unstable();
    let complete = indices.iter().enumerate().all(|(i, &index)| {
        // CAST: usize -> u32, i is bounded by total_usize which came from a u32
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let expected = i as u32 + 1;
        index == expected
    });

    if !complete {
        return GgufFileSetKind::QuantAlternatives;
    }

    // INDEX: `complete` just verified the indices cover `1..=total_usize`
    // exactly, so exactly one entry in `parsed` (same order/length as
    // `gguf`, both filtered from `filenames` together) has index `1`.
    let Some((&first_shard, _)) = gguf
        .iter()
        .zip(parsed.iter())
        .find(|&(_, &(_, index, _))| index == 1)
    else {
        // EXPLICIT: unreachable given `complete` above; degrade to
        // QuantAlternatives instead of panicking if this invariant is ever
        // violated by a future edit.
        return GgufFileSetKind::QuantAlternatives;
    };
    GgufFileSetKind::Sharded {
        // BORROW: explicit .to_owned() for &str → owned String field
        first_shard: first_shard.to_owned(),
    }
}

/// Computes the min/max size across `.gguf` files in `filenames_with_size`,
/// when they are mutually-exclusive quant alternatives (see
/// [`classify_gguf_files`]). Returns `None` when the set is `Sharded` or
/// `NotApplicable` (summing is already correct there) or when none of the
/// `.gguf` files carry a known size.
///
/// Takes `(filename, size)` pairs rather than a concrete file type so both
/// the remote listing (`repo::RepoFile`, whose `size` is `Option<u64>`) and
/// the local cache listing (`cache::CacheFileUsage`, whose `size` is always
/// known) can share one implementation. `size` is `Option<u64>` — and every
/// filename must be passed, not just the ones with a known size —
/// because classification (`Sharded` vs `QuantAlternatives`) depends on
/// seeing the *complete* file set: a genuinely sharded repo where the Hub
/// API happens not to report one shard's size would otherwise lose that
/// shard's filename before `classify_gguf_files` ever saw it, undercounting
/// the shard total and misclassifying the whole set as `QuantAlternatives`.
/// Only the size-known subset feeds the min/max once classification itself
/// has already run on every filename.
#[must_use]
pub fn gguf_size_range<'a, I>(filenames_with_size: I) -> Option<(u64, u64)>
where
    I: IntoIterator<Item = (&'a str, Option<u64>)>,
{
    let pairs: Vec<(&str, Option<u64>)> = filenames_with_size.into_iter().collect();
    let filenames: Vec<&str> = pairs.iter().map(|&(name, _)| name).collect();
    if !matches!(
        classify_gguf_files(&filenames),
        GgufFileSetKind::QuantAlternatives
    ) {
        return None;
    }

    let sizes: Vec<u64> = pairs
        .iter()
        .filter(|&&(name, _)| is_gguf_filename(name))
        .filter_map(|&(_, size)| size)
        .collect();
    let min = sizes.iter().copied().min()?;
    let max = sizes.iter().copied().max()?;
    Some((min, max))
}

/// A repo's aggregate size, aware of the `.gguf` quant-alternatives case.
///
/// `total` is always the raw sum of every listed file's size — well-defined,
/// if not always the most useful number. When `quant_alternatives` is `true`,
/// `size_min`/`size_max` give the more honest range: the smallest and
/// largest `.gguf` file, since the repo's files are mutually-exclusive
/// choices rather than parts of a whole.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct RepoSizeSummary {
    /// Sum of every listed file's size.
    pub total: u64,
    /// Whether this repo's `.gguf` files are quant alternatives (see [`GgufFileSetKind`]).
    pub quant_alternatives: bool,
    /// Smallest `.gguf` file's size, when `quant_alternatives` is `true`.
    pub size_min: Option<u64>,
    /// Largest `.gguf` file's size, when `quant_alternatives` is `true`.
    pub size_max: Option<u64>,
}

/// Fetches a repo's file listing and summarizes its size, detecting the
/// `.gguf` quant-alternatives case via [`classify_gguf_files`].
///
/// Additive alongside [`fetch_repo_total_size`] (kept unchanged for existing
/// callers/downstream consumers) rather than replacing it — this is the
/// quant-aware variant `hf-fm search --show size` uses.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the API request fails or returns a
/// non-success status. Returns [`FetchError::RepoNotFound`] if the
/// repository does not exist on the Hub.
pub async fn fetch_repo_size_summary(
    repo_id: &str,
    client: &reqwest::Client,
) -> Result<RepoSizeSummary, FetchError> {
    let files = crate::repo::list_repo_files_with_metadata(repo_id, None, None, client).await?;
    let total: u64 = files.iter().filter_map(|f| f.size).sum();

    // BORROW: explicit .as_str() instead of Deref coercion
    let sized: Vec<(&str, Option<u64>)> = files
        .iter()
        .map(|f| (f.filename.as_str(), f.size))
        .collect();
    let range = gguf_size_range(sized);

    Ok(RepoSizeSummary {
        total,
        quant_alternatives: range.is_some(),
        size_min: range.map(|(min, _)| min),
        size_max: range.map(|(_, max)| max),
    })
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

/// Picks the `.gguf` file whose header [`build_quant_candidate`] should
/// inspect for a `general.source.url`-style backlink.
///
/// A genuinely sharded `GGUF` file (see [`classify_gguf_files`]) carries its
/// metadata `KV` table in the *first* split only — `llama.cpp`'s own
/// convention; later shards' headers hold tensor info alone, no metadata.
/// Picking the smallest shard by size (very often the *last* one, not the
/// first) would inspect a header with no backlink key at all and misreport
/// [`QuantVerification::Unverified`] even when the repo genuinely names its
/// source. For a mutually-exclusive quant-alternatives set (or a lone
/// `.gguf` file), smallest-by-size stays the right choice: it is the
/// cheapest header to fetch, and every alternative carries the same
/// backlink (they are re-quantizations of the same source checkpoint).
///
/// Reads which file is shard 1 straight off [`GgufFileSetKind::Sharded`]'s
/// own `first_shard` field rather than re-deriving it here, so this can
/// never disagree with `classify_gguf_files`'s own notion of "sharded".
fn pick_backlink_representative(
    gguf_files: &[crate::repo::RepoFile],
) -> Option<&crate::repo::RepoFile> {
    // BORROW: explicit .as_str() instead of Deref coercion
    let filenames: Vec<&str> = gguf_files.iter().map(|f| f.filename.as_str()).collect();
    if let GgufFileSetKind::Sharded { first_shard } = classify_gguf_files(&filenames)
        && let Some(file) = gguf_files.iter().find(|f| f.filename == first_shard)
    {
        return Some(file);
    }
    gguf_files.iter().min_by_key(|f| f.size.unwrap_or(u64::MAX))
}

/// Lists `candidate_repo_id`'s files and, if it holds at least one `.gguf`
/// file, cross-checks a representative one's metadata backlink (see
/// [`pick_backlink_representative`]) against `base_repo_id`.
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

    let gguf_files: Vec<_> = files
        .iter()
        .filter(|f| is_gguf_filename(&f.filename))
        .cloned()
        .collect();

    let Some(representative) = pick_backlink_representative(&gguf_files) else {
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
    let results = search_models(short_name, 50, None, None, None, token).await?;

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

/// Fans out an async per-item operation through a bounded
/// `tokio::sync::Semaphore`, running at most `concurrency` futures at once.
/// Returns one `Option<R>` per input item, in the same order as `items` —
/// regardless of which task finishes first. `None` marks a per-item outcome
/// the caller should treat as absent: `f` returning `None`, or (defensively)
/// a task whose permit was never acquired or that panicked. Neither case
/// aborts the fan-out; every other item still runs to completion.
///
/// The single bounded-concurrency scaffold behind every per-repo network
/// fan-out in this crate — [`fetch_repo_sizes_concurrent`],
/// [`fetch_repo_size_summaries_concurrent`], [`fetch_tags_concurrent`], and
/// (through the `hf-fm` CLI, a separate crate that can only reach a `pub`
/// item) `quants --fits`'s bounded offload-plan inspection — so a future
/// change to permit-acquisition, panic-handling, or concurrency semantics
/// only needs to happen once. Order-preserving (rather than the map-keyed
/// shape an earlier version of this helper had) because `--fits` needs
/// every row's verdict, in row order, even the ones that produced no
/// result — a `HashMap` has no way to represent "present but absent",
/// only "absent". Callers that want a `repo_id`-keyed map instead (every
/// caller above except `--fits`) zip `items` back onto the result, which is
/// why `T` does not need `Eq + Hash` here even though every current caller's
/// `T` happens to satisfy it.
///
/// `f` is `Fn`, not `FnOnce`, since it is called once per item — shared
/// state it needs (e.g. an HTTP client) should be cloned inside the closure
/// body on each call, not moved out of the closure's own captured environment.
pub async fn fan_out_bounded<T, R, F, Fut>(
    items: Vec<T>,
    concurrency: usize,
    f: F,
) -> Vec<Option<R>>
where
    T: Send + 'static,
    R: Send + 'static,
    F: Fn(T) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Option<R>> + Send + 'static,
{
    let len = items.len();
    let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let f = Arc::new(f);
    let mut set: tokio::task::JoinSet<(usize, Option<R>)> = tokio::task::JoinSet::new();

    for (index, item) in items.into_iter().enumerate() {
        let limiter = Arc::clone(&semaphore);
        let f = Arc::clone(&f);
        set.spawn(async move {
            let Ok(_permit) = limiter.acquire_owned().await else {
                return (index, None);
            };
            (index, f(item).await)
        });
    }

    let mut slots: Vec<Option<R>> = (0..len).map(|_| None).collect();
    while let Some(joined) = set.join_next().await {
        if let Ok((index, result)) = joined
            && let Some(slot) = slots.get_mut(index)
        {
            *slot = result;
        }
    }
    slots
}

/// Zips `keys` back onto `fan_out_bounded`'s order-aligned `Vec<Option<R>>`,
/// dropping absent slots — the map-building step every `repo_id`-keyed
/// fan-out caller in this module needs after switching to the order-
/// preserving [`fan_out_bounded`].
fn zip_into_map<T: Eq + std::hash::Hash, R>(
    keys: Vec<T>,
    results: Vec<Option<R>>,
) -> HashMap<T, R> {
    keys.into_iter()
        .zip(results)
        .filter_map(|(key, result)| result.map(|r| (key, r)))
        .collect()
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
    let client = reqwest::Client::new();
    // BORROW: explicit .clone() — fan_out_bounded consumes `repo_ids`, but
    // the original ids are still needed afterward to key the returned map.
    let keys = repo_ids.clone();
    let results = fan_out_bounded(repo_ids, 8, move |repo_id| {
        let client = client.clone();
        // EXPLICIT: per-repo failure intentionally swallowed — the caller
        // renders the row with "—" rather than aborting the search.
        async move { fetch_repo_total_size(&repo_id, &client).await.ok() }
    })
    .await;
    zip_into_map(keys, results)
}

/// Fans out [`fetch_repo_size_summary`] across the given repository IDs
/// through a bounded `tokio::sync::Semaphore` (8 permits) — the quant-aware
/// counterpart `hf-fm search --show size` uses.
///
/// Per-repo failures are silently dropped from the returned map; callers
/// should render rows whose `repo_id` is absent with a placeholder (`—`).
/// This is deliberately distinct from a client-build failure (see Errors
/// below): one bad repo is an expected, per-item outcome across a fan-out
/// this wide, but a token too malformed to become an HTTP header is a
/// caller configuration error every row would otherwise fail identically
/// and silently for — worth surfacing loudly instead, the same way
/// [`search_models`] already does for the same failure mode.
///
/// # Arguments
///
/// * `repo_ids` — Owned list of model identifiers. Ownership is moved into
///   the spawned tasks so each future is `'static`.
/// * `token` — Authentication token (or `None` for anonymous requests),
///   applied to every fanned-out request via [`crate::chunked::build_client`]
///   — without this, a private or gated repo silently fails to size (the
///   same per-repo failure path as a 404, indistinguishable to the caller).
///
/// # Errors
///
/// Returns [`FetchError::Http`] if `token` cannot be turned into a valid
/// HTTP header value (via [`crate::chunked::build_client`]) — before any
/// per-repo request is attempted.
pub async fn fetch_repo_size_summaries_concurrent(
    repo_ids: Vec<String>,
    token: Option<&str>,
) -> Result<HashMap<String, RepoSizeSummary>, FetchError> {
    let client = crate::chunked::build_client(token)?;
    // BORROW: explicit .clone() — fan_out_bounded consumes `repo_ids`, but
    // the original ids are still needed afterward to key the returned map.
    let keys = repo_ids.clone();
    let results = fan_out_bounded(repo_ids, 8, move |repo_id| {
        let client = client.clone();
        // EXPLICIT: per-repo failure intentionally swallowed — the caller
        // renders the row with "—" rather than aborting the search.
        async move { fetch_repo_size_summary(&repo_id, &client).await.ok() }
    })
    .await;
    Ok(zip_into_map(keys, results))
}

/// Fans out [`fetch_model_card`] across the given repository IDs through a
/// bounded `tokio::sync::Semaphore` (8 permits) and returns a map from
/// `repo_id` to the model card's tag list.
///
/// Per-repo failures (network errors, 404s, missing models) are silently
/// dropped from the returned map. Callers that want strict semantics should
/// treat absence as "no tags known". Uses the same bounded fan-out
/// scaffold as [`fetch_repo_sizes_concurrent`], [`fan_out_bounded`].
///
/// # Arguments
///
/// * `repo_ids` — Owned list of model identifiers. Ownership is moved into
///   the spawned tasks so each future is `'static`.
#[must_use]
pub async fn fetch_tags_concurrent(repo_ids: Vec<String>) -> HashMap<String, Vec<String>> {
    // BORROW: explicit .clone() — fan_out_bounded consumes `repo_ids`, but
    // the original ids are still needed afterward to key the returned map.
    let keys = repo_ids.clone();
    let results = fan_out_bounded(repo_ids, 8, move |repo_id| async move {
        // EXPLICIT: per-repo failure intentionally swallowed — missing
        // tags mean the row simply doesn't match any --tag filter (the
        // user's listing is not aborted on a single 404 / network blip).
        fetch_model_card(&repo_id).await.ok().map(|card| card.tags)
    })
    .await;
    zip_into_map(keys, results)
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

    // ---------- classify_gguf_files ----------

    #[test]
    fn classify_gguf_files_not_applicable_for_zero_or_one_file() {
        assert_eq!(classify_gguf_files(&[]), GgufFileSetKind::NotApplicable);
        assert_eq!(
            classify_gguf_files(&["model-Q4_K_M.gguf"]),
            GgufFileSetKind::NotApplicable
        );
    }

    #[test]
    fn classify_gguf_files_recognizes_a_complete_shard_set() {
        let files = [
            "model-00001-of-00003.gguf",
            "model-00002-of-00003.gguf",
            "model-00003-of-00003.gguf",
        ];
        assert_eq!(
            classify_gguf_files(&files),
            GgufFileSetKind::Sharded {
                first_shard: "model-00001-of-00003.gguf".to_owned()
            }
        );
    }

    #[test]
    fn classify_gguf_files_recognizes_a_shard_set_regardless_of_listing_order() {
        let files = [
            "model-00003-of-00003.gguf",
            "model-00001-of-00003.gguf",
            "model-00002-of-00003.gguf",
        ];
        // `first_shard` must always be the index-1 filename, not simply the
        // first one encountered in `files`.
        assert_eq!(
            classify_gguf_files(&files),
            GgufFileSetKind::Sharded {
                first_shard: "model-00001-of-00003.gguf".to_owned()
            }
        );
    }

    #[test]
    fn classify_gguf_files_flags_quant_alternatives() {
        let files = ["model-Q4_K_M.gguf", "model-Q5_K_M.gguf", "model-Q8_0.gguf"];
        assert_eq!(
            classify_gguf_files(&files),
            GgufFileSetKind::QuantAlternatives
        );
    }

    #[test]
    fn classify_gguf_files_flags_a_missing_shard_index() {
        // Claims 3-of-3 but only two files are present — incomplete.
        let files = ["model-00001-of-00003.gguf", "model-00003-of-00003.gguf"];
        assert_eq!(
            classify_gguf_files(&files),
            GgufFileSetKind::QuantAlternatives
        );
    }

    #[test]
    fn classify_gguf_files_flags_a_duplicate_shard_index() {
        let files = [
            "model-00001-of-00003.gguf",
            "model-00001-of-00003.gguf",
            "model-00003-of-00003.gguf",
        ];
        assert_eq!(
            classify_gguf_files(&files),
            GgufFileSetKind::QuantAlternatives
        );
    }

    #[test]
    fn classify_gguf_files_flags_mismatched_prefixes() {
        let files = ["model-a-00001-of-00002.gguf", "model-b-00002-of-00002.gguf"];
        assert_eq!(
            classify_gguf_files(&files),
            GgufFileSetKind::QuantAlternatives
        );
    }

    #[test]
    fn classify_gguf_files_ignores_non_gguf_files() {
        let files = [
            "model-00001-of-00002.gguf",
            "model-00002-of-00002.gguf",
            "config.json",
            "README.md",
        ];
        assert_eq!(
            classify_gguf_files(&files),
            GgufFileSetKind::Sharded {
                first_shard: "model-00001-of-00002.gguf".to_owned()
            }
        );
    }

    // ---------- gguf_size_range ----------

    #[test]
    fn gguf_size_range_computes_min_max_for_quant_alternatives() {
        let files = vec![
            ("model-Q8_0.gguf", Some(20_000)),
            ("model-Q4_K_M.gguf", Some(10_000)),
            ("model-Q3_K_S.gguf", Some(14_000)),
        ];
        assert_eq!(gguf_size_range(files), Some((10_000, 20_000)));
    }

    #[test]
    fn gguf_size_range_returns_none_for_a_sharded_set() {
        let files = vec![
            ("model-00001-of-00002.gguf", Some(10_000)),
            ("model-00002-of-00002.gguf", Some(10_000)),
        ];
        assert_eq!(gguf_size_range(files), None);
    }

    #[test]
    fn gguf_size_range_still_classifies_sharded_when_one_shard_has_no_known_size() {
        // Regression test: a genuinely sharded 3-file set where the Hub API
        // happened not to report one shard's size must still classify as
        // Sharded (via the full filename list) and therefore return None —
        // dropping the unsized shard's filename before classification would
        // leave only 2 of 3 expected filenames, tripping the "file count
        // must equal the parsed total" check and misclassifying the whole
        // set as QuantAlternatives, which then reports a bogus min/max
        // range over only the 2 known-size shards.
        let files = vec![
            ("model-00001-of-00003.gguf", Some(5_000)),
            ("model-00002-of-00003.gguf", None),
            ("model-00003-of-00003.gguf", Some(5_000)),
        ];
        assert_eq!(gguf_size_range(files), None);
    }

    #[test]
    fn gguf_size_range_excludes_unsized_files_from_min_max_but_keeps_classifying_on_all() {
        // A quant-alternatives set where one candidate's size is unknown:
        // classification still sees all 3 filenames (so 3 mutually-exclusive
        // `.gguf` files are correctly detected), but the unsized entry is
        // excluded from the min/max computation itself.
        let files = vec![
            ("model-Q8_0.gguf", Some(20_000)),
            ("model-Q4_K_M.gguf", None),
            ("model-Q3_K_S.gguf", Some(14_000)),
        ];
        assert_eq!(gguf_size_range(files), Some((14_000, 20_000)));
    }

    // ---------- pick_backlink_representative ----------

    fn repo_file(filename: &str, size: u64) -> crate::repo::RepoFile {
        crate::repo::RepoFile {
            // BORROW: explicit .to_owned() for &str → owned String field
            filename: filename.to_owned(),
            size: Some(size),
            sha256: None,
        }
    }

    #[test]
    fn pick_backlink_representative_picks_first_shard_for_a_sharded_set() {
        // The metadata KV table (and any backlink key) lives only in the
        // first split, per llama.cpp's own convention — even though it is
        // not the smallest file here, it must still be the one chosen.
        let files = vec![
            repo_file("model-00001-of-00003.gguf", 5_000),
            repo_file("model-00002-of-00003.gguf", 5_000),
            repo_file("model-00003-of-00003.gguf", 1_000),
        ];
        assert_eq!(
            pick_backlink_representative(&files).map(|f| f.filename.as_str()),
            Some("model-00001-of-00003.gguf")
        );
    }

    #[test]
    fn pick_backlink_representative_picks_smallest_for_quant_alternatives() {
        let files = vec![
            repo_file("model-Q8_0.gguf", 20_000),
            repo_file("model-Q4_K_M.gguf", 10_000),
            repo_file("model-Q3_K_S.gguf", 14_000),
        ];
        assert_eq!(
            pick_backlink_representative(&files).map(|f| f.filename.as_str()),
            Some("model-Q4_K_M.gguf")
        );
    }

    #[test]
    fn pick_backlink_representative_picks_the_lone_file() {
        let files = vec![repo_file("model.gguf", 10_000)];
        assert_eq!(
            pick_backlink_representative(&files).map(|f| f.filename.as_str()),
            Some("model.gguf")
        );
    }

    #[test]
    fn pick_backlink_representative_returns_none_for_no_gguf_files() {
        assert!(pick_backlink_representative(&[]).is_none());
    }

    // ---------- fan_out_bounded ----------

    #[tokio::test]
    async fn fan_out_bounded_preserves_item_order() {
        let items = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let out =
            fan_out_bounded(items, 2, |item| async move { Some(format!("{item}-done")) }).await;

        assert_eq!(
            out,
            vec![
                Some("a-done".to_owned()),
                Some("b-done".to_owned()),
                Some("c-done".to_owned()),
            ]
        );
    }

    #[tokio::test]
    async fn fan_out_bounded_keeps_a_none_slot_for_per_item_failures() {
        let items = vec!["keep".to_owned(), "drop".to_owned(), "keep2".to_owned()];
        let out = fan_out_bounded(items, 2, |item| async move {
            if item == "drop" { None } else { Some(item) }
        })
        .await;

        assert_eq!(
            out,
            vec![Some("keep".to_owned()), None, Some("keep2".to_owned()),]
        );
    }

    #[tokio::test]
    async fn zip_into_map_drops_none_slots_and_keys_by_item() {
        let keys = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let results = vec![Some(1), None, Some(3)];

        let map = zip_into_map(keys, results);

        assert_eq!(map.len(), 2);
        assert_eq!(map.get("a"), Some(&1));
        assert!(!map.contains_key("b"));
        assert_eq!(map.get("c"), Some(&3));
    }

    #[tokio::test]
    async fn fan_out_bounded_respects_concurrency_limit() {
        // 6 items, concurrency 2: an atomic counter of in-flight tasks must
        // never exceed 2 at any point during the run.
        let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let max_seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let items: Vec<u32> = (0..6).collect();

        let in_flight_for_closure = Arc::clone(&in_flight);
        let max_seen_for_closure = Arc::clone(&max_seen);
        let out = fan_out_bounded(items, 2, move |item| {
            let in_flight = Arc::clone(&in_flight_for_closure);
            let max_seen = Arc::clone(&max_seen_for_closure);
            async move {
                let now = in_flight.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                max_seen.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                in_flight.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                Some(item)
            }
        })
        .await;

        assert_eq!(out.len(), 6);
        assert!(
            max_seen.load(std::sync::atomic::Ordering::SeqCst) <= 2,
            "observed more than 2 tasks in flight at once"
        );
    }

    // ---------- fetch_repo_size_summaries_concurrent ----------

    #[tokio::test]
    async fn fetch_repo_size_summaries_concurrent_reports_a_malformed_token_loudly() {
        // A token with an embedded newline can never become a valid HTTP
        // header value — `build_client` fails before any per-repo request
        // is attempted. This must surface as an `Err`, the same way
        // `search_models` already fails loudly for the identical cause,
        // rather than silently degrading to an empty map indistinguishable
        // from "every repo failed to size".
        let result =
            fetch_repo_size_summaries_concurrent(vec!["org/model".to_owned()], Some("bad\ntoken"))
                .await;
        assert!(
            result.is_err(),
            "a malformed token must be reported as an error"
        );
    }
}
