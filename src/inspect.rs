// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tensor-file header inspection (local and remote).
//!
//! Reads tensor metadata (names, shapes, dtypes, byte offsets) without
//! downloading full weight data. All four supported formats resolve
//! cache-first with an [`HttpRangeReader`] fallback — `.npz` since v0.11.0
//! ([`inspect_npz`] drives `anamnesis::inspect_npz_from_reader` over the
//! reader), `.safetensors` since v0.11.1 ([`inspect_safetensors`] drives
//! `anamnesis::parse_safetensors_header_from_reader` over the same reader),
//! `.gguf` since v0.11.2 ([`inspect_gguf`] drives
//! `anamnesis::parse_gguf_front_matter_from_reader` over the same reader —
//! anamnesis's earlier `inspect_gguf_from_reader`, 0.4.5, is summary-only
//! and insufficient for `hf-fm`'s per-tensor rendering), and `.pth` since
//! v0.11.4 ([`inspect_pth`] drives `anamnesis::parse_pth_front_matter_from_reader`
//! over the same reader — anamnesis's earlier `inspect_pth_from_reader`,
//! 0.6.x, is likewise summary-only; the full-detail counterpart shipped in
//! anamnesis 0.7.5).
//!
//! The primary types are [`TensorInfo`] (per-tensor metadata),
//! [`SafetensorsHeaderInfo`] (the format-agnostic parsed-header shape all
//! four formats return), and [`ShardedIndex`] (shard-to-tensor mapping for
//! sharded models). For cheap discovery without header parsing,
//! [`list_cached_tensor_files`] enumerates a cached repo's tensor files
//! across all four formats ([`list_cached_safetensors`] is the
//! `.safetensors`-only subset).
//!
//! The module also reads small JSON sidecars from the same cache-first /
//! HTTP-fallback path: [`AdapterConfig`] (`adapter_config.json`, for `PEFT`
//! adapters) and [`ModelConfig`] (`config.json`, the architecture parameters
//! that drive `inspect --check-gpu --context` KV-cache budgeting), via
//! [`fetch_model_config`] / [`fetch_model_config_cached`].
//!
//! Eight cache-first entry points take a repo id: the four single-file
//! inspectors ([`inspect_npz`] and its siblings), [`inspect_repo_safetensors`]
//! (whose shards are cache-first, though its listing always asks the Hub),
//! and the three `fetch_*` readers. They take a token and a revision as loose
//! arguments and read the default `HF` cache. Each has a `_with_config` twin (since v0.12.3, e.g.
//! [`inspect_npz_with_config`]) that takes the token, the revision and the
//! cache directory from a [`FetchConfig`] instead, so a caller that
//! downloads with a custom [`FetchConfig::output_dir`] inspects that same
//! cache.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;

use crate::cache;
use crate::cache_layout;
use crate::chunked;
use crate::config::FetchConfig;
use crate::error::FetchError;
use crate::http_range::{HttpRangeReader, RangeStats};

// -----------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------

/// Metadata for a single tensor from a `.safetensors` header.
///
/// This is hf-fetch-model's own type — lightweight, no quantization logic.
/// Consumers map this into their own richer types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Tensor name (e.g., `"model.layers.0.self_attn.q_proj.weight"`).
    pub name: String,
    /// Element dtype string as it appears in the header (e.g., `"F8_E4M3"`, `"BF16"`).
    pub dtype: String,
    /// Tensor shape (e.g., `[7168, 7168]`).
    pub shape: Vec<usize>,
    /// Byte offset range `[start, end)` within the data section of the file.
    pub data_offsets: (u64, u64),
}

impl TensorInfo {
    /// Total number of elements (product of shape dimensions).
    ///
    /// Returns `1` for a scalar (empty shape).
    #[must_use]
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().fold(1u64, |acc, &d| {
            // CAST: usize → u64, dimension values fit in u64
            #[allow(clippy::as_conversions)]
            let dim = d as u64;
            acc.saturating_mul(dim)
        })
    }

    /// Byte length of the tensor data (`end - start`).
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.data_offsets.1.saturating_sub(self.data_offsets.0)
    }

    /// Bytes per element for the tensor's dtype, if recognized.
    ///
    /// Returns `None` for unknown dtype strings. Recognized dtypes:
    ///
    /// | Dtype string | Bytes | Notes |
    /// |-------------|-------|-------|
    /// | `"BOOL"` | 1 | |
    /// | `"U8"`, `"I8"` | 1 | |
    /// | `"F8_E4M3"`, `"F8_E5M2"` | 1 | FP8 variants |
    /// | `"U16"`, `"I16"`, `"F16"`, `"BF16"` | 2 | |
    /// | `"U32"`, `"I32"`, `"F32"` | 4 | |
    /// | `"U64"`, `"I64"`, `"F64"` | 8 | |
    #[must_use]
    pub fn dtype_bytes(&self) -> Option<usize> {
        // BORROW: explicit .as_str() instead of Deref coercion
        match self.dtype.as_str() {
            "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => Some(1),
            "U16" | "I16" | "F16" | "BF16" => Some(2),
            "U32" | "I32" | "F32" => Some(4),
            "U64" | "I64" | "F64" => Some(8),
            _ => None,
        }
    }
}

/// Bytes per element for a model's activation dtype, as spelled in a
/// `config.json` `torch_dtype` field.
///
/// Distinct from [`TensorInfo::dtype_bytes`], which maps the *safetensors*
/// header spellings (`"BF16"`, `"F16"`, …); `config.json` uses the `PyTorch`
/// spellings (`"bfloat16"`, `"float16"`, `"float32"`, `"float8_e4m3fn"`).
/// Used to size the KV cache, whose element dtype tracks the model's
/// activations (typically `bf16` / `fp16`) independently of weight
/// quantization. Defaults to `2` when the dtype is absent or unrecognized —
/// the modern inference default and the safe assumption for KV sizing.
#[must_use]
pub fn torch_dtype_bytes(torch_dtype: Option<&str>) -> u8 {
    match torch_dtype {
        Some("float32" | "float") => 4,
        Some("float8_e4m3fn" | "float8_e5m2") => 1,
        // `bf16` / `fp16`, and any unknown or absent dtype: 2-byte activations.
        _ => 2,
    }
}

/// Quantization scheme + size estimates for a `.safetensors` file, cached or remote.
///
/// Populated via `anamnesis::InspectInfo::from(&header)` by both the
/// cache-hit path ([`inspect_safetensors_local`]) and the remote path
/// ([`inspect_safetensors`], v0.11.1+) — the two share the same
/// `safetensors_header_to_info` mapping, so quant detection works
/// identically cached or remote. Absent (`None`) when:
/// - the safetensors file has no detected quantization (`QuantScheme::Unquantized`), or
/// - the file format isn't safetensors (`GGUF` / `NPZ` / `PTH` carry no
///   quant-method metadata).
///
/// Decoupled from `anamnesis::QuantScheme` (a `#[non_exhaustive]` enum) so
/// downstream library consumers (such as `candle-mi`) aren't forced to
/// match every variant. The `scheme` field stores `QuantScheme`'s `Display`
/// output (`"FineGrainedFp8"`, `"Bnb4"`, `"Gptq"`, `"Awq"`, …); consumers
/// that need to match exact variants should call
/// `anamnesis::parse_safetensors_header` themselves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantInfo {
    /// Detected quantization scheme as the `Display` form of
    /// `anamnesis::QuantScheme` (e.g. `"FineGrainedFp8"`, `"Bnb4"`).
    pub scheme: String,
    /// Bytes stored on disk for tensor data (header excluded).
    pub stored_bytes: u64,
    /// Estimated bytes after dequantising to `BF16`. For `BnB-NF4`/`FP4`
    /// (`U8`-packed nibbles), this is `stored_bytes × 4`; for `FP8` / `GPTQ` /
    /// `AWQ` / `BnB-INT8` it's `num_elements × 2` summed over weight
    /// tensors, plus passthrough tensors copied as-is. The formula lives
    /// in `anamnesis::InspectInfo::from(&SafetensorsHeader)` — hf-fm just
    /// reads the result.
    pub dequantized_bytes: u64,
}

/// Parsed safetensors header metadata.
///
/// Marked `#[non_exhaustive]` (since v0.10.3) — the struct has been
/// growing through v0.10.x (the `quant_info` field landed in Phase C)
/// and will keep growing in v0.11.x. External library consumers should
/// pattern-match with `..` or use field reads, not exhaustive struct
/// literals.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SafetensorsHeaderInfo {
    /// All tensors in the header, in the order they appear in the JSON.
    pub tensors: Vec<TensorInfo>,
    /// Raw `__metadata__` entries, if present.
    ///
    /// For quantized models, this typically contains entries like
    /// `quant_method`, `bits`, `group_size` that tell GPTQ from AWQ without
    /// downloading weights.
    pub metadata: Option<HashMap<String, String>>,
    /// Size of the JSON header in bytes.
    pub header_size: u64,
    /// Total file size in bytes (header + data), if known.
    ///
    /// **Source:** for local files, from `std::fs::metadata().len()`. For HTTP
    /// Range requests, extracted from the `Content-Range` response header of
    /// the first request (`bytes 0-7/TOTAL` → `TOTAL`). This is free — no
    /// extra request needed.
    pub file_size: Option<u64>,
    /// Quantization scheme + size estimates (safetensors only, cached or remote).
    ///
    /// Populated whenever a non-`Unquantized` `QuantScheme` is detected —
    /// both [`inspect_safetensors_local`] (cache-hit) and
    /// [`inspect_safetensors`] (remote, v0.11.1+) go through the same
    /// anamnesis primitive. `None` for unquantized safetensors and for
    /// `GGUF` / `NPZ` / `PTH` files (no quant-method metadata).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant_info: Option<QuantInfo>,
}

impl SafetensorsHeaderInfo {
    /// Total parameter count across all tensors.
    #[must_use]
    pub fn total_params(&self) -> u64 {
        self.tensors
            .iter()
            .map(TensorInfo::num_elements)
            .fold(0u64, u64::saturating_add)
    }

    /// Returns tensors matching a dtype string (e.g., `"F8_E4M3"`).
    #[must_use]
    pub fn tensors_with_dtype(&self, dtype: &str) -> Vec<&TensorInfo> {
        self.tensors
            .iter()
            // BORROW: explicit .as_str() instead of Deref coercion
            .filter(|t| t.dtype.as_str() == dtype)
            .collect()
    }

    /// Constructs a new [`SafetensorsHeaderInfo`] from its core fields.
    ///
    /// Since v0.10.3 the struct is `#[non_exhaustive]` — this constructor is
    /// the canonical way to build one from outside the `hf-fetch-model` lib
    /// crate (e.g. the `hf-fm` binary crate, downstream consumers like
    /// `candle-mi`). Inside the lib crate, struct-literal syntax stays
    /// available for the inspect entry points.
    ///
    /// `quant_info` is typically `None`; populated only by
    /// [`inspect_safetensors_local`] for cached, quantized safetensors files.
    #[must_use]
    pub fn new(
        tensors: Vec<TensorInfo>,
        metadata: Option<HashMap<String, String>>,
        header_size: u64,
        file_size: Option<u64>,
        quant_info: Option<QuantInfo>,
    ) -> Self {
        Self {
            tensors,
            metadata,
            header_size,
            file_size,
            quant_info,
        }
    }
}

/// The source from which a header was read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InspectSource {
    /// Read from local cache (no network).
    Cached,
    /// Fetched via HTTP Range requests.
    Remote,
    /// Read from the on-disk header cache that `inspect --cache-headers`
    /// and `quants --fits --cache-headers` fill.
    ///
    /// Within [`crate::header_cache::TRUST_WINDOW`] no request was made at
    /// all; past it, a network probe read the file's current etag and
    /// confirmed the entry. Either way the full header parse, the dozens of
    /// range requests `Remote` implies, was skipped.
    CachedHeader {
        /// How long ago the entry was written or last confirmed.
        age: std::time::Duration,
    },
}

/// Parsed `model.safetensors.index.json` for a sharded model.
#[derive(Debug, Clone, Serialize)]
pub struct ShardedIndex {
    /// Mapping from tensor name to shard filename.
    pub weight_map: HashMap<String, String>,
    /// Ordered list of unique shard filenames.
    pub shards: Vec<String>,
    /// Raw metadata from the index, if present.
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// `PEFT` adapter configuration parsed from `adapter_config.json`.
///
/// Contains the key fields that identify an adapter: the `PEFT` type,
/// base model, `LoRA` rank and scaling parameters, and target modules.
/// All fields are optional because adapter configs vary across `PEFT` methods.
#[derive(Debug, Clone, Serialize)]
pub struct AdapterConfig {
    /// `PEFT` method type (e.g., `"LORA"`, `"ADALORA"`, `"IA3"`).
    pub peft_type: Option<String>,
    /// The base model this adapter was trained on.
    pub base_model_name_or_path: Option<String>,
    /// `LoRA` rank (the `r` parameter). Only meaningful for `LoRA`-family methods.
    pub r: Option<u32>,
    /// `LoRA` alpha scaling factor. Only meaningful for `LoRA`-family methods.
    pub lora_alpha: Option<f64>,
    /// List of model modules targeted by the adapter.
    pub target_modules: Vec<String>,
    /// Task type the adapter was trained for (e.g., `"CAUSAL_LM"`).
    pub task_type: Option<String>,
}

/// Attention- and cache-relevant fields parsed from a model's `config.json`.
///
/// Every field is [`Option`] because configs vary across architecture
/// families; the KV-cache estimator decides which combinations are
/// computable and which fall back to an "unavailable" verdict. Legacy
/// `n_layer` / `n_head` / `n_head_kv` spellings are absorbed by serde aliases
/// on the private deserialization struct. Drives KV-cache budgeting for
/// `inspect --check-gpu --context`.
#[derive(Debug, Clone, Default, Serialize)]
#[non_exhaustive]
pub struct ModelConfig {
    /// Architecture tag (e.g. `"llama"`, `"qwen3"`, `"gemma2"`, `"deepseek_v2"`).
    pub model_type: Option<String>,
    /// Number of transformer layers (`num_hidden_layers` / `n_layer`).
    pub num_hidden_layers: Option<u32>,
    /// Number of query attention heads (`num_attention_heads` / `n_head`).
    pub num_attention_heads: Option<u32>,
    /// Number of key/value heads for `GQA` (`num_key_value_heads` /
    /// `num_kv_heads` / `n_head_kv`). Absent ⇒ `MHA` (equals
    /// `num_attention_heads`).
    pub num_key_value_heads: Option<u32>,
    /// Explicit per-head dimension when stated (Gemma = 256, Qwen3 = 128).
    /// Absent ⇒ derived as `hidden_size / num_attention_heads`.
    pub head_dim: Option<u32>,
    /// Model hidden size, used to derive `head_dim` when it is not explicit.
    pub hidden_size: Option<u32>,
    /// Activation dtype spelling (`"bfloat16"`, `"float16"`, …); sizes the
    /// KV-cache element via [`torch_dtype_bytes`].
    pub torch_dtype: Option<String>,
    /// Sliding-window span in tokens when the model uses windowed attention.
    /// `null` / absent ⇒ full attention.
    pub sliding_window: Option<u32>,
    /// Global-attention period for mixed local/global layouts (Gemma-3:
    /// every `N`-th layer is a full-attention layer).
    pub sliding_window_pattern: Option<u32>,
    /// Explicit on/off switch for sliding-window attention — Qwen2/3 ship a
    /// `sliding_window` value but disable it with `false`.
    pub use_sliding_window: Option<bool>,
    /// `MLA` latent-KV rank (`DeepSeek`). Presence marks multi-head latent
    /// attention, where the naive KV formula does not apply.
    pub kv_lora_rank: Option<u32>,
    /// `MLA` decoupled-`RoPE` key dimension (`DeepSeek`); part of the latent-KV
    /// size used by the documented `MLA` estimate.
    pub qk_rope_head_dim: Option<u32>,
    /// Per-layer kind tags for hybrid models (`"attention"` / `"mamba"` /
    /// `"linear_attention"` / …). Primary hybrid-layout signal (Granite-4).
    pub layer_types: Option<Vec<String>>,
    /// Nemotron-H layer-layout string (`"M-M-M-M*-…"`: `*` = attention,
    /// `M` = Mamba, `-` = FFN-only). Alternative hybrid-layout signal.
    pub hybrid_override_pattern: Option<String>,
    /// Explicit indices of the attention layers (Bamba). Alternative
    /// hybrid-layout signal; the remaining layers are recurrent.
    pub attn_layer_indices: Option<Vec<u32>>,
    /// Period of full-attention layers — every `N`-th layer is attention, the
    /// rest recurrent (Qwen3-Next). Alternative hybrid-layout signal.
    pub full_attention_interval: Option<u32>,
    /// Offset before the first attention layer (Jamba): attention layers occur
    /// at index `offset`, then every `period`-th layer after. Alternative
    /// hybrid-layout signal, used together with `attn_layer_period`.
    pub attn_layer_offset: Option<u32>,
    /// Period between attention layers when combined with `attn_layer_offset`
    /// (Jamba: offset 4, period 8 ⇒ attention at layers 4, 12, 20, ...).
    pub attn_layer_period: Option<u32>,
    /// Mamba2 SSM head count (`mamba_n_heads` / `mamba_num_heads`).
    pub mamba_n_heads: Option<u32>,
    /// Mamba2 SSM per-head dimension (`mamba_d_head` / `mamba_head_dim`).
    pub mamba_d_head: Option<u32>,
    /// Mamba2 SSM state size (`mamba_d_state` / `ssm_state_size`).
    pub mamba_d_state: Option<u32>,
    /// Mamba2 causal-convolution width (`mamba_d_conv` / `conv_kernel`).
    pub mamba_d_conv: Option<u32>,
    /// Mamba2 group count for the convolution (`mamba_n_groups` / `n_groups`).
    pub mamba_n_groups: Option<u32>,
}

// -----------------------------------------------------------------------
// Cache resolution
// -----------------------------------------------------------------------

/// Resolves a cached file path for a given repo, revision, and filename.
///
/// Returns `None` if the file is not in the local cache. Every `inspect_*`
/// entry point in this module checks this first, before ever touching the
/// network — `pub` so a caller composing its own dispatch around these
/// entry points (the `hf-fm` CLI's `--cache-headers` path, which needs to
/// probe a reader for the file's etag before it can consult its own header
/// cache) can preserve that same local-cache preference instead of losing
/// it.
///
/// A `filename` that could leave the snapshot directory
/// ([`cache_layout::is_safe_repo_filename`]) is never joined to it and reads
/// as not cached: `inspect <repo> <N>` and `--pick` take the name from the
/// repo listing.
///
/// This reads the default `HF` cache. The `_with_config` entry points look in
/// a [`FetchConfig::output_dir`] instead when one is set.
#[must_use]
pub fn resolve_cached_path(repo_id: &str, revision: &str, filename: &str) -> Option<PathBuf> {
    resolve_cached_path_at(None, repo_id, revision, filename)
}

/// [`resolve_cached_path`] against `cache_root`, or against the default `HF`
/// cache when `cache_root` is `None`.
fn resolve_cached_path_at(
    cache_root: Option<&Path>,
    repo_id: &str,
    revision: &str,
    filename: &str,
) -> Option<PathBuf> {
    if !cache_layout::is_safe_repo_filename(filename) {
        return None;
    }
    let cache_dir = match cache_root {
        // BORROW: explicit .to_path_buf() so both arms yield an owned root
        Some(root) => root.to_path_buf(),
        None => cache::hf_cache_dir().ok()?,
    };
    let repo_dir = cache_layout::repo_dir(&cache_dir, repo_id);
    let commit_hash = cache::read_ref(&repo_dir, revision)?;
    let cached_path = cache_layout::pointer_path(&repo_dir, &commit_hash, filename);
    if cached_path.exists() {
        Some(cached_path)
    } else {
        None
    }
}

/// How a cache-first entry point reaches the Hub, and where it looks first.
///
/// The original entry points take `token` and `revision` as loose arguments
/// and read the default `HF` cache ([`HubAccess::loose`]); their
/// `_with_config` twins take all three from a [`FetchConfig`]
/// ([`HubAccess::from_config`]). Both forms build one of these and share a
/// single body, so they cannot drift. No `Debug`: it carries the token.
#[derive(Clone, Copy)]
struct HubAccess<'a> {
    /// Explicit token; `None` lets each request resolve one implicitly.
    token: Option<&'a str>,
    /// Git revision (branch, tag, or commit SHA); `None` means `main`.
    revision: Option<&'a str>,
    /// Root of the cache-first lookup; `None` means the default `HF` cache.
    cache_root: Option<&'a Path>,
}

impl<'a> HubAccess<'a> {
    /// The original entry points' access: loose arguments, default cache.
    const fn loose(token: Option<&'a str>, revision: Option<&'a str>) -> Self {
        Self {
            token,
            revision,
            cache_root: None,
        }
    }

    /// A `_with_config` twin's access: `config`'s token, revision and output
    /// directory. Every other setting of `config` is ignored.
    fn from_config(config: &'a FetchConfig) -> Self {
        Self {
            token: config.token(),
            revision: config.revision(),
            cache_root: config.output_dir(),
        }
    }

    /// The revision requests target: the configured one, or `main`.
    fn rev(&self) -> &'a str {
        self.revision.unwrap_or("main")
    }

    /// Where `filename` of `repo_id` sits in the cache this access reads, if
    /// it is cached there.
    fn cached_path(&self, repo_id: &str, filename: &str) -> Option<PathBuf> {
        resolve_cached_path_at(self.cache_root, repo_id, self.rev(), filename)
    }
}

// -----------------------------------------------------------------------
// Local file reading
// -----------------------------------------------------------------------

/// Inspects a single `.safetensors` file's header from a local file path.
///
/// Reads the first `8 + header_size` bytes from disk. Does not read tensor data.
///
/// # Blocking I/O
///
/// This function performs synchronous filesystem I/O. In async contexts, wrap
/// it in [`tokio::task::spawn_blocking`] so the calling task does not stall
/// the runtime — particularly important on network-mounted caches (NFS/CIFS)
/// where `read`/`stat` calls can take tens of milliseconds each.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the file cannot be read.
/// Returns [`FetchError::SafetensorsHeader`] if the header is malformed.
pub fn inspect_safetensors_local(path: &Path) -> Result<SafetensorsHeaderInfo, FetchError> {
    let file_size = std::fs::metadata(path)
        .map_err(|e| FetchError::Io {
            path: path.to_path_buf(),
            source: e,
        })?
        .len();

    // BORROW: explicit .to_string_lossy() for Path → str conversion
    let filename = path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().to_string(),
    );

    let file = std::fs::File::open(path).map_err(|e| FetchError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;

    // Cache-hit path delegates to anamnesis (v0.10.3 Phase B commit 4):
    // single source of truth for safetensors layout. The reader variant
    // reads the 8-byte u64 prefix + the JSON header bytes from the `Read`
    // impl itself, *without* requiring the data section to be present
    // (it bypasses `safetensors::SafeTensors::read_metadata` for exactly
    // this reason). Anamnesis caps the declared header length at 100 MiB
    // internally so the worst-case allocation is bounded. The remote path
    // (`inspect_safetensors`, v0.11.1) feeds the same anamnesis function
    // an `HttpRangeReader` instead of a `std::fs::File`, and shares this
    // function's `safetensors_header_to_info` mapping — the two paths
    // cannot drift.
    let header = anamnesis::parse_safetensors_header_from_reader(file).map_err(|e| {
        FetchError::SafetensorsHeader {
            // BORROW: explicit .clone() for the error variant's owned String field
            filename: filename.clone(),
            reason: format!("failed to parse safetensors header: {e}"),
        }
    })?;

    Ok(safetensors_header_to_info(header, Some(file_size)))
}

/// Maps an anamnesis `SafetensorsHeader` into the format-agnostic
/// [`SafetensorsHeaderInfo`] shape used by hf-fm's render path.
///
/// Shared by the cache-hit path ([`inspect_safetensors_local`]) and the
/// remote path ([`inspect_safetensors`]), so the two cannot drift. Derives
/// `quant_info` via `anamnesis::InspectInfo::from(&header)` (iterates the
/// already-parsed tensors and aggregates per-role byte sums — pure
/// computation, no I/O); unquantized models produce no `quant_info` so the
/// renderer suppresses the `Format:` / `Size:` lines (absence communicates
/// full precision). Preserves hf-fm's v0.10.2 sort order: anamnesis returns
/// tensors sorted alphabetically by name, but hf-fm's inspect table has
/// always been file-ordered (sorted by start offset) so users can spot
/// first/last tensors per shard at a glance.
fn safetensors_header_to_info(
    header: anamnesis::SafetensorsHeader,
    file_size: Option<u64>,
) -> SafetensorsHeaderInfo {
    // CAST: usize → u64, anamnesis caps header_size at 100 MiB so it always fits in u64
    #[allow(clippy::as_conversions)]
    let header_size = header.header_size as u64;

    let quant_info = if header.scheme == anamnesis::QuantScheme::Unquantized {
        None
    } else {
        let info = anamnesis::InspectInfo::from(&header);
        Some(QuantInfo {
            // BORROW: explicit .to_string() — anamnesis `QuantScheme` → owned `String`
            scheme: info.format.to_string(),
            stored_bytes: info.current_size,
            dequantized_bytes: info.dequantized_size,
        })
    };

    let mut tensors: Vec<TensorInfo> = header
        .tensors
        .into_iter()
        .map(|t| {
            // CAST: usize → u64, header data_offsets fit in u64 by definition (file size is u64)
            #[allow(clippy::as_conversions)]
            let start = t.data_offsets.0 as u64;
            // CAST: usize → u64, same rationale as above
            #[allow(clippy::as_conversions)]
            let end = t.data_offsets.1 as u64;
            TensorInfo {
                name: t.name,
                // BORROW: explicit .to_string() — anamnesis `Dtype` enum → owned `String`
                dtype: t.dtype.to_string(),
                shape: t.shape,
                data_offsets: (start, end),
            }
        })
        .collect();

    tensors.sort_by_key(|t| t.data_offsets.0);

    SafetensorsHeaderInfo {
        tensors,
        metadata: header.metadata,
        header_size,
        file_size,
        quant_info,
    }
}

// -----------------------------------------------------------------------
// Public API: single-file inspection
// -----------------------------------------------------------------------

/// Inspects a single `.safetensors` file's header (cache-first).
///
/// Checks the local `HF` cache first. If the file is cached, reads the
/// header from disk with zero network requests. Otherwise, opens an
/// [`HttpRangeReader`] over the file and runs
/// `anamnesis::parse_safetensors_header_from_reader` against it on a
/// blocking thread (v0.11.1 — previously a bespoke two-Range-request
/// fetcher that didn't go through anamnesis; see `safetensors_header_to_info`
/// for the mapping now shared with the cache-hit path). The safetensors
/// header is sequential at the very start of the file, so the reader's
/// 4 KiB read-ahead window typically satisfies both the 8-byte length
/// prefix and the `JSON` header in a single range fetch — a second fetch
/// only fires when the header exceeds that window. The reported
/// [`RangeStats`] additionally counts the reader's one-time access probe
/// (2 requests), so the `Source:` line typically shows 3–4 total for this
/// path. No tensor data is downloaded in either case.
///
/// The third tuple element reports the remote transfer statistics
/// ([`RangeStats`]: request count + bytes fetched); `None` on the cached
/// path. Mirrors [`inspect_npz`]'s shape so the `hf-fm` CLI renders both
/// formats' `Source:` line identically.
///
/// [`inspect_safetensors_with_config`] takes the token, the revision and the
/// cache directory from a [`FetchConfig`] instead.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the Range probe or a range request
/// fails — including gated repos, which surface as `returned status
/// 401/403` errors (the `hf-fm` CLI upgrades those into a gated-repo
/// diagnosis).
/// Returns [`FetchError::Io`] if the cached file cannot be read.
/// Returns [`FetchError::UnsafeName`] if the Hub sends a commit hash or
/// etag that is not a single plain name.
/// Returns [`FetchError::SafetensorsHeader`] if the header is malformed
/// (cached or remote).
pub async fn inspect_safetensors(
    repo_id: &str,
    filename: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    inspect_safetensors_with_access(repo_id, filename, &HubAccess::loose(token, revision)).await
}

/// [`inspect_safetensors`], with the token, the revision and the cache
/// directory taken from `config`.
///
/// The cache-first lookup reads [`FetchConfig::output_dir`] as the cache
/// root, so a file downloaded with the same `config` is found there (unset:
/// the default `HF` cache). A remote read sends [`FetchConfig::token`]
/// (unset: one is resolved implicitly) and targets [`FetchConfig::revision`]
/// (unset: `main`). The other settings of `config` have no meaning for a
/// header read and are ignored.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the Range probe or a range request
/// fails — including gated repos, which surface as `returned status
/// 401/403` errors.
/// Returns [`FetchError::Io`] if the cached file cannot be read.
/// Returns [`FetchError::UnsafeName`] if the Hub sends a commit hash or
/// etag that is not a single plain name.
/// Returns [`FetchError::SafetensorsHeader`] if the header is malformed
/// (cached or remote).
pub async fn inspect_safetensors_with_config(
    repo_id: &str,
    filename: &str,
    config: &FetchConfig,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    inspect_safetensors_with_access(repo_id, filename, &HubAccess::from_config(config)).await
}

/// Shared body of [`inspect_safetensors`] and
/// [`inspect_safetensors_with_config`].
async fn inspect_safetensors_with_access(
    repo_id: &str,
    filename: &str,
    access: &HubAccess<'_>,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    // Try local cache first.
    if let Some(cached_path) = access.cached_path(repo_id, filename) {
        let info = inspect_safetensors_local(&cached_path)?;
        return Ok((info, InspectSource::Cached, None));
    }

    // Fall back to HTTP Range requests: probe eagerly (typed errors here),
    // then hand the reader off to the shared from-reader parse path.
    let reader = HttpRangeReader::open(repo_id, access.revision, filename, access.token).await?;
    inspect_safetensors_from_reader(reader, filename).await
}

/// Parses a `.safetensors` header from an already-open [`HttpRangeReader`].
///
/// Split out of [`inspect_safetensors`] so a caller that already had a
/// reason to open a reader over the same file — `hf-fm`'s `--cache-headers`
/// path probes one for the file's current etag before consulting its header
/// cache — can reuse it here on a cache miss instead of paying for a second
/// [`HttpRangeReader::open`] round trip.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if a range request fails — including gated
/// repos, which surface as `returned status 401/403` errors (the `hf-fm`
/// CLI upgrades those into a gated-repo diagnosis).
/// Returns [`FetchError::Http`] if the parse task itself fails to join (an
/// internal panic; not expected in normal operation).
/// Returns [`FetchError::SafetensorsHeader`] if the header is malformed.
pub async fn inspect_safetensors_from_reader(
    reader: HttpRangeReader,
    filename: &str,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    let file_size = reader.total_size();

    let (parse_result, stats, transport_error) = tokio::task::spawn_blocking(move || {
        let mut reader = reader;
        // `&mut` keeps ownership here so stats and the typed transport
        // error survive the parse (std's blanket `Read` for `&mut R`).
        let result = anamnesis::parse_safetensors_header_from_reader(&mut reader);
        (result, reader.stats(), reader.take_last_error())
    })
    .await
    .map_err(|e| FetchError::Http(format!("failed to join safetensors inspect task: {e}")))?;

    match parse_result {
        Ok(header) => Ok((
            safetensors_header_to_info(header, Some(file_size)),
            InspectSource::Remote,
            Some(stats),
        )),
        // Prefer the typed transport error over anamnesis's io-flattened
        // wrapper — an HTTP 401/403 must stay recognisable for the CLI's
        // gated-repo diagnosis.
        Err(e) => Err(
            transport_error.unwrap_or_else(|| FetchError::SafetensorsHeader {
                // BORROW: explicit .to_owned() for owned String in the error variant
                filename: filename.to_owned(),
                reason: format!("failed to parse safetensors header: {e}"),
            }),
        ),
    }
}

/// Inspects a single `.safetensors` file from cache only.
///
/// Resolves the file in the local HF cache using the given `repo_id`,
/// `revision`, and `filename`. Returns an error if the file is not cached.
///
/// # Blocking I/O
///
/// Performs synchronous filesystem I/O; wrap in [`tokio::task::spawn_blocking`]
/// from async contexts. See [`inspect_safetensors_local`] for rationale.
///
/// # Errors
///
/// Returns [`FetchError::SafetensorsHeader`] if the file is not in the cache.
/// Returns [`FetchError::Io`] if the cached file cannot be read.
/// Returns [`FetchError::SafetensorsHeader`] if the header is malformed.
pub fn inspect_safetensors_cached(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
) -> Result<SafetensorsHeaderInfo, FetchError> {
    let rev = revision.unwrap_or("main");

    let cached_path = resolve_cached_path(repo_id, rev, filename).ok_or_else(|| {
        FetchError::SafetensorsHeader {
            filename: filename.to_owned(),
            reason: format!("file not found in local cache for {repo_id} ({rev})"),
        }
    })?;

    inspect_safetensors_local(&cached_path)
}

/// Inspects a `.gguf` file's metadata from the local `HuggingFace` cache.
///
/// Delegates to [`anamnesis::parse_gguf`] for the on-disk parse, then maps the
/// result into the format-agnostic [`SafetensorsHeaderInfo`] shape used by
/// hf-fm's existing render path. Tensor names, GGUF-native shape order, and
/// dtype name strings carry over directly; per-tensor `data_offsets` are
/// `(data_offset, data_offset + byte_len)` (with `byte_len = 0` for tensors
/// whose dtype has no known byte size in anamnesis yet).
///
/// **Naming note:** the returned type is still called [`SafetensorsHeaderInfo`]
/// in v0.10.x because renaming a public type is a breaking change; the
/// uniform-dispatch rename to a format-agnostic name is scheduled for v0.10.3
/// when the dispatcher extends across `.npz` / `.pth` (see the cache-management
/// roadmap). For now, treat the type name as "header / file-level inspect
/// info" regardless of format.
///
/// **Metadata surfacing:** the GGUF metadata table can contain very large
/// arrays (e.g. tokenizer.ggml.tokens with 50K+ entries). To keep `Metadata:`
/// rendering useful, this function surfaces *scalar* metadata values only —
/// strings, booleans, integers, floats — and skips arrays. The GGUF format
/// version is surfaced under the synthetic key `gguf.version`, the effective
/// alignment under `gguf.alignment`. The original `general.architecture`,
/// `general.name`, and friends pass through unchanged.
///
/// **Blocking I/O:** anamnesis's GGUF parser mmaps the file; this function is
/// synchronous and should be wrapped in [`tokio::task::spawn_blocking`] from
/// async contexts.
///
/// # Errors
///
/// Returns [`FetchError::SafetensorsHeader`] if the file is not in the cache.
/// Returns [`FetchError::SafetensorsHeader`] if anamnesis rejects the GGUF
/// file (malformed header, truncated tensor table, etc.).
pub fn inspect_gguf_cached(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
) -> Result<SafetensorsHeaderInfo, FetchError> {
    let rev = revision.unwrap_or("main");

    let cached_path = resolve_cached_path(repo_id, rev, filename).ok_or_else(|| {
        FetchError::SafetensorsHeader {
            // BORROW: explicit .to_owned() for owned String in the error variant
            filename: filename.to_owned(),
            reason: format!("file not found in local cache for {repo_id} ({rev})"),
        }
    })?;

    parse_gguf_file(&cached_path, filename)
}

/// Parses the cached `.gguf` file at `path` into the format-agnostic header
/// shape; `filename` names it in errors.
///
/// Shared by [`inspect_gguf_cached`] and the cache-hit path of
/// [`inspect_gguf`] / [`inspect_gguf_with_config`], which has already
/// resolved the path in whichever cache it reads.
fn parse_gguf_file(path: &Path, filename: &str) -> Result<SafetensorsHeaderInfo, FetchError> {
    let file_size = std::fs::metadata(path).ok().map(|m| m.len());

    let parsed = anamnesis::parse_gguf(path).map_err(|e| FetchError::SafetensorsHeader {
        // BORROW: explicit .to_owned() for owned String in the error variant
        filename: filename.to_owned(),
        reason: format!("failed to parse GGUF: {e}"),
    })?;

    Ok(gguf_front_matter_to_header_info(
        parsed.tensor_info(),
        parsed.metadata(),
        parsed.version(),
        parsed.alignment(),
        file_size,
    ))
}

/// Maps parsed `GGUF` front matter into the format-agnostic
/// [`SafetensorsHeaderInfo`] shape used by hf-fm's render path.
///
/// Shared by the cached ([`inspect_gguf_cached`], via `anamnesis::parse_gguf`
/// → `ParsedGguf::tensor_info`/`::metadata`) and remote ([`inspect_gguf`],
/// via `anamnesis::parse_gguf_front_matter_from_reader` → `GgufFrontMatter`'s
/// same-named fields) paths, so the two cannot drift.
///
/// **Metadata surfacing:** the GGUF metadata table can contain very large
/// arrays (e.g. `tokenizer.ggml.tokens` with 50K+ entries). To keep
/// `Metadata:` rendering useful, this function surfaces *scalar* metadata
/// values only — strings, booleans, integers, floats — and skips arrays.
/// The GGUF format version is surfaced under the synthetic key
/// `gguf.version`, the effective alignment under `gguf.alignment`. The
/// original `general.architecture`, `general.name`, and friends pass
/// through unchanged.
fn gguf_front_matter_to_header_info(
    tensor_infos: &[anamnesis::GgufTensorInfo],
    metadata: &HashMap<String, anamnesis::GgufMetadataValue>,
    version: u32,
    alignment: u32,
    file_size: Option<u64>,
) -> SafetensorsHeaderInfo {
    let tensors: Vec<TensorInfo> = tensor_infos
        .iter()
        .map(|info| {
            let start = info.data_offset;
            let end = info.byte_len.map_or(start, |b| start.saturating_add(b));
            TensorInfo {
                // BORROW: explicit .clone() / .to_string() to materialise owned
                // String + Vec<usize> from anamnesis's borrowed metadata
                name: info.name.clone(),
                dtype: info.dtype.to_string(),
                shape: info.shape.clone(),
                data_offsets: (start, end),
            }
        })
        .collect();

    // Stringify scalar metadata only; skip arrays (potentially huge — e.g.
    // tokenizer.ggml.tokens). Add synthetic keys for the format version and
    // alignment so they appear in the `Metadata:` block.
    let mut metadata_out: HashMap<String, String> = metadata
        .iter()
        // BORROW: explicit .clone() to materialise an owned String key from
        // the borrowed HashMap iteration
        .filter_map(|(k, v)| stringify_gguf_scalar(v).map(|s| (k.clone(), s)))
        .collect();
    // BORROW: explicit .to_owned() for owned String keys
    metadata_out.insert("gguf.version".to_owned(), version.to_string());
    metadata_out.insert("gguf.alignment".to_owned(), alignment.to_string());

    SafetensorsHeaderInfo {
        tensors,
        metadata: Some(metadata_out),
        // GGUF has no discrete "header size" like safetensors's
        // u64-length-prefix + JSON. The value is left at 0 here; consumers
        // that care can derive an approximation from `file_size` minus the
        // tensor byte sum. The `Metadata:` block's `gguf.version` /
        // `gguf.alignment` keys surface the equivalent format-level info.
        header_size: 0,
        file_size,
        // GGUF quant info (Q4_K_M etc.) is implicit in per-tensor dtypes;
        // the v0.10.3 Phase C `Format:` / `Size:` lines are safetensors-only.
        quant_info: None,
    }
}

/// Inspects a single `.gguf` file's metadata (cache-first, remote fallback).
///
/// Checks the local `HF` cache first. If the file is cached, parses it from
/// disk with zero network requests, as [`inspect_gguf_cached`] does.
/// Otherwise, opens an [`HttpRangeReader`] over the file and runs
/// `anamnesis::parse_gguf_front_matter_from_reader` against it on a blocking
/// thread (v0.11.2, on anamnesis 0.7.1's reader-generic `GgufFrontMatter` —
/// the full-detail counterpart to the summary-only `inspect_gguf_from_reader`
/// anamnesis shipped in 0.4.5). `GGUF` is front-loaded: the parser reads the
/// metadata KV table and tensor-info table in a single linear scan and never
/// touches the tensor-data segment, so a multi-GiB quantised file inspects in
/// a few MiB: tens to a couple of hundred range requests, most of them for
/// the tokenizer's vocabulary arrays (170 requests and 10.5 MiB measured on
/// `unsloth/Qwen3.5-4B-GGUF`). Mirrors [`inspect_npz`] / [`inspect_safetensors`]'s
/// shape so the `hf-fm` CLI renders all three formats' `Source:` line
/// identically.
///
/// The third tuple element reports the remote transfer statistics
/// ([`RangeStats`]: request count + bytes fetched); `None` on the cached path.
///
/// [`inspect_gguf_with_config`] takes the token, the revision and the cache
/// directory from a [`FetchConfig`] instead.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the Range probe or a range request
/// fails — including gated repos, which surface as `returned status
/// 401/403` errors (the `hf-fm` CLI upgrades those into a gated-repo
/// diagnosis).
/// Returns [`FetchError::UnsafeName`] if the Hub sends a commit hash or
/// etag that is not a single plain name.
/// Returns [`FetchError::SafetensorsHeader`] if the `GGUF` file is malformed
/// (cached or remote).
pub async fn inspect_gguf(
    repo_id: &str,
    filename: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    inspect_gguf_with_access(repo_id, filename, &HubAccess::loose(token, revision)).await
}

/// [`inspect_gguf`], with the token, the revision and the cache directory
/// taken from `config`.
///
/// The cache-first lookup reads [`FetchConfig::output_dir`] as the cache
/// root, so a file downloaded with the same `config` is found there (unset:
/// the default `HF` cache). A remote read sends [`FetchConfig::token`]
/// (unset: one is resolved implicitly) and targets [`FetchConfig::revision`]
/// (unset: `main`). The other settings of `config` have no meaning for a
/// header read and are ignored.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the Range probe or a range request
/// fails — including gated repos, which surface as `returned status
/// 401/403` errors.
/// Returns [`FetchError::UnsafeName`] if the Hub sends a commit hash or
/// etag that is not a single plain name.
/// Returns [`FetchError::SafetensorsHeader`] if the `GGUF` file is malformed
/// (cached or remote).
pub async fn inspect_gguf_with_config(
    repo_id: &str,
    filename: &str,
    config: &FetchConfig,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    inspect_gguf_with_access(repo_id, filename, &HubAccess::from_config(config)).await
}

/// Shared body of [`inspect_gguf`] and [`inspect_gguf_with_config`].
async fn inspect_gguf_with_access(
    repo_id: &str,
    filename: &str,
    access: &HubAccess<'_>,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    // Try local cache first (mirrors `inspect_npz` / `inspect_safetensors`).
    if let Some(cached_path) = access.cached_path(repo_id, filename) {
        let info = parse_gguf_file(&cached_path, filename)?;
        return Ok((info, InspectSource::Cached, None));
    }

    // Fall back to HTTP Range requests: probe eagerly (typed errors here),
    // then hand the reader off to the shared from-reader parse path.
    let reader = HttpRangeReader::open(repo_id, access.revision, filename, access.token).await?;
    inspect_gguf_from_reader(reader, filename).await
}

/// Parses a `.gguf` file's metadata from an already-open [`HttpRangeReader`].
///
/// Split out of [`inspect_gguf`] so a caller that already had a reason to
/// open a reader over the same file — `hf-fm`'s `--cache-headers` path
/// probes one for the file's current etag before consulting its header
/// cache — can reuse it here on a cache miss instead of paying for a second
/// [`HttpRangeReader::open`] round trip.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if a range request fails — including gated
/// repos, which surface as `returned status 401/403` errors (the `hf-fm`
/// CLI upgrades those into a gated-repo diagnosis).
/// Returns [`FetchError::Http`] if the parse task itself fails to join (an
/// internal panic; not expected in normal operation).
/// Returns [`FetchError::SafetensorsHeader`] if the `GGUF` file is malformed.
pub async fn inspect_gguf_from_reader(
    reader: HttpRangeReader,
    filename: &str,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    let file_size = reader.total_size();

    let (parse_result, stats, transport_error) = tokio::task::spawn_blocking(move || {
        let mut reader = reader;
        // `&mut` keeps ownership here so stats and the typed transport
        // error survive the parse (std's blanket `Read`/`Seek` for `&mut R`).
        let result = anamnesis::parse_gguf_front_matter_from_reader(&mut reader);
        (result, reader.stats(), reader.take_last_error())
    })
    .await
    .map_err(|e| FetchError::Http(format!("failed to join GGUF inspect task: {e}")))?;

    match parse_result {
        Ok(front) => Ok((
            gguf_front_matter_to_header_info(
                &front.tensor_infos,
                &front.metadata,
                front.version,
                front.alignment,
                Some(file_size),
            ),
            InspectSource::Remote,
            Some(stats),
        )),
        // Prefer the typed transport error over anamnesis's io-flattened
        // wrapper — an HTTP 401/403 must stay recognisable for the CLI's
        // gated-repo diagnosis.
        Err(e) => Err(
            transport_error.unwrap_or_else(|| FetchError::SafetensorsHeader {
                // BORROW: explicit .to_owned() for owned String in the error variant
                filename: filename.to_owned(),
                reason: format!("failed to parse GGUF: {e}"),
            }),
        ),
    }
}

/// Inspects a `.npz` file's metadata from the local `HuggingFace` cache.
///
/// Delegates to [`anamnesis::inspect_npz`] for the on-disk parse (which
/// reads only the ZIP central directory + per-entry NPY headers — no
/// tensor data), then maps the result into the format-agnostic
/// [`SafetensorsHeaderInfo`] shape used by hf-fm's existing render path.
///
/// **Synthesised offsets.** Anamnesis exposes per-tensor `byte_len` but
/// not on-disk byte offsets (NPZ tensors live inside a ZIP archive;
/// offsets are not part of the inspect surface). hf-fm's
/// `TensorInfo::data_offsets` is synthesised as cumulative `(start, end)`
/// pairs — `start = sum of previous byte_lens` — so `byte_len()`
/// (= `end - start`) renders the actual storage size. The synthetic
/// offsets are NOT on-disk truth — and the remote path ([`inspect_npz`],
/// v0.11.0) synthesises them identically, since anamnesis's inspect
/// surface does not expose archive offsets for either path.
///
/// **Metadata.** `metadata: None` — NPZ has no metadata block analogous
/// to safetensors's `__metadata__` or GGUF's KV table.
///
/// **Header size.** Always `0` — NPZ has no discrete header analogous
/// to safetensors's `u64`-length-prefix + JSON. Mirrors the GGUF convention.
///
/// **Blocking I/O:** anamnesis's NPZ parser opens the file with
/// `std::fs::File`; this function is synchronous and should be wrapped
/// in [`tokio::task::spawn_blocking`] from async contexts.
///
/// # Errors
///
/// Returns [`FetchError::SafetensorsHeader`] if the file is not in the cache.
/// Returns [`FetchError::SafetensorsHeader`] if anamnesis rejects the NPZ
/// file (malformed ZIP central directory, unsupported NPY dtype, etc.).
pub fn inspect_npz_cached(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
) -> Result<SafetensorsHeaderInfo, FetchError> {
    let rev = revision.unwrap_or("main");

    let cached_path = resolve_cached_path(repo_id, rev, filename).ok_or_else(|| {
        FetchError::SafetensorsHeader {
            // BORROW: explicit .to_owned() for owned String in the error variant
            filename: filename.to_owned(),
            reason: format!("file not found in local cache for {repo_id} ({rev})"),
        }
    })?;

    parse_npz_file(&cached_path, filename)
}

/// Parses the cached `.npz` archive at `path` into the format-agnostic
/// header shape; `filename` names it in errors.
///
/// Shared by [`inspect_npz_cached`] and the cache-hit path of
/// [`inspect_npz`] / [`inspect_npz_with_config`], which has already resolved
/// the path in whichever cache it reads.
fn parse_npz_file(path: &Path, filename: &str) -> Result<SafetensorsHeaderInfo, FetchError> {
    let file_size = std::fs::metadata(path).ok().map(|m| m.len());

    let parsed = anamnesis::inspect_npz(path).map_err(|e| FetchError::SafetensorsHeader {
        // BORROW: explicit .to_owned() for owned String in the error variant
        filename: filename.to_owned(),
        reason: format!("failed to parse NPZ: {e}"),
    })?;

    Ok(npz_info_to_header_info(parsed, file_size))
}

/// Maps an anamnesis `NPZ` inspect result into the format-agnostic
/// [`SafetensorsHeaderInfo`] shape used by hf-fm's render path.
///
/// Shared by the cached ([`inspect_npz_cached`]) and remote
/// ([`inspect_npz`]) paths, so the two cannot drift. See
/// [`inspect_npz_cached`] for the synthesised-offsets, metadata, and
/// header-size conventions this mapping implements.
fn npz_info_to_header_info(
    parsed: anamnesis::NpzInspectInfo,
    file_size: Option<u64>,
) -> SafetensorsHeaderInfo {
    let mut tensors: Vec<TensorInfo> = Vec::with_capacity(parsed.tensors.len());
    let mut cursor: u64 = 0;
    for t in parsed.tensors {
        // CAST: usize → u64, byte_len fits in u64 by definition (in-memory size).
        #[allow(clippy::as_conversions)]
        let len = t.byte_len as u64;
        let start = cursor;
        let end = cursor.saturating_add(len);
        cursor = end;
        tensors.push(TensorInfo {
            name: t.name,
            // BORROW: explicit .to_string() — anamnesis `NpzDtype` enum → owned `String`
            dtype: t.dtype.to_string(),
            shape: t.shape,
            data_offsets: (start, end),
        });
    }

    SafetensorsHeaderInfo {
        tensors,
        metadata: None,
        header_size: 0,
        file_size,
        // NPZ has no quant-method metadata; quant_info stays None.
        quant_info: None,
    }
}

/// Inspects a single `.npz` file's metadata (cache-first, remote fallback).
///
/// Checks the local `HF` cache first. If the file is cached, reads the `ZIP`
/// central directory + per-entry `NPY` headers from disk with zero network
/// requests. Otherwise, opens an [`HttpRangeReader`] over the file and runs
/// `anamnesis::inspect_npz_from_reader` against it on a blocking thread —
/// a handful of HTTP Range requests fetch the archive directory and array
/// headers; no tensor data is downloaded in either case.
///
/// The third tuple element reports the remote transfer statistics
/// ([`RangeStats`]: request count + bytes fetched); `None` on the cached
/// path. The `hf-fm` CLI renders it as provenance — e.g.
/// `remote (6 range requests, 136.0 KiB fetched)`, the live-measured cost
/// against a 72 MiB `GemmaScope` `params.npz`.
///
/// [`inspect_npz_with_config`] takes the token, the revision and the cache
/// directory from a [`FetchConfig`] instead.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the Range probe or a range request
/// fails — including gated repos, which surface as `returned status
/// 401/403` errors (the `hf-fm` CLI upgrades those into a gated-repo
/// diagnosis).
/// Returns [`FetchError::UnsafeName`] if the Hub sends a commit hash or
/// etag that is not a single plain name.
/// Returns [`FetchError::SafetensorsHeader`] if the `NPZ` archive is
/// malformed (cached or remote).
// `range_reader` is the name a caller looking for this function searched
// for (finding 1 of the candle-mi NPZ adoption report).
#[doc(alias = "range_reader")]
pub async fn inspect_npz(
    repo_id: &str,
    filename: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    inspect_npz_with_access(repo_id, filename, &HubAccess::loose(token, revision)).await
}

/// [`inspect_npz`], with the token, the revision and the cache directory
/// taken from `config`.
///
/// The cache-first lookup reads [`FetchConfig::output_dir`] as the cache
/// root, so a file downloaded with the same `config` is found there (unset:
/// the default `HF` cache). A remote read sends [`FetchConfig::token`]
/// (unset: one is resolved implicitly) and targets [`FetchConfig::revision`]
/// (unset: `main`). The other settings of `config` have no meaning for a
/// header read and are ignored.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the Range probe or a range request
/// fails — including gated repos, which surface as `returned status
/// 401/403` errors.
/// Returns [`FetchError::UnsafeName`] if the Hub sends a commit hash or
/// etag that is not a single plain name.
/// Returns [`FetchError::SafetensorsHeader`] if the `NPZ` archive is
/// malformed (cached or remote).
pub async fn inspect_npz_with_config(
    repo_id: &str,
    filename: &str,
    config: &FetchConfig,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    inspect_npz_with_access(repo_id, filename, &HubAccess::from_config(config)).await
}

/// Shared body of [`inspect_npz`] and [`inspect_npz_with_config`].
async fn inspect_npz_with_access(
    repo_id: &str,
    filename: &str,
    access: &HubAccess<'_>,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    // Try local cache first (mirrors `inspect_safetensors`).
    if let Some(cached_path) = access.cached_path(repo_id, filename) {
        let info = parse_npz_file(&cached_path, filename)?;
        return Ok((info, InspectSource::Cached, None));
    }

    // Fall back to HTTP Range requests: probe eagerly (typed errors here),
    // then hand the reader off to the shared from-reader parse path.
    let reader = HttpRangeReader::open(repo_id, access.revision, filename, access.token).await?;
    inspect_npz_from_reader(reader, filename).await
}

/// Parses a `.npz` archive's metadata from an already-open [`HttpRangeReader`].
///
/// Split out of [`inspect_npz`] so a caller that already had a reason to
/// open a reader over the same file — `hf-fm`'s `--cache-headers` path
/// probes one for the file's current etag before consulting its header
/// cache — can reuse it here on a cache miss instead of paying for a second
/// [`HttpRangeReader::open`] round trip.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if a range request fails — including gated
/// repos, which surface as `returned status 401/403` errors (the `hf-fm`
/// CLI upgrades those into a gated-repo diagnosis).
/// Returns [`FetchError::Http`] if the parse task itself fails to join (an
/// internal panic; not expected in normal operation).
/// Returns [`FetchError::SafetensorsHeader`] if the `NPZ` archive is malformed.
pub async fn inspect_npz_from_reader(
    reader: HttpRangeReader,
    filename: &str,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    let file_size = reader.total_size();

    let (parse_result, stats, transport_error) = tokio::task::spawn_blocking(move || {
        let mut reader = reader;
        // `&mut` keeps ownership here so stats and the typed transport
        // error survive the parse (std's blanket `Read`/`Seek` for `&mut R`).
        let result = anamnesis::inspect_npz_from_reader(&mut reader);
        (result, reader.stats(), reader.take_last_error())
    })
    .await
    .map_err(|e| FetchError::Http(format!("failed to join NPZ inspect task: {e}")))?;

    match parse_result {
        Ok(parsed) => Ok((
            npz_info_to_header_info(parsed, Some(file_size)),
            InspectSource::Remote,
            Some(stats),
        )),
        // Prefer the typed transport error over anamnesis's io-flattened
        // wrapper — an HTTP 401/403 must stay recognisable for the CLI's
        // gated-repo diagnosis.
        Err(e) => Err(
            transport_error.unwrap_or_else(|| FetchError::SafetensorsHeader {
                // BORROW: explicit .to_owned() for owned String in the error variant
                filename: filename.to_owned(),
                reason: format!("failed to parse NPZ: {e}"),
            }),
        ),
    }
}

/// Inspects a `.pth` file's metadata from the local `HuggingFace` cache.
///
/// Delegates to [`anamnesis::parse_pth`] for the on-disk parse, then uses
/// the metadata-only `ParsedPth::tensor_info()` view to enumerate
/// `(name, shape, dtype, byte_len)` per tensor — no further I/O beyond the
/// initial mmap. The earlier `.tensors()` method would materialise each
/// tensor's data via `Cow<'a, [u8]>`, which is unnecessary for inspect-only
/// use.
///
/// See `pth_tensor_info_to_header_info` for the synthesised-offsets,
/// metadata, and header-size conventions shared with the remote path
/// ([`inspect_pth`]).
///
/// **Blocking I/O:** anamnesis's PTH parser mmaps the file; this function
/// is synchronous and should be wrapped in [`tokio::task::spawn_blocking`]
/// from async contexts.
///
/// # Errors
///
/// Returns [`FetchError::SafetensorsHeader`] if the file is not in the cache.
/// Returns [`FetchError::SafetensorsHeader`] if anamnesis rejects the PTH
/// file (malformed pickle stream, legacy pre-1.6 raw-pickle format,
/// unsupported tensor dtype, etc.).
pub fn inspect_pth_cached(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
) -> Result<SafetensorsHeaderInfo, FetchError> {
    let rev = revision.unwrap_or("main");

    let cached_path = resolve_cached_path(repo_id, rev, filename).ok_or_else(|| {
        FetchError::SafetensorsHeader {
            // BORROW: explicit .to_owned() for owned String in the error variant
            filename: filename.to_owned(),
            reason: format!("file not found in local cache for {repo_id} ({rev})"),
        }
    })?;

    parse_pth_file(&cached_path, filename)
}

/// Parses the cached `.pth` archive at `path` into the format-agnostic
/// header shape; `filename` names it in errors.
///
/// Shared by [`inspect_pth_cached`] and the cache-hit path of
/// [`inspect_pth`] / [`inspect_pth_with_config`], which has already resolved
/// the path in whichever cache it reads.
fn parse_pth_file(path: &Path, filename: &str) -> Result<SafetensorsHeaderInfo, FetchError> {
    let file_size = std::fs::metadata(path).ok().map(|m| m.len());

    let parsed = anamnesis::parse_pth(path).map_err(|e| FetchError::SafetensorsHeader {
        // BORROW: explicit .to_owned() for owned String in the error variant
        filename: filename.to_owned(),
        reason: format!("failed to parse PTH: {e}"),
    })?;

    Ok(pth_tensor_info_to_header_info(
        parsed.tensor_info(),
        file_size,
    ))
}

/// Maps anamnesis `.pth` per-tensor info into the format-agnostic
/// [`SafetensorsHeaderInfo`] shape used by hf-fm's render path.
///
/// Shared by the cached ([`inspect_pth_cached`], via `anamnesis::parse_pth`
/// → `ParsedPth::tensor_info()`) and remote ([`inspect_pth`], via
/// `anamnesis::parse_pth_front_matter_from_reader` → `PthFrontMatter::tensors`)
/// paths, so the two cannot drift — mirrors [`npz_info_to_header_info`] /
/// [`gguf_front_matter_to_header_info`].
///
/// **Synthesised offsets.** As with NPZ, anamnesis exposes per-tensor
/// `byte_len` but not on-disk byte offsets (PTH tensors live inside a
/// ZIP archive; offsets are not part of the inspect surface). hf-fm's
/// `TensorInfo::data_offsets` is synthesised as cumulative `(start, end)`
/// pairs so `byte_len()` (= `end - start`) renders the actual storage size.
/// The remote path synthesises them identically.
///
/// **Metadata.** `metadata: None` — PTH has no metadata block analogous
/// to safetensors's `__metadata__` or GGUF's KV table. The format-level
/// `big_endian` flag (rare, near-always `false`) is not surfaced here;
/// can be added as a synthetic `pth.big_endian` key in a future patch if
/// real users request it.
///
/// **Header size.** Always `0` — PTH has no discrete header analogous
/// to safetensors's `u64`-length-prefix + JSON. Mirrors the GGUF / NPZ
/// convention.
fn pth_tensor_info_to_header_info(
    pth_tensors: Vec<anamnesis::PthTensorInfo>,
    file_size: Option<u64>,
) -> SafetensorsHeaderInfo {
    let mut tensors: Vec<TensorInfo> = Vec::with_capacity(pth_tensors.len());
    let mut cursor: u64 = 0;
    for t in pth_tensors {
        // CAST: usize → u64, byte_len fits in u64 by definition (in-memory size).
        #[allow(clippy::as_conversions)]
        let len = t.byte_len as u64;
        let start = cursor;
        let end = cursor.saturating_add(len);
        cursor = end;
        tensors.push(TensorInfo {
            name: t.name,
            // BORROW: explicit .to_string() — anamnesis `PthDtype` enum → owned `String`
            dtype: t.dtype.to_string(),
            shape: t.shape,
            data_offsets: (start, end),
        });
    }

    SafetensorsHeaderInfo {
        tensors,
        metadata: None,
        header_size: 0,
        file_size,
        // PTH has no quant-method metadata; quant_info stays None.
        quant_info: None,
    }
}

/// Inspects a single `.pth` file's metadata (cache-first, remote fallback).
///
/// Checks the local `HF` cache first. If the file is cached, parses it from
/// disk with zero network requests, as [`inspect_pth_cached`] does.
/// Otherwise, opens an [`HttpRangeReader`] over the file and runs
/// `anamnesis::parse_pth_front_matter_from_reader` against it on a blocking
/// thread (v0.11.4, on anamnesis 0.7.5's reader-generic `PthFrontMatter` —
/// the full-detail counterpart to the summary-only `inspect_pth_from_reader`
/// anamnesis shipped earlier). `.pth`'s metadata lives in a `data.pkl`
/// pickle stream located via the ZIP central directory; the parser reads
/// only that entry and never touches the tensor-data files (`data/0`,
/// `data/1`, ...), so a multi-GiB checkpoint inspects in a handful of range
/// requests. Mirrors [`inspect_npz`] / [`inspect_gguf`]'s shape so the
/// `hf-fm` CLI renders all four formats' `Source:` line identically.
///
/// The third tuple element reports the remote transfer statistics
/// ([`RangeStats`]: request count + bytes fetched); `None` on the cached
/// path.
///
/// [`inspect_pth_with_config`] takes the token, the revision and the cache
/// directory from a [`FetchConfig`] instead.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the Range probe or a range request
/// fails — including gated repos, which surface as `returned status
/// 401/403` errors (the `hf-fm` CLI upgrades those into a gated-repo
/// diagnosis).
/// Returns [`FetchError::UnsafeName`] if the Hub sends a commit hash or
/// etag that is not a single plain name.
/// Returns [`FetchError::SafetensorsHeader`] if the `PTH` file is malformed
/// (cached or remote).
pub async fn inspect_pth(
    repo_id: &str,
    filename: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    inspect_pth_with_access(repo_id, filename, &HubAccess::loose(token, revision)).await
}

/// [`inspect_pth`], with the token, the revision and the cache directory
/// taken from `config`.
///
/// The cache-first lookup reads [`FetchConfig::output_dir`] as the cache
/// root, so a file downloaded with the same `config` is found there (unset:
/// the default `HF` cache). A remote read sends [`FetchConfig::token`]
/// (unset: one is resolved implicitly) and targets [`FetchConfig::revision`]
/// (unset: `main`). The other settings of `config` have no meaning for a
/// header read and are ignored.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the Range probe or a range request
/// fails — including gated repos, which surface as `returned status
/// 401/403` errors.
/// Returns [`FetchError::UnsafeName`] if the Hub sends a commit hash or
/// etag that is not a single plain name.
/// Returns [`FetchError::SafetensorsHeader`] if the `PTH` file is malformed
/// (cached or remote).
pub async fn inspect_pth_with_config(
    repo_id: &str,
    filename: &str,
    config: &FetchConfig,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    inspect_pth_with_access(repo_id, filename, &HubAccess::from_config(config)).await
}

/// Shared body of [`inspect_pth`] and [`inspect_pth_with_config`].
async fn inspect_pth_with_access(
    repo_id: &str,
    filename: &str,
    access: &HubAccess<'_>,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    // Try local cache first (mirrors `inspect_gguf` / `inspect_npz`).
    if let Some(cached_path) = access.cached_path(repo_id, filename) {
        let info = parse_pth_file(&cached_path, filename)?;
        return Ok((info, InspectSource::Cached, None));
    }

    // Fall back to HTTP Range requests: probe eagerly (typed errors here),
    // then hand the reader off to the shared from-reader parse path.
    let reader = HttpRangeReader::open(repo_id, access.revision, filename, access.token).await?;
    inspect_pth_from_reader(reader, filename).await
}

/// Parses a `.pth` archive's metadata from an already-open [`HttpRangeReader`].
///
/// Split out of [`inspect_pth`] so a caller that already had a reason to
/// open a reader over the same file — `hf-fm`'s `--cache-headers` path
/// probes one for the file's current etag before consulting its header
/// cache — can reuse it here on a cache miss instead of paying for a second
/// [`HttpRangeReader::open`] round trip.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if a range request fails — including gated
/// repos, which surface as `returned status 401/403` errors (the `hf-fm`
/// CLI upgrades those into a gated-repo diagnosis).
/// Returns [`FetchError::Http`] if the parse task itself fails to join (an
/// internal panic; not expected in normal operation).
/// Returns [`FetchError::SafetensorsHeader`] if the `PTH` file is malformed.
pub async fn inspect_pth_from_reader(
    reader: HttpRangeReader,
    filename: &str,
) -> Result<(SafetensorsHeaderInfo, InspectSource, Option<RangeStats>), FetchError> {
    let file_size = reader.total_size();

    let (parse_result, stats, transport_error) = tokio::task::spawn_blocking(move || {
        let mut reader = reader;
        // `&mut` keeps ownership here so stats and the typed transport
        // error survive the parse (std's blanket `Read`/`Seek` for `&mut R`).
        let result = anamnesis::parse_pth_front_matter_from_reader(&mut reader);
        (result, reader.stats(), reader.take_last_error())
    })
    .await
    .map_err(|e| FetchError::Http(format!("failed to join PTH inspect task: {e}")))?;

    match parse_result {
        Ok(front) => Ok((
            pth_tensor_info_to_header_info(front.tensors, Some(file_size)),
            InspectSource::Remote,
            Some(stats),
        )),
        // Prefer the typed transport error over anamnesis's io-flattened
        // wrapper — an HTTP 401/403 must stay recognisable for the CLI's
        // gated-repo diagnosis.
        Err(e) => Err(
            transport_error.unwrap_or_else(|| FetchError::SafetensorsHeader {
                // BORROW: explicit .to_owned() for owned String in the error variant
                filename: filename.to_owned(),
                reason: format!("failed to parse PTH: {e}"),
            }),
        ),
    }
}

/// Stringifies a scalar `GgufMetadataValue` from anamnesis.
///
/// Returns `None` for array variants (potentially huge — vocab tables, merges
/// lists) and for any future `#[non_exhaustive]` variants we don't yet
/// recognise. Surfaced through the `Metadata:` block in `inspect` output by
/// [`inspect_gguf_cached`].
//
// `GgufMetadataValue` is `#[non_exhaustive]`. The explicit `V::Array(_)` arm
// and the `_ =>` catch-all both return `None`, but they document different
// intents — "array variants are deliberately skipped" vs "future unknown
// variants fall through". Clippy's `match_same_arms` flags the bodies as
// identical; the duplication is intentional.
#[allow(clippy::match_same_arms)]
fn stringify_gguf_scalar(value: &anamnesis::parse::gguf::GgufMetadataValue) -> Option<String> {
    use anamnesis::parse::gguf::GgufMetadataValue as V;
    match value {
        V::String(s) => Some(s.clone()),
        V::Bool(b) => Some(b.to_string()),
        V::U8(n) => Some(n.to_string()),
        V::I8(n) => Some(n.to_string()),
        V::U16(n) => Some(n.to_string()),
        V::I16(n) => Some(n.to_string()),
        V::U32(n) => Some(n.to_string()),
        V::I32(n) => Some(n.to_string()),
        V::U64(n) => Some(n.to_string()),
        V::I64(n) => Some(n.to_string()),
        V::F32(n) => Some(format!("{n}")),
        V::F64(n) => Some(format!("{n}")),
        V::Array(_) => None,
        _ => None,
    }
}

// -----------------------------------------------------------------------
// Public API: multi-file inspection
// -----------------------------------------------------------------------

/// Inspects all `.safetensors` files in a repository (cache-first per file).
///
/// Fetches the file listing via `list_repo_files_with_metadata()`, then
/// inspects each `.safetensors` file's header the way [`inspect_safetensors`]
/// reads one: for each file, it checks the local cache first and only makes HTTP Range
/// requests on cache miss. Returns full per-shard headers in filename order.
///
/// For a lightweight summary of sharded models (tensor counts per shard
/// without fetching individual headers), use [`fetch_shard_index()`] instead.
///
/// [`inspect_repo_safetensors_with_config`] takes the token, the revision and
/// the cache directory from a [`FetchConfig`] instead.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the metadata or Range requests fail.
/// Returns [`FetchError::RepoNotFound`] if the repository does not exist.
/// Returns [`FetchError::Io`] if a cached shard cannot be read.
/// Returns [`FetchError::UnsafeName`] if the Hub sends a commit hash or
/// etag that is not a single plain name.
/// Returns [`FetchError::SafetensorsHeader`] if a shard's header is
/// malformed.
pub async fn inspect_repo_safetensors(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<Vec<(String, SafetensorsHeaderInfo, InspectSource)>, FetchError> {
    inspect_repo_safetensors_with_access(repo_id, &HubAccess::loose(token, revision)).await
}

/// [`inspect_repo_safetensors`], with the token, the revision and the cache
/// directory taken from `config`.
///
/// Each shard's cache-first lookup reads [`FetchConfig::output_dir`] as the
/// cache root, so shards downloaded with the same `config` are found there
/// (unset: the default `HF` cache). The listing and every remote read send
/// [`FetchConfig::token`] (unset: one is resolved implicitly) and target
/// [`FetchConfig::revision`] (unset: `main`). The other settings of `config`
/// are ignored; in particular, shard headers are read four at a time
/// whatever its concurrency.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the metadata or Range requests fail.
/// Returns [`FetchError::RepoNotFound`] if the repository does not exist.
/// Returns [`FetchError::Io`] if a cached shard cannot be read.
/// Returns [`FetchError::UnsafeName`] if the Hub sends a commit hash or
/// etag that is not a single plain name.
/// Returns [`FetchError::SafetensorsHeader`] if a shard's header is
/// malformed.
pub async fn inspect_repo_safetensors_with_config(
    repo_id: &str,
    config: &FetchConfig,
) -> Result<Vec<(String, SafetensorsHeaderInfo, InspectSource)>, FetchError> {
    inspect_repo_safetensors_with_access(repo_id, &HubAccess::from_config(config)).await
}

/// Shared body of [`inspect_repo_safetensors`] and
/// [`inspect_repo_safetensors_with_config`].
async fn inspect_repo_safetensors_with_access(
    repo_id: &str,
    access: &HubAccess<'_>,
) -> Result<Vec<(String, SafetensorsHeaderInfo, InspectSource)>, FetchError> {
    let client = crate::chunked::build_client(access.token)?;
    let files =
        crate::repo::list_repo_files_with_metadata(repo_id, access.token, access.revision, &client)
            .await?;

    let safetensors_files: Vec<String> = files
        .into_iter()
        .filter(|f| f.filename.ends_with(".safetensors"))
        .map(|f| f.filename)
        .collect();

    inspect_shards(repo_id, safetensors_files, access).await
}

/// Reads every listed shard's header, four at a time, each one cache-first
/// through `access` exactly as [`inspect_safetensors`] reads one file, and
/// returns them in filename order.
///
/// Split out of [`inspect_repo_safetensors_with_access`], whose listing
/// request needs the network, so this per-shard plumbing (the owned copies
/// each spawned task rebuilds its access from) can be tested offline.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if a Range request fails or a task fails to
/// run.
/// Returns [`FetchError::Io`] if a cached shard cannot be read.
/// Returns [`FetchError::UnsafeName`] if the Hub sends a commit hash or
/// etag that is not a single plain name.
/// Returns [`FetchError::SafetensorsHeader`] if a shard's header is
/// malformed.
async fn inspect_shards(
    repo_id: &str,
    filenames: Vec<String>,
    access: &HubAccess<'_>,
) -> Result<Vec<(String, SafetensorsHeaderInfo, InspectSource)>, FetchError> {
    // BORROW: explicit .to_owned()/.to_path_buf() — every task must own what
    // it reads, so each one rebuilds its access from these copies
    let repo = repo_id.to_owned();
    let tok = access.token.map(str::to_owned);
    let rev = access.revision.map(str::to_owned);
    let root = access.cache_root.map(Path::to_path_buf);

    let mut results = fan_out_fail_fast(filenames, 4, move |filename: String| {
        // BORROW: explicit .clone() of the owned copies into this task
        let (repo, tok, rev, root) = (repo.clone(), tok.clone(), rev.clone(), root.clone());
        async move {
            // BORROW: explicit .as_deref() for the owned copies → borrowed access
            let shard_access = HubAccess {
                token: tok.as_deref(),
                revision: rev.as_deref(),
                cache_root: root.as_deref(),
            };
            // Range stats aren't part of this multi-file listing's shape
            // (only the per-file `Cached`/`Remote` provenance is); the
            // single-file `inspect_safetensors` caller in the `hf-fm` CLI
            // is what renders them.
            let (info, source, _stats) =
                inspect_safetensors_with_access(&repo, &filename, &shard_access).await?;
            Ok((filename, info, source))
        }
    })
    .await?;

    results.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(results)
}

/// Runs `run` on every item, at most `concurrency` at a time, and stops at
/// the first error without abandoning work already under way.
///
/// After the first error no further item starts, and every task already
/// running is awaited before that error is returned. Aborting them instead
/// would not stop the work: a remote header is read on a blocking thread
/// that aborting its task leaves running, and one still sending requests
/// when the caller's runtime shuts down panics inside `tokio`'s timer ("A
/// Tokio 1.x context was found, but it is being shutdown"), which printed
/// a panic on stderr after an already-reported failure until v0.12.3.
/// Results come back in completion order.
///
/// # Errors
///
/// Returns the first error any `run` returned.
/// Returns [`FetchError::Http`] if a task panics or the concurrency limiter
/// fails.
async fn fan_out_fail_fast<T, R, F, Fut>(
    items: Vec<T>,
    concurrency: usize,
    run: F,
) -> Result<Vec<R>, FetchError>
where
    T: Send + 'static,
    R: Send + 'static,
    F: Fn(T) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<R, FetchError>> + Send + 'static,
{
    let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let failed = Arc::new(AtomicBool::new(false));
    let run = Arc::new(run);
    let mut join_set = JoinSet::new();

    for item in items {
        let limiter = Arc::clone(&semaphore);
        let failed = Arc::clone(&failed);
        let run = Arc::clone(&run);
        join_set.spawn(async move {
            let _permit = limiter
                .acquire_owned()
                .await
                .map_err(|e| FetchError::Http(format!("semaphore error: {e}")))?;
            if failed.load(Ordering::Acquire) {
                return Ok(None); // EXPLICIT: an earlier item failed, so this one never starts
            }
            let result = run(item).await;
            if result.is_err() {
                failed.store(true, Ordering::Release);
            }
            result.map(Some)
        });
    }

    let mut results = Vec::new();
    let mut first_error: Option<FetchError> = None;
    while let Some(joined) = join_set.join_next().await {
        let error = match joined {
            Ok(Ok(Some(result))) => {
                results.push(result);
                continue;
            }
            Ok(Ok(None)) => continue, // EXPLICIT: skipped after the first failure
            Ok(Err(e)) => e,
            Err(e) => FetchError::Http(format!("task join error: {e}")),
        };
        failed.store(true, Ordering::Release);
        if first_error.is_none() {
            first_error = Some(error);
        }
    }

    match first_error {
        Some(e) => Err(e),
        None => Ok(results),
    }
}

/// Tensor-file extensions the `inspect` dispatcher understands.
///
/// Single source of truth shared by the cached listing
/// ([`list_cached_tensor_files`]) and the CLI's remote listing / numeric
/// index / `--pick` candidate set. Matches the per-file dispatch in
/// `hf-fm inspect` — all four formats remote or cached (`.npz` since
/// v0.11.0, `.safetensors` since v0.11.1, `.gguf` since v0.11.2, `.pth`
/// since v0.11.4).
pub const SUPPORTED_TENSOR_EXTENSIONS: [&str; 4] = ["safetensors", "gguf", "npz", "pth"];

/// Returns `true` when `filename`'s extension matches one of
/// [`SUPPORTED_TENSOR_EXTENSIONS`] (case-insensitive).
#[must_use]
pub fn is_supported_tensor_file(filename: &str) -> bool {
    Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| {
            SUPPORTED_TENSOR_EXTENSIONS
                .iter()
                .any(|supported| ext.eq_ignore_ascii_case(supported))
        })
}

/// A `(filename, size_bytes)` enumeration of tensor files in a repo,
/// paired with the commit SHA of the resolved revision (when known).
///
/// The same tuple shape serves both local and remote listings:
/// [`list_cached_tensor_files`] produces it from a cached snapshot;
/// `repo::list_repo_files_with_commit` filtered through
/// [`is_supported_tensor_file`] produces it from the `HuggingFace` API.
/// Callers that need a uniform view over "what tensor files can I inspect?"
/// regardless of source use this alias.
pub type TensorFileListing = (Vec<(String, u64)>, Option<String>);

/// Alias kept for pre-v0.10.5 callers; [`list_cached_safetensors`] returns it.
///
/// Same tuple shape as [`TensorFileListing`], restricted by convention to
/// `.safetensors` entries.
pub type SafetensorsListing = TensorFileListing;

/// Lists `.safetensors` files in the cached snapshot for `repo_id`@`revision`.
///
/// Returns `(entries, commit_sha)` where `entries` is a sorted list of
/// `(filename, size_bytes)` tuples, and `commit_sha` is the snapshot's commit
/// hash (same value stored in `refs/<revision>`). Returns empty lists when the
/// repo or revision is not cached. Unlike [`inspect_repo_safetensors_cached`],
/// this does **not** parse any headers — it is a cheap name-and-size enumeration
/// intended for discovery UI (e.g. `inspect --list --cached`).
///
/// # Blocking I/O
///
/// Performs a synchronous recursive directory walk with a `stat` call per
/// `.safetensors` entry. On local SSDs the cost is sub-millisecond; on
/// networked caches (NFS/CIFS) a large sharded repo can take seconds. Wrap
/// in [`tokio::task::spawn_blocking`] from async contexts.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the snapshot directory cannot be read.
pub fn list_cached_safetensors(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<SafetensorsListing, FetchError> {
    list_cached_matching_files(repo_id, revision, |name| name.ends_with(".safetensors"))
}

/// Lists all supported tensor files in the cached snapshot for `repo_id`@`revision`.
///
/// Multi-format sibling of [`list_cached_safetensors`]: matches every
/// extension in [`SUPPORTED_TENSOR_EXTENSIONS`] (case-insensitive) instead
/// of `.safetensors` only. Returns `(entries, commit_sha)` where `entries`
/// is a sorted list of `(filename, size_bytes)` tuples, and `commit_sha` is
/// the snapshot's commit hash (same value stored in `refs/<revision>`).
/// Returns empty lists when the repo or revision is not cached. Does **not**
/// parse any headers — it is a cheap name-and-size enumeration intended for
/// discovery UI (e.g. `inspect --list --cached`, `inspect --pick --cached`).
///
/// # Blocking I/O
///
/// Performs a synchronous recursive directory walk with a `stat` call per
/// matching entry. On local SSDs the cost is sub-millisecond; on networked
/// caches (NFS/CIFS) a large sharded repo can take seconds. Wrap in
/// [`tokio::task::spawn_blocking`] from async contexts.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the snapshot directory cannot be read.
pub fn list_cached_tensor_files(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<TensorFileListing, FetchError> {
    list_cached_matching_files(repo_id, revision, is_supported_tensor_file)
}

/// Shared body of [`list_cached_safetensors`] / [`list_cached_tensor_files`]:
/// resolves the snapshot directory and walks it with the given filename
/// predicate.
fn list_cached_matching_files(
    repo_id: &str,
    revision: Option<&str>,
    matches: fn(&str) -> bool,
) -> Result<TensorFileListing, FetchError> {
    let rev = revision.unwrap_or("main");
    let cache_dir = cache::hf_cache_dir()?;
    let repo_dir = cache_layout::repo_dir(&cache_dir, repo_id);

    let Some(commit_hash) = cache::read_ref(&repo_dir, rev) else {
        return Ok((Vec::new(), None));
    };

    let snapshot_dir = cache_layout::snapshot_dir(&repo_dir, &commit_hash);
    if !snapshot_dir.exists() {
        return Ok((Vec::new(), Some(commit_hash)));
    }

    let mut results = Vec::new();
    collect_matching_names_sizes(&snapshot_dir, "", matches, &mut results)?;
    results.sort_by(|a, b| a.0.cmp(&b.0));
    Ok((results, Some(commit_hash)))
}

/// Recursively collects `(filename, size)` pairs for files whose bare
/// entry name satisfies `matches` (extension predicates need no prefix).
fn collect_matching_names_sizes(
    dir: &Path,
    prefix: &str,
    matches: fn(&str) -> bool,
    results: &mut Vec<(String, u64)>,
) -> Result<(), FetchError> {
    let entries = std::fs::read_dir(dir).map_err(|e| FetchError::Io {
        path: dir.to_path_buf(),
        source: e,
    })?;

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        // BORROW: explicit .to_string_lossy() for OsString → str conversion
        let name = entry.file_name().to_string_lossy().to_string();

        if path.is_dir() {
            let child_prefix = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            collect_matching_names_sizes(&path, &child_prefix, matches, results)?;
        } else if matches(&name) {
            let filename = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            // `resolved_file_size`, not `entry.metadata().len()`: these
            // sizes come from a snapshot tree, where an entry is a symlink
            // into `blobs/` on the usual Unix layout. Reading the link's own
            // length listed every tensor file at a few dozen bytes, the same
            // root cause as issue #16's disk-usage under-report.
            let size = crate::cache::resolved_file_size(&path).unwrap_or(0);
            results.push((filename, size));
        }
    }

    Ok(())
}

/// Inspects all `.safetensors` files in a cached repository (no network).
///
/// Walks the snapshot directory and inspects each `.safetensors` file's
/// header from local disk. Returns results in filename order.
///
/// # Blocking I/O
///
/// Walks the snapshot directory and reads each header synchronously. In async
/// contexts, wrap in [`tokio::task::spawn_blocking`] to avoid stalling the
/// runtime — multi-shard repos on network-mounted caches can take seconds.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be read.
/// Returns [`FetchError::SafetensorsHeader`] if any header is malformed.
pub fn inspect_repo_safetensors_cached(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<Vec<(String, SafetensorsHeaderInfo)>, FetchError> {
    let rev = revision.unwrap_or("main");
    let cache_dir = cache::hf_cache_dir()?;
    let repo_dir = cache_layout::repo_dir(&cache_dir, repo_id);

    let Some(commit_hash) = cache::read_ref(&repo_dir, rev) else {
        return Ok(Vec::new());
    };

    let snapshot_dir = cache_layout::snapshot_dir(&repo_dir, &commit_hash);
    if !snapshot_dir.exists() {
        return Ok(Vec::new());
    }

    let mut results = Vec::new();
    collect_safetensors_recursive(&snapshot_dir, "", &mut results)?;
    results.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(results)
}

/// Recursively finds and inspects `.safetensors` files in a snapshot directory.
fn collect_safetensors_recursive(
    dir: &Path,
    prefix: &str,
    results: &mut Vec<(String, SafetensorsHeaderInfo)>,
) -> Result<(), FetchError> {
    let entries = std::fs::read_dir(dir).map_err(|e| FetchError::Io {
        path: dir.to_path_buf(),
        source: e,
    })?;

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        // BORROW: explicit .to_string_lossy() for OsString → str conversion
        let name = entry.file_name().to_string_lossy().to_string();

        if path.is_dir() {
            let child_prefix = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            collect_safetensors_recursive(&path, &child_prefix, results)?;
        } else if name.ends_with(".safetensors") {
            let filename = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            let info = inspect_safetensors_local(&path)?;
            results.push((filename, info));
        }
    }

    Ok(())
}

// -----------------------------------------------------------------------
// Cache-first JSON fetches
// -----------------------------------------------------------------------

/// Shared body of the cache-first `JSON` fetchers ([`fetch_shard_index`],
/// [`fetch_adapter_config`], [`fetch_model_config`] and their
/// `_with_config` twins): the cached copy of `filename` if `access` finds
/// one, else one `GET` of it from the Hub.
///
/// A file the Hub reports absent is `Ok(None)`. `what` names the file in
/// errors (`"model config"`), so each fetcher keeps its own messages, and
/// `parse` turns its text into the fetcher's type.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cached copy cannot be read.
/// Returns [`FetchError::Http`] if the request fails, including a `404`
/// that names a missing revision.
/// Returns [`FetchError::RepoNotFound`] if the Hub reports the repo missing.
/// Returns [`FetchError::SafetensorsHeader`] if `parse` rejects the content,
/// as each fetcher's parser does for malformed `JSON`.
async fn fetch_json_cache_first<T>(
    repo_id: &str,
    filename: &str,
    what: &str,
    access: &HubAccess<'_>,
    parse: fn(&str, &str) -> Result<T, FetchError>,
) -> Result<Option<T>, FetchError> {
    let rev = access.rev();

    // Try local cache first.
    if let Some(cached_path) = access.cached_path(repo_id, filename) {
        let content = std::fs::read_to_string(&cached_path).map_err(|e| FetchError::Io {
            path: cached_path,
            source: e,
        })?;
        return parse(&content, repo_id).map(Some);
    }

    // Fall back to HTTP.
    let client = chunked::build_client(access.token)?;
    let url = chunked::build_download_url(repo_id, rev, filename);

    // BORROW: explicit .as_str() instead of Deref coercion
    let response = client
        .get(url.as_str())
        .send()
        .await
        .map_err(|e| FetchError::Http(format!("failed to fetch {what} for {repo_id}: {e}")))?;

    if chunked::resolve_says_file_absent(response.status(), response.headers(), repo_id, rev)? {
        return Ok(None);
    }

    if !response.status().is_success() {
        return Err(FetchError::Http(format!(
            "{what} request for {repo_id} returned status {}",
            response.status()
        )));
    }

    let content = response
        .text()
        .await
        .map_err(|e| FetchError::Http(format!("failed to read {what} for {repo_id}: {e}")))?;

    parse(&content, repo_id).map(Some)
}

// -----------------------------------------------------------------------
// Shard index
// -----------------------------------------------------------------------

/// Raw JSON structure of `model.safetensors.index.json`.
#[derive(serde::Deserialize)]
struct RawShardIndex {
    weight_map: HashMap<String, String>,
    #[serde(default)]
    metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Fetches and parses the shard index for a sharded `.safetensors` model (cache-first).
///
/// Returns `Ok(None)` if the repo has no `model.safetensors.index.json` (i.e.,
/// the model is not sharded or uses a single `.safetensors` file).
///
/// [`fetch_shard_index_with_config`] takes the token, the revision and the
/// cache directory from a [`FetchConfig`] instead.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the index fetch fails, including a `404`
/// that names a missing revision.
/// Returns [`FetchError::RepoNotFound`] if the Hub reports the repo missing
/// (an authenticated `404 RepoNotFound`; see
/// `chunked::resolve_says_file_absent`). A missing index file is `Ok(None)`.
/// Returns [`FetchError::Io`] if a cached index cannot be read.
/// Returns [`FetchError::SafetensorsHeader`] if the index JSON is malformed.
pub async fn fetch_shard_index(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<Option<ShardedIndex>, FetchError> {
    fetch_shard_index_with_access(repo_id, &HubAccess::loose(token, revision)).await
}

/// [`fetch_shard_index`], with the token, the revision and the cache
/// directory taken from `config`.
///
/// The cache-first lookup reads [`FetchConfig::output_dir`] as the cache
/// root, so an index downloaded with the same `config` is found there
/// (unset: the default `HF` cache). A request sends [`FetchConfig::token`]
/// (unset: one is resolved implicitly) and targets [`FetchConfig::revision`]
/// (unset: `main`). The other settings of `config` are ignored.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the index fetch fails, including a `404`
/// that names a missing revision.
/// Returns [`FetchError::RepoNotFound`] if the Hub reports the repo missing.
/// A missing index file is `Ok(None)`.
/// Returns [`FetchError::Io`] if a cached index cannot be read.
/// Returns [`FetchError::SafetensorsHeader`] if the index JSON is malformed.
pub async fn fetch_shard_index_with_config(
    repo_id: &str,
    config: &FetchConfig,
) -> Result<Option<ShardedIndex>, FetchError> {
    fetch_shard_index_with_access(repo_id, &HubAccess::from_config(config)).await
}

/// Shared body of [`fetch_shard_index`] and [`fetch_shard_index_with_config`].
async fn fetch_shard_index_with_access(
    repo_id: &str,
    access: &HubAccess<'_>,
) -> Result<Option<ShardedIndex>, FetchError> {
    fetch_json_cache_first(
        repo_id,
        "model.safetensors.index.json",
        "shard index",
        access,
        parse_shard_index_json,
    )
    .await
}

/// Fetches the shard index from cache only (no network).
///
/// Returns `Ok(None)` if the index file is not cached.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cached file cannot be read.
/// Returns [`FetchError::SafetensorsHeader`] if the index JSON is malformed.
pub fn fetch_shard_index_cached(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<Option<ShardedIndex>, FetchError> {
    let rev = revision.unwrap_or("main");
    let index_filename = "model.safetensors.index.json";

    let Some(cached_path) = resolve_cached_path(repo_id, rev, index_filename) else {
        return Ok(None);
    };

    let content = std::fs::read_to_string(&cached_path).map_err(|e| FetchError::Io {
        path: cached_path,
        source: e,
    })?;

    let index = parse_shard_index_json(&content, repo_id)?;
    Ok(Some(index))
}

/// Parses shard index JSON into a `ShardedIndex`.
fn parse_shard_index_json(content: &str, repo_id: &str) -> Result<ShardedIndex, FetchError> {
    let raw: RawShardIndex =
        serde_json::from_str(content).map_err(|e| FetchError::SafetensorsHeader {
            filename: "model.safetensors.index.json".to_owned(),
            reason: format!("failed to parse shard index for {repo_id}: {e}"),
        })?;

    // Collect unique shard filenames in sorted order.
    let mut shard_set: Vec<String> = raw.weight_map.values().cloned().collect();
    shard_set.sort();
    shard_set.dedup();

    Ok(ShardedIndex {
        weight_map: raw.weight_map,
        shards: shard_set,
        metadata: raw.metadata,
    })
}

// -----------------------------------------------------------------------
// Param formatting helper
// -----------------------------------------------------------------------

/// Formats a parameter count with a compact suffix (e.g., `927.0M`, `1.02B`).
#[must_use]
pub fn format_params(count: u64) -> String {
    // CAST: u64 → f64, precision loss acceptable; value is a display-only scalar
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let val = count as f64;

    if count >= 1_000_000_000 {
        format!("{:.2}B", val / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.1}M", val / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", val / 1_000.0)
    } else {
        count.to_string()
    }
}

// -----------------------------------------------------------------------
// Adapter config
// -----------------------------------------------------------------------

/// Raw JSON structure of `adapter_config.json`.
#[derive(serde::Deserialize)]
struct RawAdapterConfig {
    #[serde(default)]
    peft_type: Option<String>,
    #[serde(default)]
    base_model_name_or_path: Option<String>,
    #[serde(default)]
    r: Option<u32>,
    #[serde(default)]
    lora_alpha: Option<f64>,
    #[serde(default)]
    target_modules: Option<AdapterTargetModules>,
    #[serde(default)]
    task_type: Option<String>,
}

/// `target_modules` in adapter configs can be a list of strings or a single string.
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum AdapterTargetModules {
    /// A list of module name strings.
    List(Vec<String>),
    /// A single module name string.
    Single(String),
}

/// Fetches and parses `adapter_config.json` for a `PEFT` adapter repository (cache-first).
///
/// Returns `Ok(None)` if the file does not exist (HTTP 404), meaning the
/// repository is not a `PEFT` adapter.
///
/// [`fetch_adapter_config_with_config`] takes the token, the revision and the
/// cache directory from a [`FetchConfig`] instead.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the request fails, including a `404`
/// that names a missing revision.
/// Returns [`FetchError::RepoNotFound`] if the Hub reports the repo missing
/// (an authenticated `404 RepoNotFound`; see
/// `chunked::resolve_says_file_absent`). A missing file is `Ok(None)`.
/// Returns [`FetchError::Io`] if a cached `adapter_config.json` cannot be
/// read.
/// Returns [`FetchError::SafetensorsHeader`] if the JSON is malformed.
pub async fn fetch_adapter_config(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<Option<AdapterConfig>, FetchError> {
    fetch_adapter_config_with_access(repo_id, &HubAccess::loose(token, revision)).await
}

/// [`fetch_adapter_config`], with the token, the revision and the cache
/// directory taken from `config`.
///
/// The cache-first lookup reads [`FetchConfig::output_dir`] as the cache
/// root, so an `adapter_config.json` downloaded with the same `config` is
/// found there (unset: the default `HF` cache). A request sends
/// [`FetchConfig::token`] (unset: one is resolved implicitly) and targets
/// [`FetchConfig::revision`] (unset: `main`). The other settings of `config`
/// are ignored.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the request fails, including a `404`
/// that names a missing revision.
/// Returns [`FetchError::RepoNotFound`] if the Hub reports the repo missing.
/// A missing file is `Ok(None)`.
/// Returns [`FetchError::Io`] if a cached `adapter_config.json` cannot be
/// read.
/// Returns [`FetchError::SafetensorsHeader`] if the JSON is malformed.
pub async fn fetch_adapter_config_with_config(
    repo_id: &str,
    config: &FetchConfig,
) -> Result<Option<AdapterConfig>, FetchError> {
    fetch_adapter_config_with_access(repo_id, &HubAccess::from_config(config)).await
}

/// Shared body of [`fetch_adapter_config`] and
/// [`fetch_adapter_config_with_config`].
async fn fetch_adapter_config_with_access(
    repo_id: &str,
    access: &HubAccess<'_>,
) -> Result<Option<AdapterConfig>, FetchError> {
    fetch_json_cache_first(
        repo_id,
        "adapter_config.json",
        "adapter config",
        access,
        parse_adapter_config_json,
    )
    .await
}

/// Fetches the adapter config from cache only (no network).
///
/// Returns `Ok(None)` if the file is not cached.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cached file cannot be read.
/// Returns [`FetchError::SafetensorsHeader`] if the JSON is malformed.
pub fn fetch_adapter_config_cached(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<Option<AdapterConfig>, FetchError> {
    let rev = revision.unwrap_or("main");
    let config_filename = "adapter_config.json";

    let Some(cached_path) = resolve_cached_path(repo_id, rev, config_filename) else {
        return Ok(None);
    };

    let content = std::fs::read_to_string(&cached_path).map_err(|e| FetchError::Io {
        path: cached_path,
        source: e,
    })?;

    let config = parse_adapter_config_json(&content, repo_id)?;
    Ok(Some(config))
}

/// Parses adapter config JSON into an [`AdapterConfig`].
fn parse_adapter_config_json(content: &str, repo_id: &str) -> Result<AdapterConfig, FetchError> {
    let raw: RawAdapterConfig =
        serde_json::from_str(content).map_err(|e| FetchError::SafetensorsHeader {
            filename: "adapter_config.json".to_owned(),
            reason: format!("failed to parse adapter config for {repo_id}: {e}"),
        })?;

    let target_modules = match raw.target_modules {
        Some(AdapterTargetModules::List(v)) => v,
        Some(AdapterTargetModules::Single(s)) => vec![s],
        None => Vec::new(),
    };

    Ok(AdapterConfig {
        peft_type: raw.peft_type,
        base_model_name_or_path: raw.base_model_name_or_path,
        r: raw.r,
        lora_alpha: raw.lora_alpha,
        target_modules,
        task_type: raw.task_type,
    })
}

/// Raw JSON structure of `config.json` (only the fields hf-fm reads for
/// KV-cache budgeting).
///
/// Serde aliases absorb the legacy GPT-NeoX / Falcon spellings. `text_config`
/// holds the nested language-model config of multimodal repos (Gemma-3) and
/// is used as a fallback when the attention dims are absent at top level.
#[derive(serde::Deserialize)]
struct RawModelConfig {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default, alias = "n_layer")]
    num_hidden_layers: Option<u32>,
    #[serde(default, alias = "n_head")]
    num_attention_heads: Option<u32>,
    #[serde(default, alias = "num_kv_heads", alias = "n_head_kv")]
    num_key_value_heads: Option<u32>,
    #[serde(default, alias = "attention_head_dim")]
    head_dim: Option<u32>,
    #[serde(default)]
    hidden_size: Option<u32>,
    #[serde(default)]
    torch_dtype: Option<String>,
    #[serde(default)]
    sliding_window: Option<u32>,
    #[serde(default)]
    sliding_window_pattern: Option<u32>,
    #[serde(default)]
    use_sliding_window: Option<bool>,
    #[serde(default)]
    kv_lora_rank: Option<u32>,
    #[serde(default)]
    qk_rope_head_dim: Option<u32>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    hybrid_override_pattern: Option<String>,
    #[serde(default)]
    attn_layer_indices: Option<Vec<u32>>,
    #[serde(default)]
    full_attention_interval: Option<u32>,
    #[serde(default)]
    attn_layer_offset: Option<u32>,
    #[serde(default)]
    attn_layer_period: Option<u32>,
    #[serde(default, alias = "mamba_num_heads")]
    mamba_n_heads: Option<u32>,
    #[serde(default, alias = "mamba_head_dim")]
    mamba_d_head: Option<u32>,
    #[serde(default, alias = "ssm_state_size")]
    mamba_d_state: Option<u32>,
    #[serde(default, alias = "conv_kernel")]
    mamba_d_conv: Option<u32>,
    #[serde(default, alias = "n_groups")]
    mamba_n_groups: Option<u32>,
    #[serde(default)]
    text_config: Option<Box<RawModelConfig>>,
}

/// Lowers a [`RawModelConfig`] into the public [`ModelConfig`].
///
/// Multimodal configs (Gemma-3) nest the language-model dims under
/// `text_config`; when the top level carries no attention dims, this recurses
/// into that nested config so the KV estimator sees the real numbers.
fn model_config_from_raw(raw: RawModelConfig) -> ModelConfig {
    if raw.num_hidden_layers.is_none()
        && raw.num_attention_heads.is_none()
        && raw.hidden_size.is_none()
        && let Some(text) = raw.text_config
    {
        return model_config_from_raw(*text);
    }

    ModelConfig {
        model_type: raw.model_type,
        num_hidden_layers: raw.num_hidden_layers,
        num_attention_heads: raw.num_attention_heads,
        num_key_value_heads: raw.num_key_value_heads,
        head_dim: raw.head_dim,
        hidden_size: raw.hidden_size,
        torch_dtype: raw.torch_dtype,
        sliding_window: raw.sliding_window,
        sliding_window_pattern: raw.sliding_window_pattern,
        use_sliding_window: raw.use_sliding_window,
        kv_lora_rank: raw.kv_lora_rank,
        qk_rope_head_dim: raw.qk_rope_head_dim,
        layer_types: raw.layer_types,
        hybrid_override_pattern: raw.hybrid_override_pattern,
        attn_layer_indices: raw.attn_layer_indices,
        full_attention_interval: raw.full_attention_interval,
        attn_layer_offset: raw.attn_layer_offset,
        attn_layer_period: raw.attn_layer_period,
        mamba_n_heads: raw.mamba_n_heads,
        mamba_d_head: raw.mamba_d_head,
        mamba_d_state: raw.mamba_d_state,
        mamba_d_conv: raw.mamba_d_conv,
        mamba_n_groups: raw.mamba_n_groups,
    }
}

/// Parses `config.json` content into a [`ModelConfig`].
fn parse_model_config_json(content: &str, repo_id: &str) -> Result<ModelConfig, FetchError> {
    let raw: RawModelConfig =
        serde_json::from_str(content).map_err(|e| FetchError::SafetensorsHeader {
            filename: "config.json".to_owned(),
            reason: format!("failed to parse model config for {repo_id}: {e}"),
        })?;

    Ok(model_config_from_raw(raw))
}

/// Fetches and parses a model's `config.json` (cache-first, then HTTP).
///
/// Returns `Ok(None)` when the repository has no `config.json` (HTTP 404) —
/// e.g. a non-model repo or a raw-weights upload.
///
/// [`fetch_model_config_with_config`] takes the token, the revision and the
/// cache directory from a [`FetchConfig`] instead.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the request fails, including a `404`
/// that names a missing revision.
/// Returns [`FetchError::RepoNotFound`] if the Hub reports the repo missing
/// (an authenticated `404 RepoNotFound`; see
/// `chunked::resolve_says_file_absent`). A missing `config.json` is `Ok(None)`.
/// Returns [`FetchError::Io`] if a cached `config.json` cannot be read.
/// Returns [`FetchError::SafetensorsHeader`] if the JSON is malformed.
pub async fn fetch_model_config(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<Option<ModelConfig>, FetchError> {
    fetch_model_config_with_access(repo_id, &HubAccess::loose(token, revision)).await
}

/// [`fetch_model_config`], with the token, the revision and the cache
/// directory taken from `config`.
///
/// The cache-first lookup reads [`FetchConfig::output_dir`] as the cache
/// root, so a `config.json` downloaded with the same `config` is found there
/// (unset: the default `HF` cache). A request sends [`FetchConfig::token`]
/// (unset: one is resolved implicitly) and targets [`FetchConfig::revision`]
/// (unset: `main`). The other settings of `config` are ignored.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the request fails, including a `404`
/// that names a missing revision.
/// Returns [`FetchError::RepoNotFound`] if the Hub reports the repo missing.
/// A missing `config.json` is `Ok(None)`.
/// Returns [`FetchError::Io`] if a cached `config.json` cannot be read.
/// Returns [`FetchError::SafetensorsHeader`] if the JSON is malformed.
pub async fn fetch_model_config_with_config(
    repo_id: &str,
    config: &FetchConfig,
) -> Result<Option<ModelConfig>, FetchError> {
    fetch_model_config_with_access(repo_id, &HubAccess::from_config(config)).await
}

/// Shared body of [`fetch_model_config`] and
/// [`fetch_model_config_with_config`].
async fn fetch_model_config_with_access(
    repo_id: &str,
    access: &HubAccess<'_>,
) -> Result<Option<ModelConfig>, FetchError> {
    fetch_json_cache_first(
        repo_id,
        "config.json",
        "model config",
        access,
        parse_model_config_json,
    )
    .await
}

/// Fetches a model's `config.json` from the local cache only (no network).
///
/// Returns `Ok(None)` if the file is not cached.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cached file cannot be read.
/// Returns [`FetchError::SafetensorsHeader`] if the JSON is malformed.
pub fn fetch_model_config_cached(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<Option<ModelConfig>, FetchError> {
    let rev = revision.unwrap_or("main");
    let config_filename = "config.json";

    let Some(cached_path) = resolve_cached_path(repo_id, rev, config_filename) else {
        return Ok(None);
    };

    let content = std::fs::read_to_string(&cached_path).map_err(|e| FetchError::Io {
        path: cached_path,
        source: e,
    })?;

    let config = parse_model_config_json(&content, repo_id)?;
    Ok(Some(config))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

    use std::io::Write as _;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use super::collect_matching_names_sizes;
    use super::is_supported_tensor_file;
    use super::resolve_cached_path;
    use crate::config::FetchConfig;
    use crate::error::FetchError;

    /// A listed name that could leave the snapshot directory reads as not
    /// cached, before the cache directory is even looked up.
    #[test]
    fn resolve_cached_path_never_joins_an_unsafe_filename() {
        for name in ["../../../etc/passwd", "/etc/passwd", "a/../../b", "c:\\x"] {
            assert_eq!(
                resolve_cached_path("org/repo", "main", name),
                None,
                "{name:?}"
            );
        }
    }

    /// `collect_matching_names_sizes` reports a plain file's own length.
    #[test]
    fn collect_sizes_reports_a_regular_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.safetensors"), vec![b'x'; 321]).unwrap();

        let mut out: Vec<(String, u64)> = Vec::new();
        collect_matching_names_sizes(tmp.path(), "", |n| n.ends_with(".safetensors"), &mut out)
            .unwrap();

        let (name, size) = out.first().unwrap();
        assert_eq!(name, "a.safetensors");
        assert_eq!(*size, 321);
    }

    /// These sizes come from a snapshot tree, so on the usual Unix layout the
    /// entry is a symlink into `blobs/`. Reading the link's own length listed
    /// every tensor file at a few dozen bytes: the same root cause as issue
    /// #16's disk-usage under-report, in the `inspect --list` path.
    #[cfg(unix)]
    #[test]
    fn collect_sizes_resolves_a_pointer_symlink() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("blob"), vec![b'x'; 4096]).unwrap();
        let snap = tmp.path().join("snap");
        std::fs::create_dir_all(&snap).unwrap();
        std::os::unix::fs::symlink("../blob", snap.join("model.safetensors")).unwrap();

        let mut out: Vec<(String, u64)> = Vec::new();
        collect_matching_names_sizes(&snap, "", |n| n.ends_with(".safetensors"), &mut out).unwrap();

        let (_, size) = out.first().unwrap();
        assert_eq!(
            *size, 4096,
            "must resolve to the target's size, not the link's"
        );
    }

    /// Builds a minimal `.npy` v1.0 header + data blob (`NumPy`'s on-disk
    /// array format), matching the exact byte layout `anamnesis`'s own
    /// reader expects — the same layout `anamnesis`'s own test suite builds
    /// via an identical private helper.
    fn make_npy_v1(header_str: &str, data: &[u8]) -> Vec<u8> {
        const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";
        let header_bytes = header_str.as_bytes();
        // Pad header to 64-byte alignment (magic=6 + version=2 + len=2 = 10).
        let total_before_pad = 10 + header_bytes.len();
        let padding = (64 - (total_before_pad % 64)) % 64;
        let padded_len = header_bytes.len() + padding;

        let mut npy = Vec::new();
        npy.extend_from_slice(NPY_MAGIC);
        npy.push(1); // major
        npy.push(0); // minor
        // CAST: usize → u16, header length is a few dozen bytes in these tests.
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        npy.extend_from_slice(&(padded_len as u16).to_le_bytes());
        npy.extend_from_slice(header_bytes);
        if padding > 0 {
            npy.extend(std::iter::repeat_n(b' ', padding - 1));
            npy.push(b'\n');
        }
        npy.extend_from_slice(data);
        npy
    }

    /// Builds a minimal in-memory `.npz` archive (a `ZIP` of `.npy` entries)
    /// from `(array_name, npy_header, data)` triples. `anamnesis::NpzInspectInfo`
    /// / `NpzTensorInfo` became `#[non_exhaustive]` in `anamnesis` 0.7.6, so
    /// tests that need one build real bytes and parse them through
    /// `anamnesis::inspect_npz_from_reader` — the same entry point
    /// [`super::inspect_npz`] uses in production — instead of hand-constructing
    /// the parsed struct via a literal.
    fn make_in_memory_npz(entries: &[(&str, &str, &[u8])]) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            for (name, header, data) in entries {
                zip.start_file(format!("{name}.npy"), options)
                    .expect("zip start_file");
                zip.write_all(&make_npy_v1(header, data))
                    .expect("zip write_all");
            }
            zip.finish().expect("zip finish");
        }
        buf
    }

    #[test]
    fn npz_info_to_header_info_synthesises_cumulative_offsets() {
        // The mapping shared by `inspect_npz_cached` (v0.10.3) and the
        // remote `inspect_npz` (v0.11.0): synthesised cumulative offsets
        // from per-tensor `byte_len`, no metadata block, zero header size.
        let bytes = make_in_memory_npz(&[
            (
                "w_enc",
                "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }",
                &[0u8; 24],
            ),
            (
                "b_dec",
                "{'descr': '<f4', 'fortran_order': False, 'shape': (4,), }",
                &[0u8; 16],
            ),
        ]);
        let parsed = anamnesis::inspect_npz_from_reader(std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(parsed.tensors.len(), 2, "sanity: both entries parsed");

        let info = super::npz_info_to_header_info(parsed, Some(1234));

        assert_eq!(info.tensors.len(), 2);
        // NPZ/ZIP entry order is not a contract this test should hardcode —
        // verify the cumulative-offset invariant generically instead of
        // assuming a specific position for each tensor.
        let mut cursor = 0u64;
        for t in &info.tensors {
            let (start, end) = t.data_offsets;
            assert_eq!(
                start, cursor,
                "tensor {} should start where the previous one ended",
                t.name
            );
            assert!(end >= start);
            cursor = end;
        }
        assert_eq!(
            cursor, 40,
            "cumulative end should equal the total bytes across both tensors"
        );
        assert_eq!(info.header_size, 0);
        assert_eq!(info.file_size, Some(1234));
        assert!(info.metadata.is_none());
        assert!(info.quant_info.is_none());

        let w_enc = info
            .tensors
            .iter()
            .find(|t| t.name == "w_enc")
            .expect("w_enc present");
        assert_eq!(w_enc.dtype, "F32");
        assert_eq!(w_enc.shape, vec![2, 3]);
        let b_dec = info
            .tensors
            .iter()
            .find(|t| t.name == "b_dec")
            .expect("b_dec present");
        assert_eq!(b_dec.shape, vec![4]);
    }

    /// Tiny big-endian-field binary builder for a minimal `GGUF` file,
    /// mirroring `anamnesis`'s own private `GgufBuilder` test helper (`GGUF`
    /// itself is little-endian; every multi-byte field below is `to_le_bytes`).
    struct GgufBuilder {
        buf: Vec<u8>,
    }

    impl GgufBuilder {
        fn new() -> Self {
            Self { buf: Vec::new() }
        }

        fn push_bytes(&mut self, bytes: &[u8]) {
            self.buf.extend_from_slice(bytes);
        }

        fn push_u32(&mut self, v: u32) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }

        fn push_u64(&mut self, v: u64) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }

        fn push_string(&mut self, s: &str) {
            // CAST: usize → u64, test fixture strings are a handful of bytes.
            #[allow(clippy::as_conversions)]
            self.push_u64(s.len() as u64);
            self.buf.extend_from_slice(s.as_bytes());
        }

        fn push_kv_string(&mut self, key: &str, value: &str) {
            self.push_string(key);
            self.push_u32(8); // GGUF metadata type: STRING
            self.push_string(value);
        }

        fn push_kv_string_array(&mut self, key: &str, values: &[&str]) {
            self.push_string(key);
            self.push_u32(9); // GGUF metadata type: ARRAY
            self.push_u32(8); // inner type: STRING
            // CAST: usize → u64, test fixture arrays are a handful of entries.
            #[allow(clippy::as_conversions)]
            self.push_u64(values.len() as u64);
            for v in values {
                self.push_string(v);
            }
        }

        fn push_tensor_info(&mut self, name: &str, shape: &[u64], dtype_disc: u32, offset: u64) {
            self.push_string(name);
            // CAST: usize → u32, test fixture shapes have at most a few dims.
            #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
            self.push_u32(shape.len() as u32);
            for &d in shape {
                self.push_u64(d);
            }
            self.push_u32(dtype_disc);
            self.push_u64(offset);
        }

        fn pad_to_alignment(&mut self, alignment: usize) {
            while !self.buf.len().is_multiple_of(alignment) {
                self.buf.push(0);
            }
        }

        fn finish(self) -> Vec<u8> {
            self.buf
        }
    }

    #[test]
    fn gguf_front_matter_to_header_info_maps_tensors_and_metadata() {
        // The mapping shared by `inspect_gguf_cached` (v0.10.2, via
        // `anamnesis::parse_gguf` → `ParsedGguf`) and the remote
        // `inspect_gguf` (v0.11.2, via
        // `anamnesis::parse_gguf_front_matter_from_reader` → `GgufFrontMatter`):
        // absolute per-tensor offsets carry over directly, scalar metadata
        // is stringified, array metadata is skipped, and the format
        // version/alignment land under synthetic `gguf.*` keys.
        //
        // `anamnesis::GgufTensorInfo`/`GgufMetadataValue` became
        // `#[non_exhaustive]` in `anamnesis` 0.7.6, so this builds a real
        // minimal `GGUF` file and parses it through
        // `anamnesis::parse_gguf_front_matter_from_reader` — the same entry
        // point [`super::inspect_gguf`] uses in production. Both tensors use
        // `F32` (dtype discriminant 0): every `GgufType` variant `anamnesis`
        // currently knows about has a tabulated `type_size`, so `byte_len:
        // None` (tested against a hand-built value in earlier versions of
        // this test) is no longer reachable through a successful real parse.
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3); // version
        b.push_u64(2); // tensor_count
        b.push_u64(2); // kv_count
        b.push_kv_string("general.architecture", "llama");
        b.push_kv_string_array("tokenizer.ggml.tokens", &["<bos>"]);
        // tensor 0 — F32 [4, 4] → 64 bytes at relative offset 0
        b.push_tensor_info("blk.0.attn_q.weight", &[4, 4], 0, 0);
        // tensor 1 — F32 [2] → 8 bytes at relative offset 64
        b.push_tensor_info("blk.0.attn_k.weight", &[2], 0, 64);
        b.pad_to_alignment(32);
        b.push_bytes(&[0u8; 64]); // blk.0.attn_q.weight data
        b.push_bytes(&[0u8; 8]); // blk.0.attn_k.weight data
        let bytes = b.finish();

        let front =
            anamnesis::parse_gguf_front_matter_from_reader(&mut std::io::Cursor::new(bytes))
                .unwrap();
        assert_eq!(front.tensor_infos.len(), 2, "sanity: both tensors parsed");
        assert_eq!(front.version, 3);
        assert_eq!(front.alignment, 32);

        let info = super::gguf_front_matter_to_header_info(
            &front.tensor_infos,
            &front.metadata,
            front.version,
            front.alignment,
            Some(9999),
        );

        assert_eq!(info.tensors.len(), 2);
        let q_proj = info
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.attn_q.weight")
            .expect("blk.0.attn_q.weight present");
        // `GgufTensorInfo::data_offset` is absolute within the file (the
        // tensor-data section start plus each tensor's relative offset), not
        // relative to the header — so anchor on the parser's own start
        // rather than hardcoding a byte count derived from this fixture's
        // exact header layout.
        let q_start = q_proj.data_offsets.0;
        assert_eq!(
            q_start % u64::from(front.alignment),
            0,
            "tensor data should start at an aligned offset"
        );
        assert_eq!(q_proj.data_offsets, (q_start, q_start + 64));
        assert_eq!(q_proj.dtype, "F32");
        let k_proj = info
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.attn_k.weight")
            .expect("blk.0.attn_k.weight present");
        assert_eq!(k_proj.data_offsets, (q_start + 64, q_start + 72));

        let meta = info.metadata.unwrap();
        assert_eq!(
            meta.get("general.architecture").map(String::as_str),
            Some("llama")
        );
        assert_eq!(meta.get("gguf.version").map(String::as_str), Some("3"));
        assert_eq!(meta.get("gguf.alignment").map(String::as_str), Some("32"));
        // Array-valued metadata (potentially huge, e.g. tokenizer vocab) is
        // skipped, not stringified.
        assert!(!meta.contains_key("tokenizer.ggml.tokens"));

        assert_eq!(info.header_size, 0);
        assert_eq!(info.file_size, Some(9999));
        assert!(info.quant_info.is_none());
    }

    #[test]
    fn is_supported_tensor_file_accepts_all_four_formats() {
        assert!(is_supported_tensor_file("model.safetensors"));
        assert!(is_supported_tensor_file("model.gguf"));
        assert!(is_supported_tensor_file("params.npz"));
        assert!(is_supported_tensor_file("weights.pth"));
    }

    #[test]
    fn is_supported_tensor_file_is_case_insensitive_on_extension() {
        assert!(is_supported_tensor_file("MODEL.SAFETENSORS"));
        assert!(is_supported_tensor_file("model.GGUF"));
    }

    #[test]
    fn is_supported_tensor_file_handles_nested_paths() {
        assert!(is_supported_tensor_file(
            "transformer/demonCORESFWNSFW_fluxV13.safetensors"
        ));
    }

    #[test]
    fn is_supported_tensor_file_rejects_other_extensions() {
        assert!(!is_supported_tensor_file("config.json"));
        assert!(!is_supported_tensor_file("model.bin"));
        assert!(!is_supported_tensor_file("archive.npy"));
        assert!(!is_supported_tensor_file("README.md"));
        assert!(!is_supported_tensor_file("no_extension"));
        // The extension must be the FINAL path segment suffix, not a
        // substring elsewhere in the name.
        assert!(!is_supported_tensor_file("model.safetensors.bak"));
    }

    // ---------- `_with_config` twins (v0.12.3) ----------
    //
    // Each test lays files out under a temporary cache root and points a
    // `FetchConfig`'s `output_dir` at it. A twin that ignored the root (or the
    // revision) would miss the cache and go to the Hub for `org/model`, which
    // fails, so a pass shows the configured root was read, with no network.

    /// The commit every fixture's ref points to.
    const FIXTURE_COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    /// Lays `bytes` out as `filename` of `repo_id` at `revision` in the cache
    /// rooted at `root`, as `hf-hub` would: a ref naming the commit, and the
    /// file in that commit's snapshot.
    fn cache_file_at(root: &Path, repo_id: &str, revision: &str, filename: &str, bytes: &[u8]) {
        let repo_dir = crate::cache_layout::repo_dir(root, repo_id);
        let ref_path = crate::cache_layout::ref_path(&repo_dir, revision);
        std::fs::create_dir_all(ref_path.parent().unwrap()).unwrap();
        std::fs::write(&ref_path, FIXTURE_COMMIT).unwrap();
        let pointer = crate::cache_layout::pointer_path(&repo_dir, FIXTURE_COMMIT, filename);
        std::fs::create_dir_all(pointer.parent().unwrap()).unwrap();
        std::fs::write(&pointer, bytes).unwrap();
    }

    /// A config whose cache root is `root`, at `revision`.
    fn config_at(root: &Path, revision: &str) -> FetchConfig {
        FetchConfig::builder()
            .revision(revision)
            // BORROW: explicit .to_path_buf() — the builder takes an owned PathBuf
            .output_dir(root.to_path_buf())
            .build()
            .unwrap()
    }

    #[test]
    fn resolve_cached_path_at_reads_the_given_root() {
        let root = tempfile::tempdir().unwrap();
        cache_file_at(root.path(), "org/model", "main", "config.json", b"{}");
        let repo_dir = crate::cache_layout::repo_dir(root.path(), "org/model");
        assert_eq!(
            super::resolve_cached_path_at(Some(root.path()), "org/model", "main", "config.json"),
            Some(crate::cache_layout::pointer_path(
                &repo_dir,
                FIXTURE_COMMIT,
                "config.json"
            ))
        );
        // Only the revision that was laid out is there.
        assert_eq!(
            super::resolve_cached_path_at(Some(root.path()), "org/model", "v2", "config.json"),
            None
        );
    }

    #[test]
    fn hub_access_takes_token_revision_and_cache_root_from_the_config() {
        let config = FetchConfig::builder()
            .token("hf_example")
            .revision("v1")
            .output_dir(PathBuf::from("custom-cache"))
            .build()
            .unwrap();
        let access = super::HubAccess::from_config(&config);
        assert_eq!(access.token, Some("hf_example"));
        assert_eq!(access.rev(), "v1");
        assert_eq!(access.cache_root, Some(Path::new("custom-cache")));

        // The original entry points: loose arguments, default cache, `main`.
        let loose = super::HubAccess::loose(None, None);
        assert_eq!(loose.rev(), "main");
        assert_eq!(loose.cache_root, None);
        let loose = super::HubAccess::loose(Some("hf_example"), Some("v2"));
        assert_eq!(loose.token, Some("hf_example"));
        assert_eq!(loose.rev(), "v2");
        assert_eq!(
            loose.cache_root, None,
            "the originals read the default cache"
        );
    }

    #[tokio::test]
    async fn npz_with_config_reads_its_output_dir_at_its_revision() {
        let root = tempfile::tempdir().unwrap();
        let npz = make_in_memory_npz(&[(
            "w_enc",
            "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }",
            &[0u8; 24],
        )]);
        // Laid out at `v1` only: the config's revision must reach the lookup.
        cache_file_at(root.path(), "org/model", "v1", "params.npz", &npz);

        let (info, source, stats) = super::inspect_npz_with_config(
            "org/model",
            "params.npz",
            &config_at(root.path(), "v1"),
        )
        .await
        .unwrap();
        assert!(matches!(source, super::InspectSource::Cached));
        assert!(stats.is_none(), "a cache hit makes no range request");
        // BORROW: explicit .as_str() to compare names as &str
        let names: Vec<&str> = info.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["w_enc"]);
    }

    #[tokio::test]
    async fn npz_with_config_finds_a_commit_pinned_snapshot_that_has_no_ref() {
        // `hf-hub` writes no `refs/<sha>` for a commit-hash revision: the
        // file sits in `snapshots/<sha>/` with no ref naming it.
        let root = tempfile::tempdir().unwrap();
        let repo_dir = crate::cache_layout::repo_dir(root.path(), "org/model");
        let pointer = crate::cache_layout::pointer_path(&repo_dir, FIXTURE_COMMIT, "params.npz");
        std::fs::create_dir_all(pointer.parent().unwrap()).unwrap();
        let npz = make_in_memory_npz(&[(
            "w_enc",
            "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }",
            &[0u8; 24],
        )]);
        std::fs::write(&pointer, npz).unwrap();

        let (_, source, stats) = super::inspect_npz_with_config(
            "org/model",
            "params.npz",
            &config_at(root.path(), FIXTURE_COMMIT),
        )
        .await
        .unwrap();
        assert!(matches!(source, super::InspectSource::Cached));
        assert!(stats.is_none(), "a cache hit makes no range request");
    }

    /// A minimal `.safetensors` file holding one `F32` tensor named `w`.
    fn minimal_safetensors() -> Vec<u8> {
        let header = br#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut file = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        file.extend_from_slice(header);
        file.extend_from_slice(&[0u8; 8]);
        file
    }

    #[tokio::test]
    async fn inspect_shards_reads_each_shard_through_the_given_access() {
        // `inspect_repo_safetensors_with_config`'s per-shard plumbing: each
        // spawned task rebuilds its access from owned copies, so the root and
        // the revision must survive that hand-off. Laid out at `v1` only.
        let root = tempfile::tempdir().unwrap();
        for shard in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ] {
            cache_file_at(
                root.path(),
                "org/model",
                "v1",
                shard,
                &minimal_safetensors(),
            );
        }
        let config = config_at(root.path(), "v1");

        let results = super::inspect_shards(
            "org/model",
            vec![
                "model-00002-of-00002.safetensors".to_owned(),
                "model-00001-of-00002.safetensors".to_owned(),
            ],
            &super::HubAccess::from_config(&config),
        )
        .await
        .unwrap();
        // BORROW: explicit .as_str() to compare names as &str
        let names: Vec<&str> = results.iter().map(|r| r.0.as_str()).collect();
        assert_eq!(
            names,
            [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ],
            "returned in filename order"
        );
        assert!(
            results
                .iter()
                .all(|r| matches!(r.2, super::InspectSource::Cached)),
            "every shard read from the configured cache"
        );
    }

    #[tokio::test]
    async fn safetensors_with_config_reads_its_output_dir() {
        let root = tempfile::tempdir().unwrap();
        let file = minimal_safetensors();
        cache_file_at(root.path(), "org/model", "main", "model.safetensors", &file);

        let (info, source, stats) = super::inspect_safetensors_with_config(
            "org/model",
            "model.safetensors",
            &config_at(root.path(), "main"),
        )
        .await
        .unwrap();
        assert!(matches!(source, super::InspectSource::Cached));
        assert!(stats.is_none());
        // BORROW: explicit .as_str() to compare names as &str
        let names: Vec<&str> = info.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["w"]);
    }

    #[tokio::test]
    async fn gguf_and_pth_with_config_parse_the_file_in_their_output_dir() {
        // No GGUF or PTH builder exists here, so each file is deliberately
        // corrupt: a parse error proves the twin found the cached file, where
        // a twin that missed the cache would fail on the network instead.
        let root = tempfile::tempdir().unwrap();
        cache_file_at(
            root.path(),
            "org/model",
            "main",
            "model.gguf",
            b"not a gguf",
        );
        cache_file_at(root.path(), "org/model", "main", "model.pth", b"not a pth");
        let config = config_at(root.path(), "main");

        let err = super::inspect_gguf_with_config("org/model", "model.gguf", &config)
            .await
            .unwrap_err();
        assert!(
            matches!(&err, FetchError::SafetensorsHeader { reason, .. }
                if reason.contains("failed to parse GGUF")),
            "expected a parse error from the cached file, got: {err}"
        );

        let err = super::inspect_pth_with_config("org/model", "model.pth", &config)
            .await
            .unwrap_err();
        assert!(
            matches!(&err, FetchError::SafetensorsHeader { reason, .. }
                if reason.contains("failed to parse PTH")),
            "expected a parse error from the cached file, got: {err}"
        );
    }

    #[tokio::test]
    async fn json_fetchers_with_config_read_their_output_dir() {
        let root = tempfile::tempdir().unwrap();
        cache_file_at(
            root.path(),
            "org/model",
            "main",
            "config.json",
            br#"{"model_type":"llama","num_hidden_layers":2}"#,
        );
        cache_file_at(
            root.path(),
            "org/model",
            "main",
            "adapter_config.json",
            br#"{"peft_type":"LORA","r":8}"#,
        );
        cache_file_at(
            root.path(),
            "org/model",
            "main",
            "model.safetensors.index.json",
            br#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        );
        let config = config_at(root.path(), "main");

        let model = super::fetch_model_config_with_config("org/model", &config)
            .await
            .unwrap()
            .expect("config.json is cached");
        assert_eq!(model.model_type.as_deref(), Some("llama"));

        let adapter = super::fetch_adapter_config_with_config("org/model", &config)
            .await
            .unwrap()
            .expect("adapter_config.json is cached");
        assert_eq!(adapter.peft_type.as_deref(), Some("LORA"));
        assert_eq!(adapter.r, Some(8));

        let index = super::fetch_shard_index_with_config("org/model", &config)
            .await
            .unwrap()
            .expect("the shard index is cached");
        assert_eq!(
            index.shards,
            [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        );
    }

    #[tokio::test]
    async fn fan_out_fail_fast_waits_for_running_work_before_returning_the_error() {
        // The hazard it exists for: a remote header read runs on a blocking
        // thread that aborting its task leaves running. Each item here works
        // on such a thread, so when the error comes back, every item that
        // started must also have finished. Item 2 fails at once; the others
        // take 100 ms.
        use std::sync::atomic::AtomicUsize;
        let started = Arc::new(AtomicUsize::new(0));
        let finished = Arc::new(AtomicUsize::new(0));
        let (s, f) = (Arc::clone(&started), Arc::clone(&finished));
        let result = super::fan_out_fail_fast((0..16).collect::<Vec<u32>>(), 4, move |i| {
            let (s, f) = (Arc::clone(&s), Arc::clone(&f));
            async move {
                s.fetch_add(1, Ordering::SeqCst);
                let i = tokio::task::spawn_blocking(move || {
                    if i != 2 {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                    f.fetch_add(1, Ordering::SeqCst);
                    i
                })
                .await
                .map_err(|e| FetchError::Http(e.to_string()))?;
                if i == 2 {
                    return Err(FetchError::Http("item 2 failed".to_owned()));
                }
                Ok(i)
            }
        })
        .await;

        let err = result.unwrap_err();
        assert!(err.to_string().contains("item 2 failed"), "{err}");
        let (started, finished) = (
            started.load(Ordering::SeqCst),
            finished.load(Ordering::SeqCst),
        );
        assert_eq!(
            started, finished,
            "every item that started finished before the error came back"
        );
        assert!(started < 16, "items queued behind the failure never start");
    }

    #[tokio::test]
    async fn fan_out_fail_fast_returns_every_result_when_nothing_fails() {
        let mut results = super::fan_out_fail_fast(
            (0..10).collect::<Vec<u32>>(),
            3,
            |i| async move { Ok(i * 2) },
        )
        .await
        .unwrap();
        results.sort_unstable();
        assert_eq!(results, (0..10).map(|i| i * 2).collect::<Vec<u32>>());
    }

    #[test]
    fn with_config_futures_are_send() {
        // A caller that spawns one onto a multi-threaded runtime needs this.
        fn assert_send<T: Send>(_: &T) {}
        let config = FetchConfig::builder().build().unwrap();
        assert_send(&super::inspect_safetensors_with_config("o/m", "f", &config));
        assert_send(&super::inspect_gguf_with_config("o/m", "f", &config));
        assert_send(&super::inspect_npz_with_config("o/m", "f", &config));
        assert_send(&super::inspect_pth_with_config("o/m", "f", &config));
        assert_send(&super::inspect_repo_safetensors_with_config("o/m", &config));
        assert_send(&super::fetch_model_config_with_config("o/m", &config));
        assert_send(&super::fetch_adapter_config_with_config("o/m", &config));
        assert_send(&super::fetch_shard_index_with_config("o/m", &config));
    }
}
