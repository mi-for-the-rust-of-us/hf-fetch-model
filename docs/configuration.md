# Configuration

All download functions accept a `FetchConfig` built via the builder pattern.

## Quick start

```rust
use hf_fetch_model::{FetchConfig, Filter};

// Preset — safetensors + config files
let config = Filter::safetensors()
    .on_progress(|e| println!("{}: {:.1}%", e.filename, e.percent))
    .build()?;

let outcome = hf_fetch_model::download_with_config(
    "google/gemma-2-2b-it".to_owned(),
    &config,
).await?;
```

## Builder methods

| Method | Description | Default |
|--------|-------------|---------|
| `.revision(rev)` | Git revision (branch, tag, SHA) | `"main"` |
| `.token(tok)` | Auth token | `HF_TOKEN`, else the `hf auth login` token file (see below) |
| `.token_from_env()` | Read `HF_TOKEN` env var explicitly | — |
| `.filter(glob)` | Include pattern (repeatable) | all files |
| `.exclude(glob)` | Exclude pattern (repeatable) | none |
| `.concurrency(n)` | Parallel file downloads | auto-tuned |
| `.output_dir(path)` | Custom cache directory, for downloads and for the `inspect` functions' `_with_config` twins | HF default |
| `.timeout_per_file(dur)` | Ceiling on any single file transfer | 300s |
| `.timeout_total(dur)` | Hard wall-clock cap on the whole batch (including retries and in-flight files), independent of the per-file timeout | unlimited |
| `.max_retries(n)` | Retries per file | 3 |
| `.verify_checksums(bool)` | SHA256 verification | true |
| `.chunk_threshold(bytes)` | Min file size for multi-connection download | auto-tuned |
| `.connections_per_file(n)` | Parallel connections per large file | auto-tuned |
| `.on_progress(closure)` | Progress callback (sync) | — |
| `.progress_channel()` | Returns `(self, ProgressReceiver)` for async consumers | — |

## Authentication

Without `.token(..)`, every request carries the token the Python tools would use: `HF_TOKEN` (or the legacy `HUGGING_FACE_HUB_TOKEN`), else the token file, which is `HF_TOKEN_PATH` or `token` in the HuggingFace home (`HF_HOME`, else `$XDG_CACHE_HOME/huggingface`, else `~/.cache/huggingface`), where `hf auth login` writes it. `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` turns that lookup off; an explicit token is still sent. `hf_fetch_model::auth::resolve_token(None)` returns the token a request would carry.

## Reading a config back

A built `FetchConfig` returns what was set: `token()`, `revision()` and `output_dir()` (since v0.12.3), beside `concurrency()`, `connections_per_file()` and `chunk_threshold()`. `token()` returns only an explicit token; `None` means each request resolves one as described above, so `auth::resolve_token(config.token())` gives the token a request would actually carry.

```rust
use hf_fetch_model::{FetchConfig, auth};

let config = FetchConfig::builder().revision("v1.0").build()?;
assert_eq!(config.revision(), Some("v1.0"));
assert_eq!(config.output_dir(), None); // the default HF cache
let effective = auth::resolve_token(config.token());
```

## Filter presets

Five presets cover common patterns:

```rust
// Safetensors weights + configs (*.safetensors, *.json, *.txt)
let config = Filter::safetensors().build()?;

// GGUF quantized models (*.gguf, *.json, *.txt)
let config = Filter::gguf().build()?;

// NumPy weights, e.g. GemmaScope transcoders (*.npz, *.npy, config.yaml, *.json, *.txt)
let config = Filter::npz().build()?;

// PyTorch .bin weights (pytorch_model*.bin, *.json, *.txt)
let config = Filter::pth().build()?;

// Config files only, no weights (*.json, *.txt, *.md)
let config = Filter::config_only().build()?;
```

Presets are starting points — chain additional builder calls to customize:

```rust
let config = Filter::safetensors()
    .revision("v1.0")
    .token_from_env()
    .concurrency(2)
    .build()?;
```

## Single-file download

```rust
use hf_fetch_model::FetchConfig;

let config = FetchConfig::builder()
    .on_progress(|e| println!("{}: {:.1}%", e.filename, e.percent))
    .build()?;

let outcome = hf_fetch_model::download_file(
    "mntss/clt-gemma-2-2b-426k".to_owned(),
    "W_enc_0.safetensors",
    &config,
).await?;
```

## Async progress channel

For async consumers (GUI/TUI apps), use `progress_channel()` instead of `on_progress()` to receive updates via a `tokio::sync::watch` receiver:

```rust
use hf_fetch_model::{FetchConfig, Filter};

let (builder, mut progress_rx) = Filter::safetensors()
    .progress_channel();
let config = builder.build()?;

// Spawn a task to consume progress updates
tokio::spawn(async move {
    while progress_rx.changed().await.is_ok() {
        let event = progress_rx.borrow();
        println!("{}: {:.1}%", event.filename, event.percent);
    }
});

let outcome = hf_fetch_model::download_with_config(
    "google/gemma-2-2b-it".to_owned(),
    &config,
).await?;
```

Both `on_progress()` and `progress_channel()` can be active simultaneously — the callback fires synchronously in the download task, while the watch channel coalesces updates for async polling.

## Blocking wrappers

For non-async callers, every download function has a `_blocking` variant:

```rust
let outcome = hf_fetch_model::download_blocking(
    "google/gemma-2-2b-it".to_owned(),
)?;

let outcome = hf_fetch_model::download_file_blocking(
    "mntss/clt-gemma-2-2b-426k".to_owned(),
    "config.yaml",
    &config,
)?;
```

## Download plan (dry-run API)

Before downloading, you can compute a plan that shows what needs downloading vs. what is already cached:

```rust
use hf_fetch_model::{FetchConfig, Filter, download_plan};

let config = Filter::safetensors().build()?;
let plan = download_plan("google/gemma-2-2b-it", &config).await?;

println!("Total: {} bytes", plan.total_bytes);
println!("Cached: {} bytes", plan.cached_bytes);
println!("To download: {} bytes ({} files)",
    plan.download_bytes, plan.files_to_download());
```

The plan also computes an optimized config based on the file size distribution:

```rust
// Get recommended settings (concurrency, connections/file, chunk threshold)
let optimized = plan.recommended_config()?;

// Or customize from the recommended baseline
let custom = plan.recommended_config_builder()
    .concurrency(2)
    .build()?;

// Execute with the optimized config
let outcome = hf_fetch_model::download_with_plan(&plan, &optimized).await?;
```

When no explicit plan is used, `download_with_config()` internally computes a plan and applies recommended settings for any fields the caller did not explicitly set.

## Cache and download outcome

All download functions return `DownloadOutcome<T>`, which distinguishes cache hits from network downloads:

```rust
let outcome = hf_fetch_model::download(
    "google/gemma-2-2b-it".to_owned(),
).await?;

if outcome.is_cached() {
    println!("Resolved from local cache (no network).");
}

let path = outcome.into_inner();
```

## Header inspection

Read tensor metadata without downloading full weight files. The library checks the local cache first and only makes HTTP Range requests on cache miss. Since v0.12.3 the eight cache-first entry points that take a repo id, and their `_with_config` twins, are re-exported at the crate root as well as living in `inspect`.

```rust
use hf_fetch_model::inspect;
use std::path::Path;

// Single file: cache-first; on a miss, typically 3–4 HTTP Range requests,
// counted in `stats` (`None` when the header came from the cache)
let (info, source, stats) = hf_fetch_model::inspect_safetensors(
    "google/gemma-2-2b-it",
    "model-00001-of-00002.safetensors",
    None,  // token
    None,  // revision (defaults to "main")
).await?;

for tensor in &info.tensors {
    println!("{}: {:?} {}", tensor.name, tensor.shape, tensor.dtype);
}
println!("{} tensors, {} params", info.tensors.len(), info.total_params());

// Local file only (no network)
let info = inspect::inspect_safetensors_local(Path::new("model.safetensors"))?;

// Cache-only (fail if not cached)
let info = inspect::inspect_safetensors_cached("google/gemma-2-2b-it", "model.safetensors", None)?;

// Shard index for sharded models (1 request instead of 2×N)
if let Some(index) = inspect::fetch_shard_index("google/gemma-2-2b-it", None, None).await? {
    println!("{} shards, {} tensors", index.shards.len(), index.weight_map.len());
}
```

`TensorInfo` provides helpers: `num_elements()` (product of shape), `byte_len()` (data size from offsets), and `dtype_bytes()` (bytes per element for known dtypes).

All types derive `serde::Serialize` for JSON output.

The other three formats take the same arguments and return the same triple: `inspect_gguf`, `inspect_npz` and `inspect_pth`, each cache-first with a remote fallback over HTTP Range.

Those four, `inspect_repo_safetensors`, `fetch_shard_index`, `fetch_model_config` and `fetch_adapter_config` each have a `_with_config` twin (since v0.12.3) that takes the token, the revision and the cache directory from a `FetchConfig`. Its cache-first lookup reads `output_dir`, the cache a download with the same config writes to, so files you downloaded into a custom directory are inspected from disk instead of over the network. The config's other settings have no meaning for a header read and are ignored:

```rust
use hf_fetch_model::FetchConfig;
use std::path::PathBuf;

let config = FetchConfig::builder()
    .output_dir(PathBuf::from("/data/hf-cache"))
    .build()?;

let (info, source, stats) = hf_fetch_model::inspect_safetensors_with_config(
    "google/gemma-2-2b-it",
    "model-00001-of-00002.safetensors",
    &config,
).await?;
```

The cache-only functions (`inspect_*_cached`, `fetch_*_cached`, `list_cached_*`) take no config and always read the default cache.

## Shared HTTP client

`repo::list_repo_files_with_metadata()` takes a `&reqwest::Client` parameter to reuse TCP/TLS connections. Use `build_client()` to create one with the standard connect timeout and auth headers:

```rust
use hf_fetch_model::{build_client, repo};

let client = build_client(Some("hf_..."))?;
let files = repo::list_repo_files_with_metadata(
    "google/gemma-2-2b-it", Some("hf_..."), None, &client,
).await?;
```

## Cache disk usage

```rust
use hf_fetch_model::cache;

// Per-repo file breakdown (sorted by size descending)
let files = cache::cache_repo_usage("google/gemma-2-2b-it")?;
for f in &files {
    println!("{}: {} bytes", f.filename, f.size);
}

// Cache-wide summary (all repos)
let summaries = cache::cache_summary()?;
for s in &summaries {
    println!("{}: {} bytes, {} files", s.repo_id, s.total_size, s.file_count);
}
```

The two answer different questions. `cache_repo_usage` lists a repo's logical files, each at its own size. Since v0.12.2, `total_size` is the bytes the repo's whole directory occupies on disk, which is what deleting it would free. The two can differ: on Windows, where snapshot files are usually copies of their blobs rather than links, the repo holds those bytes twice.
