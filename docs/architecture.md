# Architecture

<!-- Last updated: 2026-10-12, hf-fm v0.12.3 (lib.rs re-exports the inspect entry points; inspect's _with_config twins; http_range as the substrate) -->

## Stack

```
candle-mi
  model downloads, NPZ inspection, progress bars
         │ dependency (feature "indicatif")
hf-fetch-model   (the library; the hf-fm CLI is behind feature "cli")
  • repo file listing, download plan (dry-run, plan-to-config optimization)
  • file filtering (glob patterns, presets)
  • parallel file orchestration
  • multi-connection Range downloads, resumable across runs
  • progress callbacks, checksum verification, retry with backoff, timeouts
  • tensor-header inspection (.safetensors / .gguf / .npz / .pth),
    cached or remote over HTTP Range, with an opt-in header cache
  • small-file peek, model search, model card / README, quant siblings
  • cache disk usage, status, verify, gc, delete, clean-partial
         │ dependencies
         ├── hf-hub 1.0      repo info, single-connection downloads
         │                   (plain HTTP, or Xet for Xet-backed files),
         │                   HF cache layout, auth
         ├── anamnesis 0.7   tensor-format parsers, quant-scheme detection
         ├── reqwest 0.13    hf-fetch-model's own HTTP clients: chunked
         │                   downloads, range reads, Hub API calls
         └── hypomnesis 0.2  GPU memory for `inspect --check-gpu`
                             (optional: only feature "cli" builds it)
```

## What hf-fetch-model adds over hf-hub

`hf-hub` provides repository metadata and single-file downloads with HuggingFace cache compatibility. hf-fetch-model wraps it and adds:

- **Repo-level orchestration**: download all files in a model repository with one call
- **Multi-connection chunked downloads**: large files are split into chunks downloaded in parallel via HTTP Range requests, each request re-resolved to a freshly signed CDN URL (concurrency and connection count auto-tuned by the download plan optimizer based on file size distribution)
- **Resume across runs**: an interrupted chunked download keeps its partial file and a per-chunk progress sidecar, and the next run resumes each chunk where it stopped
- **File filtering**: glob patterns and presets to select which files to download
- **Progress reporting**: per-file callbacks, an async watch channel, and optional `indicatif` progress bars
- **Checksum verification**: SHA256 against HuggingFace LFS metadata, at download time and later with `cache verify`
- **Retry with backoff**: exponential backoff + jitter for transient failures
- **Timeout control**: per-file and overall time limits
- **Tensor-header inspection**: read tensor metadata (names, shapes, dtypes, offsets) of `.safetensors`, `.gguf`, `.npz` and `.pth` files, from the local cache or remotely via HTTP Range requests without downloading the weights, with quantization-scheme detection and an opt-in on-disk cache of parsed remote headers
- **Tensor layout comparison**: compare tensor structures between two models (only-in-A, only-in-B, dtype/shape differences, matching) via the CLI `diff` subcommand, and `config.json` architecture fields via `diff-config`
- **GPU-fit verdict**: whether a model fits the local GPU, from its tensor sizes and a KV-cache budget at a given context (CLI `inspect --check-gpu`), and an offload-aware fit plan across a base model's quantized siblings (CLI `quants --fits`)
- **Small-file peek**: print a config, README or `.gz` sidecar from a repo without downloading it
- **Cache management**: download state (complete / partial / missing / excluded) per file; disk usage per repo and globally, counted as bytes on disk; garbage collection by age and size budget; delete; partial-download cleanup
- **Model search and discovery**: query the HuggingFace Hub API with multi-term filtering, model card metadata, and server-side library/pipeline filtering; discover new model families and a base model's quant siblings
- **Model card and README display**: fetch and display model metadata and README text via the CLI `info` subcommand
- **Adapter config detection**: auto-detect `adapter_config.json` in PEFT adapter repos during `inspect` and display base model, rank, alpha, and target modules

## Module layout

| Module | Visibility | Role |
|--------|-----------|------|
| `lib.rs` | public | Top-level download functions and re-exports, including the eight cache-first `inspect` entry points that take a repo id and their `_with_config` twins |
| `config` | public | `FetchConfig` builder and `Filter` presets |
| `plan` | public | `DownloadPlan`, `FilePlan`, plan-to-config optimization |
| `download` | public | Download orchestration, `DownloadOutcome` |
| `repo` | public | Repository file listing via the HF API |
| `discover` | public | Search, model card, README fetch, `GateStatus`, family and quant-sibling discovery |
| `inspect` | public | Tensor-header inspection for all four formats, cached and remote, each remote-capable entry point with a `_with_config` twin that reads a `FetchConfig`'s token, revision and cache directory; `TensorInfo`, `SafetensorsHeaderInfo`, `ShardedIndex`, `AdapterConfig` |
| `http_range` | public | `HttpRangeReader`: the `Read + Seek` over HTTP Range requests that remote inspection runs on, with request and byte caps; the substrate under `inspect`'s entry points, for a reader-based parser of your own |
| `header_cache` | public | Opt-in on-disk cache of parsed remote headers (`--cache-headers`), with its one-hour trust window |
| `peek` | public | Small-file reads (`hf-fm peek`), with optional gzip decoding |
| `cache` | public | Cache inspection, status, disk usage, verification and cleanup; its private `cache::dedupe` submodule (`cache/dedupe.rs`) replaces duplicated snapshot copies with hard links (`plan_dedupe` / `apply_dedupe`, re-exported from `cache`) and holds the one routine that swaps a pointer for a link, which the download path also uses |
| `cache_layout` | public | Path construction for the HF cache layout and hf-fetch-model's sidecars |
| `auth` | public | `resolve_token`: the token every request carries, explicit or found the way `huggingface_hub` finds one (`HF_TOKEN`, the `hf auth login` token file) |
| `checksum` | public | SHA256 verification |
| `error` | public | `FetchError` and `FileFailure` |
| `progress` | public | `ProgressEvent`, `ProgressReceiver` and `IndicatifProgress` |
| `chunked` | private | Multi-connection Range download engine and the shared HTTP clients (`build_client` is re-exported) |
| `chunked_state` | private | Per-chunk progress sidecar that makes chunked downloads resumable |
| `atomic_write` | private | Atomic writes for the on-disk JSON sidecars |
| `retry` | private | Retry policy with exponential backoff |

Two more source files belong to the CLI binary only, which includes them from `src/bin/main.rs`: `format.rs` (display formatting shared across subcommands) and `gpu_check.rs` (the `inspect --check-gpu` verdict, the one caller of `hypomnesis`).
