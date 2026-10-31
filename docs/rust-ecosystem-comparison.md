# Rust Ecosystem Comparison

**Date surveyed:** October 2026
**hf-fetch-model version at survey:** v0.12.2
**Scope:** Rust crates and binaries only. Python tooling (`huggingface-cli`/`hf`, `accelerate`) is out of scope for the matrix by design: this comparison exists to help downstream Rust consumers evaluate alternatives in their own ecosystem. A few Python analogs are named in prose as cross-ecosystem context, but they are never scored in the matrix.

> **Revision note (October 2026).** Re-baselined from v0.10.6 to v0.12.2, and every competitor re-verified against crates.io, GitHub and, where a claim needed it, the tool's source. Several claims were already wrong in June:
> - safetensors-browser reads Hub headers over HTTP Range;
> - mistral.rs has had `cache list` / `cache delete` since January;
> - apr-cli's `serve plan` budgets the KV cache;
> - `apr diff` compares model files;
> - gguf-rs ships a CLI, and llm_hunter is library-only;
> - the Python cache RFC closed in November 2025.
>
> Since June:
> - `hf-hub` 1.0 shipped stable and absorbed `huggingface_hub_rust`;
> - `rust-hf-downloader` went from 1.4 to 2.14;
> - `mistral.rs` added a Range-based `uqff inspect` and a fit-and-quant recommender;
> - `llmfit` grew to 37.8k stars;
> - a new tool, `sift`, reads real headers for a per-quant fit verdict, as hf-fm does.
>
> Four of the "gaps unique to hf-fm" below are narrower as a result. Items verified only from a README are marked so.

---

## What hf-fetch-model does (the comparison reference)

`hf-fetch-model` (binary alias `hf-fm`) is a Rust CLI binary AND embeddable library for working with HuggingFace model repos. The 13 capability areas used as comparison axes:

1. **MultiDL** — multi-connection chunked downloads with auto-tuned concurrency. Every chunk request re-resolves to a freshly signed CDN URL, which Xet-backed repos require. Downloads resume across interruptions via `.chunked.part.state` sidecars. Also: per-file/total timeouts (hard wall-clock cap, including in-flight files, since v0.10.5), glob filters (`--filter`/`--exclude`), filter presets (`safetensors`/`gguf`/`npz`/`pth`/`config-only`), gated-model preflight auth, and `--dry-run`.
2. **DLFile** — single-file downloads with glob expansion (`download-file org/model "pytorch_model-*.bin"`), with `--dry-run` since v0.10.7.
3. **Inspect** — header reading for all four tensor formats, cached or remote via `HttpRangeReader` (HTTP Range requests, no weight bytes). Remote support arrived per format: `.npz` (v0.11.0), `.safetensors` (v0.11.1), `.gguf` (v0.11.2), `.pth` (v0.11.4).
   - **Output:** tensor names, shapes, dtypes, metadata.
   - **Views:** `--tree` (numeric sibling collapsing `[0..N]`, tolerating up to 3 outliers per edge since v0.12.0), `--dtypes`, and `--group-by <PATTERN>` (a MATCHED / OTHER byte split with a per-MoE-layer expert cost, v0.12.1).
   - **Selection:** `--limit`, `--filter`, `--list`, `--pick`, `--json`.
   - **Quantization detection:** for safetensors, cached (v0.10.3) and remote (v0.11.1).
   - **Sharded aggregation** across `model.safetensors.index.json`.
   - **`--cache-headers`** keeps parsed remote headers on disk (v0.12.1); entries younger than an hour are used without a request (v0.12.2).
   - **`peek`** covers the non-tensor files: configs, READMEs, `.gz` sidecars (v0.11.5).
4. **Diff** — tensor-layout comparison between two models, remote or cached (only-in-A, only-in-B, dtype/shape diffs, matching). Options: **`--dtypes`** side-by-side histograms (v0.10.2), `--limit` (v0.10.7), and `--collapse` pattern tables (v0.12.0). **`diff-config`** compares `config.json` architecture fields (v0.12.0).
5. **Du** — cache disk usage with a numbered list, drill-down by index, `--age`, `--tree` (box-drawing connectors), partial-download markers (`●`) and `--json` (v0.10.7). GGUF quant alternatives show as a size range (v0.12.1). Since v0.12.2 each repo counts its bytes on disk: blobs, Windows snapshot copies, and sidecars.
6. **CacheMgmt** — `cache delete`, `cache clean-partial`, `cache gc --older-than/--max-size --except --dry-run`, `cache verify` (re-checks SHA256 against HF LFS metadata, with streaming spinner progress), `cache path` (prints the snapshot path for shell substitution).
7. **Search** — Hub search with multi-term filtering, `--library`, `--pipeline`, `--tag`, and `--exact` for a full-id match with model card display.
8. **Info** — model card + README display.
9. **Status** — per-repo file-level cache state (Complete / Partial / Missing / Excluded), with per-file partial attribution (v0.10.5) and `--json` (v0.10.7).
10. **ListFiles** — remote file listing with sizes + SHA256, optional `--show-cached` cross-reference, `--json`.
11. **Discover** — `model_type` grouping over the local cache and Hub discovery of new families. **`quants`** (v0.12.1) aggregates a base model's quantized sibling repos: it finds them by name and confirms them by GGUF metadata backlinks.
12. **Lib** — embeddable Rust library API, consumed by `candle-mi`. Built on `hf-hub` 1.0 (a direct dependency since v0.11.3), `anamnesis` (tensor-format parsers) and `hypomnesis` (GPU memory, used by the CLI).
13. **VRAMFit** — "does this model fit on my GPU?" from metadata only (no weight download). See the dedicated section below.
    - **`inspect --check-gpu`:** v0.10.1. KV-cache budgeting at `--context N` and hybrid Mamba/attention accounting came in v0.10.4, and a `Spilling:` capability line in v0.12.0.
    - **`quants --fits <SIZE> [--reserve <SIZE>]`:** v0.12.1. An offload-aware plan across quantized siblings, which reads real `.gguf` headers for over-budget candidates; `--cache-headers` since v0.12.2.

The bin is `hf-fm` (alias `hf-fetch-model`); installed via `cargo install hf-fetch-model --features cli`. GPU detection is delegated to `hypomnesis` 0.2.14 (NVML / DXGI / `nvidia-smi`, plus Metal on macOS).

---

## Per-tool detail

### A. Direct competitors (Rust HuggingFace clients/CLIs)

**hf-hub** — [github.com/huggingface/hf-hub](https://github.com/huggingface/hf-hub) | [crates.io/crates/hf-hub](https://crates.io/crates/hf-hub)

**v1.0.0 (2026-07-10)**, stable, ~336 stars, official HuggingFace-maintained, **~560 reverse deps** (crates.io API).

- **Lineage.** 1.0 is a complete rewrite ported from `huggingface_hub_rust` ([PR #150](https://github.com/huggingface/hf-hub/pull/150)), so HF now has one official Rust lineage, not two. 1.0.0 was published from the commit tagged `v1.0.0-rc.2`; there is no `v1.0.0` tag and no GitHub Release for it.
- **Covers:**
  - repo metadata and CRUD;
  - file download, including `snapshot_download` with allow/ignore globs and file-level concurrency, and ranged streams;
  - `list_tree` with LFS sha256 and sizes;
  - **Hub search** (`list_models` with `search` / `author` / `filter` / `sort`);
  - `scan_cache()` (per-repo size and revisions);
  - Xet transfers, Buckets, sync + async APIs.
- **The `hfrs` CLI** exposes most of this but is not on crates.io (`cargo install --git`).
- **Does NOT cover:** tensor-header inspection, diff, cache gc / verify / tree view.
- **Downstream notes.** hf-fm depends on it directly since v0.11.3. The unreleased 1.1.0 on `main` removes every environment-variable read (`HF_TOKEN`, `HF_HOME`, `HF_HUB_CACHE`, `HF_ENDPOINT`) from the library ([PR #193](https://github.com/huggingface/hf-hub/pull/193)), and `hf-hub = "1.0"` will accept it.

**huggingface-hub** (a.k.a. `huggingface_hub_rust`) — [github.com/huggingface/huggingface_hub_rust](https://github.com/huggingface/huggingface_hub_rust) | [crates.io/crates/huggingface-hub](https://crates.io/crates/huggingface-hub)
Still **v0.0.0 (2026-03-26)**, 24 stars, no feature commit since 2026-04-15. Superseded in practice: its code became `hf-hub` 1.0, by the same publisher, and `hf-hub`'s README now carries its tagline. Its own README has no redirect notice. Dropped from the matrix.

**rust-hf-downloader** — [crates.io/crates/rust-hf-downloader](https://crates.io/crates/rust-hf-downloader)

**v2.14.0 (2026-10-07)**, 2.0.0 on 2026-09-23, 37 stars, very active. Binary-only (not a library).

- **Downloads:** up to 8 parallel chunks per file, one file at a time, with resume, SHA256 verification, gated-model auth and multi-part GGUF grouping.
- **Interfaces:** a TUI, plus one-shot `download` and `search` subcommands with NDJSON `--json` and stable exit codes (headless mode returned in 2.3.0 after 2.0.0 removed it).
- **New since June:** `hf-cache sync`, which writes the real `~/.cache/huggingface/hub` layout (SHA256-gated, atomic, with `--include`/`--exclude`), `hf-cache path`, and a self-`update` command.
- **Config path** is per-platform now (`~/.config/jreb`, `~/Library/Application Support/jreb`, `%APPDATA%\jreb`).
- **Lacks:** tensor inspection, diff, cache gc / verify / du, library API.

**modelhub** — [github.com/di-osc/modelhub](https://github.com/di-osc/modelhub) — **NEW**

**v0.2.2 (2026-10-08)**, first published 2026-07-03, 1 star. Library + CLI for **both HuggingFace and ModelScope**, the closest new match to hf-fm's library-plus-CLI shape.

- `download`: a whole repo with `-j`, or one file.
- `list`: across its own, the HF and the ModelScope caches.
- `check`: completeness by existence and size, not SHA.
- `clear`, and incremental `upload`.
- No tensor inspection or Hub search.

**xhf** — [github.com/3axap4eHko/xhf](https://github.com/3axap4eHko/xhf) — **NEW**

**v0.0.2 (2026-09-15)**, 1 star, binary-only.

- `search users|orgs|repos` with `--json`.
- `repo tree` with globs, and `repo cat`.
- `download` with glob include / `!` exclude, byte-range parallel chunks within one file, and resumable `.part` files.

**model-hub** — [github.com/waitsalt/model-hub](https://github.com/waitsalt/model-hub) | [crates.io/crates/model-hub](https://crates.io/crates/model-hub)
**v0.1.1 (2026-04-15)**, no activity since. Async downloader for HuggingFace and ModelScope: concurrent transfers, retries, resume, path-traversal protection, token auth, file filtering. No cache management, inspect, or search.

**rustyface** — [crates.io/crates/rustyface](https://crates.io/crates/rustyface) | [github.com/AspadaX/RustyFace](https://github.com/AspadaX/RustyFace)
**v0.1.3 (2025-07-30)**, unchanged. Lightweight whole-repo downloader: concurrent downloads (`--tasks N`), SHA-256 verification, mirror support (default `hf-mirror.com`). No glob filters, inspect, search, or cache management.

**facecrab** / **ogenius** — [crates.io/crates/facecrab](https://crates.io/crates/facecrab) | [crates.io/crates/ogenius](https://crates.io/crates/ogenius) | [github.com/tmzt/rusty-genius](https://github.com/tmzt/rusty-genius)

Both **v0.1.6 (2026-02-08)**, unchanged.

- **facecrab:** resolves HF repo IDs to local cached paths and keeps a `registry.toml` index. Async via `surf` (async-std); crates.io counts 427 code lines in 3 files.
- **ogenius:** "a simpler alternative to Ollama", with `chat`, `download <repo>` (via facecrab) and an OpenAI-compatible `serve`. No inspect, diff, cache or search.

**hugging-face-client** (v0.6.0, 2025-05-14) and **hf-hub-enfer** (v0.3.2, 2024-12-06) remain inactive and abandoned respectively.

### B. Tensor format inspectors in Rust

**safetensors-browser** — [github.com/danieldk/safetensors-browser](https://github.com/danieldk/safetensors-browser) | [crates.io/crates/safetensors-browser](https://crates.io/crates/safetensors-browser)

**v0.2.0 (2026-02-19)**, by Daniël de Kok (a HuggingFace engineer), ~956 LOC, binary-only. Its repository, unlisted on crates.io, settles what the June survey could not confirm. It is a **remote Hub inspector** whose only input is a repo ID plus `--revision`:

- It fetches `config.json` and the shard index through `hf-hub` 0.5.
- It reads each shard's header with two HTTP Range requests (`bytes=0-7`, then the header).
- It caches header stubs locally.
- It shows them in a ratatui TUI with fuzzy search, AWQ/GPTQ-aware: it reads `quantization_config` and shows dequantized `nn.Linear` shapes.

Safetensors only. These features date from early 2025.

**safetensors_explorer** — [github.com/EricLBuehler/safetensors_explorer](https://github.com/EricLBuehler/safetensors_explorer) | [crates.io](https://crates.io/crates/safetensors_explorer)
Crate **v0.2.0 (2025-07-28)**, ~64 stars, by Eric Buehler (also `mistral.rs`). Interactive TUI over local `.safetensors` and `.gguf` files: hierarchical tree, fuzzy search, sharded-index detection, multi-file view. `main` has unreleased work (BitNet GGUF, array value types). Local files only, binary-only.

**gguf-rs-lib / `gguf-cli`** (ThreatFlux) — [github.com/ThreatFlux/gguf](https://github.com/ThreatFlux/gguf) | [crates.io/crates/gguf-rs-lib](https://crates.io/crates/gguf-rs-lib)

**v0.3.2 (2026-10-06)**, ~6 stars, active. Library + `gguf-cli`, which is not published to crates.io and ships as GitHub release binaries.

- **Subcommands:** `info`, `tensors --filter`, `metadata` (json/yaml/toml/table), `validate` (recursive) and, new, **`compare`** of two GGUF files' metadata and tensor descriptors.
- **Quant geometry:** exact for TQ1_0 / TQ2_0 / MXFP4 / NVFP4 / Q1_0 / Q2_0.
- **Scope:** local files only.

**gguf-rs** (Zack Shen) — [github.com/zackshen/gguf](https://github.com/zackshen/gguf) | [crates.io/crates/gguf-rs](https://crates.io/crates/gguf-rs)
**v0.1.8 (2026-06-15)**, 21 stars. A GGUF parsing library (format versions 1–3, mmap, async) that also ships a **`gguf` CLI** (`gguf model.gguf`, `--tensors`) and writes GGUF files. Unrelated to ThreatFlux's crate despite the name. Local files only.

**inspector-gguf** — [github.com/FerrisMind/inspector-gguf](https://github.com/FerrisMind/inspector-gguf)
**v0.3.1 (2025-12-15)**, unchanged. GUI (egui) + CLI + library; exports CSV/YAML/Markdown/HTML/PDF; tokenizer and chat-template analysis. Local files only.

**llm_hunter** — [github.com/jpegleg/llm_hunter](https://github.com/jpegleg/llm_hunter)
**v0.3.5 (2026-04-14)**. A **library** (crates.io lists no binary) for forensic identification of model files: GGUF-focused, with entropy profiling, container-format signatures and family/quant fingerprinting, emitting JSON reports. The CLI that uses it is a separate project ([giant-spellbook](https://github.com/jpegleg/giant-spellbook)). Not structural inspection in hf-fm's sense.

**safetensors-cli** — [github.com/gzsombor/safetensors-cli](https://github.com/gzsombor/safetensors-cli)
**v0.1.0 (2023-06-17)**, ~57 LOC. Dormant on features; Renovate keeps its dependencies current (through 2026-06-09). Local files only.

**safetensors** (canonical Rust crate) — [github.com/safetensors/safetensors](https://github.com/safetensors/safetensors) | [crates.io](https://crates.io/crates/safetensors)
crates.io **v0.8.0 (2026-06-09)**, ~3.9k stars; `v0.9.0-rc.1` (2026-09-23) on GitHub only. The repository moved to the new `safetensors` organization (created 2026-04-07), and `huggingface/safetensors` now redirects to it. Library only.

**candle's `tensor-tools`** — see candle below: a local `ls` / `print` across six formats.

**Newer inspection libraries** (no CLI, local files): `engram-parser` 0.3.0 (GGUF + safetensors), the `mlmf-*` 0.5.11 crates (safetensors, GGUF, GPTQ/AWQ layouts). Also `assay` ([mlab-sh/assay](https://github.com/mlab-sh/assay), GitHub only), an offline integrity scanner for safetensors / GGUF / pickle with a `compare` mode; README only.

### C. Adjacent: ML inference frameworks with download tooling

**candle** — [github.com/huggingface/candle](https://github.com/huggingface/candle)

candle-core **v0.11.0 (2026-06-26)**, ~21.2k stars. Active. Pulls models via `hf-hub` per example (0.11.0 pins `hf-hub` 0.5; `main` has moved to 1.0).

- **No centralized download CLI.**
- The workspace's **`tensor-tools`** binary (`ls` / `print` / `quantize` / `dequantize`) reads local safetensors, npz, ggml, gguf, pth and pickle files. Its last crates.io publish was 0.9.0-alpha.5 (2025-04).

**mistral.rs** — [github.com/EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs)

GitHub **v0.9.4 (2026-09-24)**, ~7.7k stars; the crates.io crates lag at 0.8.1. Beyond `run` / `serve` / `quantize` / `doctor` / `bench`, it now covers three of hf-fm's axes:

- **`cache list` / `cache delete -m <id>`** over the HF hub cache, since v0.7.0 (2026-01): per-model size and last use. No gc, verify or partial-download handling.
- **`uqff inspect | report | verify`** (v0.8.18, 2026-06-21): a TUI explorer and structure checks for UQFF artifacts. It accepts an HF model id and reads remote files over HTTP Range. UQFF format only.
- **`tune`:** a fit-and-quant recommender.
  - It fetches only `config.json` and sizes each quantization level from the model's dimensions.
  - For each level it reports Fits / Hybrid / TooLarge, VRAM use and the largest context that fits, with multi-GPU device mapping and `--json`.
  - `--quant auto` applies its pick, and `--quant <level>` selects a matching UQFF or GGUF file by name.

The web UI is on by default; `--no-ui` disables it.

**apr-cli** — [docs.rs/crate/apr-cli/latest](https://docs.rs/crate/apr-cli/latest) | part of [paiml/aprender](https://github.com/paiml/aprender)

**v0.70.3 (2026-10-09)**, published in lockstep with `aprender` now, ~130 stars. An inference orchestration tool centered on its own APR format.

**`apr serve plan hf://org/model`** emits a **READY / WARNINGS / BLOCKED** fit verdict from `config.json` alone:

- **Weights:** a parameter count estimated from config, always sized at Q4 (`--quant` changes only the label).
- **KV cache:** plain GQA, f16, at `--seq-len × --batch-size`.
- **Also budgeted:** activations and a 512 MB overhead.
- **VRAM source:** `nvidia-smi memory.total` (nameplate capacity, first GPU).

On hf-fm's other axes:

- `apr pull` downloads from HF into apr's own cache, with a BLAKE3 manifest that `apr pull --verify` re-checks offline.
- `apr diff` compares two local model files across APR, GGUF and SafeTensors (`--weights`, `--values`, `--json`).
- `apr tensors` / `apr inspect` read local files.

**burn** — [github.com/tracel-ai/burn](https://github.com/tracel-ai/burn)
**v0.22.0 (2026-10-06)**, ~16.1k stars. No CLI for HF model management; weights load through in-code APIs. Out of scope.

**llm crate** — [crates.io/crates/llm](https://crates.io/crates/llm) | [github.com/graniet/llm](https://github.com/graniet/llm)
**v1.3.8 (2026-04-19)**, unchanged: a multi-backend LLM API client (OpenAI/Anthropic/Gemini/Ollama/…), not a local inference engine. Listed only because the crates.io name once belonged to the archived `rustformers/llm`.

### D. Adjacent: GPU-fit / VRAM estimation

**llmfit** — [github.com/AlexsJones/llmfit](https://github.com/AlexsJones/llmfit) | [crates.io/crates/llmfit](https://crates.io/crates/llmfit)

**v1.1.17 (2026-10-08)**, 1.0.0 on 2026-07-09, **~37.8k stars**, very active. A TUI + CLI + library (`llmfit-core`) that answers "which models fit my hardware?" with a **Perfect / Good / Marginal / Too Tight** verdict (a utilization ratio).

- **Weights:** `params × bits-per-param`. Parameter counts come from a built-in catalog that `llmfit update` refreshes from HF metadata; it never reads file headers.
- **KV cache counts toward the fit**, at `--max-context` or `plan --context N`, with a choice of KV quant. Hybrid models count only their full-attention layers (since 1.1.11).
- **VRAM detection** is broad: NVIDIA, AMD, Intel Arc, Apple, Ascend, Jetson, and summing across GPUs.
- **Quant choice:** picks the best quant from Q8_0 down to Q2_K. `llmfit download <repo>` picks a GGUF by its real file size against the budget, without reserving room for KV.

**sift** — [github.com/maxgfr/sift](https://github.com/maxgfr/sift) — **NEW**

**v0.5.1 (2026-09-11)**, 7 releases since August, prebuilt binaries and a Homebrew tap, not on crates.io, 0 stars. **The closest peer to hf-fm's `--check-gpu` and `quants --fits`.**

- **`sift fit <hf-repo>[:QUANT] --ctx N`** reads the real GGUF headers over HTTP Range; safetensors repos are sized from the header's `data_offsets` plus `config.json`.
- **Output:** a per-quant table of size, fit (yes / tight / no), measured bits-per-weight and estimated tok/s, with KV at `--ctx` and a recommended quant.
- **Header cache:** in `~/.sift/cache`, revalidated by ETag.
- **`sift inspect`** works remotely too.

**Smaller fit tools since June** (catalog- or config-based):
- `rigspark` 3.0.2: offline catalog, yes / slow / no verdict.
- `ridgepoint`: Rust core, published on PyPI; models serving-engine memory, with GQA and MLA KV.
- `paddock` 0.2.0: Apple Silicon.
- `snug` (GitHub only): `snug recommend <repo>` downloads the largest GGUF that fits, with KV from `config.json` at `--context`.

### E. Adjacent: cache management

**hf-store-rs** — [github.com/dinoml/hf-store-rs](https://github.com/dinoml/hf-store-rs) — **NEW** *(README only)*
**v0.1.0 source tag (2026-07-22)**, no crate, no binaries, no commits since, 0 stars. A library with a thin CLI, `hf-store fetch | inspect | verify | gc plan | gc execute`, working on the canonical HF cache layout (conformance-tested against `huggingface_hub` v1.24.0). The first Rust tool found with HF-cache-aware verify and gc; its implementation depth is unverified.

**llmserve** — [crates.io/crates/llmserve](https://crates.io/crates/llmserve) | [github.com/AlexsJones/llmserve](https://github.com/AlexsJones/llmserve)
crates.io **v0.0.7 (2026-04-12)**; GitHub has v0.0.10 (2026-07-10), ~408 stars. TUI for orchestrating inference servers. It discovers models in LM Studio, llama.cpp, Ollama and llmfit caches, but in the HF hub cache only `mlx-community` repos. No gc or verify.

**diskard** — [github.com/connectwithprakash/diskard](https://github.com/connectwithprakash/diskard)
**v0.2.0 (2026-02-19)**, dormant since March. A general-purpose disk cleaner with the HF cache as 1 of 18 recognizers: scan and bulk delete, no verify, no revision-aware pruning.

**Also seen** (0 stars, small):
- `hf-cache-reader` 0.1.0: read-only HF cache scanner.
- `modelwarden` 0.5.0: inventories Ollama, HF and NAS models by SHA-256 against its own manifests.
- `gguf-janitor`: Windows; dedupes GGUFs across tools with NTFS hard links.

Plus the HF-cache listings in `hf-hub` (`scan_cache`), `mistral.rs` (`cache list`), `modelhub` (`list`) and `rust-hf-downloader` (`hf-cache`).

### Cross-ecosystem context (Python — not scored in the matrix)

- **`hf cache`** — `huggingface_hub`'s cache CLI was redesigned by [RFC #3432](https://github.com/huggingface/huggingface_hub/issues/3432) (closed November 2025, shipped in 1.0.0). At 2.2.0 it has four subcommands:
  - `ls` (filters, sort, JSON);
  - `rm` (repos, revisions, or single files via `hf://` URIs);
  - `prune` (detached revisions, `.incomplete` leftovers, unreferenced shared blobs);
  - `verify` (checksums against the Hub; fails on missing or extra files only when asked).
  `cache verify` and `clean-partial` are hf-fm's analogs.
- **Shared blob store** — since `huggingface_hub` 1.32.0 (2026-09-17), Xet files are stored once at `<cache>/blobs/<2-hex>/<xet_hash>`, and each repo's `blobs/<etag>` becomes a symlink into that store. Any tool that measures or deletes the HF cache now has to know this layout.
- **`hf-mem`** — [github.com/alvarobartt/hf-mem](https://github.com/alvarobartt/hf-mem) v0.5.6 (2026-09-07), ~942 stars: memory estimates from safetensors/GGUF headers read over HTTP Range, the same trick hf-fm's `inspect` uses. Its `--experimental` mode adds the KV cache (sliding-window and hybrid aware) and an MoE breakdown. It installs as an `hf` CLI extension.
- **`accelerate estimate-memory`** — still in accelerate 1.15.0: loads the model on a meta device and reports per-dtype memory. The reference point any VRAM-fit tool is measured against.

### Notable observations

- `hf-hub` has **~560 reverse deps**, dominated by libraries (`tokenizers`, `fastembed`, `lancedb`, `candle-*`). HF's two Rust clients are now one: `hf-hub` 1.0 *is* the `huggingface_hub_rust` code.
- **The inference engines are growing hf-fm's axes inside their own CLIs:**
  - `mistral.rs`: cache list/delete, Range-based UQFF inspection, a fit-and-quant recommender;
  - `apr-cli`: a cross-format `diff`, a fit gate with KV.
  Each serves its own engine's workflow, not the HF repo as such.
- One author (**AlexsJones**) ships both `llmfit` (now one of the most-starred Rust ML tools) and `llmserve`, still as two binaries.
- `rust-hf-downloader` is still the closest direct competitor on download/search, and now writes the real HF cache layout, but remains binary-only with no tensor inspection.

---

## Comparison matrix

Cells: ✓ = present · ◐ = partial · — = absent.

Capability headers (the 13 hf-fm areas above):

| # | Header | Capability |
|---|---|---|
| 1 | MultiDL | multi-connection chunked download with resume + filters/presets |
| 2 | DLFile | single-file download with glob expansion |
| 3 | Inspect | tensor-header inspection (tree, dtypes, sharded aggregation, JSON; remote and cached) |
| 4 | Diff | tensor-layout diff between two models |
| 5 | Du | cache disk-usage drill-down + tree |
| 6 | CacheMgmt | cache delete / clean-partial / gc / verify (SHA256) / path |
| 7 | Search | Hub search with library/pipeline/tag filters |
| 8 | Info | model card / README display |
| 9 | Status | per-repo file-level cache state |
| 10 | ListFiles | remote file listing with sizes + SHA256 |
| 11 | Discover | `model_type` grouping / Hub family discovery / quant siblings |
| 12 | Lib | embeddable Rust library API |
| 13 | VRAMFit | "does this model fit my GPU?" verdict from metadata only (no weight download) |

| Tool | MultiDL | DLFile | Inspect | Diff | Du | CacheMgmt | Search | Info | Status | ListFiles | Discover | Lib | VRAMFit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **hf-fetch-model v0.12.2** *(reference)* | ✓ | ✓ | ✓ 4 fmts remote+cached, tree, dtypes, group-by, quant, sharded, JSON, header cache | ✓ +dtypes, collapse, config | ✓ tree, bytes on disk | ✓ delete/clean-partial/gc/verify/path | ✓ | ✓ | ✓ | ✓ | ✓ +quant siblings | ✓ | ✓ +KV/ctx, quant fit plan |
| hf-hub 1.0.0 | ◐ file-level concurrency | ✓ globs | — | — | ◐ `scan_cache` | — | ✓ `list_models` | ◐ metadata | — | ✓ `list_tree` + LFS sha | — | ✓ | — |
| modelhub 0.2.2 *(new)* | ◐ | ✓ | — | — | ◐ list | ◐ check by size, clear | — | — | ◐ | — | — | ✓ (HF+ModelScope) | — |
| rust-hf-downloader 2.14.0 | ✓ 8 chunks/file | ✓ | — | — | — | ◐ writes/locates HF cache | ✓ TUI + CLI | — | — | ◐ | — | — bin only | — |
| xhf 0.0.2 *(new)* | ✓ resumable chunks | ✓ | — | — | — | — | ◐ | — | — | ◐ tree | — | — bin only | — |
| model-hub 0.1.1 | ✓ concurrent+resume | ◐ no glob | — | — | — | — | — | — | — | — | — | ✓ (HF+ModelScope) | — |
| rustyface 0.1.3 | ◐ concurrent, no chunks | — repo-only | — | — | — | — | — | — | — | — | — | — bin only | — |
| facecrab 0.1.6 | — single-conn | ✓ resolve-by-id | — | — | — | ◐ registry.toml | — | — | — | — | — | ✓ | — |
| ogenius 0.1.6 | — | ✓ via facecrab | — | — | — | — | — | — | — | — | — | ◐ via facecrab | — |
| hugging-face-client 0.6.0 | ◐ basic | ◐ basic | — | — | — | — | — | — | — | ◐ | — | ✓ | — |
| hf-hub-enfer 0.3.2 *(abandoned)* | ◐ | ◐ | — | — | — | — | — | — | — | ◐ | — | ✓ | — |
| safetensors-browser 0.2.0 | — | — | ✓ remote via Range, safetensors only, sharded, AWQ/GPTQ | — | — | — | — | — | — | — | — | — bin only | — |
| safetensors_explorer 0.2.0 | — | — | ✓ local TUI tree, sharded, fuzzy | — | — | — | — | — | — | — | — | — bin only | — |
| gguf-rs-lib 0.3.2 *(ThreatFlux)* | — | — | ◐ GGUF only, local | ◐ `compare`, GGUF, local | — | — | — | — | — | — | — | ✓ | — |
| gguf-rs 0.1.8 *(Zack Shen)* | — | — | ◐ GGUF only, local CLI | — | — | — | — | — | — | — | — | ✓ | — |
| inspector-gguf 0.3.1 | — | — | ◐ GGUF only, GUI+CLI | — | — | — | — | — | — | — | — | ✓ | — |
| llm_hunter 0.3.5 | — | — | ◐ forensic ID (GGUF + entropy) | — | — | — | — | — | — | — | — | ✓ lib only | — |
| safetensors-cli 0.1.0 | — | — | ◐ minimal | — | — | — | — | — | — | — | — | — bin only | — |
| safetensors 0.8.0 | — | — | — no CLI | — | — | — | — | — | — | — | — | ✓ format only | — |
| candle 0.11.0 | — via hf-hub | — via hf-hub | ◐ `tensor-tools`, 6 fmts, local | — | — | — | — | — | — | — | — | ✓ framework | — |
| mistral.rs 0.9.4 | ◐ via hf-hub | ◐ `--quant` file pick | ◐ `uqff inspect`, UQFF only, remote | — | ◐ `cache list` | ◐ `cache delete` | — | — | — | — | — | ✓ | ✓ `tune`, config.json-based, multi-GPU |
| apr-cli 0.70.3 | ◐ `apr pull`, own cache | — | ◐ local, APR/GGUF/safetensors | ✓ `apr diff`, 3 fmts, local | — | ◐ own cache, BLAKE3 verify | — | — | — | — | — | ✓ via aprender | ✓ `serve plan`, config.json + GQA KV |
| llmfit 1.1.17 | — | ◐ `download`, GGUF by size | — | — | — | — | ◐ `hf-search` | — | — | — | ◐ model browser | ✓ lib+bin | ✓ multi-vendor, KV/ctx, quant pick |
| sift 0.5.1 *(new)* | — | — | ◐ remote GGUF/safetensors | — | — | — | — | — | — | — | — | — not published | ✓ real headers, KV/ctx, per-quant |
| burn 0.22.0 | — | — | — | — | — | — | — | — | — | — | — | ✓ framework | — |
| hf-store-rs 0.1.0 *(new, README only)* | — | ◐ fetch | ◐ | — | — | ◐ verify, gc | — | — | — | — | — | ✓ | — |
| llmserve 0.0.7 | — | — | — | — | ◐ discovery | ◐ multi-cache discovery | — | — | ◐ status | — | ◐ across caches | — bin only | — |
| diskard 0.2.0 | — | — | — | — | ◐ scan | ◐ delete-only (no verify/gc-by-rev) | — | — | — | — | — | — bin only | — |

---

## Gaps unique to hf-fm in the surveyed Rust ecosystem

Narrower than in June, and stated as such:

- **Remote inspection of all four tensor formats, in one CLI and one library.** Others now read headers over HTTP Range, each for one format family: `safetensors-browser` (safetensors), `sift` (GGUF and safetensors, for its fit verdict) and `mistral.rs` (UQFF). None reads `.npz` or `.pth` remotely. None pairs remote reads with JSON output, dtype histograms, `--group-by` byte splits and sibling-collapsing trees, or exposes them as a library. Locally, candle's `tensor-tools` reads six formats.
- **HF-aware cache lifecycle with LFS-checksum verify.** `hf-store-rs` (v0.1.0, README only) is the first Rust tool found with HF-cache-aware verify and gc. Others list or delete: `hf-hub` `scan_cache`, `mistral.rs` `cache list/delete`, `modelhub` (completeness by size). None of those verifies SHA256 against the Hub's LFS metadata as `cache verify` does, and none has `clean-partial` or revision-aware gc.
- **`diff` between two Hub repos without downloading them.** Diffing is no longer unique in kind: `apr diff` compares two local model files across three formats, and `gguf-cli compare` two local GGUFs. hf-fm's `diff` is still the only one comparing two *remote* repos from their headers, with dtype histograms and pattern collapsing, plus `diff-config` at the architecture level.
- **`du --tree` over the HF cache with index drill-down**, counting each repo's bytes on disk. Others list per-repo sizes; none renders the tree or drills down.
- **Quantized-sibling aggregation with an MoE offload plan** (`quants --fits`). `sift` and `llmfit` pick a quant *within* a repo or catalog; none found gathers a base model's sibling repos and plans `--n-cpu-moe` offload from their real expert tensors.

Library + CLI parity is no longer unique: `modelhub` (HF + ModelScope), `llmfit` (`llmfit-core`) and `hf-hub` (with its unpublished `hfrs`) all pair the two. hf-fm's distinction is narrower: the same inspect, cache and fit functions its CLI uses are public API.

The closest competitor footprint is now a set of six or more separate tools whose union still does not cover four-format remote inspection, remote-repo `diff`, LFS-verified HF cache management or sibling-quant offload planning:
- `hf-hub` (lib, search, download);
- `rust-hf-downloader` (download CLI);
- `safetensors-browser` (remote safetensors inspect);
- `sift` or `llmfit` (fit);
- `apr` (local diff);
- `hf-store-rs` (verify/gc).

## VRAMFit — a crowded niche

In June the question was whether hf-fm's `--check-gpu` had peers; in October there are at least four, and one of them reads real headers too.

| | hf-fm `inspect --check-gpu` / `quants --fits` | sift `fit` | llmfit | mistral.rs `tune` | apr `serve plan` |
|---|---|---|---|---|---|
| Status | shipped v0.10.1; KV v0.10.4; `quants --fits` v0.12.1 | shipped 2026-08 | shipped, very active | shipped | shipped |
| Weights from | **exact tensor byte lengths in the file headers**, remote or cached, all four formats | **real GGUF headers over Range**; safetensors `data_offsets` | params × bits-per-param, from a catalog | `config.json` dimensions per quant level | `config.json` param estimate, always at Q4 |
| KV-cache | **✓ at `--context N`**: GQA, sliding window, hybrid Mamba/attention; MLA detected and skipped | ✓ at `--ctx` | ✓ at a chosen context and KV quant; hybrid-aware | ✓ reports the largest context that fits | ✓ plain GQA, f16 |
| Quantization | exact footprints; `quants --fits` plans across sibling repos with an `--n-cpu-moe` offload plan | per-quant table, measured bits-per-weight, recommends one | picks best quant Q8_0 → Q2_K | Fits / Hybrid / TooLarge per level; `--quant auto` applies it | label only |
| Live VRAM source | `hypomnesis` (NVML / DXGI / `nvidia-smi` / Metal), one device per run | not verified | multi-vendor, sums GPUs | its runtime's device mapping, multi-GPU | `nvidia-smi memory.total` (nameplate, first GPU) |
| Shape | flags on `inspect` and `quants` | standalone CLI with an ETag-validated header cache | TUI + CLI + library | subcommand of an inference engine | gate inside an inference pipeline |
| Primary question | "should I download this model, and which sibling quant fits?" | "which quant of this repo fits, and how fast?" | "which models fit this machine?" | "how should I run this model here?" | "should I serve this model?" |

**Where hf-fm stands:**
- **Header-exact sizing.** hf-fm and `sift` are the two that size weights from real file headers rather than parameter counts.
- **What only hf-fm has:**
  - an architecture-aware KV model (sliding window, hybrid Mamba/attention) across all four formats;
  - planning across a base model's *sibling repos* rather than within one, with MoE expert offload.
- **What the others have that hf-fm lacks:**
  - `sift`: throughput estimates and a measured bits-per-weight;
  - `llmfit`: breadth (multi-vendor, multi-GPU, a model catalog) and by far the largest user base;
  - `mistral.rs`: the verdict wired straight into the engine that runs the model;
  - hf-fm's multi-GPU sweep is planned for v0.14.0.

---

## Strategic read

The `hf-hub` reverse-dependency graph (~560 crates, mostly libraries) still shows library composition rather than user-facing tools, and HF consolidated its Rust story into a single client. Around it, hf-fm's axes are being picked up one or two at a time:
- **inference engines** (`mistral.rs`, `apr`) grow cache, inspect, diff and fit commands for their own workflows;
- **standalone fit tools** (`llmfit`, `sift`) compete on the VRAM question;
- **downloaders** (`rust-hf-downloader`, `modelhub`, `xhf`) add cache awareness;
- **new single-purpose crates** (`hf-store-rs`, `safetensors-browser`) take one corner each.

None is a library-and-CLI superset.

hf-fm's position is the integrated one: one CLI and one library covering download, four-format remote inspection, remote diff, HF-cache lifecycle and fit planning across quantized siblings. The individual niches now have dedicated tools, and in fit planning a very popular one, so the integration and the header-exact precision are what to defend, not "only one."

---

## Sources

- [hf-hub GitHub](https://github.com/huggingface/hf-hub) / [crates.io](https://crates.io/crates/hf-hub) (1.0.0) / [reverse deps API](https://crates.io/api/v1/crates/hf-hub/reverse_dependencies) / [PR #150](https://github.com/huggingface/hf-hub/pull/150) / [PR #193](https://github.com/huggingface/hf-hub/pull/193)
- [huggingface_hub_rust](https://github.com/huggingface/huggingface_hub_rust) / [huggingface-hub crate](https://crates.io/crates/huggingface-hub)
- [rust-hf-downloader on lib.rs](https://lib.rs/crates/rust-hf-downloader) / [crates.io](https://crates.io/crates/rust-hf-downloader)
- [modelhub](https://github.com/di-osc/modelhub) / [xhf](https://github.com/3axap4eHko/xhf) / [model-hub](https://github.com/waitsalt/model-hub)
- [rustyface](https://github.com/AspadaX/RustyFace) / [facecrab](https://crates.io/crates/facecrab) / [ogenius](https://crates.io/crates/ogenius) / [rusty-genius](https://github.com/tmzt/rusty-genius)
- [hugging-face-client](https://crates.io/crates/hugging-face-client) / [hf-hub-enfer](https://crates.io/crates/hf-hub-enfer)
- [safetensors-browser](https://github.com/danieldk/safetensors-browser) / [safetensors_explorer](https://github.com/EricLBuehler/safetensors_explorer)
- [gguf-rs-lib / ThreatFlux](https://github.com/ThreatFlux/gguf) / [gguf-rs (Zack Shen)](https://github.com/zackshen/gguf) / [inspector-gguf](https://github.com/FerrisMind/inspector-gguf)
- [llm_hunter](https://github.com/jpegleg/llm_hunter) / [safetensors-cli](https://github.com/gzsombor/safetensors-cli) / [safetensors](https://github.com/safetensors/safetensors)
- [candle](https://github.com/huggingface/candle) ([tensor-tools](https://github.com/huggingface/candle/tree/main/tensor-tools)) / [mistral.rs releases](https://github.com/EricLBuehler/mistral.rs/releases)
- [apr-cli on docs.rs](https://docs.rs/crate/apr-cli/latest) / [aprender monorepo](https://github.com/paiml/aprender)
- [burn](https://github.com/tracel-ai/burn) / [llm crate (graniet/llm)](https://github.com/graniet/llm)
- [llmfit](https://github.com/AlexsJones/llmfit) / [sift](https://github.com/maxgfr/sift) / [llmserve](https://github.com/AlexsJones/llmserve) / [diskard](https://github.com/connectwithprakash/diskard) / [hf-store-rs](https://github.com/dinoml/hf-store-rs)
- Python context: [huggingface_hub cache CLI](https://github.com/huggingface/huggingface_hub/blob/v2.2.0/src/huggingface_hub/cli/cache.py) / [RFC #3432](https://github.com/huggingface/huggingface_hub/issues/3432) / [shared blob store](https://github.com/huggingface/huggingface_hub/blob/v2.2.0/src/huggingface_hub/utils/_shared_blobs.py) / [hf-mem](https://github.com/alvarobartt/hf-mem) / [accelerate estimate-memory](https://huggingface.co/docs/accelerate/main/en/usage_guides/model_size_estimator)
