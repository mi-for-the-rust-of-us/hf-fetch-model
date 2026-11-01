# hf-fetch-model

[![CI](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/actions/workflows/ci.yml/badge.svg)](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/hf-fetch-model)](https://crates.io/crates/hf-fetch-model)
[![docs.rs](https://img.shields.io/docsrs/hf-fetch-model)](https://docs.rs/hf-fetch-model)
[![Rust](https://img.shields.io/badge/rust-1.91%2B-orange)](https://www.rust-lang.org)
[![License](https://img.shields.io/crates/l/hf-fetch-model)](LICENSE-MIT)

A Rust library and CLI for downloading and inspecting HuggingFace models. Multi-connection parallel downloads, file filtering, checksum verification, retry — plus remote tensor-header inspection (`.safetensors`, NumPy `.npz`, `.gguf`, PyTorch `.pth`) and structural comparison between models, all without downloading weight data.

> 🆕 **`0.12.5` tells you which part of your cache is the same bytes twice, and what moving a repo to another disk really freed.** On Windows `hf-hub` leaves a second full copy of every snapshot file beside its blob; on the maintainer's cache that was 154 GiB of 763 GiB, and no command said so. `hf-fm du` now has a `DUP` column, a footer and `--sort dup`, `du <repo>` splits the gap into duplicated copies, orphan blobs, partial downloads and bookkeeping, and `--verify-duplicates` hashes each copy against its blob. `du` also lists dataset and Space repos (`datasets/org/name`), takes bare names like `gpt2` and index numbers everywhere (`status` included), and shows a repo moved behind a junction or symlink with its target and a per-disk total. `cache delete` takes several repos, `--dry-run`, and `--with-target` to remove a relocated repo's bytes along with its link (by default only the link goes). See the [CHANGELOG](CHANGELOG.md).

> 🚀 **`0.12.4` puts a floor under hf-fm's TLS stack, so a library user can no longer resolve a certificate verifier that accepts distrusted intermediates.** RUSTSEC-2026-0382 (`rustls-webpki` below 0.103.16) reaches every connection hf-fm makes, and `rustls` 0.23.46 alone still allowed it. `Cargo.toml` now requires both, with no code behind either, because Cargo never reads a dependency's lockfile and a lockfile only protects the binary. The same release drops the `zip` dev-dependency (three more advisories), moves `hypomnesis` to 0.2.15 so `--check-gpu` reports a broken NVIDIA driver by its cause instead of as "no GPU", and no longer names `DXGI` in its Linux no-GPU message. See the [CHANGELOG](CHANGELOG.md).

## Table of contents

- [Install](#install)
- [Commands](#commands)
- [Try it](#try-it)
- [Inspect & compare](#inspect--compare)
- [Quant fit planning](#quant-fit-planning)
- [Disk usage](#disk-usage)
- [Library quick start](#library-quick-start)
- [Documentation](#documentation)
- [Used by](#used-by)
- [License](#license)
- [Development](#development)

> **New to hf-fm?**
> - **I want to know which model to download** → [Inspect before you download](docs/tutorials/inspect-before-downloading.md): read tensor shapes, size, and architecture without pulling a single weight byte.
> - **I want to know which quant fits my GPU** → [Pick a quant that fits before you download it](docs/tutorials/pick-a-quant-that-fits.md): aggregate a model's quant siblings into one table and plan `MoE` CPU offload before choosing.
> - **I want to manage the models on my disk** → [Clean up before your disk fills](docs/tutorials/clean-up-before-your-disk-fills.md): see what the cache holds, then reclaim space safely.
>
> Common questions live in the [FAQ](docs/FAQ.md); every flag is in the [CLI Reference](docs/cli-reference.md).

## Install

```sh
cargo install hf-fetch-model --features cli
```

Verify the install:

```sh
hf-fm --version
```

### Upgrading from a previous version

`cargo install` skips the build when any version of the binary is already on disk, even if crates.io has a newer release. Use `--force` to upgrade:

```sh
cargo install hf-fetch-model --features cli --force
hf-fm --version
```

Without `--force`, a stale local registry index can cause `cargo install` to exit `0` silently — the install command appears to succeed but the binary on `PATH` is unchanged. See [FAQ → *How do I upgrade hf-fm?*](docs/FAQ.md#how-do-i-upgrade-hf-fm-why-does-cargo-install-silently-keep-the-old-version) for more.

## Commands

| Command | Description |
|---------|-------------|
| `hf-fm <REPO_ID>` *(default)* | Download a model (multi-connection, auto-tuned) |
| `hf-fm cache clean-partial` | Remove interrupted downloads' temp files: hf-fm's `.chunked.part`, and the `.incomplete` files `hf-hub` / `huggingface_hub` leave (a download still running is skipped) |
| `hf-fm cache dedupe [REPO_ID\|N]...` | Reclaim the duplicated copies (Windows): replace each snapshot file that equals its blob with a hard link to it, keeping every model; `--dry-run` to preview |
| `hf-fm cache delete <REPO_ID\|N>...` | Delete cached repos: several per call, `--dry-run` to preview; a repo moved to another disk behind a link loses the link only |
| `hf-fm cache gc --older-than/--max-size` | Garbage-collect cached models by age and/or size budget |
| `hf-fm cache path <REPO_ID\|N>` | Print snapshot directory path (for scripting) |
| `hf-fm cache verify <REPO_ID\|N>` | Re-verify SHA256 digests of cached files against HF LFS metadata |
| `hf-fm diff <REPO_A> <REPO_B>` | Compare tensor layouts between two models |
| `hf-fm diff-config <REPO_A> <REPO_B>` | Compare `config.json` architecture fields between two models |
| `hf-fm discover` | Find new model families on the Hub |
| `hf-fm download-file <REPO_ID> <FILE>` | Download a single file (or glob pattern) |
| `hf-fm du [REPO_ID\|N]` | Show cache disk usage (by name or `#` index): models, datasets and Spaces, with the duplicated copies named |
| `hf-fm inspect <REPO_ID> [FILE]` | Inspect tensor headers (names, shapes, dtypes) without downloading weights — safetensors/NPZ/GGUF/PTH remote or cached; add `--check-gpu [--context N]` for a GPU-fit verdict (with KV-cache budgeting), `--pick` to choose the file interactively, or `--cache-headers` to keep parsed remote headers on disk for the next run |
| `hf-fm list-families` | List model families in local cache |
| `hf-fm list-files <REPO_ID>` | List remote files (sizes, SHA256) without downloading |
| `hf-fm peek <REPO_ID> <FILE>` | Print a small file's content (config, README, `.gz` sidecar) without downloading — `--head`/`--tail` bound the read, `--gunzip` decodes, `--max` caps the size (no tensor formats — use `inspect` for those) |
| `hf-fm quants <REPO_ID>` | Aggregate a base model's quant sibling repos into one sorted table; add `--fits <SIZE> [--reserve <SIZE>]` for an offload-aware VRAM fit plan, and `--cache-headers` to reuse the headers it reads |
| `hf-fm search <QUERY>` | Search the HuggingFace Hub for models |
| `hf-fm status [REPO_ID]` | Per-repo: per-file download status (complete / partial / missing / excluded). With no `REPO_ID`: a table of all cached repos, each marked `ok` or `PARTIAL`. |

See [CLI Reference](docs/cli-reference.md) for all flags and output examples.

## Try it

```
$ hf-fm search mistral,3B,instruct
Models matching "mistral,3B,instruct" (by downloads):

  hf-fm mistralai/Ministral-3-3B-Instruct-2512           (159,700 downloads)
  hf-fm mistralai/Ministral-3-3B-Instruct-2512-BF16      (62,600 downloads)
  hf-fm mistralai/Ministral-3-3B-Instruct-2512-GGUF      (32,700 downloads)
  ...

$ hf-fm search llama --tag gguf --limit 3
Models matching "llama" (by downloads):

  hf-fm bartowski/Llama-3.2-3B-Instruct-GGUF             (489,856 downloads)  [text-generation]
  hf-fm bartowski/Meta-Llama-3.1-8B-Instruct-GGUF        (237,791 downloads)  [text-generation]
  hf-fm MaziyarPanahi/Meta-Llama-3.1-8B-Instruct-GGUF    (184,847 downloads)  [text-generation]

$ hf-fm search fp4 --tag bitsandbytes --show tags,size --limit 3
Models matching "fp4" (by downloads):

  hf-fm HF-Quantization/Llama-3.2-1B-BNB-FP4-BF16     (5 downloads)  [transformers, text-generation]  1.50 GiB  tags: transformers, safetensors, llama, 4-bit, bitsandbytes
  hf-fm saxman/Qwen3-Coder-30B-A3B-Instruct-bnb-fp4   (4 downloads)  [transformers, text-generation]  18.20 GiB  tags: transformers, safetensors, qwen3, 4-bit, bitsandbytes
  hf-fm ema1234/qwen_mcqa_bnb_fp4                     (2 downloads)  [transformers, text-generation]  548.00 MiB  tags: transformers, safetensors, qwen3, 4-bit, bitsandbytes

$ hf-fm search mistralai/Ministral-3-3B-Instruct-2512 --exact
Exact match:

  hf-fm mistralai/Ministral-3-3B-Instruct-2512           (159,700 downloads)

  License:      apache-2.0
  Pipeline:     text-generation
  Library:      vllm
  Languages:    en, fr, es, de, it, pt, nl, zh, ja, ko, ar

$ hf-fm list-files mistralai/Ministral-3-3B-Instruct-2512 --preset safetensors
  File                                               Size      SHA256
  model-00001-of-00002.safetensors                 3.68 GiB    a1b2c3d4e5f6
  model-00002-of-00002.safetensors                 2.88 GiB    f6e5d4c3b2a1
  config.json                                        856 B     —
  ...
  7 files, 6.57 GiB total

$ hf-fm mistralai/Ministral-3-3B-Instruct-2512 --preset safetensors --dry-run
  Repo:     mistralai/Ministral-3-3B-Instruct-2512
  Revision: main

  File                                               Size      Status
  model-00001-of-00002.safetensors                 3.68 GiB    to download
  model-00002-of-00002.safetensors                 2.88 GiB    to download
  ...
  Total: 6.57 GiB (7 files, 0 cached, 7 to download)

  Recommended config:
    concurrency:        2
    connections/file:   8
    chunk threshold:  100 MiB

$ hf-fm mistralai/Ministral-3-3B-Instruct-2512 --preset safetensors
Downloaded to: ~/.cache/huggingface/hub/models--mistralai--Ministral-3-3B.../snapshots/...
  6.57 GiB in 18.2s (369.1 MiB/s)

# Download to flat layout (files directly in ./models/)
$ hf-fm mistralai/Ministral-3-3B-Instruct-2512 --preset safetensors --flat --output-dir ./models

# Download sharded PyTorch files by glob
$ hf-fm download-file org/model "pytorch_model-*.bin"
```

## Inspect & compare

```
$ hf-fm inspect EleutherAI/pythia-1.4b model.safetensors --cached --filter "layers.0."
  Repo:     EleutherAI/pythia-1.4b
  File:     model.safetensors
  Source:   cached
  Header:   41.5 KiB (JSON), 2.73 GiB total
  Metadata: format=pt

  Tensor                                             Dtype    Shape                    Size     Params
  gpt_neox.layers.0.attention.dense.weight           F16      [2048, 2048]         8.00 MiB       4.2M
  gpt_neox.layers.0.mlp.dense_h_to_4h.weight         F16      [8192, 2048]        32.00 MiB      16.8M
  ...
  ──────────────────────────────────────────────────────────────────────────────────────────────────────
  Showing 15 of 364 tensors matching filter "layers.0.".
  Param counts: 54.6M matching filter, 1.52B total.

$ hf-fm inspect google/gemma-4-E2B-it model.safetensors --tree --filter "embed"
  Repo:     google/gemma-4-E2B-it
  File:     model.safetensors
  Source:   remote (4 range requests, 257.8 KiB fetched)
  Header:   257.8 KiB (JSON), 9.54 GiB total
  Metadata: format=pt

  └── model.
      ├── embed_audio.embedding_projection.weight   BF16  [1536, 1536]  4.50 MiB
      ├── embed_vision.embedding_projection.weight  BF16  [1536, 768]   2.25 MiB
      ├── language_model.
      │   ├── embed_tokens.weight            BF16  [262144, 1536]  768.00 MiB
      │   └── embed_tokens_per_layer.weight  BF16  [262144, 8960]    4.38 GiB
      └── vision_tower.patch_embedder.
          ├── input_proj.weight         BF16  [768, 768]        1.12 MiB
          └── position_embedding_table  BF16  [2, 10240, 768]  30.00 MiB
  Showing 6 of 2011 tensors matching filter "embed".
  Param counts: 2.77B matching filter, 5.12B total.

$ hf-fm inspect poolside/Laguna-XS-2.1-GGUF Q4_K_M.gguf --group-by 'blk.*.ffn_*_exps.weight'
  Group                    Tensors        Size  Percent
  MATCHED (blk.*.ffn_*_exps.weight)   117   17.68 GiB    93.6%
  OTHER                                561    1.20 GiB     6.4%
  ────────────────────────────────────────────────────────────
  TOTAL                                678   18.88 GiB   100.0%

  per-MoE-layer expert cost: 0.45 GiB (39 layers)

$ hf-fm diff RedHatAI/Llama-3.2-1B-Instruct-FP8 casperhansen/llama-3.2-1b-instruct-awq --cached --summary
  A: RedHatAI/Llama-3.2-1B-Instruct-FP8
  B: casperhansen/llama-3.2-1b-instruct-awq
  ──────────────────────────────────────────────────────────────────────────────────────────────
  A: 371 tensors | B: 370 tensors | only-A: 337 | only-B: 336 | differ: 34 | match: 0

$ hf-fm diff openai/gpt-oss-20b openai/gpt-oss-120b --dtypes
  A: openai/gpt-oss-20b
  B: openai/gpt-oss-120b

  Dtype  A Tensors     A Size  B Tensors      B Size      Δ Size
  U8           192  18.91 GiB        288  113.46 GiB  +94.55 GiB
  BF16         630   6.72 GiB        942    8.07 GiB   +1.35 GiB
  ──────────────────────────────────────────────────────────────
  A: 822 tensors, 25.63 GiB | B: 1230 tensors, 121.54 GiB | Δ: +408 tensors, +95.90 GiB

$ hf-fm diff openai/gpt-oss-20b openai/gpt-oss-120b --collapse --limit 3
  A: openai/gpt-oss-20b
  B: openai/gpt-oss-120b

  Only in B (408 tensors, 31 patterns):
  Pattern                                           Tensors       Bytes
  block.{N}.mlp.mlp{N}_weight.blocks                     24   17.80 GiB
  model.layers.{N}.mlp.experts.gate_up_proj_blocks       12   11.87 GiB
  model.layers.{N}.mlp.experts.down_proj_blocks          12    5.93 GiB
    … showing 3 of 31 (limit 3)

  Dtype/shape differences (384 tensors, 13 patterns):
  Pattern                                           Tensors     A Bytes     B Bytes      Δ Bytes
  block.{N}.mlp.mlp{N}_weight.blocks                     48    8.90 GiB   35.60 GiB   +26.70 GiB
  model.layers.{N}.mlp.experts.gate_up_proj_blocks       24    5.93 GiB   23.73 GiB   +17.80 GiB
  model.layers.{N}.mlp.experts.down_proj_blocks          24    2.97 GiB   11.87 GiB    +8.90 GiB
    … showing 3 of 13 (limit 3)

  ──────────────────────────────────────────────────────────────────────
  only-A: 0 tensors (0 patterns) | only-B: 408 tensors (31 patterns) | differ: 384 tensors (13 patterns)

$ hf-fm diff-config openai/gpt-oss-20b openai/gpt-oss-120b
  A: openai/gpt-oss-20b
  B: openai/gpt-oss-120b

  Field              A                                                              B
  num_hidden_layers  24                                                             36
  layer_types        sliding_attention, full_attention, sliding_attention, full_a…  sliding_attention, full_attention, sliding_attention, full_a…

  2 of 23 fields differ

$ hf-fm inspect meta-llama/Llama-3.2-3B --cached --check-gpu --context 32768
  ...existing tensor table...

  Model weights:  5.98 GiB  (BF16, 3.21B params)
  KV cache @ ctx=32768:  3.50 GiB  (BF16)
  Total:          9.48 GiB  (weights + KV)
  GPU 0:          NVIDIA GeForce RTX 5060 Ti — 15.93 GiB VRAM
                  free: 13.68 GiB, used: 2.25 GiB
  Fit:            ✓ 4.20 GiB headroom (weights + KV; runtime extra)
  Spilling:       not sampled (platform supports detection)
```

Inspect reads tensor metadata via HTTP Range requests — no weight data downloaded: typically 3–4 requests per `.safetensors` file (2 of them the reader's one-time access probe), a handful (reported live on the `Source:` line, e.g. `remote (6 range requests, 136.0 KiB fetched)`) per NumPy `.npz` archive (remote NPZ since v0.11.0), a similar handful per `.gguf` file (remote GGUF since v0.11.2, e.g. `remote (7 range requests, 1.94 MiB fetched)` on an 84 MiB quantized model, `--tree`/`--dtypes` included, and since v0.12.3 under 20 even for a 12 MiB tokenizer-heavy header that took 201: see the [before-and-after measurements](docs/case-studies/remote-gguf-read-ahead.md)), and likewise per `.pth` checkpoint (remote PTH since v0.11.4, e.g. `remote (9 range requests, 132.0 KiB fetched)` on a 364 MiB PyTorch `state_dict` — only the ZIP-archived `data.pkl` pickle stream is read, never the tensor-data files). When a repo has many tensor files, `--list` prints a numbered table (pass the `#` back as the `FILE` argument) and `--pick` chooses interactively, narrowing first by a case-insensitive substring — both cover every format inspect reads (`.safetensors` / `.gguf` / `.npz` / `.pth`). The `--tree` flag shows the hierarchical namespace with numeric sibling groups auto-collapsed to `[0..N]` for structural discovery. The `--group-by <PATTERN>` flag rolls tensors up into a MATCHED / OTHER byte split against a glob pattern instead — the tool for "what fraction of this file is the MoE expert weights" (`blk.*.ffn_*_exps.weight` on a GGUF checkpoint), with a `per-MoE-layer expert cost` line once the matched tensor names carry a single, unambiguous layer index. Add `--cache-headers` to persist a remote header to a local sidecar, keyed on `(repo, revision, filename, etag)`, so a repeat inspect of the same file is free (`Source: cached header (age: ...)`) — off by default, so a plain remote inspect never touches local disk without it. The `--check-gpu` flag adds a one-line GPU-fit verdict using [`hypomnesis`](https://crates.io/crates/hypomnesis) (NVML on Linux/Windows, DXGI on Windows, Metal on macOS); composes with `--json`. Add `--context N` to fold in the KV cache at a context length and get a real `weights + KV` verdict — the difference between "fits" and "OOM at token 8000" on a consumer card. The estimate is parameter-driven from the model's `config.json` (`GQA`, sliding-window, and hybrid Mamba/attention all handled; `MLA` is detected and skipped); see the [FAQ entry on GPU fit](docs/FAQ.md#how-do-i-know-if-a-model-fits-on-my-gpu) for the formula and its limitations. Diff compares tensor names, dtypes, and shapes between any two models (remote or cached); `--dtypes` swaps the per-tensor body for a side-by-side per-dtype histogram with a signed Δ Size column — the high-leverage view for scaled-sibling pairs. `--collapse` groups only-A / only-B / dtype-shape-differences by numeric-segment pattern (`model.layers.{N}.mlp.gate_proj.weight`) into a `Pattern / Tensors / Bytes` table, the built-in counterpart to the `jq` recipe below. See the [FAQ entry on comparing two models](docs/FAQ.md#how-do-i-compare-two-huggingface-models-structurally) for that `jq` recipe, which uses the `byte_count` field in `--json` output for cases `--collapse`'s digit-run heuristic doesn't cover. `diff-config` complements `diff` at the architecture level: a field-by-field comparison of `config.json` (layer counts, hidden size, GQA/sliding-window/hybrid-layout fields) instead of tensor headers — the tool to reach for when `diff --dtypes` shows a size jump and you want to know *why* (more layers? wider hidden dim? a different attention pattern?) without eyeballing two raw `config.json` files.

## Quant fit planning

```
$ hf-fm quants google/gemma-2-2b-it
Searching for quant siblings of google/gemma-2-2b-it...
8 repos found, 0 verified via GGUF backlink

  ARTIFACT                                     SIZE  REPO                                      BITS
  gemma-2-2b-it.IQ1_S.gguf               793.61 MiB  MaziyarPanahi/gemma-2-2b-it-GGUF          ~1.6
  gemma-2-2b-it.Q2_K.gguf                  1.15 GiB  MaziyarPanahi/gemma-2-2b-it-GGUF          ~2.6
  gemma-2-2b-it-IQ3_M.gguf                 1.30 GiB  bartowski/gemma-2-2b-it-GGUF              ~3.7
  ...
  gemma-2-2b-it-f32.gguf                   9.74 GiB  bartowski/gemma-2-2b-it-GGUF              ~32.0

$ hf-fm quants google/gemma-2-2b-it --fits 9.5GiB
...
  gemma-2-2b-it.fp16.gguf                  4.88 GiB   4.88 GiB  full GPU
  gemma-2-2b-it-f32.gguf                   9.74 GiB          —  does not fit (no MoE experts to offload)
```

`quants` aggregates a base model's quant sibling repos — discovered by searching the Hub for the base model's short name and keeping results whose repo ID contains it, cross-checked against each `.gguf` candidate's own `general.source.url` / `general.base_model.*.repo_url` metadata backlink where present. No dedicated Hub endpoint exists for "sibling repos", so this is best-effort: a candidate with no backlink, or whose backlink check itself failed (network error, gated repo), is still listed rather than silently dropped — only an explicit backlink *mismatch* excludes a candidate. `--fits <SIZE>` adds a `RESIDENT`/`PLAN` column pair; candidates that already fit under budget are never inspected (`full GPU`, no header fetch), and only over-budget `MoE` `GGUF` files trigger a header fetch to compute a `--n-cpu-moe N` offload plan against the internal `blk.*.*_exps.weight` expert-tensor pattern — the plan a naive size-vs-budget comparison would miss, since a checkpoint 40% over the raw VRAM budget can still be viable once its expert tensors (touched only a few times per token) move to system RAM. `--reserve <SIZE>` carves out headroom (KV cache, runtime) from the budget before the comparison. See [Pick a quant that fits before you download it](docs/tutorials/pick-a-quant-that-fits.md) for the walkthrough this feature was built from, and the [FAQ](docs/FAQ.md#what-fraction-of-a-gguf-file-is-the-moe-expert-weights) for the underlying `inspect --group-by` rollup.

## Disk usage

```
$ hf-fm du
Cache: /home/user/.cache/huggingface/hub

    #        SIZE  REPO                                             FILES
    1    4.89 GiB  google/gemma-2-2b-it                                 8
    2    2.73 GiB  EleutherAI/pythia-1.4b                               4  ●
    3  360.18 MiB  google/gemma-scope-2b-pt-res                         2
  ───────────────────────────────────────────────────────────────────────
  7.97 GiB  total (3 repos, 14 files)
  ● = partial downloads

$ hf-fm du 2
Cache: /home/user/.cache/huggingface/hub

  EleutherAI/pythia-1.4b:

    #      SIZE  FILE
    1  2.02 MiB  tokenizer.json
    2     570 B  config.json
    3     396 B  tokenizer_config.json
    4      99 B  special_tokens_map.json
  ──────────────────────────────────────
  2.02 MiB  listed above (4 files)
  2.73 GiB  total on disk

  duplicated copies 0 B · orphan blobs 0 B · partial downloads 2.73 GiB · bookkeeping 40 B

  ● partial downloads — run `hf-fm status EleutherAI/pythia-1.4b` for details

$ hf-fm du --age
Cache: /home/user/.cache/huggingface/hub

    #        SIZE  REPO                                             FILES  AGE
    1    4.89 GiB  google/gemma-2-2b-it                                 8  2 days ago
    2    2.73 GiB  EleutherAI/pythia-1.4b                               4  1 month ago      ●
    3  360.18 MiB  google/gemma-scope-2b-pt-res                         2  3 months ago
  ────────────────────────────────────────────────────────────────────────────────────────
  7.97 GiB  total (3 repos, 14 files)
  ● = partial downloads

$ hf-fm du --tree
Cache: /home/user/.cache/huggingface/hub

  ├── google/gemma-2-2b-it  .  .  .  .  .  .  .  .  .    4.89 GiB  (8 files)
  │   ├── model-00001-of-00002.safetensors               4.65 GiB
  │   ├── model-00002-of-00002.safetensors             229.54 MiB
  │   ├── tokenizer.json                                16.71 MiB
  │   ├── tokenizer_config.json                          45.9 KiB
  │   ├── model.safetensors.index.json                   23.7 KiB
  │   ├── config.json                                       838 B
  │   ├── special_tokens_map.json                           636 B
  │   └── generation_config.json                            187 B
  ├── EleutherAI/pythia-1.4b   .  .  .  .  .  .  .  .    2.73 GiB  (4 files)  ●
  │   ├── tokenizer.json                                 2.02 MiB
  │   ├── config.json                                       570 B
  │   ├── tokenizer_config.json                             396 B
  │   └── special_tokens_map.json                            99 B
  └── google/gemma-scope-2b-pt-res   .  .  .  .  .  .  360.18 MiB  (2 files)
      ├── layer_0/width_16k/average_l0_105/params.npz  288.13 MiB
      └── embedding/width_4k/average_l0_44/params.npz   72.04 MiB
  ──────────────────────────────────────────────────────────────────────────
  7.97 GiB  total (3 repos, 14 files)
  ● = partial downloads

$ hf-fm cache path google/gemma-2-2b-it
/home/user/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
```

### Duplicated copies, datasets, and repos moved to another disk

On Windows `hf-hub` never tries a symlink, so until 0.12.6 every file was stored twice: as its blob and as a full copy in `snapshots/`. `du` counts both (they occupy the disk) and, since 0.12.5, names the second copy: a `DUP` column beside `SIZE` (only when some repo has copies, so the Linux and macOS output above is unchanged), and a footer. The listing also covers datasets and Spaces, so its total is the cache's, and a repo moved to another disk behind a junction or symlink is marked with its target and the footer splits the bytes by volume. From a 763 GiB Windows cache, before `cache dedupe` (below):

```
$ hf-fm du
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub

    #                      SIZE         DUP  REPO                                                     FILES
    1                159.05 GiB           -  mntss/clt-gemma-2-2b-2.5M                                   53
    2                 78.92 GiB   38.30 GiB  bluelightai/clt-qwen3-1.7b-base-20k                         88
    3                 46.09 GiB   22.42 GiB  bluelightai/clt-qwen3-0.6b-base-20k                         88
    4                 34.54 GiB           -  datasets/TheQweaker/rented_boxes                          1952  ●
    5                 32.39 GiB   15.60 GiB  bluelightai-dev/clt-Qwen3-0.6B-Base-16k-test                87
…
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
  763.22 GiB  total (72 repos, 2839 files)
  By volume: C: 762.71 GiB · D: 525.44 MiB
  154.16 GiB of 763.22 GiB are duplicated copies (21 repos)
  Note: a duplicated copy is a snapshot file stored again beside its blob, which is how hf-hub writes on Windows. `du <repo>` shows each repo's share; `--verify-duplicates` hashes the copies to confirm them. `hf-fm cache dedupe` replaces them with hard links to their blobs, keeping every model.
  ● = partial downloads
  → = relocated: the repo's directory is a link to where its bytes are

$ hf-fm du microsoft/Phi-3.5-mini-instruct
…
   20      195 B  generation_config.json
  ────────────────────────────────────────────────
   7.12 GiB  listed above (20 files)
  14.24 GiB  total on disk

  duplicated copies 7.12 GiB · orphan blobs 0 B · partial downloads 0 B · bookkeeping 140 B
  `hf-fm cache dedupe microsoft/Phi-3.5-mini-instruct` replaces the duplicated copies with hard links to their blobs.
```

Since 0.12.6 `cache dedupe` reclaims them: each copy is compared with its blob byte for byte and replaced with a hard link to it, so the model keeps every file and the second copy goes. The preview pairs by length and reads nothing, hence "up to"; the result is what it reclaimed, and every file it left alone is named with its reason. On the same cache:

```
$ hf-fm cache dedupe --yes
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub

  up to   1.11 GiB  Qwen/Qwen3-0.6B-Base  (1 file)
  up to   9.69 GiB  bartowski/Mistral-7B-Instruct-v0.3-GGUF  (5 files)
…
  up to   7.12 GiB  microsoft/Phi-3.5-mini-instruct  (2 files)
…

  Up to 155.00 GiB in 231 files across 21 repos. Each file is compared with its blob before it is replaced.

  Reclaimed 154.16 GiB: 228 files replaced by hard links to their blobs.
  Compared 309.23 GiB in 1 min 34 s (3.29 GiB/s).
  Left alone: 3 files (864.40 MiB):
    google/gemma-scope-2b-pt-transcoders  50eec2f25c60545a9a74c1c3a26a0afdd0b4b872\layer_0\width_16k\average_l0_76\params.npz: no blob of its length holds the same bytes
…
```

`C:` gained 154.16 GiB, `du`'s total fell from 763.22 GiB to 609.07 GiB, its `DUP` column is gone, and `du microsoft/Phi-3.5-mini-instruct` now prints one `7.12 GiB  total (20 files)`. The three files left alone have the length of the repo's blobs but no blob of their own. New downloads on Windows no longer store LFS files twice: the chunked path makes a hard link where a symlink is refused, and the `hf-hub` path's copy is linked to its blob afterwards.

`du --sort dup` puts the repos with the most copies first (`#` stays each repo's rank by size, which `du <N>` takes), and `du --verify-duplicates` hashes every copy against its blob's name to confirm it. A dataset or Space is addressed behind its Hub prefix, `hf-fm du datasets/wikitext`; a root-level repo by its bare name, `hf-fm du gpt2`. `cache delete` takes several repos and a `--dry-run`, and on a repo behind a link removes the link only (`--with-target` deletes the bytes too):

```
$ hf-fm cache delete julien-c/dummy-unknown EleutherAI/pythia-70m --dry-run
  julien-c/dummy-unknown  (276.7 KiB, 8 files)
  EleutherAI/pythia-70m  (607 B, 1 file)
  Dry run: would delete 2 repos, freeing 277.2 KiB. Nothing was deleted.

$ hf-fm cache delete gpt2 --dry-run
  gpt2  (525.44 MiB, 7 files)  → D:\hf-cache\models--gpt2
  Only the link is removed: its 525.44 MiB stay at D:\hf-cache\models--gpt2. `--with-target` deletes them too.
  Dry run: would delete 1 repo, freeing 0 B. Nothing was deleted.
```

The [cache tutorial](docs/tutorials/clean-up-before-your-disk-fills.md) walks through all of it.

## Library quick start

```rust
let outcome = hf_fetch_model::download(
    "google/gemma-2-2b-it".to_owned(),
).await?;

println!("Model at: {}", outcome.inner().display());
```

Read a file's tensor metadata without downloading its weights. On this 288 MiB archive, not yet cached, it took 8 range requests and 84 KiB:

```rust
let (header, _source, _stats) = hf_fetch_model::inspect_npz(
    "google/gemma-scope-2b-pt-transcoders",
    "layer_0/width_16k/average_l0_115/params.npz",
    None, // token: resolved implicitly when `None`
    None, // revision: `main`
).await?;

for t in &header.tensors {
    println!("{} {} {:?}", t.name, t.dtype, t.shape); // W_enc F32 [2304, 16384], …
}
```

`inspect_safetensors`, `inspect_gguf` and `inspect_pth` take the same arguments, and each reads the local cache first. `inspect_repo_safetensors` reads every shard of a repo, each from the cache when it is there, after listing the repo on the Hub. Each also has a `_with_config` twin that takes a `FetchConfig` instead, whose token, revision and `output_dir` it uses, so a custom cache directory you download into is the one it inspects.

Filter, progress, auth, and more via the builder — see [Configuration](docs/configuration.md).

## Documentation

| Topic | |
|-------|---|
| [CLI Reference](docs/cli-reference.md) | All subcommands, flags, and output examples |
| [FAQ](docs/FAQ.md) | Common questions — installation, auth, cache location, discovery, errors |
| [Inspect tutorial](docs/tutorials/inspect-before-downloading.md) | Walkthrough: read tensor metadata, size, and architecture without downloading weights |
| [Quant tutorial](docs/tutorials/pick-a-quant-that-fits.md) | Walkthrough: compare a base model's quantized siblings and plan which one fits your GPU before downloading it (`quants --fits`, `--cache-headers`) |
| [Cache tutorial](docs/tutorials/clean-up-before-your-disk-fills.md) | Walkthrough: see what the cache holds, then reclaim disk space safely (`du` → `status` → `cache dedupe` → `cache gc`) |
| [Case studies](docs/case-studies/) | Real investigations where `inspect` did the diagnostic work — per-layer shape variation, OOM forensics from a crash log, and remote GGUF inspect before and after growing read-ahead (up to 11.7× faster) |
| [Search](docs/search.md) | Comma filtering, `--exact`, model card metadata |
| [Configuration](docs/configuration.md) | Builder API, presets, progress callbacks |
| [Architecture](docs/architecture.md) | How hf-fetch-model relates to `hf-hub` and `candle-mi` |
| [Diagnostics](docs/diagnostics.md) | `--verbose` output, `tracing` setup for library users |
| [Upstream differences](docs/upstream-differences.md) | Where hf-fetch-model diverges from Python `huggingface_hub`/`hf_transfer` |
| [Candle example](examples/candle_inspect.rs) | Inspect tensor layouts before downloading — for candle users |
| [Changelog](CHANGELOG.md) | Release history and migration notes |

## Used by

- [candle-mi](https://github.com/mi-for-the-rust-of-us/candle-mi) — Mechanistic interpretability toolkit for language models

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT License](LICENSE-MIT) at your option.

## Development

- Exclusively developed with [Claude Code](https://claude.com/product/claude-code) (dev)
- Git workflow managed with [Fork](https://fork.dev/)
- All code follows [CONVENTIONS.md](CONVENTIONS.md), derived from [Amphigraphic-Strict](https://github.com/PCfVW/Amphigraphic-Strict)'s [Grit](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit) — a strict Rust subset designed to improve AI coding accuracy.
- CI gates every push and PR: `cargo fmt --check`, `clippy --all-targets --all-features -D warnings`, and the test suite on **Linux and Windows**, plus a [`cargo audit`](https://github.com/rustsec/rustsec) security pass and a [`cargo-deny`](https://embarkstudios.github.io/cargo-deny/) check of licences, banned crates and sources. Vulnerability disclosure: see [SECURITY.md](SECURITY.md).
