# Pick a quant that fits before you download it

*Aggregate a model's quant sibling repos into one table, then plan CPU-expert offload before choosing — without downloading a single candidate.*

*~1,250 words · about 5 min read*

<!-- Last updated: 2026-09-16, hf-fm v0.12.1 -->

<!--
STYLE CONVENTIONS for editing this tutorial — keep growth consistent.

1. Tone: match the FAQ and the other two tutorials. Conversational,
   address the reader as "you", short paragraphs over bullet lists where
   prose works.
2. Reproducibility: like the disk-usage tutorial (not the inspect one),
   there is nothing to pin. `quants`' sibling discovery is a live Hub
   search — the exact repo set for any base model drifts as new quants
   get published. The `google/gemma-2-2b-it` walkthrough's output blocks
   were captured 2026-09-16 and are real, verified command output at
   that date; the reader's row count and exact sizes WILL differ. The
   `--n-cpu-moe` offload-plan numbers in the "the case this feature was
   built for" section are NOT a fresh capture — they are the worked
   example from the dogfooding report this feature originated from,
   reused here as illustration and labeled as such, since reproducing a
   12+ GiB MoE GGUF download-free capture is impractical to redo on
   every doc refresh.
3. Output blocks: paste exact output, do not paraphrase. Trim a long
   table with `…` and note the trim.
4. Length budget: under 300 lines total, including embedded outputs.
   Update the word count + reading-time line at the top whenever the
   prose changes non-trivially (250 wpm).
5. Word count = total words in this file excluding code blocks and
   HTML comments. Reading time = word count / 250, rounded to the
   nearest minute, minimum 1.
-->

The third tutorial in the docs effort, and the first one built from a single dogfooding session rather than a feature already in daily use. If a step is confusing or your output looks structurally different from what's shown here, please open an issue on [GitHub](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues).

## Contents

- [The seven-call problem](#the-seven-call-problem)
- [The 30-second answer](#the-30-second-answer)
- [Reading the table](#reading-the-table)
- [Where discovery can be wrong](#where-discovery-can-be-wrong)
- [`--fits`: a plan, not a boolean](#--fits-a-plan-not-a-boolean)
- [The case this feature was built for](#the-case-this-feature-was-built-for)
- [Going deeper: `inspect --group-by`](#going-deeper-inspect---group-by)
- [What you've learned](#what-youve-learned)

## The seven-call problem

Say you want to run a model on a 16 GiB card, and the model ships in a handful of quantizations across several repos — `bartowski/<model>-GGUF`, `mradermacher/<model>-i1-GGUF`, maybe an `NVFP4` or `INT4` safetensors repo too. "Which one fits" sounds like a one-line question. Answering it with `hf-fm` alone, before this feature existed, took a `search` call to find the candidates and one `list-files` call per repo to read their sizes — then manual transcription into a table, by hand, outside the tool. A real [dogfooding session](../dogfooding-feedbacks/hf-fm-dogfooding-vram-fit-laguna-session.md) sizing `poolside/Laguna-XS-2.1`'s quants this way needed seven calls for one answer, and even then the size comparison alone was misleading: a checkpoint 40% over the raw VRAM budget could still be the right choice if most of its bytes are `MoE` expert weight that can live in system RAM instead.

`hf-fm quants` collapses that into one command, and `--fits` turns the size comparison into an offload-aware plan.

## The 30-second answer

```sh
hf-fm quants google/gemma-2-2b-it
```

```
Searching for quant siblings of google/gemma-2-2b-it...
8 repos found, 0 verified via GGUF backlink

  ARTIFACT                                     SIZE  REPO                                      BITS
  gemma-2-2b-it.IQ1_S.gguf               793.61 MiB  MaziyarPanahi/gemma-2-2b-it-GGUF          ~1.6
  gemma-2-2b-it.IQ1_M.gguf               833.32 MiB  MaziyarPanahi/gemma-2-2b-it-GGUF          ~1.8
  gemma-2-2b-it.IQ2_XS.gguf              956.10 MiB  MaziyarPanahi/gemma-2-2b-it-GGUF          ~2.3
  gemma-2-2b-it.Q2_K.gguf                  1.15 GiB  MaziyarPanahi/gemma-2-2b-it-GGUF          ~2.6
  …
  gemma-2-2b-it-bnb-4bit                   2.07 GiB  unsloth/gemma-2-2b-it-bnb-4bit            ?
  gemma-2-2b-it-yarn-32k                   4.87 GiB  theblackcat102/gemma-2-2b-it-yarn-32k     ?
  gemma-2-2b-it                            4.87 GiB  unsloth/gemma-2-2b-it                     ?
  gemma-2-2b-it.fp16.gguf                  4.88 GiB  MaziyarPanahi/gemma-2-2b-it-GGUF          ?
  gemma-2-2b-it                            9.74 GiB  Efficient-Large-Model/gemma-2-2b-it       ?
  gemma-2-2b-it-f32.gguf                   9.74 GiB  bartowski/gemma-2-2b-it-GGUF              ~32.0
```

No repo ID was typed except the base model's. `quants` found eight sibling repos on its own, listed every `.gguf` file and every safetensors-only repo's total as one sorted table, and looked up an approximate bits-per-weight figure from each filename.

## Reading the table

Three columns need a second look:

- **ARTIFACT** is either a `.gguf` filename (one row per file — a single repo can hold dozens of quant variants) or, for a candidate repo with no `.gguf` file at all, the repo's short name standing in for the whole thing (`gemma-2-2b-it-bnb-4bit`, `gemma-2-2b-it-yarn-32k` above — the size is the sum of that repo's `.safetensors` files).
- **BITS** is an approximation read off the filename's quant-scheme token (`Q4_K_M` → `~4.85`, `IQ3_XS` → `~3.3`). A `?` means the artifact's name didn't match any known scheme — common for `bnb-4bit`-style suffixes or repos that are really just a full-precision mirror, not a quant at all (more on that below).
- The stderr line — `8 repos found, 0 verified via GGUF backlink` — is the discovery summary, always printed before the table so a slow multi-repo search doesn't look like a hang.

## Where discovery can be wrong

There is no HuggingFace Hub endpoint for "find the quant siblings of this repo". `quants` combines two signals instead: a **naming match** (any repo whose ID contains the base model's short name) builds the candidate pool, and, for `.gguf` candidates, the file's own metadata — `general.source.url` / `general.base_model.*.repo_url` — is checked against the base repo to raise confidence.

The `0 verified via GGUF backlink` in the run above is honest, not broken: none of the eight candidates' `.gguf` metadata happened to carry a recognized backlink key, so every row is a naming match standing alone. Look closely at the table and two rows are not quants at all — `unsloth/gemma-2-2b-it` and `Efficient-Large-Model/gemma-2-2b-it` are full-precision mirrors that happen to share the base model's name. This is the accepted tradeoff of a naming-based search: recall over precision, with the `BITS` column's `?` and a low verified count as the signal to sanity-check a row before trusting it. A candidate is only ever silently *excluded* when its backlink explicitly names a different repo — a failed check (network error, a gated repo) always keeps the candidate rather than hiding it.

## `--fits`: a plan, not a boolean

```sh
hf-fm quants google/gemma-2-2b-it --fits 9.5GiB
```

```
…
  gemma-2-2b-it.fp16.gguf                  4.88 GiB   4.88 GiB  full GPU
  gemma-2-2b-it                            9.74 GiB          —  does not fit (no offload mechanism for this format)
  gemma-2-2b-it-f32.gguf                   9.74 GiB          —  does not fit (no MoE experts to offload)
  gemma-2-2b-it-abliterated-f32.gguf       9.74 GiB          —  does not fit (no MoE experts to offload)
```

Every row already under the 9.5 GiB budget renders `full GPU` instantly — no network call, because there is nothing to compute. Only the three rows over budget get inspected, and Gemma-2-2B is a dense model with no `MoE` experts, so there's nothing to offload: `does not fit` is the honest answer, and the reason column says exactly why. The safetensors-only row skips inspection entirely (`no offload mechanism for this format` — offload planning only applies to `.gguf` files, since the resident/offload split comes from a tensor-name rollup GGUF's `blk.N.*` naming makes possible).

## The case this feature was built for

Gemma-2-2B is dense, so it can only ever say yes or no. The reason `--fits` exists is `MoE` checkpoints, where the size-vs-budget comparison alone is misleading. From the dogfooding session that motivated this feature — `poolside/Laguna-XS-2.1-GGUF`'s `Q4_K_M`, a 12.06 GiB **file** (not the 18.88 GiB `total` `list-files` misleadingly reported for the whole multi-quant repo) with 93.6% of its bytes in expert tensors spread over 39 layers:

```
ARTIFACT                  SIZE       RESIDENT   PLAN
...i1-IQ3_XS.gguf         12.80 GiB  12.80 GiB  full GPU
...Q4_K_M.gguf            18.88 GiB  12.54 GiB  --n-cpu-moe 14  (6.34 GiB -> RAM)
```

*(Illustrative — the exact numbers above are from the dogfooding report, not a fresh capture; see the style note at the top of this file.)* A naive size comparison rejects `Q4_K_M` outright at 18.88 GiB against a 16 GiB card. `--fits` doesn't: it inspects the file's `MoE` expert-tensor rollup, computes that offloading 14 of 39 layers' experts to system RAM brings the resident footprint to 12.54 GiB, and reports the exact `--n-cpu-moe` flag to hand `llama.cpp`. That's the difference between a scalar filter and a plan.

## Going deeper: `inspect --group-by`

`--fits` computes its rollup against a fixed internal pattern (`blk.*.*_exps.weight`, `llama.cpp`'s own `MoE` expert-tensor convention) so you never have to supply one. To see the same rollup directly — a different pattern, or just to understand one specific file before comparing it to siblings — reach for `inspect --group-by` on that file alone:

```sh
hf-fm inspect poolside/Laguna-XS-2.1-GGUF Q4_K_M.gguf --group-by 'blk.*.ffn_*_exps.weight'
```

This is the same building block `--fits` uses internally, exposed as its own command — see the [FAQ entry](../FAQ.md#what-fraction-of-a-gguf-file-is-the-moe-expert-weights) for the full output shape.

## What you've learned

| Question | Command |
|----------|---------|
| What quants exist for this model? | `quants <repo>` |
| Which one fits my card? | `quants <repo> --fits <SIZE> [--reserve <SIZE>]` |
| Can I trust a row's naming match? | Check `BITS` (`?` = unrecognized scheme) and the stderr verified count |
| What fraction of one file is `MoE` expert weight? | `inspect <repo> <file> --group-by 'blk.*.ffn_*_exps.weight'` |
| For scripting | `quants <repo> --json` / `quants <repo> --fits <SIZE> --json` |

One model in one sentence: `quants` gathers the candidates and sizes them, `--fits` turns "too big" into "here's the offload plan" for the ones where the answer is actually in doubt — never wasting a header fetch on a row that trivially fits.

For the deeper "will this one file fit" question against your actual live VRAM (not a budget you supply by hand), see [`inspect --check-gpu`](../FAQ.md#how-do-i-know-if-a-model-fits-on-my-gpu). For everything else about reading a model before downloading it, see the companion tutorial, [Inspect before you download](inspect-before-downloading.md).
