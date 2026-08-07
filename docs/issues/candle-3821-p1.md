# candle #3821 — reply 1 (Draft)

- **Target issue:** https://github.com/huggingface/candle/pull/3821
- **Status:** Draft (not yet posted)
- **Context:** PR #3821 self-answers [#3820](https://github.com/huggingface/candle/issues/3820)'s open question (the reporter, astorise, opened both, four hours apart) by hardcoding `general.architecture = "qwen3_5"` (with an underscore) as the only string `Architecture::from_name` recognizes for the dense model, in `candle-transformers/src/models/quantized_lm.rs` lines 97 and 116 (head commit `ceb78ca8aee88bc6d34c0b232e7b0e4272dbe7f2` at draft time). `hf-fm inspect` against real, currently-hosted checkpoints reports `general.architecture=qwen35`, no underscore. The PR has zero reviews, zero comments, and no CI runs at draft time, so nobody has flagged the mismatch yet. This supersedes our earlier drafts for [#3820](candle-3820-p1.md) directly on the issue: the more useful target now is the PR that would ship the wrong string, not the issue whose question that PR already (incorrectly) tried to answer.
- **Outcome:** —
- **Lesson / Leverage angle:** A new variant of the pattern [#3530](https://github.com/huggingface/candle/issues/3530) established: instead of "one binary fact blocks a ready-to-submit PR," this is "the PR got submitted anyway and the binary fact would have caught the bug before merge." Worth keeping as a distinct leverage angle for future archive entries: an open, unreviewed PR is sometimes a better target than the issue it closes, especially when it ships within hours of the issue and plausibly wasn't checked against a real file. Also notable: sibling PR #3838 (fixes #3837) independently re-derives the correct `qwen35.` metadata namespace via a fallback in its own field reader, without anyone connecting that back to #3821's dispatch-level string; the two PRs are inconsistent with each other, not just with the real files.
- **Accuracy flags:** `general.architecture=qwen35` and `qwen35moe` are copied verbatim from live `hf-fm inspect` output against `unsloth/Qwen3.5-4B-GGUF` and `unsloth/Qwen3.5-35B-A3B-GGUF`, re-verified live at draft time (2026-08-07), not guessed. The `quantized_lm.rs` line numbers and the `SUPPORTED_ARCHITECTURES`/`from_name` snippets are copied from the PR's actual head commit via the GitHub contents API, not from the PR description. The `quantized_qwen3_5.rs` fallback in PR #3838 (`s.replace("qwen3.", "qwen35.")`, lines 645 to 654 at its own head commit `6d94f28dfc470d97669765dbe76252e0224b6eaa`) was read the same way. The claim that no code path handles `qwen35moe` at all was checked by grep across the full PR #3821 diff, not inferred from the summary; the failure mode for a real MoE checkpoint under the current code is an explicit "unsupported gguf architecture" rejection, not a silent misroute, which is the safer of the two failure modes but still blocks loading.

---

Ran `hf-fm inspect` (v0.11.2, remote GGUF support, reads the metadata KV table over HTTP Range, no download) against the checkpoints [#3820](https://github.com/huggingface/candle/issues/3820) is about, to check the `general.architecture` string this PR registers:

```
$ hf-fm inspect unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-UD-IQ2_XXS.gguf

  Repo:     unsloth/Qwen3.5-4B-GGUF
  File:     Qwen3.5-4B-UD-IQ2_XXS.gguf
  Source:   remote (170 range requests, 10.50 MiB fetched)
  Size:     1.42 GiB
  Metadata:
    general.architecture=qwen35
    …
```

```
$ hf-fm inspect unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q3_K_S.gguf

  Repo:     unsloth/Qwen3.5-35B-A3B-GGUF
  File:     Qwen3.5-35B-A3B-Q3_K_S.gguf
  Source:   remote (170 range requests, 10.50 MiB fetched)
  Size:     14.22 GiB
  Metadata:
    general.architecture=qwen35moe
    …
```

`Architecture::from_name` currently matches on `"qwen3_5"` (with an underscore, `quantized_lm.rs:116`, also listed in `SUPPORTED_ARCHITECTURES` at line 97), but both real checkpoints report `qwen35` and `qwen35moe`, no underscore. As written, `from_gguf` would reject both files with "unsupported gguf architecture", not dispatch them to `quantized_qwen3_5::ModelWeights`.

Interestingly, the sibling PR #3838 already gets the field-level namespace right: its `md_get` closure in `quantized_qwen3_5.rs` (lines 645 to 654) tries `"qwen3.xxx"` first, and falls back to `"qwen35.xxx"` when the first lookup misses, which correctly resolves the real files' `qwen35.*` metadata keys once a `ModelWeights` is actually constructed. That fallback just never gets reached, because this PR's `Architecture::from_name` gate rejects the file one step earlier, before `from_gguf` ever calls into `quantized_qwen3_5.rs`.

Two things this suggests for `Architecture::from_name` and `SUPPORTED_ARCHITECTURES`: recognize `"qwen35"` (matching what real Unsloth-quantized checkpoints actually write), and possibly keep `"qwen3_5"` as an accepted alias too in case some other conversion path does emit the underscore form; and add a distinct arm for `"qwen35moe"` rather than leaving it unhandled entirely, since it resolves to neither the existing `Qwen3Moe`/`quantized_qwen3_moe` arm (a structurally different, full-attention model, as [#3820](https://github.com/huggingface/candle/issues/3820) itself already points out) nor the plain `qwen35` dense arm.

One open question from our earlier read of [#3820](https://github.com/huggingface/candle/issues/3820) is already resolved by PR #3838, worth confirming here rather than re-raising: neither checkpoint's metadata has a distinct `partial_rotary_factor` key, only `rope.dimension_count` (64 on both variants); #3838's `rotary_dim` computation reads `qwen3(5).rope.dimension_count` directly and falls back to the full head dimension when absent, so the GGUF path does not need `partial_rotary_factor` plumbed through separately from what [#3837](https://github.com/huggingface/candle/issues/3837) already covers for the safetensors path.

Happy to inspect other quant variants, or dump full tensor names/shapes via `--tree`, if that helps land this or #3838; no download needed either way.

PS: `hf-fm` is a small Rust CLI for HuggingFace repos (no Python dependency, no weight data fetched). If you'd like to verify independently: `cargo install hf-fetch-model --features cli` installs it, and the two commands above are exactly what produced the output pasted here.
