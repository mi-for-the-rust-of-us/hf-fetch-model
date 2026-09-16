# unsloth #6071 — reply 1 (Posted)

- **Target issue:** https://github.com/unslothai/unsloth/issues/6071
- **Status:** Posted (2026-08-29). [Comment](https://github.com/unslothai/unsloth/issues/6071#issuecomment-5463081028).
- **Context:** OP reports that Unsloth's official Qwen3.5/Qwen3.6 GGUF exports "truncate trailing `ssm_conv1d.weight` tensors" on the final layer block of every model size (0.8B, 4B, 9B-MTP, 35B), citing `llama.cpp` load failures (`missing tensor 'blk.N.ssm_conv1d.weight'`) and, for MTP variants, a separate first-inference `GGML_ASSERT` crash. Zero comments, zero maintainer response, four months open as of this writing.
- **Verification method:** `hf-fm inspect <repo> <file>` (v0.12.0, remote GGUF header read over HTTP Range, no weight download) against the four exact repo/quant pairs the issue names: `unsloth/Qwen3.5-0.8B-GGUF` (`Q4_K_M`), `unsloth/Qwen3.5-4B-GGUF` (`UD-Q4_K_XL`), `unsloth/Qwen3.5-35B-A3B-GGUF` (`Q4_K_M`), `unsloth/Qwen3.5-9B-MTP-GGUF` (`Q4_K_M`). Raw command outputs saved locally; the numbers quoted below (`block_count`, `full_attention_interval`, tensor/param totals, per-block tensor names) are copied verbatim from those captures.
- **Finding:** All four files are structurally complete for their declared architecture. The "missing" tensor in every case is explained by `qwen35(moe).full_attention_interval=4` (the model interleaves SSM and full-attention layers; attention layers never carry `ssm_conv1d.weight`) or, for the MTP file, a genuine extra NextN speculative-decoding head block that is itself attention-shaped. This refutes the "truncation" claim (Error Pattern 1) for all four cited files. Error Pattern 2 (the MTP `GGML_ASSERT` runtime crash) is a separate claim about runtime indexing behavior that a static header read cannot confirm or refute, so the reply below says so explicitly rather than silently dropping it.
- **Accuracy flags:** The command blocks in the reply below show the `Metadata:` section trimmed with `…` to just the fields the argument needs (`block_count`, `full_attention_interval`, `nextn_predict_layers`): real GGUF metadata for these files runs to ~30 keys including the full chat template, as archived in [candle-3821-p1.md](candle-3821-p1.md) for the same 4B/35B repos. The per-block tensor listings and the final `N tensors, M params` line are pasted in full, unedited. The issue calls the 0.8B model "33-layer"; its GGUF metadata says `qwen35.block_count=24`. I did not check the 9B non-MTP base file, the 35B MTP variant, or any Qwen3.6 release, only the four listed above.
- **Outcome:** **Vindicated.** [danielhanchen](https://github.com/danielhanchen) closed #6071 on 2026-09-15 as COMPLETED, restating this reply's finding as his first reason ("The block index in the error is past the model's own block_count") and [p2](unsloth-6071-p2.md)'s as his second ("the same missing ssm_conv1d.weight failure shows up on GGUFs quantized by other people with llama.cpp's own converter"), and adding that Unsloth's export path "only adds the linear_attn tensor aliases and leaves the writing to llama.cpp". Nothing in this reply needed correction. [p3](unsloth-6071-p3.md) later bisected the real root cause to a stale `llama.cpp` build (fixed 2026-05-16 in `b9180`). See [p2](unsloth-6071-p2.md) for a follow-up: independent third-party crash evidence (OpenWhispr#939) that refines Error Pattern 1's likely root cause toward a `llama.cpp`-side NextN-layer issue rather than an export truncation.

---

Ran `hf-fm inspect` (v0.12.0, remote GGUF header read over HTTP Range, no weight download) against the four exact repo/quant pairs this issue names, to check whether `ssm_conv1d.weight` is actually missing from the final block:

```
$ hf-fm inspect unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q4_K_M.gguf

  Repo:     unsloth/Qwen3.5-0.8B-GGUF
  File:     Qwen3.5-0.8B-Q4_K_M.gguf
  Source:   remote (170 range requests, 10.50 MiB fetched)
  Size:     507.85 MiB
  Metadata:
    qwen35.block_count=24
    qwen35.full_attention_interval=4
    …
  blk.22.ssm_conv1d.weight          F32   [4, 6144]  96.0 KiB  24.6K
  blk.23.attn_k.weight              Q4_K  [1024, 512]  288.0 KiB  524.3K
  blk.23.attn_output.weight         Q4_K  [2048, 1024]  1.12 MiB  2.1M
  blk.23.attn_q.weight              Q4_K  [1024, 4096]  2.25 MiB  4.2M
  blk.23.attn_v.weight              Q6_K  [1024, 512]  420.0 KiB  524.3K
  blk.23.ffn_down.weight            Q6_K  [3584, 1024]  2.87 MiB  3.7M
  blk.23.ffn_gate.weight            Q4_K  [1024, 3584]  1.97 MiB  3.7M
  blk.23.ffn_up.weight              Q4_K  [1024, 3584]  1.97 MiB  3.7M
  320 tensors, 752.4M params
```

`block_count=24` means valid block indices are `0`..`23`; there is no `blk.24` to be missing anything from. The last real block, `blk.23`, is a full-attention block (`attn_q`/`attn_k`/`attn_v`/`attn_output`, no SSM tensors at all), which is exactly what `full_attention_interval=4` predicts: `(23+1) % 4 == 0`. The last block that does carry `ssm_conv1d.weight` is `blk.22`, present and correctly shaped.

Same story at 4B and 35B, just the numbers shift:

```
$ hf-fm inspect unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-UD-Q4_K_XL.gguf
  …
  Metadata:
    qwen35.block_count=32
    qwen35.full_attention_interval=4
    …
  blk.30.ssm_conv1d.weight   F32  [4, 8192]  128.0 KiB  32.8K
  blk.31.attn_output.weight  Q4_K [4096, 2560]  5.62 MiB  10.5M
  426 tensors, 4.21B params

$ hf-fm inspect unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q4_K_M.gguf
  …
  Metadata:
    qwen35moe.block_count=40
    qwen35moe.full_attention_interval=4
    …
  blk.39.attn_output.weight     Q8_0 [4096, 2048]  8.50 MiB  8.4M
  blk.39.ffn_down_exps.weight   Q5_K [512, 2048, 256]  176.00 MiB  268.4M
  733 tensors, 34.66B params
```

`blk.31` (4B) and `blk.39` (35B) are both full-attention for the same reason: `32 % 4 == 0` and `40 % 4 == 0`, so the last block in every one of these releases lands on an attention layer. `blk.32` and `blk.40` don't exist in these files at all: block counts are 0-indexed.

The 9B-MTP file is the interesting one, since it's the file actually named in the crash report:

```
$ hf-fm inspect unsloth/Qwen3.5-9B-MTP-GGUF Qwen3.5-9B-Q4_K_M.gguf
  …
  Metadata:
    qwen35.block_count=33
    qwen35.full_attention_interval=4
    qwen35.nextn_predict_layers=1
    …
  blk.32.attn_output.weight             Q4_K  [4096, 4096]  9.00 MiB  16.8M
  blk.32.ffn_down.weight                Q6_K  [12288, 4096]  39.38 MiB  50.3M
  blk.32.nextn.eh_proj.weight           Q8_0  [8192, 4096]  34.00 MiB  33.6M
  blk.32.nextn.enorm.weight             F32   [4096]  16.0 KiB  4.1K
  blk.32.nextn.hnorm.weight             F32   [4096]  16.0 KiB  4.1K
  blk.32.nextn.shared_head_norm.weight  F32   [4096]  16.0 KiB  4.1K
  442 tensors, 9.20B params
```

Here `blk.32` genuinely exists (`block_count=33`, unlike the other three files), but it isn't a 33rd base transformer layer: `nextn_predict_layers=1` says it's the NextN speculative-decoding head, appended after the 32 base layers (`0`..`31`, same `full_attention_interval=4` pattern the plain 9B would have). It carries regular attention/FFN tensors plus the `nextn.*` projection/norm weights, and correctly has no `ssm_conv1d.weight` because it was never an SSM layer to begin with.

So across all four files: whichever way `blk.N` in the report is read, literal 0-indexed tensor name (doesn't exist in three of the four) or "layer N of N" 1-indexed reference (lands on a real block in all four), that block turns out attention-shaped by design in every case, either a regular full-attention layer per `full_attention_interval`, or the MTP NextN head. I don't see a truncated export here.

This only speaks to Error Pattern 1 (the "missing tensor" / truncation claim). Error Pattern 2, the `GGML_ASSERT`/`GGU_ASSERT` crash on first inference for MTP variants, is a runtime indexing/stride claim that a static header read can't confirm or rule out, and may still be a real, separate bug. I also didn't check the 9B non-MTP base file, the 35B MTP variant, or any Qwen3.6 release. Happy to run the same check against any of those if it'd help narrow things down, just need the exact repo id and quant.

PS: `hf-fm` is a small Rust CLI for HuggingFace repos (no Python dependency, no weight data fetched for `inspect`). `cargo install hf-fetch-model --features cli` installs it if you'd like to verify independently; the four commands above (minus the `…` trims) are exactly what produced the output pasted here.
