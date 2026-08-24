<!--
CASE-STUDY PREP NOTE — NOT FOR POSTING, NOT AN INDEX ENTRY.

Leading underscore in the filename marks this as draft raw material, kept out
of the case-studies index in README.md. It captures the diagnostic path
(including the wrong turns) while fresh, so that when candle #2875's thread
settles the case study is pure synthesis, not re-derivation.

When writing the real case study:
  1. Rename/replace with docs/case-studies/fp8-quantize-error-already-fixed.md
  2. Add the row to docs/case-studies/README.md's index table
  3. Fill the "Outcome / reception" section from the settled thread
  4. Delete this prep note
Source archive: docs/issues/candle-2875-p1.md (stale), candle-2875-p2.md (posted correction)
-->

# PREP: The FP8 quantize error that had already fixed itself

**Status:** raw material, outcome pending. Do not publish as-is.
**Source thread:** candle [#2875](https://github.com/huggingface/candle/issues/2875) · posted correction [comment](https://github.com/huggingface/candle/issues/2875#issuecomment-5392212864) (2026-08-24)
**hf-fm workflow demonstrated:** `inspect --dtypes` + `--filter` as a *hypothesis-falsifier* — deciding NOT to run an expensive reproduction.

---

## 1. Symptom (reporter's words)

@AlpineVibrations, quantizing an FP8 Flux finetune ([`little-lake-studios/demoncore-flux`](https://huggingface.co/little-lake-studios/demoncore-flux)):

```
tensor-tools quantize demonCORESFWNSFW_fluxV13.safetensors --out-file Q8_0.gguf --quantization q8_0
Error: unsupported safetensor dtype F8_E4M3
```

## 2. The timeline that explains everything (all dates verified via GitHub API)

| Date | Event | How verified |
|------|-------|--------------|
| **2025-04-10** | Issue #2875 filed. Candle had **no** fp8 support; the error was real and correct. | `gh issue view 2875 --json createdAt` |
| 2025-06-11 | PR [#2989](https://github.com/huggingface/candle/pull/2989) "fp8 support" opened | `gh pr view 2989 --json createdAt` |
| **2025-08-04** | #2989 **merged** — adds `DType::F8E4M3` + safetensors load. Never linked back to #2875, which stayed open. | `gh pr view 2989 --json mergedAt` |
| **2026-03-31** | candle **0.10.0** released, containing the fix | annotated-tag `0.10.0` → commit date; `compare af5a69ee...0.10.0` returned **"ahead"** (0.10.0 contains the fp8 commit) |
| 2026-04-15 | Our first comment — **stale**: restated candle's April-2025 state as if current | `gh issue view 2875 --json comments` |
| 2026-08-24 | Corrected follow-up posted | comment 5392212864 |

**The one-sentence root cause: the issue simply predates the fix.** Filed ~2 months before the fix PR even opened, ~4 months before it merged; sat orphaned for a year because nothing closed it.

## 3. Diagnostic path — including the wrong turns (this is the valuable part)

### Wrong turn A — our own stale answer
Our April 2026 comment gave a *correct* `--dtypes` breakdown but a *wrong* conclusion ("candle would need to add an `F8E4M3` variant to `candle_core::DType`"). The variant had existed for 8 months. **Lesson: the empirical half was re-checkable and right; the claim about candle's current state was asserted from memory and wrong.** A source re-check before posting would have caught it. (See [candle-2875-p1.md](../issues/candle-2875-p1.md).)

### Source archaeology — dating the fix
Traced the live error string `unsupported safetensor dtype F8_E4M3` to `Error::UnsupportedSafeTensorDtype` in `candle-core/src/error.rs`, then found it is **no longer raised** for F8_E4M3 because `safetensors.rs` now maps it. Pinned on `main` head `81f247a8985e0b5b6c7c7c5b35c07dc685e005e9`:
- `safetensors.rs:56` — `st::Dtype::F8_E4M3 => Ok(DType::F8E4M3)` (`TryFrom`)
- `safetensors.rs:305` — `st::Dtype::F8_E4M3 => convert_::<float8::F8E4M3>(...)` (`convert()`, the path `candle::safetensors::load` uses)
- `quantized/mod.rs:547` — `let src = src.to_dtype(crate::DType::F32)?...` (`QTensor::quantize` upcasts to F32 first)

End to end on current candle: `tensor-tools quantize` → `run_quantize_safetensors` → `candle::safetensors::load` (loads F8_E4M3) → `QTensor::quantize` (upcasts F8E4M3→F32) → Q8_0. The error path is gone.

### Wrong turn B — the "candle is silently wrong" hypothesis (refuted for ~193 KiB)
Hypothesis: candle #2989's fp8 is deliberately *naive* — its own PR says scale-dependent ops (matmul) "can't be implemented because they require a scale tensor." anamnesis's [`remember/fp8.rs`](../../../anamnesis/src/remember/fp8.rs) does fine-grained **128×128 block-scale** E4M3 dequant. So *if* demoncore-flux is fine-grained FP8 (separate `weight_scale`/`scale_inv` tensors), candle's plain `to_dtype(F32)` upcast would **drop the scales and produce a wrong Q8_0** — a second, real candle bug, and an anamnesis differentiator.

Refuted by header inspection alone:
```
$ hf-fm inspect little-lake-studios/demoncore-flux transformer/demonCORESFWNSFW_fluxV13.safetensors --filter weight_scale
  Showing 0 of 1443 tensors matching filter "weight_scale".
$ hf-fm inspect ... --filter scale        # 152 hits, ALL of the form *.key_norm.scale / *.query_norm.scale
```
The 152 `*.scale` tensors are **QK-norm layer weights**, not quantization scales; zero `weight_scale`/`scale_inv`/`input_scale`/`qscale`. hf-fm classifies the file as **`Format: Per-tensor FP8 (E4M3)`**. So there are no scales to drop: candle's F32 upcast is exact, and the Q8_0 is faithful. **No second bug. Hypothesis dead.**

### Wrong turn C — reproduce-it-to-be-sure, and the anamnesis-fixture idea
Two reasons to download 16.34 GiB and run the quantize locally were considered and both killed:
- **"Confirm candle now works"** — the load path and F32 upcast are source-provable; running it would only confirm the un-interesting plain-cast case. Cost/benefit fails.
- **"Use the download to validate anamnesis's block-scale path"** — the model has no block scales (Wrong turn B), so it would exercise anamnesis's *plain* path, not its differentiator. Wrong fixture.

**RAM/disk sanity check that gated the decision** (the reproduction was costed before being declined; machine: 63.9 GiB RAM, 50.6 GiB free, 1789 GiB disk free):
- Download: **16.34 GiB** safetensors (file total per inspect header).
- Peak RAM ≈ input map (~16 GiB, tensors held in native dtype) + Q8_0 output accreting (~18 GB) + transient per-tensor F32 scratch (~2–4 GB) ≈ **~36–40 GB**. Fits, but not free. (Earlier worry of a 68 GB all-F32-resident peak was wrong: candle upcasts per-tensor via rayon, not all at once.)
- Disk: ~34 GB (in + out).

Total cost actually paid to settle all of the above: **~193 KiB, 4 range requests, seconds.**

## 4. Transferable workflow — the takeaway

**"Inspect before you *reproduce*."** One level up from the existing [inspect-before-downloading tutorial](../tutorials/inspect-before-downloading.md): before running an expensive experiment (a 16 GiB download + a 40 GB-RAM quantize), read the header to verify the experiment would actually *test your hypothesis*. Here it falsified two hypotheses and prevented three expensive actions for a fraction of a megabyte. **The first case study whose hero is inaction — and the first where inspection corrected its own author.**

One-line pattern for the reader who hits a similar wall:
```
hf-fm inspect <repo> <file> --dtypes          # what dtypes / how big / what format?
hf-fm inspect <repo> <file> --filter weight_scale   # are there quantization scales a naive cast would drop?
```

## 5. Outcome / reception — PENDING

> Fill after the thread settles (per case-studies/README: written after reception, honest about how far it got).
> - Did @AlpineVibrations re-run on ≥ 0.10.0? Did it work? Was #2875 closed?
> - Did a maintainer weigh in / close as fixed-by-#2989?
> - If it went quiet: say so. The transferable lesson is the workflow, not the close state.

## 6. Loose ends to double-check at write time
- Re-confirm the three source line numbers still resolve (rebase drift) or re-pin to a fresh SHA and note both.
- Re-run the two `hf-fm inspect` commands live for a fresh capture (byte counts drift trivially; keep them honest).
- Confirm the em-dash-free house punctuation in the final prose (per project style).
- Cross-link the finished case study from candle-2875-p2.md's Lesson bullet.
