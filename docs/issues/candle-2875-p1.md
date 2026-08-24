# candle #2875 — reply 1 (Posted, superseded)

- **Target issue:** https://github.com/huggingface/candle/issues/2875
- **Status:** Posted (2026-04-15) — **superseded by [candle-2875-p2.md](candle-2875-p2.md)** (2026-08-24)
- **Context:** [#2875](https://github.com/huggingface/candle/issues/2875) reports `Error: unsupported safetensor dtype F8_E4M3` from `tensor-tools quantize` on an FP8 Flux finetune ([`little-lake-studios/demoncore-flux`](https://huggingface.co/little-lake-studios/demoncore-flux)). The issue was filed *before* candle had any fp8 support. Our reply gave a correct dtype breakdown via `hf-fm inspect --dtypes` but then drew a **wrong, out-of-date conclusion**: that candle "would require … adding an `F8E4M3` variant to `candle_core::DType`."
- **Outcome:** No reporter response to this comment. On re-examination (2026-08-24) the diagnosis was found stale: fp8 support had landed in [#2989](https://github.com/huggingface/candle/pull/2989) (merged 2025-08-04), released in **candle 0.10.0** (2026-03-31) — ~8 months before this comment on `main`, ~2 weeks before it in a stable release. The `F8E4M3` variant we said candle "would need to add" already existed. Corrected in [p2](candle-2875-p2.md).
- **Lesson / Leverage angle:** The empirical half (the `--dtypes` output) was right and useful; the *interpretive* half was asserted from memory of candle's state, not re-checked against current source. The transferable rule: **a claim about what the upstream code does today must be traced against today's source before posting**, exactly as [candle-3821-p1.md](candle-3821-p1.md) traced `quantized_lm.rs` line-by-line. This entry is the negative example that motivates the [inspect-before-you-reproduce case study](../case-studies/) — inspection told us *what* the file is; we failed to inspect *what candle already was*.
- **Accuracy flags:** The `--dtypes` table below is verbatim real `hf-fm inspect` output and remains accurate (re-verified 2026-08-24: 948 F8_E4M3 / 16.53B params / 15.40 GiB). The conclusion sentence ("would require candle to first support loading F8_E4M3 tensors…") is the part that was wrong at posting time.

---

Candle's `tensor-tools quantize` doesn't support `F8_E4M3` yet; this is a missing dtype conversion in candle, not a problem with the model.

Here's the dtype breakdown for that file:

`$ hf-fm inspect little-lake-studios/demoncore-flux transformer/demonCORESFWNSFW_fluxV13.safetensors --dtypes`

  `Dtype    Tensors       Params       Size`
  `F8_E4M3      948       16.53B  15.40 GiB`
  `BF16         244        83.8M 159.87 MiB`
  `F16          199       161.6M 308.22 MiB`
  `F32           52       131.8M 502.77 MiB`
  `───────────────────────────────────────────`
  `1443 tensors, 16.91B params`

948 out of 1443 tensors (66%) are `F8_E4M3`, so this isn't a single outlier — the model is predominantly FP8. Quantizing it would require candle to first support loading F8_E4M3 tensors, which would mean adding an `F8E4M3` variant to `candle_core::DType` and the corresponding conversion paths...

PS: `hf-fm inspect` reads safetensors headers via HTTP Range requests, i.e. no weight data downloaded; if needed: `cargo install hf-fetch-model --features cli` (requires v0.9.6 or later).
