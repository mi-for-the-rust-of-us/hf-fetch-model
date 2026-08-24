# candle #2875 — reply 2 (Posted)

- **Target issue:** https://github.com/huggingface/candle/issues/2875
- **Posted comment:** https://github.com/huggingface/candle/issues/2875#issuecomment-5392212864
- **Status:** Posted (2026-08-24) — corrects [candle-2875-p1.md](candle-2875-p1.md)
- **Context:** Our first reply ([p1](candle-2875-p1.md), 2026-04-15) said candle "would need to add an `F8E4M3` variant to `candle_core::DType`." That was already false: fp8 support landed in [#2989](https://github.com/huggingface/candle/pull/2989) (merged 2025-08-04), released in **candle 0.10.0** (2026-03-31). This reply corrects the record, traces the current `tensor-tools quantize` path on `main` (`81f247a`) to show the error no longer originates there, and confirms via `hf-fm inspect` that the specific file is plain per-tensor FP8 (no `weight_scale`/`scale_inv`), so candle's F32 upcast is faithful.
- **Outcome:** _Pending_ (as of 2026-08-24). Awaiting reporter (@AlpineVibrations) / maintainer response; ideally a re-run on ≥ 0.10.0 and a close.
- **Lesson / Leverage angle:** First archive entry whose leverage is **inaction**. Remote inspection prevented three expensive actions for ~193 KiB fetched: (1) a wrong "second candle bug" report — the `--filter weight_scale → 0 tensors` check proved candle's plain upcast is correct here; (2) a 16 GiB download + ~40 GB-RAM local quantize to "confirm" the un-interesting plain-cast case; (3) a 16 GiB download to validate anamnesis's block-scale FP8 path against a model that has no block scales and so wouldn't exercise it. The transferable pattern, **"inspect before you *reproduce*"** (verify the expensive experiment will actually test your hypothesis before running it), is the seed of the planned [case study](../case-studies/). Distinct from the existing [inspect-before-downloading tutorial](../tutorials/inspect-before-downloading.md), which stops at "inspect before you download."
- **Accuracy flags:** The `#2989` merge date (2025-08-04) and candle 0.10.0 release date (2026-03-31) were read from the GitHub API (PR merge commit; annotated-tag → commit date), and 0.10.0 was confirmed to contain the fp8 commit via `compare` (status "ahead"). The three source references are pinned to `main` head `81f247a8985e0b5b6c7c7c5b35c07dc685e005e9` and their line numbers were verified against the contents API at that SHA: `safetensors.rs:56` (`TryFrom` arm), `safetensors.rs:305` (`convert()` arm), `quantized/mod.rs:547` (`QTensor::quantize` upcast to F32). The two `hf-fm inspect` blocks are verbatim live output (2026-08-24): `--dtypes` (192.8 KiB fetched, "Per-tensor FP8 (E4M3)") and `--filter weight_scale` ("Showing 0 of 1443"). **Not reproduced end-to-end** — the claim that the quantize now succeeds is source-verified (load path handles F8_E4M3; `QTensor::quantize` upcasts to F32 first), not confirmed by running the 16 GiB quantize; the reply is worded to state exactly that ("no longer originates from this path"), not to overclaim a successful run.

---

**Follow-up, correcting my earlier comment.** My April note above is out of date: `F8_E4M3` has been supported since [#2989 "fp8 support"](https://github.com/huggingface/candle/pull/2989) (merged 2025-08-04), first in a stable release with **candle 0.10.0** (2026-03-31). So the fix predated my comment by ~8 months on `main` and ~2 weeks in a release; my apologies for the noise!

The `unsupported safetensor dtype F8_E4M3` error no longer originates from the `tensor-tools quantize` path. Traced on `main` (`81f247a`):

- `run_quantize_safetensors` loads via `candle::safetensors::load`, whose `convert()` now maps the dtype instead of rejecting it: [`safetensors.rs:305`](https://github.com/huggingface/candle/blob/81f247a8985e0b5b6c7c7c5b35c07dc685e005e9/candle-core/src/safetensors.rs#L305) (`st::Dtype::F8_E4M3 => convert_::<float8::F8E4M3>(...)`), backed by the `TryFrom` arm at [`safetensors.rs:56`](https://github.com/huggingface/candle/blob/81f247a8985e0b5b6c7c7c5b35c07dc685e005e9/candle-core/src/safetensors.rs#L56).
- `QTensor::quantize` upcasts the source to F32 before quantizing: [`quantized/mod.rs:547`](https://github.com/huggingface/candle/blob/81f247a8985e0b5b6c7c7c5b35c07dc685e005e9/candle-core/src/quantized/mod.rs#L547) (`let src = src.to_dtype(crate::DType::F32)?...`).

I checked the specific file this issue is about to confirm candle's plain upcast is faithful here, i.e. that the weights carry no separate quantization scales that a naive F32 cast would drop:

```
$ hf-fm inspect little-lake-studios/demoncore-flux transformer/demonCORESFWNSFW_fluxV13.safetensors --dtypes
  Source:   remote (4 range requests, 192.8 KiB fetched)
  Format:   Per-tensor FP8 (E4M3)
  Dtype    Tensors       Params       Size
  F8_E4M3      948       16.53B  15.40 GiB
  BF16         244        83.8M 159.87 MiB
  F16          199       161.6M 308.22 MiB
  F32           52       131.8M 502.77 MiB
  1443 tensors, 16.91B params

$ hf-fm inspect little-lake-studios/demoncore-flux transformer/demonCORESFWNSFW_fluxV13.safetensors --filter weight_scale
  Showing 0 of 1443 tensors matching filter "weight_scale".
```

No `weight_scale` / `scale_inv` tensors (the only `*.scale` entries are QK-norm layer weights), so this is plain per-tensor FP8: the F32 upcast is exact, and the resulting Q8_0 is faithful.

@AlpineVibrations — updating candle to >= 0.10.0 and re-running your original `tensor-tools quantize demonCORESFWNSFW_fluxV13.safetensors --out-file Q8_0.gguf --quantization q8_0` should now work. Worth closing if it does.

(reminder: `hf-fm inspect` reads safetensors headers over HTTP Range; the checks above fetched ~193 KiB, no weight data. `cargo install hf-fetch-model --features cli`.)
