# unsloth #6071 — reply 2 (Posted)

- **Target issue:** https://github.com/unslothai/unsloth/issues/6071
- **Status:** Posted (2026-08-29), **partially inaccurate — see Accuracy flags and [p3](unsloth-6071-p3.md), posted 2026-09-16**. [Comment](https://github.com/unslothai/unsloth/issues/6071#issuecomment-5463138984). The central structural finding stands and was adopted by the maintainer; the framing of the cited OpenWhispr thread does not.
- **Context:** User (PCfVW) spotted a cross-reference GitHub surfaces on #6071: [OpenWhispr/openwhispr#939](https://github.com/OpenWhispr/openwhispr/issues/939) (dakotahp, **opened** 2026-06-12; **closed 2026-07-09 as resolved** — p2 as posted mislabelled the creation date as the closure date) hits the exact `llama_model_load: error loading model: missing tensor 'blk.32.ssm_conv1d.weight'` error against a real `llama-server` crash log, for a 9B `Q4_K_M` file. This is independent, third-party, production evidence that the crash is real, separate from [p1](unsloth-6071-p1.md)'s finding that the four files checked there are structurally complete.
- **Verification method:** Identified the exact crashed file from OpenWhispr's own source: `gh search code` in `OpenWhispr/openwhispr` for `qwen3.5-9b-q4_k_m` locates `src/models/modelRegistryData.json`, whose entry for that id gives `"hfRepo": "bartowski/Qwen_Qwen3.5-9B-GGUF"` and `"fileName": "Qwen_Qwen3.5-9B-Q4_K_M.gguf"`, matching dakotahp's cached path (`/Users/dpena/.cache/openwhispr/models/Qwen_Qwen3.5-9B-Q4_K_M.gguf`) exactly. Then `hf-fm inspect` (v0.12.0) against that exact repo/file, and against `unsloth/Qwen3.5-9B-GGUF` (the plain, non-MTP-named 9B repo p1 didn't check) for contrast.
- **Finding:** The crashed file is a `bartowski` quant, not an Unsloth one, and it is structurally identical to Unsloth's own `Qwen3.5-9B-MTP-GGUF` from [p1](unsloth-6071-p1.md): `block_count=33`, `nextn_predict_layers=1`, `blk.32` is the NextN speculative-decoding head. Unsloth's plain (non-MTP-named) `Qwen3.5-9B-GGUF`, by contrast, ships without the NextN head at all (`block_count=32`, no `blk.32`). So the crash reproduces on an independently-produced GGUF from a different quantizer whenever the NextN head is present, which points away from "Unsloth's export pipeline truncates tensors" (p1's finding stands: none of the four files it checked are truncated) and toward a `llama.cpp`-side layer-type classification issue specific to the NextN/MTP head on hybrid `qwen35` models, not an export-side bug in any one tool.
- **Accuracy flags:** The repo/file identification rests on an exact three-way match (`hfRepo`, `fileName`, and the id `qwen3.5-9b-q4_k_m` all agree with dakotahp's log), not a byte-for-byte hash check, since the OpenWhispr issue doesn't include one. One inconsistency worth naming: OpenWhispr's registry lists `sizeBytes: 5889811552` (~5.49 GiB) for this model, while `hf-fm list-files` currently reports the live file at 5.75 GiB; this is plausibly just a stale hardcoded estimate in their registry (a common pattern, unrelated to the crash), not evidence of a different file, but it's flagged rather than silently ignored. The claim that a `llama.cpp`-side NextN-layer classification issue is the likely root cause is a hypothesis grounded in the structural evidence (both an Unsloth-produced and a bartowski-produced GGUF fail identically whenever the NextN head is present, and neither fails without it), not a confirmed diagnosis; `llama.cpp`'s own model-loading source was not read for this reply. **[p3](unsloth-6071-p3.md) later read it: the hypothesis was correct**, confirmed at [`qwen35.cpp:20`](https://github.com/ggml-org/llama.cpp/blob/b81c2cdd748dc2704d5989cf03936325554c12d3/src/models/qwen35.cpp#L20) pre-fix, guarded by PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) (2026-05-16, build `b9180`).
- **Four defects found on re-reading (2026-09-16), all from reading only the opening post of a cited thread:**
  1. **Wrong closure date**, corrected in Context above: OpenWhispr#939 closed 2026-07-09, not 2026-06-12.
  2. **Wrong closure *status* implied publicly.** The posted comment says only "(closed, but cross-referenced here by GitHub)", reading as an unresolved corroborating crash. It closed as **fixed**, in OpenWhispr 1.7.4 via [openwhispr#995](https://github.com/OpenWhispr/openwhispr/pull/995), by bumping the bundled `llama-server`.
  3. **Uncredited prior art.** [navarro165](https://github.com/navarro165) posted the same mechanism in that very thread on 2026-06-26, two months before p2, with a stronger test: the *same* file, sha256-pinned, failing on `b8857` and generating normally on `b9820`. p2 presented it as a fresh hypothesis.
  4. **Inferior practical advice.** p2 steered readers to the non-MTP `unsloth/Qwen3.5-9B-GGUF`. That is a workaround for an already-fixed bug; updating `llama.cpp` past `b9180` keeps the MTP file working.
- **Lesson:** this is a textbook violation of the read-fully practice. The conclusive answer was sitting in the comment bodies of the one external thread p2 rests on. Reading it would have resolved #6071 on 2026-08-29 instead of leaving it to close three weeks later on a half-right mechanism.
- **Outcome:** **Adopted, with our framing corrected in [p3](unsloth-6071-p3.md).** [danielhanchen](https://github.com/danielhanchen) closed #6071 on 2026-09-15 as COMPLETED, citing this reply's cross-quantizer evidence as his second reason and confirming Unsloth's export path "only adds the linear_attn tensor aliases and leaves the writing to llama.cpp". He suggested raising it upstream; p3 shows there is nothing left to raise, the fix having landed 2026-05-16.

---

cc @dakotahp, whose report in [OpenWhispr/openwhispr#939](https://github.com/OpenWhispr/openwhispr/issues/939) (closed, but cross-referenced here by GitHub) hits the exact same tensor name from a real crash, not a paraphrase:

```
llama_model_load: error loading model: missing tensor 'blk.32.ssm_conv1d.weight'
llama_model_load_from_file_impl: failed to load model
common_init_from_params: failed to load model '/Users/dpena/.cache/openwhispr/models/Qwen_Qwen3.5-9B-Q4_K_M.gguf'
```

OpenWhispr's own model registry (`src/models/modelRegistryData.json`) pins that exact filename to `bartowski/Qwen_Qwen3.5-9B-GGUF`, not an Unsloth repo. So this is a second, independently-produced GGUF (different quantizer, presumably `llama.cpp`'s own converter) hitting the identical failure:

```
$ hf-fm inspect bartowski/Qwen_Qwen3.5-9B-GGUF Qwen_Qwen3.5-9B-Q4_K_M.gguf
  …
  Metadata:
    qwen35.block_count=33
    qwen35.full_attention_interval=4
    qwen35.nextn_predict_layers=1
    …
  blk.32.attn_output.weight             Q8_0  [4096, 4096]  17.00 MiB  16.8M
  blk.32.ffn_down.weight                Q8_0  [12288, 4096]  51.00 MiB  50.3M
  blk.32.nextn.eh_proj.weight           Q8_0  [8192, 4096]   34.00 MiB  33.6M
  blk.32.nextn.enorm.weight             F32   [4096]         16.0 KiB   4.1K
  blk.32.nextn.hnorm.weight             F32   [4096]         16.0 KiB   4.1K
  blk.32.nextn.shared_head_norm.weight  F32   [4096]         16.0 KiB   4.1K
  442 tensors, 9.20B params
```

Same shape as Unsloth's own `Qwen3.5-9B-MTP-GGUF` from the previous comment: `block_count=33`, `nextn_predict_layers=1`, `blk.32` is the NextN head (attention-shaped, no `ssm_conv1d.weight` by design, same as before). For contrast, Unsloth's *other* 9B repo, the plain one without "MTP" in the name, doesn't ship that head at all:

```
$ hf-fm inspect unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf
  …
  Metadata:
    qwen35.block_count=32
    qwen35.full_attention_interval=4
    …
  427 tensors, 8.95B params
```

`block_count=32`, no `blk.32`, no `nextn.*` tensors anywhere. A user loading this file would never hit the `blk.32.ssm_conv1d.weight` error, since `llama.cpp` would only ever ask for blocks `0`..`31`.

So the crash is real, reproduced independently by a different app against a different quantizer's file, but it tracks the presence of the NextN/MTP head, not who produced the GGUF. Both an Unsloth-produced MTP file and a bartowski-produced file fail the same way when that head is present; neither Unsloth's plain 9B nor the other three files checked in the previous comment (which don't have a NextN head at all) show any sign of it. That's consistent with a `llama.cpp`-side issue in how it classifies the NextN layer's type for hybrid `qwen35` models (my read: `(32+1) % 4 != 0`, so if the loader applies the ordinary `full_attention_interval` layer-type rule to the NextN head instead of special-casing it, it would wrongly expect that layer to carry SSM weights), rather than a truncation bug in any particular export pipeline. I haven't read `llama.cpp`'s model-loading source to confirm that mechanism though, so take it as a hypothesis the evidence points toward, not a diagnosis.

Practical note for anyone hitting this: since Unsloth publishes both variants, `unsloth/Qwen3.5-9B-GGUF` (no NextN head, 32 layers) looks like it should load fine, versus `unsloth/Qwen3.5-9B-MTP-GGUF` or `bartowski/Qwen_Qwen3.5-9B-GGUF` (both 33 layers, NextN head present) which reproduce dakotahp's exact error by this reasoning. I haven't run `llama-server` myself to confirm the plain file actually loads, only that it doesn't contain the tensor structure that appears to trigger the failure.
