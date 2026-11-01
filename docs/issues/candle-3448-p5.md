# candle #3448 — reply 5 (Posted)

- **Target issue:** https://github.com/huggingface/candle/issues/3448
- **Status:** Posted 2026-04-26 12:41 UTC as [this comment](https://github.com/huggingface/candle/issues/3448#issuecomment-4322047232). Archived on 2026-11-01, verbatim from the posted text: no draft of it had been kept here, and the archive's [p4](candle-3448-p4.md), a different draft written for the same moment, was never posted.
- **Context:** @AlpineVibrations reported on 2026-04-21 that [p3](candle-3448-p3.md)'s fix left "the same error as I reported above". Instead of asking for their diff (p4's approach), this reply reproduced the problem on a fresh clone of candle against a local cache of `google/gemma-4-E2B-it`, inspected the safetensors layout with `hf-fm`, and reported a picture "clearer now and bigger than my earlier comments suggested".
- **Outcome:** No reply as of 2026-11-01; this is the last comment on the thread. [PR #3608](https://github.com/huggingface/candle/pull/3608) (eddyb, "gemma4/text: get E2B, E4B, 12B and 31B to load"), a draft linked from the issue, has been idle since 2026-06-12.

---

**Quick follow-up.**

I cloned `candle` fresh, ran the example end-to-end against a local cache of `google/gemma-4-E2B-it`, and inspected the safetensors layout with `hf-fm`. The picture is clearer now and bigger than my earlier comments suggested (sorry!).

### Reproduction (matches the OP)

`$ git clone https://github.com/huggingface/candle.git`
`$ cd candle`
`$ cargo run --example gemma4 -- --cpu --prompt "test" --model-id google/gemma-4-E2B-it`
`...`
`retrieved the files in 176ms`
`Error: cannot find tensor model.embed_tokens.weight`

Same error @AlpineVibrations reported. The example's text-only path calls `TextModel::new(&config, vb)`, and `text.rs:532` does
`let vb_m = vb.pp("model");`, looking for `model.embed_tokens.weight`.

In this multimodal repo the actual tensor lives at `model.language_model.embed_tokens.weight`. Adding `vb.pp("language_model")`
in the text-only call site fixes this *immediate* error, but it's only the first of three.

### What `hf-fm inspect` reveals

`gemma-4-E2B-it` is architecturally closer to `gemma-3n` than to `gemma-3`; every text-decoder layer carries **AltUp/Laurel per-layer-input** weights that candle's `gemma4/text.rs` doesn't load and doesn't consume in the forward pass:

`$ hf-fm inspect google/gemma-4-E2B-it model.safetensors --filter "language_model.layers.0"`

`Tensor                                                          Dtype    Shape              Size     Params`
`model.language_model.layers.0.input_layernorm.weight            BF16     [1536]          3.0 KiB       1.5K`
`model.language_model.layers.0.layer_scalar                      BF16     [1]                 2 B          1   ← MISSING`
`model.language_model.layers.0.mlp.down_proj.weight              BF16     [1536, 6144]  18.00 MiB       9.4M`
`model.language_model.layers.0.mlp.gate_proj.weight              BF16     [6144, 1536]  18.00 MiB       9.4M`
`model.language_model.layers.0.mlp.up_proj.weight                BF16     [6144, 1536]  18.00 MiB       9.4M`
`model.language_model.layers.0.per_layer_input_gate.weight       BF16     [256, 1536]   768.0 KiB     393.2K   ← MISSING`
`model.language_model.layers.0.per_layer_projection.weight       BF16     [1536, 256]   768.0 KiB     393.2K   ← MISSING`
`model.language_model.layers.0.post_attention_layernorm.weight   BF16     [1536]          3.0 KiB       1.5K`
`model.language_model.layers.0.post_feedforward_layernorm.weight BF16     [1536]          3.0 KiB       1.5K`
`model.language_model.layers.0.post_per_layer_input_norm.weight  BF16     [1536]          3.0 KiB       1.5K   ← MISSING`
`model.language_model.layers.0.pre_feedforward_layernorm.weight  BF16     [1536]          3.0 KiB       1.5K`
`model.language_model.layers.0.self_attn.k_norm.weight           BF16     [256]             512 B        256`
`model.language_model.layers.0.self_attn.k_proj.weight           BF16     [256, 1536]   768.0 KiB     393.2K`
`model.language_model.layers.0.self_attn.o_proj.weight           BF16     [1536, 2048]   6.00 MiB       3.1M`
`model.language_model.layers.0.self_attn.q_norm.weight           BF16     [256]             512 B        256`
`model.language_model.layers.0.self_attn.q_proj.weight           BF16     [2048, 1536]   6.00 MiB       3.1M`
`model.language_model.layers.0.self_attn.v_proj.weight           BF16     [256, 1536]   768.0 KiB     393.2K`
`17/2011 tensors, 36.2M/5.12B params (filter: "language_model.layers.0")`

Four missing per-layer tensors per decoder layer (×35 layers), and one missing global table:

`$ hf-fm inspect google/gemma-4-E2B-it model.safetensors --filter "embed_tokens_per_layer"`

`Tensor                                             Dtype    Shape                Size     Params`
`model.language_model.embed_tokens_per_layer.weight BF16     [262144, 8960]   4.38 GiB      2.35B`
`1/2011 tensor, 2.35B/5.12B params (filter: "embed_tokens_per_layer")`

`embed_tokens_per_layer` is a `[vocab_size, num_layers × hidden_size_per_layer_input]` = `[262144, 35 × 256]` = `[262144, 8960]` lookup, ~2.35B parameters, providing a separate per-layer 256-dim embedding for each token. The "Effective 2B" naming convention is the giveaway, same trick as `gemma-3n-E2B`: 5.12B total params, ~2B "effective" at inference because the per-layer embedding lookup is sparse-access. The HF `transformers` reference for the forward flow is in the `Gemma4Text*` modules of `modeling_gemma4_text.py` (the `Gemma4TextScaledWordEmbedding` plus the per-layer-input branch inside `Gemma4DecoderLayer`).

### Three bugs, ordered

1. **Tensor name prefix** (text-only path, `text.rs:532`): `vb.pp("language_model")` missing. Reproduces immediately above.
2. **Per-layer MLP doubling:** `Gemma4TextConfig` is missing `num_kv_shared_layers` and `use_double_wide_mlp`; layers `15..34` (last 20 of 35) need `2 × intermediate_size` = `12288`. Details in [my earlier comment]. The `[12288, 1536]` shape on layer 15 is exactly what AlpineVibrations hit after the prefix fix.
3. **AltUp/Laurel forward pass:** the substantive lift. The four per-layer tensors plus `embed_tokens_per_layer` need to be loaded, and at each layer the residual must be projected into the per-layer space, gated against the per-layer embedding, normalized, scaled by `layer_scalar`, and mixed back. None of this is in `gemma4/text.rs` today. Even if (1)+(2) succeed, the forward pass produces incorrect logits without (3).

### Suggested fixing order

Practical fixing order would be:

- (1) + (2) standalone: small change, lets `gemma-4-E2B-it` *load* cleanly, with a clear `TODO: per-layer-input flow not implemented` comment and a runtime check that bails out with a descriptive error rather than producing garbage. A few hours.
- (3): the real fix, with a logit-equality test against the Python reference on a fixed input. Real architectural work, ~a day or two of careful porting + verification.

Hoping this helps...
