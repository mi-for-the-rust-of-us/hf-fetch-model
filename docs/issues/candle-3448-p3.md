# candle #3448 — reply 3 (Posted)

- **Target issue:** https://github.com/huggingface/candle/issues/3448
- **Status:** Posted 2026-04-19 06:57 UTC as [this comment](https://github.com/huggingface/candle/issues/3448#issuecomment-4275381957) (this line said 2026-04-16, and the title "draft", until 2026-11-01). The body below was then replaced with the posted text; the archived draft differed in wording only (dashes, one emoji, "unblocks" softened to "should unblock").
- **Context:** Follow-up to @AlpineVibrations' reported shape mismatch on `model.language_model.layers.15.mlp.gate_proj.weight` (expected `[6144, 1536]`, got `[12288, 1536]`). Posted because [p2](candle-3448-p2.md) had been sitting for nearly four days with no reply, and subsequent investigation revealed its diagnosis was oversimplified.
- **Verified against:** [transformers `modeling_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py) — the layer-15 boundary in the user's error matches the `first_kv_shared_layer_idx = 35 - 20 = 15` calculation exactly.
- **Outcome:** @AlpineVibrations replied on 2026-04-21: "I tried that and it's still the same error as I reported above. I only added what you specified and it didn't seam to fix it." Answered by [p5](candle-3448-p5.md) (posted); the draft [p4](candle-3448-p4.md) was not used.

---

**Following up:** it's not hardcoded; but candle's gemma4 implementation is missing **two config fields** entirely.

### Root cause (verified against [transformers' `modeling_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py))

The reference Python implementation does:

```python
first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
```

For `gemma-4-E2B-it`:
- `num_hidden_layers = 35`
- `num_kv_shared_layers = 20`
- `first_kv_shared_layer_idx = 35 - 20 = 15`
- Layers 15..34 (last 20) use **2 × intermediate_size = 12288**
- Layers 0..14 (first 15) use **intermediate_size = 6144**

The boundary at index 15 exactly explains your error: layer 15 is the first KV-shared (double-wide) layer, which is why you hit the shape mismatch there.

### What candle is missing

candle's [`Gemma4TextConfig`](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/gemma4/config.rs) is missing two fields:

- `num_kv_shared_layers: usize`
- `use_double_wide_mlp: bool`

And [`DecoderLayer::new()` in `gemma4/text.rs`](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/gemma4/text.rs) passes a fixed `cfg.intermediate_size` to every MLP (no per-layer branching), so every layer gets 6144, and the weight file's 12288 for the last 20 layers fails to load.

## Proposed fix

```rust
// config.rs — in Gemma4TextConfig
#[serde(default)]
pub num_kv_shared_layers: usize,
#[serde(default)]
pub use_double_wide_mlp: bool,

// text.rs — in DecoderLayer::new(), replace the MLP instantiation
let first_kv_shared_layer_idx = cfg.num_hidden_layers.saturating_sub(cfg.num_kv_shared_layers);
let is_kv_shared = first_kv_shared_layer_idx > 0 && layer_idx >= first_kv_shared_layer_idx;
let effective_intermediate = if cfg.use_double_wide_mlp && is_kv_shared {
    cfg.intermediate_size * 2
} else {
    cfg.intermediate_size
};
let mlp = MLP::new(
    cfg.hidden_size,
    effective_intermediate,
    cfg.hidden_activation,
    false,
    vb.pp("mlp"),
)?;
```

This is a candle model-code bug in `candle-transformers/src/models/gemma4/`, not an example bug; fixing it should unblock every Gemma 4 variant (E2B, E4B, etc.) not just the specific example...
