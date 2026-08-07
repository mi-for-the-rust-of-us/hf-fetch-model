# candle #3820 — reply 2 (Draft)

- **Target issue:** https://github.com/huggingface/candle/issues/3820
- **Status:** Draft (not yet posted); superseded before posting by [candle-3821-p1.md](candle-3821-p1.md), which targets PR #3821 directly once live checking found the reporter had already tried to answer this question there, with the wrong string
- **Context:** Same evidence as [p1](candle-3820-p1.md), plus one addition: a closing pointer to `cargo install hf-fetch-model --features cli`, so astorise (or anyone else reading the thread) can reproduce the two `hf-fm inspect` commands independently rather than take the pasted output on trust. Matches the established archive convention (see the closing PS in [candle-3530-p1.md](candle-3530-p1.md)). Zero comments on the issue, no linked PRs, as of 2026-08-07.
- **Outcome:** —
- **Lesson / Leverage angle:** Same as [p1](candle-3820-p1.md): first candle-issue application of hf-fm v0.11.2's remote GGUF inspect, and the cleanest fit yet for the "one binary fact blocks a ready-to-submit PR" pattern that made [#3530](https://github.com/huggingface/candle/issues/3530) convert. The verifiability pointer matters more here than usual: astorise reads as a careful, hands-on contributor across a long run of issues, exactly the kind of reader who would want to check `general.architecture=qwen35`/`qwen35moe` themselves before wiring a PR around it.
- **Accuracy flags:** Same as [p1](candle-3820-p1.md): the two `general.architecture` values and every metadata key quoted below are copied verbatim from live `hf-fm inspect` output against real, currently-hosted Hub repos (`unsloth/Qwen3.5-4B-GGUF` and `unsloth/Qwen3.5-35B-A3B-GGUF`), first run 2026-08-06, not guessed or inferred from documentation. Independently re-verified live on 2026-08-07 (repo, filenames, `general.architecture` values, and the request/byte counts all unchanged). The `partial_rotary_factor` observation in the reply below is flagged inline as an open question, not asserted as fact; it has not been checked against llama.cpp's conversion script or a real `config.json`.

---

Ran `hf-fm inspect`'s new remote GGUF support (v0.11.2, reads the metadata KV table and tensor-info table over HTTP Range, no download) against both variants on the Hub:

```
$ hf-fm inspect unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-UD-IQ2_XXS.gguf

  Repo:     unsloth/Qwen3.5-4B-GGUF
  File:     Qwen3.5-4B-UD-IQ2_XXS.gguf
  Source:   remote (170 range requests, 10.50 MiB fetched)
  Size:     1.42 GiB
  Metadata:
    general.architecture=qwen35
    general.base_model.0.name=Qwen3.5 4B
    general.base_model.0.organization=Qwen
    …
    qwen35.attention.head_count=16
    qwen35.attention.head_count_kv=4
    qwen35.attention.key_length=256
    qwen35.attention.layer_norm_rms_epsilon=0.000001
    qwen35.attention.value_length=256
    qwen35.block_count=32
    qwen35.context_length=262144
    qwen35.embedding_length=2560
    qwen35.feed_forward_length=9216
    qwen35.full_attention_interval=4
    qwen35.rope.dimension_count=64
    qwen35.rope.freq_base=10000000
    qwen35.ssm.conv_kernel=4
    qwen35.ssm.group_count=16
    qwen35.ssm.inner_size=4096
    qwen35.ssm.state_size=128
    qwen35.ssm.time_step_rank=32
    …
  [chat template + per-tensor table omitted, not relevant here]
```

```
$ hf-fm inspect unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q3_K_S.gguf

  Repo:     unsloth/Qwen3.5-35B-A3B-GGUF
  File:     Qwen3.5-35B-A3B-Q3_K_S.gguf
  Source:   remote (170 range requests, 10.50 MiB fetched)
  Size:     14.22 GiB
  Metadata:
    general.architecture=qwen35moe
    general.base_model.0.name=Qwen3.5 35B A3B
    general.base_model.0.organization=Qwen
    …
    qwen35moe.attention.head_count=16
    qwen35moe.attention.head_count_kv=2
    qwen35moe.attention.key_length=256
    qwen35moe.attention.layer_norm_rms_epsilon=0.000001
    qwen35moe.attention.value_length=256
    qwen35moe.block_count=40
    qwen35moe.context_length=262144
    qwen35moe.embedding_length=2048
    qwen35moe.expert_count=256
    qwen35moe.expert_feed_forward_length=512
    qwen35moe.expert_shared_feed_forward_length=512
    qwen35moe.expert_used_count=8
    qwen35moe.full_attention_interval=4
    qwen35moe.rope.dimension_count=64
    qwen35moe.rope.freq_base=10000000
    qwen35moe.ssm.conv_kernel=4
    qwen35moe.ssm.group_count=16
    qwen35moe.ssm.inner_size=4096
    qwen35moe.ssm.state_size=128
    qwen35moe.ssm.time_step_rank=32
    …
  [chat template + per-tensor table omitted, not relevant here]
```

So the two spellings are `qwen35` (dense) and `qwen35moe` (MoE), confirming your note that `qwen3moe` isn't a substitute: it's a third, distinct string from both, not a spelling variant of either.

For [#3837](https://github.com/huggingface/candle/issues/3837)'s sparse-MoE config, the MoE file's own metadata already has the numbers: 256 experts, top-8 routing (`expert_used_count`), a 512-wide routed *and* shared expert FFN, and `full_attention_interval=4` (matching the `layer_types` alternation both of you describe). The Gated-DeltaNet/SSM fields [#3832](https://github.com/huggingface/candle/issues/3832) discusses (`ssm.conv_kernel`, `ssm.group_count`, `ssm.inner_size`, `ssm.state_size`, `ssm.time_step_rank`) are present under the same keys on both the dense and MoE checkpoints.

One thing worth double-checking rather than trusting from this alone: I don't see a distinct `partial_rotary_factor` key in either file's metadata, only `rope.dimension_count` (64 on both variants) and `rope.freq_base`. It's possible the GGUF conversion tooling already bakes the partial rotary into `rope.dimension_count` directly, rather than candle needing to read a separate field, but I haven't confirmed that against llama.cpp's conversion script or a real `config.json`; worth checking before assuming [#3837](https://github.com/huggingface/candle/issues/3837) needs to plumb `partial_rotary_factor` through at all on the GGUF path specifically (it may only matter for the dense-safetensors loader).

Happy to inspect other quant variants, or dump full tensor names/shapes via `--tree`, if that's useful for either PR; no download needed either way.

PS: `hf-fm` is a small Rust CLI for HuggingFace repos (no Python dependency, no weight data fetched). If you'd like to verify independently: `cargo install hf-fetch-model --features cli` installs it, and the two commands above are exactly what produced the output pasted here.
