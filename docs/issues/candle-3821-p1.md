# candle #3821 — reply 1 (Draft)

- **Target PR:** https://github.com/huggingface/candle/pull/3821
- **Status:** Draft (not yet posted)
- **Context:** [PR #3821](https://github.com/huggingface/candle/pull/3821) self-answers [#3820](https://github.com/huggingface/candle/issues/3820)'s open question (the reporter, astorise, opened both, four hours apart) by hardcoding `general.architecture = "qwen3_5"` (with an underscore) as the only string `Architecture::from_name` recognizes for the dense model, in `candle-transformers/src/models/quantized_lm.rs` lines [97](https://github.com/huggingface/candle/blob/ceb78ca8aee88bc6d34c0b232e7b0e4272dbe7f2/candle-transformers/src/models/quantized_lm.rs#L97) and [116](https://github.com/huggingface/candle/blob/ceb78ca8aee88bc6d34c0b232e7b0e4272dbe7f2/candle-transformers/src/models/quantized_lm.rs#L116) (head commit [`ceb78ca8`](https://github.com/huggingface/candle/commit/ceb78ca8aee88bc6d34c0b232e7b0e4272dbe7f2) at draft time). `hf-fm inspect` against real, currently-hosted checkpoints reports `general.architecture=qwen35`, no underscore. The PR has zero reviews, zero comments, and no CI runs at draft time, so nobody has flagged the mismatch yet. This supersedes our earlier [#3820 reply drafts](candle-3820-p1.md): the more useful target now is the PR that would ship the wrong string, not the issue whose question that PR already (incorrectly) tried to answer.
- **Outcome:** —
- **Lesson / Leverage angle:** A new variant of the pattern [#3530](https://github.com/huggingface/candle/issues/3530) established: instead of "one binary fact blocks a ready-to-submit PR," this is "the PR got submitted anyway and the binary fact would have caught the bug before merge." Worth keeping as a distinct leverage angle for future archive entries: an open, unreviewed PR is sometimes a better target than the issue it closes, especially when it ships within hours of the issue and plausibly wasn't checked against a real file. Also notable: sibling [PR #3838](https://github.com/huggingface/candle/pull/3838) (fixes [#3837](https://github.com/huggingface/candle/issues/3837)) independently re-derives the correct `qwen35.` metadata namespace via a fallback in its own field reader, without anyone connecting that back to [PR #3821](https://github.com/huggingface/candle/pull/3821)'s dispatch-level string; the two PRs are inconsistent with each other, not just with the real files.
- **Accuracy flags:** `general.architecture=qwen35` and `qwen35moe` are copied verbatim from live `hf-fm inspect` output against `unsloth/Qwen3.5-4B-GGUF` and `unsloth/Qwen3.5-35B-A3B-GGUF`, re-verified live at draft time (2026-08-07) and independently cross-checked against the raw file bytes fetched directly over HTTP Range with `curl`, bypassing hf-fm and anamnesis entirely, not guessed. The `quantized_lm.rs` line numbers and the `SUPPORTED_ARCHITECTURES`/`from_name` snippets are copied from the PR's actual head commit ([`ceb78ca8aee88bc6d34c0b232e7b0e4272dbe7f2`](https://github.com/huggingface/candle/commit/ceb78ca8aee88bc6d34c0b232e7b0e4272dbe7f2)) via the GitHub contents API, not from the PR description. The `quantized_qwen3_5.rs` fallback in [PR #3838](https://github.com/huggingface/candle/pull/3838) (`s.replace("qwen3.", "qwen35.")`, lines [645 to 654](https://github.com/huggingface/candle/blob/6d94f28dfc470d97669765dbe76252e0224b6eaa/candle-transformers/src/models/quantized_qwen3_5.rs#L645-L654) at its own head commit [`6d94f28dfc470d97669765dbe76252e0224b6eaa`](https://github.com/huggingface/candle/commit/6d94f28dfc470d97669765dbe76252e0224b6eaa)) was read the same way. The claim that no code path handles `qwen35moe` at all was checked by grep across the full [PR #3821](https://github.com/huggingface/candle/pull/3821) diff, not inferred from the summary; the failure mode for a real MoE checkpoint under the current code is an explicit "unsupported gguf architecture" rejection, not a silent misroute, which is the safer of the two failure modes but still blocks loading. The two collapsed `--tree` dumps in the reply body are the full, unedited output of the same two live `hf-fm inspect` commands, captured fresh at draft time (2026-08-07); nothing in them is trimmed or hand-edited, unlike the `…`-elided metadata blocks earlier in this file.

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

`Architecture::from_name` currently matches on `"qwen3_5"` (with an underscore, [`quantized_lm.rs:116`](https://github.com/huggingface/candle/blob/ceb78ca8aee88bc6d34c0b232e7b0e4272dbe7f2/candle-transformers/src/models/quantized_lm.rs#L116), also listed in `SUPPORTED_ARCHITECTURES` at [line 97](https://github.com/huggingface/candle/blob/ceb78ca8aee88bc6d34c0b232e7b0e4272dbe7f2/candle-transformers/src/models/quantized_lm.rs#L97)), but both real checkpoints report `qwen35` and `qwen35moe`, no underscore. As written, `from_gguf` would reject both files with "unsupported gguf architecture", not dispatch them to `quantized_qwen3_5::ModelWeights`.

Interestingly, the sibling [PR #3838](https://github.com/huggingface/candle/pull/3838) already gets the field-level namespace right: its `md_get` closure in `quantized_qwen3_5.rs` (lines [645 to 654](https://github.com/huggingface/candle/blob/6d94f28dfc470d97669765dbe76252e0224b6eaa/candle-transformers/src/models/quantized_qwen3_5.rs#L645-L654)) tries `"qwen3.xxx"` first, and falls back to `"qwen35.xxx"` when the first lookup misses, which correctly resolves the real files' `qwen35.*` metadata keys once a `ModelWeights` is actually constructed. That fallback just never gets reached, because this PR's `Architecture::from_name` gate rejects the file one step earlier, before `from_gguf` ever calls into `quantized_qwen3_5.rs`.

Two things this suggests for `Architecture::from_name` and `SUPPORTED_ARCHITECTURES`: recognize `"qwen35"` (matching what real Unsloth-quantized checkpoints actually write), and possibly keep `"qwen3_5"` as an accepted alias too in case some other conversion path does emit the underscore form; and add a distinct arm for `"qwen35moe"` rather than leaving it unhandled entirely, since it resolves to neither the existing `Qwen3Moe`/`quantized_qwen3_moe` arm (a structurally different, full-attention model, as [#3820](https://github.com/huggingface/candle/issues/3820) itself already points out) nor the plain `qwen35` dense arm.

One open question from our earlier read of [#3820](https://github.com/huggingface/candle/issues/3820) is already resolved by [PR #3838](https://github.com/huggingface/candle/pull/3838), worth confirming here rather than re-raising: neither checkpoint's metadata has a distinct `partial_rotary_factor` key, only `rope.dimension_count` (64 on both variants); its `rotary_dim` computation reads `qwen3(5).rope.dimension_count` directly and falls back to the full head dimension when absent, so the GGUF path does not need `partial_rotary_factor` plumbed through separately from what [#3837](https://github.com/huggingface/candle/issues/3837) already covers for the safetensors path.

<details>
<summary>Full tensor tree, dense Qwen3.5-4B (664 lines), click to expand</summary>

```
$ hf-fm inspect unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-UD-IQ2_XXS.gguf --tree

  Repo:     unsloth/Qwen3.5-4B-GGUF
  File:     Qwen3.5-4B-UD-IQ2_XXS.gguf
  Source:   remote (170 range requests, 10.50 MiB fetched)
  Size:     1.42 GiB
  Metadata:
    general.architecture=qwen35
    general.base_model.0.name=Qwen3.5 4B
    general.base_model.0.organization=Qwen
    general.base_model.0.repo_url=https://huggingface.co/Qwen/Qwen3.5-4B
    general.base_model.count=1
    general.basename=Qwen3.5-4B
    general.file_type=19
    general.license=apache-2.0
    general.license.link=https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE
    general.name=Qwen3.5-4B
    general.quantization_version=2
    general.quantized_by=Unsloth
    general.repo_url=https://huggingface.co/unsloth
    general.size_label=4B
    general.type=model
    gguf.alignment=32
    gguf.version=3
    quantize.imatrix.chunks_count=80
    quantize.imatrix.dataset=unsloth_calibration_Qwen3.5-4B.txt
    quantize.imatrix.entries_count=248
    quantize.imatrix.file=Qwen3.5-4B-GGUF/imatrix_unsloth.gguf
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
    tokenizer.chat_template=
      {%- set image_count = namespace(value=0) %}
      {%- set video_count = namespace(value=0) %}
      {%- macro render_content(content, do_vision_count, is_system_content=false) %}
          {%- if content is string %}
              {{- content }}
          {%- elif content is iterable and content is not mapping %}
              {%- for item in content %}
                  {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                      {%- if is_system_content %}
                          {{- raise_exception('System message cannot contain images.') }}
                      {%- endif %}
                      {%- if do_vision_count %}
                          {%- set image_count.value = image_count.value + 1 %}
                      {%- endif %}
                      {%- if add_vision_id %}
                          {{- 'Picture ' ~ image_count.value ~ ': ' }}
                      {%- endif %}
                      {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
                  {%- elif 'video' in item or item.type == 'video' %}
                      {%- if is_system_content %}
                          {{- raise_exception('System message cannot contain videos.') }}
                      {%- endif %}
                      {%- if do_vision_count %}
                          {%- set video_count.value = video_count.value + 1 %}
                      {%- endif %}
                      {%- if add_vision_id %}
                          {{- 'Video ' ~ video_count.value ~ ': ' }}
                      {%- endif %}
                      {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
                  {%- elif 'text' in item %}
                      {{- item.text }}
                  {%- else %}
                      {{- raise_exception('Unexpected item type in content.') }}
                  {%- endif %}
              {%- endfor %}
          {%- elif content is none or content is undefined %}
              {{- '' }}
          {%- else %}
              {{- raise_exception('Unexpected content type.') }}
          {%- endif %}
      {%- endmacro %}
      {%- if not messages %}
          {{- raise_exception('No messages provided.') }}
      {%- endif %}
      {%- if tools and tools is iterable and tools is not mapping %}
          {{- '<|im_start|>system\n' }}
          {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
          {%- for tool in tools %}
              {{- "\n" }}
              {{- tool | tojson }}
          {%- endfor %}
          {{- "\n</tools>" }}
          {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
          {%- if messages[0].role == 'system' %}
              {%- set content = render_content(messages[0].content, false, true)|trim %}
              {%- if content %}
                  {{- '\n\n' + content }}
              {%- endif %}
          {%- endif %}
          {{- '<|im_end|>\n' }}
      {%- else %}
          {%- if messages[0].role == 'system' %}
              {%- set content = render_content(messages[0].content, false, true)|trim %}
              {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
          {%- endif %}
      {%- endif %}
      {%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
      {%- for message in messages[::-1] %}
          {%- set index = (messages|length - 1) - loop.index0 %}
          {%- if ns.multi_step_tool and message.role == "user" %}
              {%- set content = render_content(message.content, false)|trim %}
              {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
                  {%- set ns.multi_step_tool = false %}
                  {%- set ns.last_query_index = index %}
              {%- endif %}
          {%- endif %}
      {%- endfor %}
      {%- if ns.multi_step_tool %}
          {{- raise_exception('No user query found in messages.') }}
      {%- endif %}
      {%- for message in messages %}
          {%- set content = render_content(message.content, true)|trim %}
          {%- if message.role == "system" %}
              {%- if not loop.first %}
                  {{- raise_exception('System message must be at the beginning.') }}
              {%- endif %}
          {%- elif message.role == "user" %}
              {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
          {%- elif message.role == "assistant" %}
              {%- set reasoning_content = '' %}
              {%- if message.reasoning_content is string %}
                  {%- set reasoning_content = message.reasoning_content %}
              {%- else %}
                  {%- if '</think>' in content %}
                      {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                      {%- set content = content.split('</think>')[-1].lstrip('\n') %}
                  {%- endif %}
              {%- endif %}
              {%- set reasoning_content = reasoning_content|trim %}
              {%- if loop.index0 > ns.last_query_index %}
                  {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
              {%- else %}
                  {{- '<|im_start|>' + message.role + '\n' + content }}
              {%- endif %}
              {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
                  {%- for tool_call in message.tool_calls %}
                      {%- if tool_call.function is defined %}
                          {%- set tool_call = tool_call.function %}
                      {%- endif %}
                      {%- if loop.first %}
                          {%- if content|trim %}
                              {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                          {%- else %}
                              {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                          {%- endif %}
                      {%- else %}
                          {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                      {%- endif %}
                      {%- if tool_call.arguments is mapping %}
                          {%- for args_name in tool_call.arguments %}
                              {%- set args_value = tool_call.arguments[args_name] %}
                              {{- '<parameter=' + args_name + '>\n' }}
                              {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                              {{- args_value }}
                              {{- '\n</parameter>\n' }}
                          {%- endfor %}
                      {%- endif %}
                      {{- '</function>\n</tool_call>' }}
                  {%- endfor %}
              {%- endif %}
              {{- '<|im_end|>\n' }}
          {%- elif message.role == "tool" %}
              {%- if loop.previtem and loop.previtem.role != "tool" %}
                  {{- '<|im_start|>user' }}
              {%- endif %}
              {{- '\n<tool_response>\n' }}
              {{- content }}
              {{- '\n</tool_response>' }}
              {%- if not loop.last and loop.nextitem.role != "tool" %}
                  {{- '<|im_end|>\n' }}
              {%- elif loop.last %}
                  {{- '<|im_end|>\n' }}
              {%- endif %}
          {%- else %}
              {{- raise_exception('Unexpected message role.') }}
          {%- endif %}
      {%- endfor %}
      {%- if add_generation_prompt %}
          {{- '<|im_start|>assistant\n' }}
          {%- if enable_thinking is defined and enable_thinking is true %}
              {{- '<think>\n' }}
          {%- else %}
              {{- '<think>\n\n</think>\n\n' }}
          {%- endif %}
      {%- endif %}
    tokenizer.ggml.eos_token_id=248046
    tokenizer.ggml.model=gpt2
    tokenizer.ggml.padding_token_id=248055
    tokenizer.ggml.pre=qwen35

  ├── blk.
  │   ├── 0.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             Q2_K     [9216, 2560]  7.38 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 1.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             Q2_K     [9216, 2560]  7.38 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 10.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 11.
  │   │   ├── attn_k.weight               IQ2_XXS  [2560, 1024]  660.0 KiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_output.weight          IQ2_XXS  [4096, 2560]  2.58 MiB
  │   │   ├── attn_q.weight               IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               IQ3_XXS  [2560, 1024]  980.0 KiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   └── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   ├── 12.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_S    [9216, 2560]  7.21 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 13.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_S    [9216, 2560]  7.21 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 14.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 15.
  │   │   ├── attn_k.weight               IQ2_XXS  [2560, 1024]  660.0 KiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_output.weight          IQ2_XXS  [4096, 2560]  2.58 MiB
  │   │   ├── attn_q.weight               IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               IQ3_XXS  [2560, 1024]  980.0 KiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   └── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   ├── 16.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_S    [9216, 2560]  7.21 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 17.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 18.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 19.
  │   │   ├── attn_k.weight               IQ2_XXS  [2560, 1024]  660.0 KiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_output.weight          IQ2_XXS  [4096, 2560]  2.58 MiB
  │   │   ├── attn_q.weight               IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               IQ3_XXS  [2560, 1024]  980.0 KiB
  │   │   ├── ffn_down.weight             IQ3_S    [9216, 2560]  9.67 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   └── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   ├── 2.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ3_S    [9216, 2560]  9.67 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 20.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 21.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_S    [9216, 2560]  7.21 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 22.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 23.
  │   │   ├── attn_k.weight               IQ2_XXS  [2560, 1024]  660.0 KiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_output.weight          IQ2_XXS  [4096, 2560]  2.58 MiB
  │   │   ├── attn_q.weight               IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               IQ3_XXS  [2560, 1024]  980.0 KiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   └── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   ├── 24.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 25.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_S    [9216, 2560]  7.21 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 26.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 27.
  │   │   ├── attn_k.weight               IQ2_XXS  [2560, 1024]  660.0 KiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_output.weight          IQ2_XXS  [4096, 2560]  2.58 MiB
  │   │   ├── attn_q.weight               IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               IQ3_XXS  [2560, 1024]  980.0 KiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   └── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   ├── 28.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 29.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 3.
  │   │   ├── attn_k.weight               IQ2_XXS  [2560, 1024]  660.0 KiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_output.weight          IQ2_XXS  [4096, 2560]  2.58 MiB
  │   │   ├── attn_q.weight               IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               IQ3_XXS  [2560, 1024]  980.0 KiB
  │   │   ├── ffn_down.weight             IQ3_S    [9216, 2560]  9.67 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   └── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   ├── 30.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 31.
  │   │   ├── attn_k.weight               IQ2_XXS  [2560, 1024]  660.0 KiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_output.weight          IQ2_XXS  [4096, 2560]  2.58 MiB
  │   │   ├── attn_q.weight               IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               IQ3_XXS  [2560, 1024]  980.0 KiB
  │   │   ├── ffn_down.weight             IQ3_S    [9216, 2560]  9.67 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   └── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   ├── 4.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             Q2_K     [9216, 2560]  7.38 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 5.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             Q2_K     [9216, 2560]  7.38 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 6.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ3_S    [9216, 2560]  9.67 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   ├── 7.
  │   │   ├── attn_k.weight               IQ2_XXS  [2560, 1024]  660.0 KiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_output.weight          IQ2_XXS  [4096, 2560]  2.58 MiB
  │   │   ├── attn_q.weight               IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               IQ3_XXS  [2560, 1024]  980.0 KiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   └── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   ├── 8.
  │   │   ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │   │   ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │   │   ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │   │   ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │   │   ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │   │   ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  │   └── 9.
  │       ├── attn_gate.weight            IQ3_XXS  [2560, 4096]  3.83 MiB
  │       ├── attn_norm.weight            F32      [2560]  10.0 KiB
  │       ├── attn_qkv.weight             IQ2_XXS  [2560, 8192]  5.16 MiB
  │       ├── ffn_down.weight             IQ2_XXS  [9216, 2560]  5.80 MiB
  │       ├── ffn_gate.weight             IQ2_XXS  [2560, 9216]  5.80 MiB
  │       ├── ffn_up.weight               IQ2_XXS  [2560, 9216]  5.80 MiB
  │       ├── post_attention_norm.weight  F32      [2560]  10.0 KiB
  │       ├── ssm_a                       F32      [32]  128 B
  │       ├── ssm_alpha.weight            Q8_0     [2560, 32]  85.0 KiB
  │       ├── ssm_beta.weight             Q8_0     [2560, 32]  85.0 KiB
  │       ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │       ├── ssm_dt.bias                 F32      [32]  128 B
  │       ├── ssm_norm.weight             F32      [128]  512 B
  │       └── ssm_out.weight              Q4_K     [4096, 2560]  5.62 MiB
  ├── output_norm.weight  F32   [2560]  10.0 KiB
  └── token_embd.weight   Q5_K  [2560, 248320]  416.80 MiB
  426 tensors, 4.21B params
```

</details>

<details>
<summary>Full tensor tree, MoE Qwen3.5-35B-A3B (985 lines), click to expand</summary>

```
$ hf-fm inspect unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q3_K_S.gguf --tree

  Repo:     unsloth/Qwen3.5-35B-A3B-GGUF
  File:     Qwen3.5-35B-A3B-Q3_K_S.gguf
  Source:   remote (170 range requests, 10.50 MiB fetched)
  Size:     14.22 GiB
  Metadata:
    general.architecture=qwen35moe
    general.base_model.0.name=Qwen3.5 35B A3B
    general.base_model.0.organization=Qwen
    general.base_model.0.repo_url=https://huggingface.co/Qwen/Qwen3.5-35B-A3B
    general.base_model.count=1
    general.basename=Qwen3.5-35B-A3B
    general.file_type=11
    general.license=apache-2.0
    general.license.link=https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/LICENSE
    general.name=Qwen3.5-35B-A3B
    general.quantization_version=2
    general.quantized_by=Unsloth
    general.repo_url=https://huggingface.co/unsloth
    general.sampling.temp=1
    general.sampling.top_k=20
    general.sampling.top_p=0.95
    general.size_label=35B-A3B
    general.type=model
    gguf.alignment=32
    gguf.version=3
    quantize.imatrix.chunks_count=76
    quantize.imatrix.dataset=unsloth_calibration_Qwen3.5-35B-A3B.txt
    quantize.imatrix.entries_count=510
    quantize.imatrix.file=Qwen3.5-35B-A3B-GGUF/imatrix_unsloth.gguf
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
    tokenizer.chat_template=
      {%- set image_count = namespace(value=0) %}
      {%- set video_count = namespace(value=0) %}
      {%- macro render_content(content, do_vision_count, is_system_content=false) %}
          {%- if content is string %}
              {{- content }}
          {%- elif content is iterable and content is not mapping %}
              {%- for item in content %}
                  {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                      {%- if is_system_content %}
                          {{- raise_exception('System message cannot contain images.') }}
                      {%- endif %}
                      {%- if do_vision_count %}
                          {%- set image_count.value = image_count.value + 1 %}
                      {%- endif %}
                      {%- if add_vision_id %}
                          {{- 'Picture ' ~ image_count.value ~ ': ' }}
                      {%- endif %}
                      {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
                  {%- elif 'video' in item or item.type == 'video' %}
                      {%- if is_system_content %}
                          {{- raise_exception('System message cannot contain videos.') }}
                      {%- endif %}
                      {%- if do_vision_count %}
                          {%- set video_count.value = video_count.value + 1 %}
                      {%- endif %}
                      {%- if add_vision_id %}
                          {{- 'Video ' ~ video_count.value ~ ': ' }}
                      {%- endif %}
                      {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
                  {%- elif 'text' in item %}
                      {{- item.text }}
                  {%- else %}
                      {{- raise_exception('Unexpected item type in content.') }}
                  {%- endif %}
              {%- endfor %}
          {%- elif content is none or content is undefined %}
              {{- '' }}
          {%- else %}
              {{- raise_exception('Unexpected content type.') }}
          {%- endif %}
      {%- endmacro %}
      {%- if not messages %}
          {{- raise_exception('No messages provided.') }}
      {%- endif %}
      {%- if tools and tools is iterable and tools is not mapping %}
          {{- '<|im_start|>system\n' }}
          {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
          {%- for tool in tools %}
              {{- "\n" }}
              {{- tool | tojson }}
          {%- endfor %}
          {{- "\n</tools>" }}
          {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
          {%- if messages[0].role == 'system' %}
              {%- set content = render_content(messages[0].content, false, true)|trim %}
              {%- if content %}
                  {{- '\n\n' + content }}
              {%- endif %}
          {%- endif %}
          {{- '<|im_end|>\n' }}
      {%- else %}
          {%- if messages[0].role == 'system' %}
              {%- set content = render_content(messages[0].content, false, true)|trim %}
              {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
          {%- endif %}
      {%- endif %}
      {%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
      {%- for message in messages[::-1] %}
          {%- set index = (messages|length - 1) - loop.index0 %}
          {%- if ns.multi_step_tool and message.role == "user" %}
              {%- set content = render_content(message.content, false)|trim %}
              {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
                  {%- set ns.multi_step_tool = false %}
                  {%- set ns.last_query_index = index %}
              {%- endif %}
          {%- endif %}
      {%- endfor %}
      {%- if ns.multi_step_tool %}
          {{- raise_exception('No user query found in messages.') }}
      {%- endif %}
      {%- for message in messages %}
          {%- set content = render_content(message.content, true)|trim %}
          {%- if message.role == "system" %}
              {%- if not loop.first %}
                  {{- raise_exception('System message must be at the beginning.') }}
              {%- endif %}
          {%- elif message.role == "user" %}
              {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
          {%- elif message.role == "assistant" %}
              {%- set reasoning_content = '' %}
              {%- if message.reasoning_content is string %}
                  {%- set reasoning_content = message.reasoning_content %}
              {%- else %}
                  {%- if '</think>' in content %}
                      {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                      {%- set content = content.split('</think>')[-1].lstrip('\n') %}
                  {%- endif %}
              {%- endif %}
              {%- set reasoning_content = reasoning_content|trim %}
              {%- if loop.index0 > ns.last_query_index %}
                  {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
              {%- else %}
                  {{- '<|im_start|>' + message.role + '\n' + content }}
              {%- endif %}
              {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
                  {%- for tool_call in message.tool_calls %}
                      {%- if tool_call.function is defined %}
                          {%- set tool_call = tool_call.function %}
                      {%- endif %}
                      {%- if loop.first %}
                          {%- if content|trim %}
                              {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                          {%- else %}
                              {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                          {%- endif %}
                      {%- else %}
                          {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                      {%- endif %}
                      {%- if tool_call.arguments is mapping %}
                          {%- for args_name in tool_call.arguments %}
                              {%- set args_value = tool_call.arguments[args_name] %}
                              {{- '<parameter=' + args_name + '>\n' }}
                              {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                              {{- args_value }}
                              {{- '\n</parameter>\n' }}
                          {%- endfor %}
                      {%- endif %}
                      {{- '</function>\n</tool_call>' }}
                  {%- endfor %}
              {%- endif %}
              {{- '<|im_end|>\n' }}
          {%- elif message.role == "tool" %}
              {%- if loop.previtem and loop.previtem.role != "tool" %}
                  {{- '<|im_start|>user' }}
              {%- endif %}
              {{- '\n<tool_response>\n' }}
              {{- content }}
              {{- '\n</tool_response>' }}
              {%- if not loop.last and loop.nextitem.role != "tool" %}
                  {{- '<|im_end|>\n' }}
              {%- elif loop.last %}
                  {{- '<|im_end|>\n' }}
              {%- endif %}
          {%- else %}
              {{- raise_exception('Unexpected message role.') }}
          {%- endif %}
      {%- endfor %}
      {%- if add_generation_prompt %}
          {{- '<|im_start|>assistant\n' }}
          {%- if enable_thinking is defined and enable_thinking is false %}
              {{- '<think>\n\n</think>\n\n' }}
          {%- else %}
              {{- '<think>\n' }}
          {%- endif %}
      {%- endif %}
    tokenizer.ggml.eos_token_id=248046
    tokenizer.ggml.model=gpt2
    tokenizer.ggml.padding_token_id=248055
    tokenizer.ggml.pre=qwen35

  ├── blk.
  │   ├── 0.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 1.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 10.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 11.
  │   │   ├── attn_k.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_output.weight          Q8_0     [4096, 2048]  8.50 MiB
  │   │   ├── attn_q.weight               Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   └── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   ├── 12.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 13.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 14.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 15.
  │   │   ├── attn_k.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_output.weight          Q8_0     [4096, 2048]  8.50 MiB
  │   │   ├── attn_q.weight               Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   └── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   ├── 16.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 17.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 18.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 19.
  │   │   ├── attn_k.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_output.weight          Q8_0     [4096, 2048]  8.50 MiB
  │   │   ├── attn_q.weight               Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   └── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   ├── 2.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 20.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 21.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 22.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 23.
  │   │   ├── attn_k.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_output.weight          Q8_0     [4096, 2048]  8.50 MiB
  │   │   ├── attn_q.weight               Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   └── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   ├── 24.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 25.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 26.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 27.
  │   │   ├── attn_k.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_output.weight          Q8_0     [4096, 2048]  8.50 MiB
  │   │   ├── attn_q.weight               Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   └── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   ├── 28.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 29.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 3.
  │   │   ├── attn_k.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_output.weight          Q8_0     [4096, 2048]  8.50 MiB
  │   │   ├── attn_q.weight               Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   └── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   ├── 30.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 31.
  │   │   ├── attn_k.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_output.weight          Q8_0     [4096, 2048]  8.50 MiB
  │   │   ├── attn_q.weight               Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   └── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   ├── 32.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 33.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 34.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 35.
  │   │   ├── attn_k.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_output.weight          Q8_0     [4096, 2048]  8.50 MiB
  │   │   ├── attn_q.weight               Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   └── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   ├── 36.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 37.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 38.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 39.
  │   │   ├── attn_k.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_output.weight          Q8_0     [4096, 2048]  8.50 MiB
  │   │   ├── attn_q.weight               Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   └── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   ├── 4.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 5.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 6.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   ├── 7.
  │   │   ├── attn_k.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── attn_k_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_output.weight          Q8_0     [4096, 2048]  8.50 MiB
  │   │   ├── attn_q.weight               Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── attn_q_norm.weight          F32      [256]  1.0 KiB
  │   │   ├── attn_v.weight               Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   └── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   ├── 8.
  │   │   ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │   │   ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │   │   ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │   │   ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │   │   ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │   │   ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │   │   ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │   │   ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │   │   ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │   │   ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │   │   ├── ssm_a                       F32      [32]  128 B
  │   │   ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │   │   ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │   │   ├── ssm_dt.bias                 F32      [32]  128 B
  │   │   ├── ssm_norm.weight             F32      [128]  512 B
  │   │   └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  │   └── 9.
  │       ├── attn_gate.weight            Q8_0     [2048, 4096]  8.50 MiB
  │       ├── attn_norm.weight            F32      [2048]  8.0 KiB
  │       ├── attn_qkv.weight             Q8_0     [2048, 8192]  17.00 MiB
  │       ├── ffn_down_exps.weight        IQ3_S    [512, 2048, 256]  110.00 MiB
  │       ├── ffn_down_shexp.weight       Q8_0     [512, 2048]  1.06 MiB
  │       ├── ffn_gate_exps.weight        IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │       ├── ffn_gate_inp.weight         F32      [2048, 256]  2.00 MiB
  │       ├── ffn_gate_inp_shexp.weight   F32      [2048]  8.0 KiB
  │       ├── ffn_gate_shexp.weight       Q8_0     [2048, 512]  1.06 MiB
  │       ├── ffn_up_exps.weight          IQ3_XXS  [2048, 512, 256]  98.00 MiB
  │       ├── ffn_up_shexp.weight         Q8_0     [2048, 512]  1.06 MiB
  │       ├── post_attention_norm.weight  F32      [2048]  8.0 KiB
  │       ├── ssm_a                       F32      [32]  128 B
  │       ├── ssm_alpha.weight            Q8_0     [2048, 32]  68.0 KiB
  │       ├── ssm_beta.weight             Q8_0     [2048, 32]  68.0 KiB
  │       ├── ssm_conv1d.weight           F32      [4, 8192]  128.0 KiB
  │       ├── ssm_dt.bias                 F32      [32]  128 B
  │       ├── ssm_norm.weight             F32      [128]  512 B
  │       └── ssm_out.weight              Q8_0     [4096, 2048]  8.50 MiB
  ├── output.weight       Q6_K  [2048, 248320]  397.85 MiB
  ├── output_norm.weight  F32   [2048]  8.0 KiB
  └── token_embd.weight   Q6_K  [2048, 248320]  397.85 MiB
  733 tensors, 34.66B params
```

</details>

PS: `hf-fm` is a small Rust CLI for HuggingFace repos (no Python dependency, no weight data fetched). If you'd like to verify independently: `cargo install hf-fetch-model --features cli` installs it, and the two commands above are exactly what produced the output pasted here.
