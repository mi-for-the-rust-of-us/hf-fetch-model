# Issue replies archive

This folder archives comments drafted or posted on upstream issues (primarily [huggingface/candle](https://github.com/huggingface/candle/issues), [unslothai/unsloth](https://github.com/unslothai/unsloth/issues), and related repos) where `hf-fetch-model`'s inspection features helped diagnose a problem. The dual purpose is:

1. **Reference**: keep a searchable trail of which issues we engaged with, what we said, and what happened.
2. **Case-study source material**: the v0.10.0 roadmap includes a first-batch of `docs/case-studies/` written from these replies (see [cache-management-roadmap.md](../roadmaps/cache-management-roadmap.md)). The archive is the raw material; case studies synthesize them into narrative form. First two shipped in v0.10.5: [per-layer shape variation in Gemma 4](../case-studies/gemma4-per-layer-shape-variation.md) (from the #3448 thread) and [reconstructing an OOM from a crash log](../case-studies/qwen3-next-memory-forensics.md) (from #3530), both written to be honest about how far the threads actually got.

## Thread status

*Checked against GitHub on 2026-11-01.* One row per thread. Threads that are closed, or where nothing is left for us to do, are listed under **Done**; threads with an upstream issue or PR still open are under **Open**, with whatever would move them on. The per-file list is [below](#current-archive).

### Open

| Thread | Opened by | Last activity | Attached PR | Next step for us |
|---|---|---|---|---|
| [candle #3617](https://github.com/huggingface/candle/issues/3617): unbounded pickle-VM working set | us | our comment, 2026-07-30 | **our [#3628](https://github.com/huggingface/candle/pull/3628)**: open, no review since 2026-06-18 | Ask for a review on #3628: it is now the only open fix for #3620's memo replay too (see next row) |
| [candle #3620](https://github.com/huggingface/candle/issues/3620): pickle memo bomb | professor-moody | our comment, 2026-06-30 | [#3688](https://github.com/huggingface/candle/pull/3688) **closed unmerged** by its author 2026-08-21, after astorise noted that #3628 covers the same attack | Same as above |
| [candle #3530](https://github.com/huggingface/candle/issues/3530): memory left unused on Spark | sempervictus | our [p3](candle-3530-p3.md), 2026-11-01, answering maintainer ivarflakstad's 2026-08-16 question (do our numbers look right, to close) | none | None: the maintainer decides whether to close |
| [candle #3619](https://github.com/huggingface/candle/issues/3619): `Long1` panic | professor-moody | our comment, 2026-06-30 | [#3626](https://github.com/huggingface/candle/pull/3626) (satyakwok): open, no review | None |
| [candle #3448](https://github.com/huggingface/candle/issues/3448): Gemma 4 load | AlpineVibrations | our comment ([p5](candle-3448-p5.md)), 2026-04-26 | [#3608](https://github.com/huggingface/candle/pull/3608) (eddyb): draft, idle since 2026-06-12 | None |
| [candle #3401](https://github.com/huggingface/candle/issues/3401): LightOnOCR-2-1B | Abhinav5132 | our comment, 2026-04-16 | [#3772](https://github.com/huggingface/candle/pull/3772) (the reporter): open | None |
| [candle #2875](https://github.com/huggingface/candle/issues/2875): F8_E4M3 quantize error | AlpineVibrations | our comment, 2026-08-24 | none | None |

### Done

| Thread | How it ended | Files |
|---|---|---|
| [hf-fetch-model #21](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues/21): reclaim duplicated Windows copies | closed as completed by us, 2026-11-01, after v0.12.6 | p1–p3 |
| [hf-fetch-model #16](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues/16): `du` ignores `blobs/` | closed as completed, 2026-10-02 (by a commit keyword, explained in p2) | p1–p2 |
| [unsloth #6071](https://github.com/unslothai/unsloth/issues/6071): truncated SSM tensors in Qwen3.5/3.6 GGUFs | closed as completed, 2026-09-15, adopting our findings | p1–p3 |
| [candle PR #3821](https://github.com/huggingface/candle/pull/3821): Qwen3.5 GGUF dispatch | closed unmerged by astorise, 2026-08-17; the work moved to a separate crate | p1 |
| [candle #3820](https://github.com/huggingface/candle/issues/3820): same, as an issue | still open upstream, but its only PR is #3821 above; nothing for us | p1–p2, never posted |
| hf-hub, Windows cache directory | never filed: [hf-hub #193](https://github.com/huggingface/hf-hub/pull/193), merged 2026-08-05, made it moot | [not-filed draft](hf-hub-windows-cache-dir-not-filed.md) |

**Refreshing this table.** For each thread, `gh api repos/<owner>/<repo>/issues/<n>` gives `state`, `state_reason` and `closed_at`; `.../issues/<n>/comments` gives the last activity; `.../issues/<n>/timeline` lists cross-referenced PRs (events of type `cross-referenced` whose source is a pull request), and `gh api repos/<owner>/<repo>/pulls/<n>` gives a PR's state, merge and reviews. Update the "checked" date with the table, and move a row to **Done** when its thread closes; the files themselves never move, so no link breaks.

## File naming

```
<upstream-repo>-<issue-number>-p<N>.md
```

- `<upstream-repo>` — short identifier (`candle`, `hf-hub`, `transformers`, …)
- `<issue-number>` — the upstream issue/PR number (no `#`)
- `p<N>` — post/reply index within that thread (`p1` = our first comment, `p2` = second, …)

Examples: `candle-3448-p1.md`, `candle-3448-p3.md`, `candle-3401-p1.md`.

## File structure

Each file starts with a lightweight metadata header followed by the reply body verbatim:

```markdown
# <upstream-repo> #<issue-number> — reply <N> (<status>)

- **Target issue:** <URL>
- **Status:** <Posted | Draft | Superseded>  (plus date or superseding file)
- **Context:** one or two sentences: who asked, what was stuck, what we showed
- **Outcome:** (for posted) what the OP/maintainer said in response
- **Lesson / Leverage angle:** (optional) anything worth noting for future replies

---

<reply body exactly as posted, or as drafted>
```

The separator line (`---`) marks the boundary between metadata and the reply body, so the body is pastable into GitHub as-is.

## Status taxonomy

| Status | Meaning |
|--------|---------|
| **Posted** | Sent to the upstream issue; matches what's live there verbatim |
| **Draft** | Written but not yet posted (pending review, release timing, etc.) |
| **Superseded** | Was posted, but later investigation revealed a better diagnosis. The metadata must link to the superseding file. Do **not** delete superseded files — they document our learning curve and protect against re-making the same mistake. |

## Flagging practice

When archiving, **flag anything that's not 100% faithful or not 100% correct**:

- **Posted content with inaccuracies**: note them in the metadata. If a diagnosis turned out to be wrong or oversimplified, say so. Link to the corrected follow-up. Example: [candle-3448-p2.md](candle-3448-p2.md) — posted but oversimplified.
- **Cosmetic edits vs literal tool output**: if the archived reply trims or reformats `hf-fm` output (e.g., brace-notation shortcuts like `{k,o,q,v}_proj`), flag it so the archive isn't mistaken for raw output. Example: [candle-3401-p1.md](candle-3401-p1.md).
- **Verified vs guessed claims**: if a claim was verified against a specific reference (transformers' source, a paper, a config file), note the source. If something was a reasoned guess, say so explicitly. Example: [candle-3448-p3.md](candle-3448-p3.md) — "verified against `modeling_gemma4.py`".
- **Best-guess inputs (when the OP omits key context)**: some replies have to identify the repo, config, or model from a fingerprint when the OP didn't share it. This is a *different* dimension from "verified vs guessed claims" above — the claims downstream may be perfectly verified, but they rest on a guessed input premise. Flag prominently in the metadata that downstream takeaways are conditional on the guess being right; distinguish *wrong variant* (numbers shift ≤ a few %, takeaways hold) from *wrong family* (direction of argument holds, exact numbers shift more); include an explicit recovery hint inviting the OP to share the repo line, `config.json`, or `AutoConfig.from_pretrained(...).to_dict()` output. Example: [candle-3530-p2.md](candle-3530-p2.md) — guessed `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` from the crash log fingerprint; takeaways framed as conditional with a two-failure-mode caveat at the end.

## Workflow

1. **Before posting**: draft the reply as `<repo>-<issue>-p<N>.md` with status `Draft`. Review offline. Post.
2. **After posting**: update the status to `Posted`, add the posting date, and fill in `Outcome:` (blank for now; fill it in when the OP/maintainer replies).
3. **If superseded**: mark as `Superseded`, add a `See: <newer-file>` line in metadata, but keep the original file intact.

## Current archive

| File | Target | Status |
|------|--------|--------|
| [candle-3448-p1.md](candle-3448-p1.md) | candle #3448 — Gemma 4 download/load | Posted (2026-04-14) |
| [candle-3448-p2.md](candle-3448-p2.md) | candle #3448 — shape mismatch | Posted (2026-04-15), superseded by p3 |
| [candle-3448-p3.md](candle-3448-p3.md) | candle #3448 — `use_double_wide_mlp` bug | Posted (2026-04-19) |
| [candle-3448-p4.md](candle-3448-p4.md) | candle #3448 — ask for the reporter's diff after p3 "didn't fix it" | Draft, never posted; superseded by p5 |
| [candle-3448-p5.md](candle-3448-p5.md) | candle #3448 — fresh-clone reproduction; the picture is bigger than p1–p3 said | Posted (2026-04-26); archived 2026-11-01 from the live text |
| [candle-3401-p1.md](candle-3401-p1.md) | candle #3401 — LightOnOCR-2-1B | Posted |
| [candle-3530-p1.md](candle-3530-p1.md) | candle #3530 — 80B NVFP4 fit / double-mapping diagnostic ask | Posted (2026-05-12) |
| [candle-3530-p2.md](candle-3530-p2.md) | candle #3530 — best-guess repo identification + corrected on-disk math | Posted (2026-05-13) |
| [candle-3530-p3.md](candle-3530-p3.md) | candle #3530 — answering the maintainer: figures re-checked, the ~14 GiB is a remainder, the 18 GiB KV pool is the configured context (xinfer at `cf9b63b`) | Posted (2026-11-01) |
| [candle-3617-p1.md](candle-3617-p1.md) | candle #3617 — `.pth` pickle-VM DoS (issue we opened) | Posted |
| [candle-3617-p2.md](candle-3617-p2.md) | candle #3617 — PR #3628 announced | Posted |
| [candle-3617-p3.md](candle-3617-p3.md) | candle #3617 — consolidating sibling reports #3619/#3620 | Posted |
| [candle-3619-p1.md](candle-3619-p1.md) | candle #3619 — `LONG1` overflow panic, already fixed by #3628 | Posted |
| [candle-3620-p1.md](candle-3620-p1.md) | candle #3620 — pickle memo-bomb, already bounded by #3628 | Posted |
| [candle-3820-p1.md](candle-3820-p1.md) | candle #3820 — Qwen3.5 GGUF `general.architecture` strings, live-verified | Draft, superseded by candle-3821-p1 |
| [candle-3820-p2.md](candle-3820-p2.md) | candle #3820 — same, plus verification pointer | Draft, superseded by candle-3821-p1 |
| [candle-3821-p1.md](candle-3821-p1.md) | candle PR #3821 — `qwen3_5` vs. real `qwen35`/`qwen35moe` string mismatch, live-verified | Posted, confirmed by astorise |
| [candle-2875-p1.md](candle-2875-p1.md) | candle #2875 — FP8 `tensor-tools quantize` error; correct dtype breakdown, stale conclusion | Posted (2026-04-15), superseded by p2 |
| [candle-2875-p2.md](candle-2875-p2.md) | candle #2875 — corrected: fixed by #2989/0.10.0; per-tensor FP8, upcast faithful | Posted (2026-08-24) |
| [unsloth-6071-p1.md](unsloth-6071-p1.md) | unsloth #6071 — Qwen3.5/3.6 GGUF "truncated SSM tensor" claim, refuted across 4 files via remote GGUF header inspection | Posted |
| [unsloth-6071-p2.md](unsloth-6071-p2.md) | unsloth #6071 — cross-repo evidence (OpenWhispr#939) refines root cause toward a `llama.cpp` NextN-layer issue | Posted, partially inaccurate (see p3) |
| [unsloth-6071-p3.md](unsloth-6071-p3.md) | unsloth #6071 — bisected to a stale `llama.cpp` build (fixed 2026-05-16, `b9180`); confirms p2's hypothesis in source and corrects p2's framing | Posted (2026-09-16) |
| [hf-fetch-model-16-p1.md](hf-fetch-model-16-p1.md) | hf-fetch-model #16 (issue we opened) — `du` ignores `blobs/`; 189 GiB of 825 GiB under-reported | Posted (2026-10-01); Unix claim corrected by p2 |
| [hf-fetch-model-16-p2.md](hf-fetch-model-16-p2.md) | hf-fetch-model #16 — accidental auto-close explained; fix measured byte-exact; Unix correction; fixes 2/3 moved to #21 | Posted (2026-10-10) |
| [hf-fetch-model-21-p1.md](hf-fetch-model-21-p1.md) | hf-fetch-model #21 (issue we opened) — reclaim duplicated snapshot copies on Windows (hard-link pointers, reclaim command) | Posted (2026-10-10) |
| [hf-fetch-model-21-p2.md](hf-fetch-model-21-p2.md) | hf-fetch-model #21 — duplication measured by content, both open questions tested, the v0.12.5/v0.12.6 plan | Posted (2026-10-31); archived 2026-11-01 from the live text |
| [hf-fetch-model-21-p3.md](hf-fetch-model-21-p3.md) | hf-fetch-model #21 — v0.12.6 shipped: `cache dedupe`, 154.16 GiB reclaimed on the reporting cache | Posted (2026-11-01); issue closed |
| [hf-hub-windows-cache-dir-not-filed.md](hf-hub-windows-cache-dir-not-filed.md) | hf-hub — Windows cache-directory bug | Never filed; made moot by hf-hub #193 (merged 2026-08-05) |
