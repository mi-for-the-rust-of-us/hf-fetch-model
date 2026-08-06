# candle #3617 — reply 3 (Posted)

- **Target issue:** https://github.com/huggingface/candle/issues/3617
- **Status:** Posted (2026-07-30) — [issuecomment-5129575862](https://github.com/huggingface/candle/issues/3617#issuecomment-5129575862)
- **Context:** Six-week follow-up after [p2](candle-3617-p2.md). PR [#3628](https://github.com/huggingface/candle/pull/3628) still had zero maintainer reviews/comments. This post surfaces, in one place, that #3628 also fixes the two independently-filed sibling reports ([#3619](candle-3619-p1.md), [#3620](candle-3620-p1.md)) that had each accumulated their own separate replies by this point, and notes Sébastien Astori's independent confirmation (on PR [#3688](https://github.com/huggingface/candle/pull/3688), the competing #3620 fix) plus his adoption of #3628 into his own fork ahead of official review.
- **Outcome:** No maintainer response as of 2026-08-06 (verified live at drafting time for this archive entry).
- **Lesson / Leverage angle:** When a PR stalls for weeks with zero engagement, a single consolidating comment that names every issue it closes (with links) lowers the bar for a maintainer to act — they don't have to reconstruct the cross-issue picture themselves before merging.
- **Accuracy flags:** None.

---

FYI for whoever picks this up: PR [#3628](https://github.com/huggingface/candle/pull/3628), which fixes this issue, also already fixes two more independently-filed reports from the same window, neither formally linked to it: [#3619](https://github.com/huggingface/candle/issues/3619) (`LONG1` overflow panic, the "bonus" fix libFuzzer surfaced while validating this PR's other guards) and [#3620](https://github.com/huggingface/candle/issues/3620) (the pickle memo-bomb / algorithmic-complexity DoS).

I flagged the [#3619](https://github.com/huggingface/candle/issues/3619) overlap directly a month ago, in [a comment there](https://github.com/huggingface/candle/issues/3619#issuecomment-4844689184), and offered to add Closes #3619 to this PR if maintainers wanted all three vectors ([#3617](https://github.com/huggingface/candle/issues/3617) / [#3619](https://github.com/huggingface/candle/issues/3619) / [#3620](https://github.com/huggingface/candle/issues/3620)) tracked through one change. No response since, and the PR body still doesn't reference it.

More recently, and independently, a third party reached a similar conclusion for the [#3620](https://github.com/huggingface/candle/issues/3620) side: on 2026-07-29, [Sébastien Astori commented on #3688](https://github.com/huggingface/candle/pull/3688#issuecomment-5117494997) (the competing memo_get-specific fix for [#3620](https://github.com/huggingface/candle/issues/3620)), noting that this PR's byte-based working-set cap already bounds the same amplification [#3688](https://github.com/huggingface/candle/pull/3688) targets, down to naming the rejects_memo_replay_amplification test that covers it. He merged this PR into his own serving-oriented fork ([astorise/candle#37](https://github.com/astorise/candle/pull/37)) the next day.

Happy to add Closes #3619 (and, pending a maintainer's own check against [#3620](https://github.com/huggingface/candle/issues/3620)'s PoC, possibly Closes #3620) to this PR if that's useful for consolidating the tracking. The offer from a month ago still stands.
