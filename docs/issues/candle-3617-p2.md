# candle #3617 — reply 2 (Posted)

- **Target issue:** https://github.com/huggingface/candle/issues/3617
- **Status:** Posted (2026-06-18) — [issuecomment-4740892210](https://github.com/huggingface/candle/issues/3617#issuecomment-4740892210)
- **Context:** Follow-up to the issue body ([p1](candle-3617-p1.md)), posted the same day PR [#3628](https://github.com/huggingface/candle/pull/3628) opened. Announces the PR and summarizes its three controls (working-set floor, depth cap, `BINUNICODE` payload cap) plus the `LONG1` overflow libFuzzer surfaced as a bonus find while validating the other guards — the same bug independently reported four days later in [#3619](https://github.com/huggingface/candle/issues/3619).
- **Outcome:** See [p3](candle-3617-p3.md) — no maintainer engagement in the six weeks between this post and the next.
- **Lesson / Leverage angle:** Posting the PR announcement directly on the originating issue (rather than relying on GitHub's automatic cross-link) keeps the thread self-contained for anyone reading the issue in isolation.
- **Accuracy flags:** None.

---

Opened #3628 with the fix. It mirrors the `GGUF_MAX_VALUE_DEPTH` control from #3585, applied to the pickle `Object`:

- a cumulative working-set floor (charged *before* memo clones, so over-budget `BINGET` replays are rejected pre-allocation) for the [CWE-1325](https://cwe.mitre.org/data/definitions/1325.html) amplification,
- a construction-depth cap (64) for the [CWE-674](https://cwe.mitre.org/data/definitions/674.html) recursion / `Drop` overflow,
- a per-payload `BINUNICODE` cap ([CWE-770](https://cwe.mitre.org/data/definitions/770.html)).

Always-on, O(1) per opcode, no public API change, legitimate files unaffected. Five unit tests cover the three vectors above plus a positive parse.

While fuzzing the change, libFuzzer also surfaced a pre-existing `LONG1` arithmetic-overflow panic ([CWE-190](https://cwe.mitre.org/data/definitions/190.html), crafted `n_bytes >= 9`) — fixed in the same PR. The candle adaptation was libFuzzer-validated locally (seeded with the PoC shapes, RSS-bounded; 6491 runs, zero crashes).
