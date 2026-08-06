# candle #3620 — reply 1 (Posted)

- **Target issue:** https://github.com/huggingface/candle/issues/3620
- **Status:** Posted (2026-06-30) — [issuecomment-4844687691](https://github.com/huggingface/candle/issues/3620#issuecomment-4844687691)
- **Context:** professor-moody independently reported a pickle memo-bomb: `Stack::memo_get`/`memo_put` deep-clone the entire `PyObject` tree on every `BinGet`/`BinPut`, so a crafted pickle that fetches and recombines the same memo slot doubles the node count per cycle (O(2^N) CPU/memory from linear input growth — the same amplification principle as our own #3617's CWE-1325 finding, filed one day earlier, independently). By the time we replied, AnkitNakhawa had already volunteered to implement a fix (2026-06-21) and opened a competing node-budget PR, [#3688](https://github.com/huggingface/candle/pull/3688) (ready for review 2026-06-30, same day as our comment). Our reply explained that PR #3628's byte-based working-set cap already bounds this exact amplification, with the `memo_get` guard code quoted verbatim.
- **Outcome:** [#3688](https://github.com/huggingface/candle/pull/3688) is still open, unreviewed, as of 2026-08-06. On 2026-07-29 Sébastien Astori independently reached the same overlap conclusion in [a comment on #3688](https://github.com/huggingface/candle/pull/3688#issuecomment-5117494997) (naming the specific `rejects_memo_replay_amplification` test in #3628 that covers the same PoC), and merged #3628 into his own serving-oriented fork ([astorise/candle#37](https://github.com/astorise/candle/pull/37)) the next day, ahead of official review. See [`anamnesis/docs/PLAN-candle-3617-pickle-hardening.md`](../../../anamnesis/docs/PLAN-candle-3617-pickle-hardening.md) for the full downstream-adoption record.
- **Lesson / Leverage angle:** Two independently-authored fixes (byte-budget vs. node-count) converging on the same root cause, within days of each other, is a strong signal the bug is real and the fix is overdue for maintainer attention — not evidence of confusion. Worth a light nudge toward consolidation (e.g. a comment pointing #3688's author/reviewer at the overlap) rather than treating it as competing claims.
- **Accuracy flags:** None — the `memo_get` code snippet quoted below is copied verbatim from PR #3628's source at the time of posting.

---

This `BINGET`-replay amplification is already bounded in [#3628](https://github.com/huggingface/candle/pull/3628) (open, currently set to close [#3617](https://github.com/huggingface/candle/issues/3617)).

Flagging it here because there's a separate node-budget fix in flight for this thread — the two approaches are worth comparing before either lands.

**How [#3628](https://github.com/huggingface/candle/pull/3628) bounds it.** It adds a cumulative working-set floor (`PICKLE_MAX_WORKING_SET`, 512 MiB): every pushed value's heap, plus the deep size of every memo clone, is charged to it.

The key detail is *when*: the memo clone is charged **before** the clone happens, so an over-budget replay is rejected without ever allocating the duplicate.

That's exactly the `memo_get` site whose dev-TODO your report quotes — *"Maybe we should use refcounting rather than doing potential large clones here"* ([CWE-1325](https://cwe.mitre.org/data/definitions/1325.html)):

```rust
// Clones a memoised value for `BINGET` replay, charging its deep heap to the
// working set *before* the clone so an over-budget replay is rejected
// without allocating the duplicate. Returns the value and its depth.
fn memo_get(&mut self, id: u32) -> Result<(Object, u32)> {
    let (bytes, depth) = match self.memo.get(&id) {
        None => crate::bail!("missing object in memo {id}"),
        Some((obj, depth)) => (Self::deep_size(obj), *depth),
    };
    self.charge(bytes)?;                       // <- charged before the clone
    match self.memo.get(&id) {
        None => crate::bail!("missing object in memo {id}"),
        Some((obj, _)) => Ok((obj.clone(), depth)),
    }
}
```

**Both paths you report hit the same floor:**

- **Path A** — the `Tuple2` doubling grows the charged heap every cycle, so the floor trips long before the O(2^N) CPU stall.
- **Path B** — the `Build`-dict doubling does the same, well before the ~92 GB RSS.

The `BINPUT` side (`memo_put`) charges its `deep_size` too.

There's a `rejects_memo_replay_amplification` regression test built from your Path A shape, and the design mirrors the cargo-fuzzed [anamnesis](https://github.com/PCfVW/anamnesis) pickle VM (libFuzzer: 6491 runs, zero crashes, RSS bounded ~1 GB).

**vs. the node-budget approach:** [#3628](https://github.com/huggingface/candle/pull/3628) bounds by **heap bytes** rather than node count, and folds this amplification together with the recursion/`Drop` depth cap ([#3617](https://github.com/huggingface/candle/issues/3617)) and the `LONG1` shift-overflow panic ([#3619](https://github.com/huggingface/candle/issues/3619)) into one change.

So it could close the whole pickle cluster ([#3617](https://github.com/huggingface/candle/issues/3617) / [#3619](https://github.com/huggingface/candle/issues/3619) / [#3620](https://github.com/huggingface/candle/issues/3620)) in a single merge. Happy to defer to whichever the maintainers prefer.
