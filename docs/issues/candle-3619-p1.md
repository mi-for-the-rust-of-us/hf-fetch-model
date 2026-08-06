# candle #3619 — reply 1 (Posted)

- **Target issue:** https://github.com/huggingface/candle/issues/3619
- **Status:** Posted (2026-06-30) — [issuecomment-4844689184](https://github.com/huggingface/candle/issues/3619#issuecomment-4844689184)
- **Context:** professor-moody (CRUCIBLE security-advisory series) independently reported a deterministic panic in `candle-core`'s pickle `Long1` opcode handler: an attacker-controlled `n_bytes` with no `> 8` guard reaches `(byte as i64) << 64` when `n_bytes >= 9`, and Rust's shift-left panics unconditionally on a shift ≥ the target width (CWE-190). Filed one day after our own #3617 (same file, same class of DoS), independently — no cross-reference either way at filing time.
- **Outcome:** No reply from professor-moody or a maintainer since. Still open as of 2026-08-06.
- **Lesson / Leverage angle:** An independent, fuzzing-derived discovery of a bug our own fuzzing had *also* found (while validating PR #3628's guards) is strong convergent evidence the fix is real and worth landing — two unrelated processes hit the identical panic site. Good precedent for citing "already fixed in an open PR" replies: point at the exact commit's guard code, not just the PR title, so a maintainer skimming the thread can verify the claim without opening the diff.
- **Accuracy flags:** None — the PR #3628 code snippet quoted below is copied verbatim from the actual PR source at the time of posting.

---

Heads-up — this exact `LONG1` overflow is already fixed in [#3628](https://github.com/huggingface/candle/pull/3628) (open, currently set to close [#3617](https://github.com/huggingface/candle/issues/3617)).

It surfaced the same way you describe: while libFuzzing the working-set / depth / payload bounds that [#3628](https://github.com/huggingface/candle/pull/3628) adds to `pickle.rs`, a crafted `n_bytes >= 9` hit `v |= (byte as i64) << (i * 8)` at `i == 8` and panicked with *"attempt to shift left with overflow"* ([CWE-190](https://cwe.mitre.org/data/definitions/190.html)).

Since it's the same crafted-`.pth` availability class, [#3628](https://github.com/huggingface/candle/pull/3628) guards it in the same change — rejecting `LONG1` values wider than 8 bytes before the shift, which is the `n_bytes > 8` check you suggest:

```rust
OpCode::Long1 => {
    let n_bytes = r.read_u8()?;
    // LONG1 is arbitrary-precision in Python; candle stores it as an
    // i64, so a value wider than 8 bytes is unrepresentable. Reject
    // it instead of panicking on the shift overflow a crafted
    // `n_bytes >= 9` triggered (`<< (i * 8)` with `i * 8 >= 64`).
    // CWE-190, integer overflow: https://cwe.mitre.org/data/definitions/190.html
    if n_bytes > 8 {
        crate::bail!("pickle: LONG1 value too large ({n_bytes} bytes, max 8 for i64)")
    }
    let mut v = 0;
    // Decode the next n bytes in little endian
    for i in 0..n_bytes {
        v |= (r.read_u8()? as i64) << (i * 8);
    }
    ...
}
```

This mirrors how the [anamnesis](https://github.com/PCfVW/anamnesis) pickle VM (the fuzzed reference parser [#3628](https://github.com/huggingface/candle/pull/3628)'s design is based on) already handles it. There's a `rejects_oversized_long1` regression test alongside it, and the same libFuzzer harness that first caught it now runs clean (6491 runs, zero crashes, RSS bounded ~1 GB).

So this can ride along with [#3628](https://github.com/huggingface/candle/pull/3628) rather than needing a separate fix.

Happy to add a `Closes #3619` reference to that PR if the maintainers would prefer to track all three pickle vectors ([#3617](https://github.com/huggingface/candle/issues/3617) / [#3619](https://github.com/huggingface/candle/issues/3619) / [#3620](https://github.com/huggingface/candle/issues/3620)) through the one change.
