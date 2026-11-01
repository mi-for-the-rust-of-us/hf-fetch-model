# hf-fetch-model #21 — reply 3 (Posted)

- **Target issue:** https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues/21
- **Status:** Posted 2026-11-01 10:47 UTC as [this comment](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues/21#issuecomment-6556908103), after v0.12.6 was published (tag `v0.12.6` on `16f0e60`, Trusted Publishing, CI green on all seven jobs). The issue was then closed as completed by hand: no commit carries a closing keyword (every one says "Part of #21").
- **Context:** Reports what v0.12.6 shipped against [reply 1](hf-fetch-model-21-p1.md)'s "Done means" and [reply 2](hf-fetch-model-21-p2.md)'s plan, with the run on the maintainer's cache.
- **Accuracy flags:**
  - **Measured on 2026-11-01** on the maintainer's Windows 11 machine (NTFS). The reclaim run, its dry run and the `du` captures around it used a `--release` build of `ccb3288`; `du`'s warm wall time was five runs each of v0.12.5's release build against a `--release` build of `482b7e5` (980 ms) and of `ccb3288` after the run (965 ms). Free space on `C:` is PowerShell's `(Get-PSDrive C).Free` before and after the run; the totals are `du --json`'s `total_bytes` and the sum of its per-repo `duplicated_bytes`.
  - **Python checks** were run with `huggingface_hub` 1.19.0: `scan_cache_dir` and `try_to_load_from_cache` on the real cache; the forced re-download in a throwaway `HF_HOME` holding `julien-c/dummy-unknown`, after `cache dedupe`.
  - **`cache verify`** was run on four of the 21 touched repos, not all (it hashes every LFS file of a repo).
  - **The review fixes** (`e8cd2f8`) came after the real run; none changes what that run did (they affect error classification, leftovers after a crash, the timeout placement of prevention, and the reclaimed count for a copy hard-linked under several names, none of which occurred in it).
- **Outcome:** Issue closed as completed on posting; no response yet.

---

Shipped in v0.12.6. Both proposals, Windows hard-link recognition, and the tests asked for above are in; measured on the cache this issue was filed from.

**What landed.**
- **Recognition first** ([`482b7e5`](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/commit/482b7e5)). `physical_id` reads the volume serial number and file index on Windows through `winapi-util` (the closer fit of the two candidates in the plan: it exposes them directly and holds no handle), so `du`, `DUP`, `cache delete` and `cache gc` count a hard-linked pointer once. The identity is `(volume, index, length)` on every platform, because a 64-bit index is not guaranteed unique on ReFS. The two hard-link walk tests lost their `#[cfg(unix)]`; one of them is the Windows test "Done means" asked for. Cost: `du` takes 980 ms instead of 700 ms on this cache (72 repos, 2,839 files), warm, with identical output.
- **Proposal 2, reclaim** ([`686b33b`](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/commit/686b33b), [`ccb3288`](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/commit/ccb3288)): `hf-fm cache dedupe [REPO_ID...] [--dry-run] [--yes] [--verify]`, and `cache::plan_dedupe` / `cache::apply_dedupe` in the library. The preview pairs by length and reads nothing ("up to"); the run compares each copy with its blob byte for byte and replaces it with a hard link made under a temporary name and renamed over it, so the pointer is never missing. One deviation from the proposal: byte equality rather than hashing the copy against the blob's name. It already refuses a copy that differs from its blob, works the same for non-LFS files, and stops at the first differing byte; `--verify` adds the hash for those who want it. A repo with a partial download or a held download lock is skipped.
- **Proposal 1, one copy on write** ([`072a8b7`](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/commit/072a8b7), [`e8cd2f8`](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/commit/e8cd2f8)). Where a symlink is refused, hf-fm's chunked path makes a hard link before falling back to a copy. Since `hf-hub` copies regardless (the point of reply 1's last open question), its copy is replaced after the download with a hard link to `blobs/<sha256>` when the listing names the file's SHA-256 and the two are byte-equal, never by name or length alone. On a fresh `julien-c/dummy-unknown`, the `hf-hub` path now leaves 2.2 KiB of copies (its non-LFS files) instead of 276.5 KiB, and the chunked path none.

**The run on the maintainer's cache** (`cache dedupe --yes`, 21 repos):

| | Before | After |
|---|---|---|
| `du` total | 763.22 GiB | 609.07 GiB |
| `DUP` | 154.16 GiB (21 repos) | 0 |
| Free space on `C:` | 762.68 GiB | 916.84 GiB |

It replaced 228 files, reclaiming 154.16 GiB (165,524,494,120 bytes, the `DUP` figure to the byte), and compared 309.23 GiB in 1 min 34 s (3.29 GiB/s). Free space grew by 165,524,770,816 bytes, the difference being cluster slack over 228 files, and `du`'s total fell by exactly the reclaimed bytes. The preview said up to 155.00 GiB: `google/gemma-scope-2b-pt-transcoders` holds 27 snapshot files of one length but only 24 blobs, so three files stored in the snapshot alone paired by length, were compared, and were left alone with "no blob of its length holds the same bytes".

**The open questions, checked on the result.** `huggingface_hub` 1.19.0's `scan_cache_dir` reports one warning on the deduplicated cache, which predates it (a repo holding only hf-fm's header cache, with no `snapshots/`); `try_to_load_from_cache` returns a deduplicated `microsoft/Phi-3.5-mini-instruct` shard whose header parses. A forced re-download (`hf_hub_download(force_download=True)`) over hard-linked pointers left every blob hashing to its name, and `hf-fm download-file` reads them as cached. `cache verify` passes on four of the touched repos.

What stays a copy: non-LFS files written by `hf-hub` (no SHA-256 to name their blob), and the older revision's pointer when a new revision's download renames a fresh blob over the one it linked; `cache dedupe` reclaims both. The walkthrough is in the [cache tutorial](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/blob/main/docs/tutorials/clean-up-before-your-disk-fills.md#reclaiming-the-copies-keeping-every-model-cache-dedupe).

AI usage disclosure: this comment was drafted with Claude Opus 5.5. The measurements above were run on the maintainer's machine during the implementation.
