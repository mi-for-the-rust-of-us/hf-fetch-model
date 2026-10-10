# hf-fetch-model #16 — reply 1 (Posted)

- **Target issue:** https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues/16
- **Status:** Posted (2026-10-01). The body below matches what is live verbatim.
- **Context:** Found while freeing disk after anamnesis Phase 7.10 downloaded
  `distaste447/zeta-2.1-NVFP-GGUF` (5.18 GiB): `hf-fm du` said 5.18 GiB, the
  directory held 10.36 GiB. A scan of the whole cache then showed the gap is
  general, not specific to that repo.
- **Outcome:** Suggested fixes 1 and 4 landed in `3667323`
  (`fix(cache): count blob bytes and resolve pointer sizes in disk
  accounting`), with no public API change: `CachedModelSummary::total_size`
  and `repo_disk_usage` were already documented and named for bytes on disk,
  and `du`'s footer already promised "real bytes on disk", so the fix made
  three existing promises true. Seventeen tests came with it.
  **Suggested fixes 2 and 3 are NOT done, and they are the ones that reclaim
  disk.** The accounting fix makes the 189.25 GiB visible; it does not free a
  byte of it.
  **The issue was closed by mistake on 2026-10-02**, as COMPLETED with no
  comment, by the keyword `Closes #16.` in `3667323`'s message, which had not
  been agreed. Reply 2 ([p2](hf-fetch-model-16-p2.md), posted 2026-10-10) explains this,
  corrects this post's Unix claim (flag 5), and moves fixes 2 and 3 to a new
  issue, [#21](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues/21) ([p1](hf-fetch-model-21-p1.md)).
  **A gap in fix 1 itself was found on 2026-10-10**, by a consistency pass
  over that reply: the single-repo view `du <repo>`, which is this post's own
  example command, still sums the logical per-file listing, so it still
  under-reports on a copy layout. Fixed in `e5949c7` ("fix(du): report bytes on disk
  in du <repo>", pushed 2026-10-10), which unblocked reply 2.
  The fix is on `main` but unreleased (crates.io still serves 0.12.1).
- **Lesson / Leverage angle:** The first measurement ("du reports half") was
  true for one repo and wrong as a rule; the whole-cache scan (189 GiB of
  825 GiB, 0x to 2x per repo) is what the issue states. Also, deletion frees the
  right bytes and only *reports* the wrong number, which changes the severity.
- **Accuracy flags:**
  1. **Root cause read in the source, not inferred from totals:**
     `walk_repo_files` opens `snapshots_dir` only (`src/cache.rs:904` at
     `c9b9179`, identical to the local tree), and `cache_summary` /
     `repo_disk_usage` / `run_cache_delete` all use it.
  2. **Which component wrote the snapshot copies is not established**, and the
     issue says so rather than blaming hf-fm's downloader. **Resolved while
     fixing it:** both write copies on Windows, by design and with the reason
     documented in each. hf-fm's `chunked::symlink_or_copy` tries
     `std::os::windows::fs::symlink_file` and falls back to `std::fs::copy`
     when it fails, which it does without `SeCreateSymbolicLinkPrivilege`.
     hf-hub 1.0's `cache::storage::create_pointer_symlink` does not try at
     all: on Windows it copies **unconditionally** (`#[cfg(windows)]
     std::fs::copy`, `src/cache/storage.rs:88-95`), and its comment reads,
     verbatim: "On Windows, copies the blob instead of creating a symlink
     because symlinks require elevated privileges." (An earlier version of
     this flag paraphrased that comment inside quotation marks, and read as
     if hf-hub also fell back; corrected 2026-10-10.) Neither ever creates a
     hard link, which is also why the fix can skip hard-link dedup on
     Windows.
  3. **Personal path redacted** to `%USERPROFILE%`.
  4. **Not verified:** that `cache gc --size` budgets against the same figure;
     the issue says "if". **Resolved 2026-10-10:** it does. `compute_gc_plan`
     sums `CachedModelSummary::total_size`, and its `Freed` line adds the same
     per-repo sizes (see [p2](hf-fetch-model-16-p2.md)).
  5. **The posted body understates the bug, and fix 1's "Totals on the usual
     Unix layout are unchanged" is wrong.** Established only while fixing it:
     sizes came from `DirEntry::metadata`, which `std` documents as
     "equivalent to calling `symlink_metadata`", so on a symlinked layout it
     returned the length of the link's *target path* rather than the file's
     size, and the blob went uncounted too. Measured on Linux through a
     verbatim-extracted harness: a 1 MB file reported **20 bytes**, the length
     of `../../blobs/deadbeef`. So the symlink layout reported a rounding
     error, not "unchanged" and not "half" — the Windows copy layout this
     issue was filed from is the *milder* of the two. A reply 2 owes the
     thread this correction. The same root cause also had two further
     expressions the body does not mention, `inspect --list`'s per-file sizes
     and the post-download summary line (7 bytes for a 1 MB download), both
     found by a consistency pass and fixed in the same commit.

---

## Summary

`hf-fm du` (and `du --tree`, `du --json`, and the `cache_summary` data that `status` and `cache gc` share) sums only the files under each repo's `snapshots/` directory. Files under `blobs/` are never counted. Wherever a snapshot entry is a **copy** of its blob rather than a link to it, the bytes on disk are under-reported, and the footer's statement that *"the total above still reflects real bytes on disk across every cached file"* does not hold.

On one Windows 11 machine (hf-fm 0.12.1), across a 76-repo cache:

| | GiB |
|---|---:|
| `du` total (snapshot files) | 636.16 |
| `blobs/`, not counted | 189.25 |
| **actual bytes on disk** | **825.41** |

For a single repo the error ranges from nothing to 2x. After `hf-fm download-file distaste447/zeta-2.1-NVFP-GGUF zeta-2.1-NVFP4.gguf --revision 3cb915940e83fede1fb4e961df81d901acd71773`:

```
$ hf-fm du distaste447/zeta-2.1-NVFP-GGUF
    #        SIZE  FILE
    1    5.18 GiB  zeta-2.1-NVFP4.gguf
    5.18 GiB  total (1 file)
```

while the repo directory holds the 5.18 GiB blob (`blobs/e853bfbc...`) **and** a 5.18 GiB snapshot file, 10.36 GiB in all. `du --tree` and `du --json` report the same 5.18 GiB (`"size": 5561543232`).

## Root cause

[`walk_repo_files`](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/blob/c9b9179/src/cache.rs#L904) opens `snapshots_dir` only, and is the walk behind both [`cache_summary`](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/blob/c9b9179/src/cache.rs#L571) (whose doc says it "counts files + sizes in each snapshot") and [`repo_disk_usage`](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/blob/c9b9179/src/cache.rs#L650). That is correct where snapshot entries are symlinks into `blobs/` (the usual Unix layout), and wrong wherever they are not.

On this machine **none** of the 933 snapshot files is a link: none has a symlink `LinkType`, and `fsutil hardlink list` prints a single path for each, so every one occupies its own bytes. Which component wrote them as copies (hf-fm's downloader or the `hf-hub` layer below it) I have not established; the accounting is wrong either way.

## How to measure it (PowerShell)

```powershell
$hub = "$env:USERPROFILE\.cache\huggingface\hub"
$blobs = 0; $copies = 0
foreach ($r in Get-ChildItem $hub -Directory -Filter "models--*") {
  $b = Join-Path $r.FullName "blobs"
  if (Test-Path $b) { $blobs += (Get-ChildItem $b -File -Force | Measure-Object Length -Sum).Sum }
  $s = Join-Path $r.FullName "snapshots"
  if (Test-Path $s) {
    foreach ($f in Get-ChildItem $s -Recurse -File -Force) {
      $links = (fsutil hardlink list $f.FullName 2>$null | Measure-Object).Count
      if (-not $f.LinkType -and $links -le 1) { $copies += $f.Length }
    }
  }
}
"blobs {0:N2} GiB + snapshot copies {1:N2} GiB = {2:N2} GiB on disk" -f ($blobs/1GB), ($copies/1GB), (($blobs+$copies)/1GB)
```

## Why it matters

`du` is what a user runs to decide what to delete, and the same number is what deletion reports back. [`run_cache_delete`](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/blob/c9b9179/src/bin/main.rs#L4396) removes the whole repo directory with `remove_dir_all`, so it **does** free the blobs, but its preview and its `Deleted. Freed ...` line come from `repo_disk_usage`. For the repo above it would free 10.36 GiB and print `Freed 5.18 GiB`. If `cache gc --size` budgets against the same figure, it frees less than it needs to. The error always makes the cache look smaller than it is.

## Suggested fixes

1. **Count bytes on disk:** every regular file under `blobs/` and `snapshots/`, each physical file once (a snapshot entry that is a symlink or hard link to a blob adds nothing). Totals on the usual Unix layout are unchanged. Then the footer's "real bytes on disk" becomes true, or it should be reworded.
2. **Store one copy:** where symlinks are unavailable, hard-link the snapshot entry to its blob (NTFS supports hard links without Developer Mode) and copy only if that fails, or drop the blob once the snapshot copy is verified.
3. **Reclaim existing duplicates:** an option (under `cache`, or on `gc`) that replaces a snapshot copy with a hard link to a byte-identical blob.
4. A test that builds a cache with a blob plus a snapshot copy and checks `du`'s total, so the layout difference is covered on every platform.

## Environment

hf-fm 0.12.1 (`cargo install`), Windows 11 Pro (x86_64-pc-windows-msvc), cache at the default `%USERPROFILE%\.cache\huggingface\hub`.
