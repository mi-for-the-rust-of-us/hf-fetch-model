# hf-fetch-model #21 — reply 2 (Posted)

- **Target issue:** https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues/21
- **Status:** Posted 2026-10-31 10:37 UTC as [this comment](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues/21#issuecomment-6542314176). Archived on 2026-11-01 from the posted text, verbatim, when [reply 3](hf-fetch-model-21-p3.md) was drafted; no draft of it had been kept here.
- **Context:** The plan for v0.12.5 and v0.12.6, after the duplication was measured by content and both of [reply 1](hf-fetch-model-21-p1.md)'s open questions were tested (a hard-linked pointer and a dropped blob, against `huggingface_hub` 1.19.0 and hf-fm `main`). The cache-management roadmap's v0.12.6 section cites this comment for that table.
- **Outcome:** No response. Superseded in substance by [reply 3](hf-fetch-model-21-p3.md), which reports the release.

---

Update, 2026-10-31: new evidence, both open questions measured, and a plan.

**1. The duplication, measured by content (2026-10-30).** On the same Windows cache, an independent scan found 23 repos with 184.1 GiB stored twice (22.6 % of 816.32 GiB), consistent with the 184.07 GiB above. It is a second copy, not a link: `sha256sum` of one 1.6 GB snapshot file equals the name of its blob, `fsutil hardlink list` shows a single path, and no snapshot file in the cache has a link count above 1. After seven unreferenced repos were deleted (two of them among the 23), the same scan gives 21 repos and 154.2 GiB.

**2. The two open questions, tested (Windows 11, NTFS, `huggingface_hub` 1.19.0, hf-fm `main`).** A scratch cache with the three layouts, each a 3 MiB file:

| Layout | `scan_cache_dir` | `try_to_load_from_cache` | `hf-fm du` |
|---|---|---|---|
| copy (today) | 3 MiB, no warnings | found | 6.00 MiB |
| hard-linked pointer | 3 MiB, no warnings | found | **6.00 MiB** |
| pointer only, blob dropped | 3 MiB, no warnings | found | 3.00 MiB |

So `huggingface_hub` tolerates a hard-linked pointer, and also a missing blob. With a real small repo, `hf-fm download-file` reports "Cached" for both layouts, since it resolves the pointer and never needs the blob. And the constraint in the description is confirmed: `du` counts the hard-linked pointer twice, because `physical_id` is `None` on Windows. Not tested: the large-file (chunked) path, and what a later revision update does when the blob is missing (probably a re-download).

**3. Plan.**
- v0.12.5 makes `du` name the duplicates (a `DUP` column and a footer), detected with stable `std` only, and lists `datasets--` repos. That needs none of this issue's prerequisites, so it does not close it.
- v0.12.6 carries this issue: Windows hard-link recognition, `cache dedupe` (dry-run by default, content checked before anything is replaced), and hard-link-before-copy on hf-fm's own writes.
- Design: hard links, not dropping the blob. Dropping works for both tools, but it is irreversible and means a re-download when another revision shares the blob; a hard link keeps the blob. For recognition, `same-file` is already in `Cargo.lock` (an Android build-dependency through `walkdir`), and so is `winapi-util`, which exposes `volume_serial_number`, `file_index` and `number_of_links` directly, so it is the closer fit; settling which one is part of that release.

Until then: please do not hard-link the copies by hand on Windows. The volume would free the bytes while `du` keeps reporting them and `cache delete` keeps promising to free them twice.

AI usage disclosure: this comment was drafted with Claude Sonnet 5.5. The measurements above were run on the maintainer's machine for this comment.


