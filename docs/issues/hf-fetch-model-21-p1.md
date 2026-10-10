# hf-fetch-model #21 — reply 1 (Posted)

- **Target issue:** https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues/21 (an issue we opened, so this "reply 1" is the opening post, as in [candle-3617-p1.md](candle-3617-p1.md)).
- **Status:** Posted 2026-10-10 13:16 UTC, after `e5949c7` (#16's `du <repo>` fix) was pushed and before [#16's reply 2](hf-fetch-model-16-p2.md), which links to it.
- **Title:** `Reclaim duplicated snapshot copies on Windows (hard-link pointers, and a reclaim command)`
- **Context:** Split out of [#16](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues/16): its suggested fixes 1 and 4 (accounting) landed in [`3667323`](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/commit/3667323142aa994d8dd6a97748d5ce02dc262f06); fixes 2 and 3, which would free disk, did not.
- **Accuracy flags:**
  - **Measured on 2026-10-10**, by the per-repo script recorded in [hf-fetch-model-16-p2.md](hf-fetch-model-16-p2.md): the 815.06 / 630.98 / 184.07 GiB figures, the 23-of-75 count, that every blob byte sits in one of those 23 repos, each holding at least as many bytes of copies as of blobs (184.07 GiB of blobs beside 194.88 GiB of snapshot files in all), and that the other 52 repos hold 436.10 GiB under `snapshots/` with no blob at all.
  - **Not measured: how much of the 184.07 GiB is actually reclaimable.** That needs every snapshot copy hashed against its blob, which was not done, so the post says "up to".
  - **Read in source:**
    - hf-fm's `symlink_or_copy` tries a symlink and copies when that fails.
    - `hf-hub` 1.0's `create_pointer_symlink` (`src/cache/storage.rs:64-97`) copies **unconditionally** on Windows. Its doc comment, quoted verbatim: "On Windows, copies the blob instead of creating a symlink because symlinks require elevated privileges. This means `find_cached_etag` (which uses `read_link`) cannot determine the cached etag on Windows, effectively disabling conditional-request (If-None-Match) optimization."
    - The Windows identity constraint: `std::os::windows::fs::MetadataExt::file_index` / `volume_serial_number` / `number_of_links` are `#[unstable(feature = "windows_by_handle")]`.
    - The `unsafe_code = "forbid"` lint.
  - **Reasoned, not tested:**
    - That `hf-hub` would not notice a hard-linked pointer. `read_link` fails on a hard link exactly as it does on the copy it already handles, so it should behave as it does today.
    - That non-LFS blobs are named by their git object id rather than a SHA-256 of their content. That is the Hugging Face cache convention as generally documented, but it was not checked here.
  - **Not established: how those 52 repos came to have no blob.** Another writer (an older one, or another tool) is the likely explanation, but nothing here shows which, so the post makes no claim about it.
  - **Open questions, not checked:** whether `hf-hub` or the Python `huggingface_hub` tolerates a missing blob (relevant only to the "drop the blob" variant), and how a hard-linked pointer behaves under `huggingface_hub`'s own cache scanning.
- **Outcome:** Open; no response yet. Cross-referenced from #16 by its reply 2.

---

Follow-up to #16, which fixed `du`'s accounting (suggested fixes 1 and 4) but not the duplication itself (fixes 2 and 3).

**The problem.** On Windows, the snapshot entries that hf-fm and `hf-hub` write are usually full copies of their blobs, so a model's bytes sit on disk twice. `hf-hub` always copies there: it never tries a symlink, because creating one needs elevated privileges. hf-fm's own `symlink_or_copy` tries a symlink first and falls back to `std::fs::copy` when it's refused, which it is without Developer Mode or elevation. On the machine #16 was filed from, the fixed `du` now shows this honestly: 815.06 GiB on disk, of which 630.98 GiB is under `snapshots/` and 184.07 GiB under `blobs/`. The duplication is confined to 23 of the 75 repos, which hold all 184.07 GiB of blobs beside 194.88 GiB of snapshot copies; the other 52 keep their files under `snapshots/` alone, with no blob, so they hold nothing twice. Up to 184.07 GiB is therefore reclaimable. The exact figure needs each copy hashed against its blob, which hasn't been done.

**Proposals (from #16):**

1. **Store one copy on write.** When a symlink can't be created, try a hard link to the blob before falling back to a copy. NTFS supports hard links without Developer Mode or elevation, and a pointer and its blob normally sit in the same repo directory, so on the same volume, which a hard link requires.
2. **Reclaim existing copies.** Add a command (under `cache`, or on `gc`) that replaces a snapshot copy with a hard link to its blob, after checking the two are byte-identical. For LFS files the blob's name is the SHA-256 of its content, so checking means hashing the copy and comparing it with the name. Small non-LFS files are named differently, so those would be compared byte for byte instead.

**The constraint that has to be solved first.** Once hf-fm creates hard links, `du` has to recognise them, or every hard-linked pointer is counted twice. The accounting recognises hard links only on Unix today, through `(st_dev, st_ino)`. On Windows, the stable way to tell that two paths are one file needs `std`'s `file_index` / `volume_serial_number`, which are unstable (`windows_by_handle`). The alternative is calling `GetFileInformationByHandle` directly, but that is `unsafe`, and this crate sets `unsafe_code = "forbid"`. Options I can see:

- **The `same-file` crate**, which hides the `unsafe` behind a safe API. It's small, but it would be a new compiled dependency on every desktop build: it's in the lockfile today only as an Android build-dependency.
- **Use the number of links on the file**, which `std` also keeps unstable on Windows, so this has the same problem.
- **Wait for `windows_by_handle` to stabilise.**

**Open questions:**

- Does anything downstream depend on the blob staying a separate file? `hf-hub` should not notice a hard link: on Windows it already treats its pointers as plain files, and its own comment says it cannot follow them back to their blob there. The Python `huggingface_hub` also scans the cache and reports sizes; how it treats a hard-linked pointer hasn't been checked.
- #16 also floated "drop the blob once the snapshot copy is verified". That would save the same space without hard links, but whether `hf-hub` and `huggingface_hub` cope with a missing blob hasn't been checked, so it isn't proposed here.
- Proposal 1 changes only hf-fm's own writes. Every file `hf-hub` downloads on Windows is still a copy, privileges or not, so proposal 2 is needed regardless.

**Done means:** both proposals, Windows hard-link recognition in `du`, and tests covering all three layouts (copy, symlink, hard link) on both platforms. That includes a Windows test that a hard-linked pointer is counted once.

AI usage disclosure: this issue was drafted with Claude Opus 5.5, from the investigation behind #16. I reviewed it, and the figures and source references above are checked against the repository, `hf-hub` 1.0's source and the cache they describe.
