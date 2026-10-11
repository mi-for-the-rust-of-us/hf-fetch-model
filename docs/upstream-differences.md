# Upstream differences

Where hf-fetch-model diverges from Python's `huggingface_hub`, from `hf_transfer`, and from `hf-hub`, the Rust client it builds on, and why. When upstream catches up on a point, the entry is removed, and the note at the end says when and why.

**Last reviewed:** October 2026, against `huggingface_hub` 2.2.0 (2026-10-08), `hf_transfer` 0.1.9 and `hf-hub` 1.0.0, reading each one's source at that release. `huggingface_hub` has not used `hf_transfer` since its v1.0.0: it now downloads Xet-backed files through `hf_xet` and everything else over plain HTTP, so the `hf_transfer` points below concern only code that still calls it directly.

---

## A fresh signed URL for every chunk request

`huggingface_hub`'s plain-HTTP path takes the signed CDN URL once, from the `Location` of its first `HEAD`. A `403` is not retried, and its retries after a network error reuse the same URL, so a URL that expires before the transfer finishes fails the download. `hf_transfer` likewise reuses one redirected URL for every chunk and every retry. (`huggingface_hub`'s Xet path does refresh its token before it expires, and `hf_xet` re-fetches its URLs after a `403`, so Xet downloads are not affected.)

hf-fetch-model sends every chunk request to the Hub's `/resolve` URL and follows the redirect, so each request carries a freshly signed CDN URL and expiry cannot interrupt a long download. This is required anyway for Xet-backed repos, whose CDN URLs are signed for the exact `Range` header that minted them: a URL minted for one range fails with `403` on any other. `reqwest` drops the `Authorization` header on the cross-host redirect, so the token never reaches the CDN.

(Until July 2026, hf-fetch-model instead parsed `X-Amz-Expires` from the URL and re-probed before a download that might outlast it. Per-chunk resolution replaced that when Xet-backed repos made a single URL unusable for a whole file.)

**Files:** `chunked.rs` (`RangeInfo::resolve_url`), `download.rs` (`download_single_file_chunked`).

---

## Windows blob deduplication

The HF cache links `snapshots/<commit>/<filename>` to `blobs/<etag>`. On Windows without `SeCreateSymbolicLinkPrivilege` (the default for non-admin users), Python's `huggingface_hub` cannot create the symlink and **moves** a newly downloaded blob out of `blobs/` into `snapshots/`. Its cache guide says as much: without symlinks the cache keeps files in `snapshots/` directly, so a file may be downloaded again for every revision that contains it.

hf-fetch-model tries a symlink first and falls back to a **copy**, keeping the blob in `blobs/<etag>` for the next revision. The cost is the file held twice on disk, which `du` reports since v0.12.2. `hf-hub` 1.0 also copies on Windows (it does not try a symlink there at all), so files `hf-hub` writes for hf-fetch-model keep their blob too.

**File:** `chunked.rs` (`symlink_or_copy`).

---

## Resume across interrupted runs

`huggingface_hub` deletes a failed download's temporary file since v1.18.0 (2026-06): each download writes to a temporary name unique to that attempt, removed in a `finally` block on any failure, so an interrupted download starts again from zero on the next run. (Up to v1.17.0 it kept `<etag>.incomplete` and resumed from it.) `hf_transfer` removes its output file on any error. `hf-hub` 1.0 leaves its `<blob>.incomplete` file behind on failure, but recreates it from zero on the next attempt.

hf-fetch-model keeps an interrupted chunked download's `.chunked.part` file, with a per-chunk progress sidecar, and the next run resumes each chunk from its last checkpoint through `Range` requests, provided the file's etag has not changed. This holds for timeouts, Ctrl-C and transient I/O errors alike: the temp file's guard keeps the partial by default and removes it only when its bytes are known to be unusable. A partial whose etag, size or sidecar schema no longer matches is deleted before the download restarts. `hf-fm cache clean-partial` removes the partials a user gives up on.

**Files:** `chunked.rs` (`TempFileGuard`, `prepare_or_resume_temp_file`), `chunked_state.rs`.

---

## TCP connect timeout

`hf-hub` 1.0 builds its `reqwest::Client` with no connect timeout, and `hf_transfer` sets none either. A dead peer (SYN sent, no SYN-ACK) then stalls until the OS gives up, typically after two minutes or more on Linux and longer on Windows. `huggingface_hub` is not affected: its single 10-second timeout covers the connect phase too.

hf-fetch-model sets a 30-second connect timeout on its own two HTTP clients (`build_client`, `build_no_redirect_client`), which serve chunked downloads, the metadata listing, the range probes, remote `inspect` and `info`. Since v0.12.2 it also hands `hf-hub` a client with the same timeout (`build_hf_hub_client`), so `hf-hub`'s repo listing and single-connection download requests have it too. One request still goes without: the `HEAD` that `hf-hub` sends through a second, internal client before a single-connection download, which a caller cannot replace; the per-file timeout (300 seconds by default) bounds that download as a whole.

**File:** `chunked.rs` (`CONNECT_TIMEOUT`, `build_hf_hub_client`).

---

## Removed in the October 2026 review

Three entries from the April 2026 review no longer describe a difference:

- **POSIX symlink race tolerance.** `huggingface_hub` catches `FileExistsError` when the existing link already points at the same blob, and it already did in 0.34.x, so the entry was wrong when written. `hf-hub` 1.0 ignores `AlreadyExists` too.
- **Disk space check without a cache scan.** hf-fetch-model, like `huggingface_hub`, asks for the free space once and only warns; this was a convergence, not a divergence.
- **Shared HTTP client for metadata requests.** `huggingface_hub` shares one pooled client across its calls, and `hf-hub` 1.0's `HFClient` holds one `reqwest::Client` for every request made through it, so all three reuse connections.
