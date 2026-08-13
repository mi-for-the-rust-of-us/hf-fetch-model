# hf-hub — Windows cache-dir bug (Draft, **NOT POSTED — superseded upstream**)

- **Target issue:** none filed. <https://github.com/huggingface/hf-hub/issues>
- **Status:** **Do not post.** Drafted 2026-08-13, retired the same day after checking upstream.
- **Context:** Found while porting `hf-fetch-model` to `hf-hub` 1.0 (hf-fm v0.11.3). The Windows CI runner failed `cache_layout_matches_hf_hub`; root cause was `hf-hub` 1.0.0 resolving its default cache root from `HOME` alone, which Windows does not set.
- **Outcome:** The offending code was already deleted upstream in [PR #193](https://github.com/huggingface/hf-hub/pull/193), merged 2026-08-05 — eight days before we hit it. Reporting it would describe code that no longer exists.

---

## The bug (real, in the published 1.0.0)

`hf-hub` 1.0.0's `src/constants.rs`:

```rust
pub(crate) fn dirs_or_home() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string())
}
```

Windows does not set `HOME` (it uses `USERPROFILE`), so `hf_home()` falls through to `/tmp`. Verified with a standalone repro run from **native PowerShell** (Git Bash / MSYS / WSL set `HOME` themselves and mask it entirely — which is why this survived local testing and only surfaced on the windows-latest CI runner):

```
HOME        = Err(NotPresent)
USERPROFILE = Ok("C:\\Users\\Eric JACOPIN")
hf_home()          = /tmp/.cache/huggingface
resolve_cache_dir()= /tmp/.cache/huggingface\hub
```

Consequences on Windows for any 1.0.0 consumer that does not pass an explicit cache directory: downloads land in `C:\tmp\.cache\huggingface\hub`, invisible to Python's `huggingface_hub` (which uses `Path.home()` → `USERPROFILE`) and to every other tool expecting the standard location.

## Why it is not worth reporting

Upstream checked on 2026-08-13:

- **12 open issues, none related.** Searches for `windows`, `USERPROFILE`, `HOME`, `cache dir`, `hf_home`, `tmp` across all issues (open and closed) and PRs turned up nothing about this. The three `windows` hits are a 401 auth bug ([#120](https://github.com/huggingface/hf-hub/issues/120), hf-hub 0.4.3), an unrelated download issue, and a symlink-fallback PR. No mention of `USERPROFILE` anywhere in the repo's issues or PRs.
- **The code is gone on `main`.** [PR #193 "Remove environment configuration from hf-hub"](https://github.com/huggingface/hf-hub/pull/193) (merged 2026-08-05, +132/−304) deleted `dirs_or_home`, `hf_home`, and `resolve_cache_dir` outright, along with their re-exports. Its stated goal: *"make hf-hub client construction deterministic and explicit, without environment-variable reads"*, with env-based token / endpoint / cache configuration preserved one layer up in the `hfrs` CLI. Bumps the crate to **1.1.0**, which is not yet on crates.io (1.0.0 is still `max_stable`).

So the bug is real in the only published 1.0.x, but it is already fixed-by-deletion in the unreleased 1.1.0, and it was a deliberate architectural change rather than a response to this defect. Filing it now would be noise.

## What this means for hf-fm (the useful part)

**Our fix is not merely a 1.0.0 workaround — it is mandatory under 1.1.0 too, for a different reason.**

`hf-hub` 1.1.0's default becomes the *relative* path `.cache/huggingface/hub`, resolved against the **current working directory** ([`client.rs`](https://github.com/huggingface/hf-hub/blob/main/hf-hub/src/client.rs), `build()`), with no home-directory logic and no environment reads at all — its own unit test asserts exactly that:

```rust
let client = HFClientBuilder::new().build().unwrap();
assert_eq!(client.cache_dir(), std::path::Path::new(".cache/huggingface/hub"));
```

Had hf-fm relied on hf-hub's default, the failure mode would simply have changed shape on the next upgrade: from "everything in `C:\tmp`" to "a fresh cache under whatever directory the user happened to run `hf-fm` from". Because `build_model_repo` now always passes an explicit `cache_dir` ([src/lib.rs](../../src/lib.rs)), hf-fm is immune to both.

Two follow-on notes for whoever takes the 1.1.0 bump:

1. **hf-fm becomes the sole provider of `HF_HOME` support for its users.** 1.1.0 removes env-var handling from the library; `cache::hf_cache_dir()` reads `HF_HOME` and is what every hf-fm command already uses, so the user-facing contract is unchanged. Do not "simplify" by deferring to hf-hub's default.
2. **Re-check `cache_layout_matches_hf_hub` on Windows after the bump.** That test is what caught this, and it is the only thing standing between a cache-path regression and silent misplacement of downloads.

## Reproduction (kept for reference)

`hf-hub = "1.0"`, run from PowerShell or `cmd.exe`, **not** Git Bash:

```rust
fn main() {
    println!("HOME        = {:?}", std::env::var("HOME"));
    println!("USERPROFILE = {:?}", std::env::var("USERPROFILE"));
    println!("hf_home()          = {}", hf_hub::hf_home().display());
    println!("resolve_cache_dir()= {}", hf_hub::resolve_cache_dir().display());
}
```

Not added to the test suite: it needs a native-shell run to be meaningful, and both functions disappear in 1.1.0. If we ever want CI coverage of the underlying property, the durable form is a `#[cfg(windows)]` assertion that hf-fm's *own* resolved download cache sits under `USERPROFILE` — which is what `cache_layout_matches_hf_hub` already effectively checks.
