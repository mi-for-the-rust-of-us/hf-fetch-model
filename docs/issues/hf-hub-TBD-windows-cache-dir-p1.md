# hf-hub #TBD — reply 1 (Draft)

- **Target issue:** to be filed at <https://github.com/huggingface/hf-hub/issues> (new issue, number unknown)
- **Status:** Draft, 2026-08-13 — awaiting maintainer review before posting
- **Context:** Found while porting `hf-fetch-model` from `hf-hub` 0.5 to 1.0 (hf-fm v0.11.3). The Windows CI runner failed a cache-layout integration test; root cause is `hf-hub` 1.0 resolving its default cache root from `HOME` alone, which Windows does not set.
- **Outcome:** (pending)

---

## Title

`hf_home()` resolves to `/tmp` on Windows — `HOME` is not a Windows environment variable

## Body

### Summary

`hf_home()` derives the default Hugging Face home directory from the `HOME` environment variable, falling back to `/tmp`. Windows does not set `HOME`; the equivalent is `USERPROFILE`. As a result, on Windows every `hf-hub` 1.0 consumer that does not explicitly set a cache directory reads and writes `/tmp/.cache/huggingface/hub` — i.e. `C:\tmp\.cache\huggingface\hub` — instead of `C:\Users\<name>\.cache\huggingface\hub`.

This diverges from Python's `huggingface_hub`, which resolves the same directory via `Path.home()` (i.e. `USERPROFILE` on Windows), so the Rust and Python clients no longer share a cache on that platform.

### Source

[`src/constants.rs`](https://github.com/huggingface/hf-hub/blob/main/src/constants.rs):

```rust
pub(crate) fn dirs_or_home() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string())
}

pub fn hf_home() -> std::path::PathBuf {
    if let Ok(path) = std::env::var(HF_HOME) {
        return std::path::PathBuf::from(path);
    }
    if let Ok(xdg) = std::env::var(XDG_CACHE_HOME) {
        return std::path::PathBuf::from(xdg).join("huggingface");
    }
    let home = dirs_or_home();
    std::path::PathBuf::from(format!("{home}/.cache/huggingface"))
}
```

### Reproduction

`hf-hub = "1.0"`, run from **PowerShell or `cmd.exe`** (not Git Bash / MSYS / WSL, which set `HOME` themselves and will mask the bug):

```rust
fn main() {
    println!("HOME        = {:?}", std::env::var("HOME"));
    println!("USERPROFILE = {:?}", std::env::var("USERPROFILE"));
    println!("hf_home()          = {}", hf_hub::hf_home().display());
    println!("resolve_cache_dir()= {}", hf_hub::resolve_cache_dir().display());
}
```

Observed on Windows 11, `rustc` 1.97.1, `hf-hub` 1.0.0:

```
HOME        = Err(NotPresent)
USERPROFILE = Ok("C:\\Users\\Eric JACOPIN")
hf_home()          = /tmp/.cache/huggingface
resolve_cache_dir()= /tmp/.cache/huggingface\hub
```

Expected `hf_home()` on this machine: `C:\Users\Eric JACOPIN\.cache\huggingface`.

### Impact

1. **Cache is not shared with Python `huggingface_hub` on Windows.** Anything downloaded by the Rust client is invisible to the Python client and vice-versa, so users silently re-download models they already have.
2. **Downloads land on the system drive root** (`C:\tmp\...`) regardless of where the user's profile lives, which on multi-drive setups is often the smallest volume — and `C:\tmp` is not a conventional Windows location, so the space is hard to find and reclaim.
3. **Silent, not loud.** Nothing errors; the files simply go somewhere unexpected. We only caught it because an integration test asserts the on-disk layout matches what a separate cache-introspection code path expects.
4. **The `/tmp` fallback is also questionable on Unix**, independently of Windows: if `HOME` is unset (some daemon/container contexts), a world-writable shared directory is a surprising place to put a model cache.

### Suggested fix

Use the `dirs`/`home` crate (or `USERPROFILE` explicitly on Windows) rather than `HOME` alone:

```rust
pub(crate) fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME")
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| {
            #[cfg(windows)]
            { std::env::var_os("USERPROFILE").filter(|s| !s.is_empty()).map(std::path::PathBuf::from) }
            #[cfg(not(windows))]
            { None }
        })
}
```

…and build the path with `PathBuf::join` rather than `format!("{home}/.cache/huggingface")`, which currently produces mixed separators on Windows (`C:\Users\x/.cache/huggingface`). That works for filesystem calls but makes any string comparison against a natively-constructed path fail, which is its own trap for downstream code.

`hf-hub` 0.5 was not affected: it used `dirs::home_dir()`, which handles `USERPROFILE`.

### Note

Reported from downstream [`hf-fetch-model`](https://github.com/mi-for-the-rust-of-us/hf-fetch-model), which has worked around it in v0.11.3 by always passing an explicit `cache_dir` to `HFClientBuilder`. The workaround is easy for a library that already owns its cache-path logic, but a plain `HFClient::new()` user on Windows gets the wrong directory with no indication anything is off.

---

## Notes for us (not part of the issue body)

- The workaround lives in `build_model_repo` ([src/lib.rs](../../src/lib.rs)): always `.cache_dir(...)`, using `config.output_dir` when set and `cache::hf_cache_dir()` otherwise.
- Guarded by the existing `cache_layout_matches_hf_hub` test in [tests/integration.rs](../../tests/integration.rs), which is what caught this on the windows-latest runner.
- Repro crate kept out of tree deliberately (needs a native-shell run to be meaningful; Git Bash sets `HOME` and hides the bug). If we ever want it in CI, it has to be a `#[cfg(windows)]` test that asserts `hf_hub::hf_home()` starts with `USERPROFILE` — worth adding only if upstream declines the fix.
