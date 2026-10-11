// SPDX-License-Identifier: MIT OR Apache-2.0

//! Which `HuggingFace` token a request carries.
//!
//! [`resolve_token`] follows `huggingface_hub`'s rules for an *implicit*
//! token, as read in its v2.2.0 source (`utils/_auth.py`, `constants.py`), so
//! hf-fm authenticates the way the Python tools on the same machine do:
//!
//! 1. A non-empty token the caller passes (`FetchConfigBuilder::token`,
//!    `token_from_env`, `--token`) is used exactly as given, not cleaned, so
//!    a malformed one fails loudly instead of being silently repaired.
//! 2. Otherwise, if `HF_HUB_DISABLE_IMPLICIT_TOKEN` holds a true value (`1`,
//!    `ON`, `YES` or `TRUE`, in any case), no token is sent.
//! 3. Otherwise, the first of these that yields a token: `HF_TOKEN`; the
//!    legacy `HUGGING_FACE_HUB_TOKEN`; the token file, which is
//!    `HF_TOKEN_PATH`, or else `token` in the `HuggingFace` home: `HF_HOME`,
//!    or else `huggingface` under `XDG_CACHE_HOME`, or else
//!    `~/.cache/huggingface`. The token file is where `hf auth login` writes.
//!
//! A token is cleaned as `huggingface_hub` cleans it: carriage returns and
//! newlines removed, surrounding whitespace trimmed, and an empty result
//! counts as none. Two small differences are deliberate: an environment
//! variable set to the empty string counts as unset (Python would use the
//! empty path), and of the shell expansions Python applies to `HF_HOME` /
//! `HF_TOKEN_PATH` only a leading `~` is applied, not `$VAR`. Python's other
//! token sources are not followed: the OIDC exchange (`HF_OIDC_RESOURCE`),
//! Google Colab secrets, and the refresh of an expiring OAuth login.
//!
//! Every client this crate builds resolves its token here, and so does the
//! `hf-hub` client it hands single-connection downloads to, so one download
//! never mixes authenticated and anonymous requests. Until v0.12.2 only
//! `hf-hub` 1.0's own client looked for an implicit token, with rules of its
//! own; `hf-hub` 1.1 drops that lookup entirely.

use std::path::{Path, PathBuf};

/// The values `huggingface_hub` reads as true in a boolean environment
/// variable (compared after upper-casing).
const TRUE_VALUES: [&str; 4] = ["1", "ON", "YES", "TRUE"];

/// The token a request should carry: `explicit` when it is non-empty,
/// otherwise the implicit token found by the rules in the [module
/// docs](self), or `None` for an anonymous request.
///
/// This is what every client the crate builds uses, so calling it answers
/// "which token will hf-fm send?" exactly. Reading the token file is the
/// only I/O; a missing or unreadable file counts as no token.
#[must_use]
pub fn resolve_token(explicit: Option<&str>) -> Option<String> {
    let home = dirs::home_dir();
    resolve_token_with(explicit, &|name| std::env::var(name).ok(), home.as_deref())
}

/// [`resolve_token`] with the environment and the home directory passed
/// in, so every rule can be tested without touching the process
/// environment (`std::env::set_var` is `unsafe`, which this crate forbids).
fn resolve_token_with(
    explicit: Option<&str>,
    env: &dyn Fn(&str) -> Option<String>,
    home: Option<&Path>,
) -> Option<String> {
    // An explicit token is sent exactly as given, as `huggingface_hub` does:
    // cleaning it would hide a malformed one (a stray newline) that should
    // fail loudly. Only the empty string, which the CLI passes along from an
    // exported but empty `HF_TOKEN`, counts as no token.
    if let Some(token) = explicit.filter(|t| !t.is_empty()) {
        // BORROW: explicit .to_owned() for &str → owned String
        return Some(token.to_owned());
    }
    if env("HF_HUB_DISABLE_IMPLICIT_TOKEN").is_some_and(|value| is_true(&value)) {
        return None;
    }
    // As in `huggingface_hub`: the legacy variable is consulted only when
    // `HF_TOKEN` is unset or empty, and the one chosen is cleaned afterwards,
    // so a blank `HF_TOKEN` falls through to the token file.
    set(env("HF_TOKEN"))
        .or_else(|| env("HUGGING_FACE_HUB_TOKEN"))
        .as_deref()
        .and_then(clean_token)
        .or_else(|| {
            let path = token_path(env, home)?;
            let text = std::fs::read_to_string(path).ok()?;
            // BORROW: explicit .as_str() instead of Deref coercion
            clean_token(text.as_str())
        })
}

/// The token file: `HF_TOKEN_PATH`, or `token` in the `HuggingFace` home.
fn token_path(env: &dyn Fn(&str) -> Option<String>, home: Option<&Path>) -> Option<PathBuf> {
    match set(env("HF_TOKEN_PATH")) {
        // BORROW: explicit .as_str() instead of Deref coercion
        Some(path) => Some(expand_tilde(path.as_str(), home)),
        None => Some(hf_home(env, home)?.join("token")),
    }
}

/// The `HuggingFace` home: `HF_HOME`, or `huggingface` under
/// `XDG_CACHE_HOME`, or `~/.cache/huggingface`.
fn hf_home(env: &dyn Fn(&str) -> Option<String>, home: Option<&Path>) -> Option<PathBuf> {
    if let Some(hf_home) = set(env("HF_HOME")) {
        // BORROW: explicit .as_str() instead of Deref coercion
        return Some(expand_tilde(hf_home.as_str(), home));
    }
    if let Some(xdg) = set(env("XDG_CACHE_HOME")) {
        // BORROW: explicit .as_str() instead of Deref coercion
        return Some(expand_tilde(xdg.as_str(), home).join("huggingface"));
    }
    Some(home?.join(".cache").join("huggingface"))
}

/// An environment value, with the empty string read as unset.
fn set(value: Option<String>) -> Option<String> {
    value.filter(|v| !v.is_empty())
}

/// Replaces a leading `~` (alone, or followed by a separator) with `home`.
fn expand_tilde(path: &str, home: Option<&Path>) -> PathBuf {
    match (path.strip_prefix('~'), home) {
        (Some(""), Some(home)) => home.to_path_buf(),
        (Some(rest), Some(home)) if rest.starts_with(['/', '\\']) => {
            home.join(rest.trim_start_matches(['/', '\\']))
        }
        _ => PathBuf::from(path),
    }
}

/// `huggingface_hub`'s `_clean_token`: drop `\r` and `\n`, trim, and treat
/// an empty result as no token.
fn clean_token(raw: &str) -> Option<String> {
    let token = raw.replace(['\r', '\n'], "");
    let token = token.trim();
    // BORROW: explicit .to_owned() for &str → owned String
    (!token.is_empty()).then(|| token.to_owned())
}

/// `huggingface_hub`'s `_is_true`: upper-cased, one of [`TRUE_VALUES`].
fn is_true(value: &str) -> bool {
    // BORROW: explicit .as_str() instead of Deref coercion
    TRUE_VALUES.contains(&value.to_uppercase().as_str())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use std::collections::HashMap;

    /// An environment of exactly the given variables.
    fn env_of(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
        // BORROW: explicit .to_owned() for owned map entries
        let map: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        move |name: &str| map.get(name).cloned()
    }

    /// Writes `contents` to `{dir}/{rel}`, creating parent directories.
    fn write_file(dir: &Path, rel: &str, contents: &str) -> PathBuf {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn an_explicit_token_wins_over_every_implicit_source() {
        let home = tempfile::tempdir().unwrap();
        write_file(home.path(), ".cache/huggingface/token", "hf_file");
        let env = env_of(&[
            ("HF_TOKEN", "hf_env"),
            ("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1"),
        ]);
        assert_eq!(
            resolve_token_with(Some("hf_explicit"), &env, Some(home.path())).as_deref(),
            Some("hf_explicit")
        );
    }

    #[test]
    fn an_empty_explicit_token_falls_through_to_the_implicit_one() {
        let env = env_of(&[("HF_TOKEN", "hf_env")]);
        assert_eq!(
            resolve_token_with(Some(""), &env, None).as_deref(),
            Some("hf_env")
        );
    }

    #[test]
    fn an_explicit_token_is_sent_exactly_as_given() {
        // Cleaning it would hide a malformed token that must fail loudly
        // (see `fetch_repo_size_summaries_concurrent_reports_a_malformed_token_loudly`).
        let env = env_of(&[("HF_TOKEN", "hf_env")]);
        for given in ["bad\ntoken", "  hf_padded  ", "\r\n"] {
            assert_eq!(
                resolve_token_with(Some(given), &env, None).as_deref(),
                Some(given),
                "{given:?}"
            );
        }
    }

    #[test]
    fn the_disable_flag_reads_true_values_like_huggingface_hub() {
        for value in ["1", "true", "TRUE", "Yes", "on"] {
            let env = env_of(&[
                ("HF_TOKEN", "hf_env"),
                ("HF_HUB_DISABLE_IMPLICIT_TOKEN", value),
            ]);
            assert_eq!(resolve_token_with(None, &env, None), None, "{value:?}");
        }
        for value in ["0", "false", "no", "", " 1"] {
            let env = env_of(&[
                ("HF_TOKEN", "hf_env"),
                ("HF_HUB_DISABLE_IMPLICIT_TOKEN", value),
            ]);
            assert_eq!(
                resolve_token_with(None, &env, None).as_deref(),
                Some("hf_env"),
                "{value:?}"
            );
        }
    }

    #[test]
    fn hf_token_comes_before_the_legacy_variable_and_the_file() {
        let home = tempfile::tempdir().unwrap();
        write_file(home.path(), ".cache/huggingface/token", "hf_file");
        let env = env_of(&[
            ("HF_TOKEN", "hf_env"),
            ("HUGGING_FACE_HUB_TOKEN", "hf_legacy"),
        ]);
        assert_eq!(
            resolve_token_with(None, &env, Some(home.path())).as_deref(),
            Some("hf_env")
        );
    }

    #[test]
    fn the_legacy_variable_is_used_only_when_hf_token_is_unset_or_empty() {
        let home = tempfile::tempdir().unwrap();
        write_file(home.path(), ".cache/huggingface/token", "hf_file");
        for env in [
            env_of(&[("HUGGING_FACE_HUB_TOKEN", "hf_legacy")]),
            env_of(&[("HF_TOKEN", ""), ("HUGGING_FACE_HUB_TOKEN", "hf_legacy")]),
        ] {
            assert_eq!(
                resolve_token_with(None, &env, Some(home.path())).as_deref(),
                Some("hf_legacy")
            );
        }
        // A blank (not empty) `HF_TOKEN` is chosen over the legacy variable,
        // cleans to nothing, and so falls through to the file, as in Python.
        let blank = env_of(&[("HF_TOKEN", "  "), ("HUGGING_FACE_HUB_TOKEN", "hf_legacy")]);
        assert_eq!(
            resolve_token_with(None, &blank, Some(home.path())).as_deref(),
            Some("hf_file")
        );
    }

    #[test]
    fn the_token_file_defaults_to_the_home_cache_and_is_cleaned() {
        let home = tempfile::tempdir().unwrap();
        write_file(home.path(), ".cache/huggingface/token", "  hf_file\r\n");
        assert_eq!(
            resolve_token_with(None, &env_of(&[]), Some(home.path())).as_deref(),
            Some("hf_file")
        );
    }

    #[test]
    fn hf_home_and_xdg_cache_home_move_the_token_file() {
        let root = tempfile::tempdir().unwrap();
        let hf_home = root.path().join("hf");
        let xdg = root.path().join("xdg");
        write_file(&hf_home, "token", "hf_from_hf_home");
        write_file(&xdg, "huggingface/token", "hf_from_xdg");
        write_file(root.path(), ".cache/huggingface/token", "hf_from_default");

        let both = env_of(&[
            ("HF_HOME", hf_home.to_str().unwrap()),
            ("XDG_CACHE_HOME", xdg.to_str().unwrap()),
        ]);
        assert_eq!(
            resolve_token_with(None, &both, Some(root.path())).as_deref(),
            Some("hf_from_hf_home")
        );
        let xdg_only = env_of(&[("XDG_CACHE_HOME", xdg.to_str().unwrap())]);
        assert_eq!(
            resolve_token_with(None, &xdg_only, Some(root.path())).as_deref(),
            Some("hf_from_xdg")
        );
        // Empty values count as unset, so the default applies.
        let empty = env_of(&[("HF_HOME", ""), ("XDG_CACHE_HOME", "")]);
        assert_eq!(
            resolve_token_with(None, &empty, Some(root.path())).as_deref(),
            Some("hf_from_default")
        );
    }

    #[test]
    fn hf_token_path_comes_before_hf_home() {
        let root = tempfile::tempdir().unwrap();
        let explicit_file = write_file(root.path(), "elsewhere/my-token", "hf_from_path");
        write_file(root.path(), "hf/token", "hf_from_hf_home");
        let env = env_of(&[
            ("HF_TOKEN_PATH", explicit_file.to_str().unwrap()),
            ("HF_HOME", root.path().join("hf").to_str().unwrap()),
        ]);
        assert_eq!(
            resolve_token_with(None, &env, Some(root.path())).as_deref(),
            Some("hf_from_path")
        );
    }

    #[test]
    fn a_leading_tilde_is_the_home_directory() {
        let home = tempfile::tempdir().unwrap();
        write_file(home.path(), "custom-hf/token", "hf_tilde");
        let env = env_of(&[("HF_HOME", "~/custom-hf")]);
        assert_eq!(
            resolve_token_with(None, &env, Some(home.path())).as_deref(),
            Some("hf_tilde")
        );
        assert_eq!(expand_tilde("~", Some(home.path())), home.path());
        // Only a leading `~`, and only before a separator.
        assert_eq!(
            expand_tilde("~user/x", Some(home.path())),
            PathBuf::from("~user/x")
        );
        assert_eq!(
            expand_tilde("a/~/b", Some(home.path())),
            PathBuf::from("a/~/b")
        );
    }

    #[test]
    fn no_source_means_no_token() {
        let home = tempfile::tempdir().unwrap();
        assert_eq!(
            resolve_token_with(None, &env_of(&[]), Some(home.path())),
            None
        );
        assert_eq!(resolve_token_with(None, &env_of(&[]), None), None);
        write_file(home.path(), ".cache/huggingface/token", " \n");
        assert_eq!(
            resolve_token_with(None, &env_of(&[]), Some(home.path())),
            None
        );
    }
}
