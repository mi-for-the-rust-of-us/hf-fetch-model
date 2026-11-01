// SPDX-License-Identifier: MIT OR Apache-2.0

//! Centralized `hf-hub` cache path construction.
//!
//! All paths follow the `hf-hub` 1.0 cache layout:
//! `{cache_root}/models--org--name/{snapshots,blobs,refs}/...`
//!
//! This module is the single source of truth for cache directory structure.
//! When `hf-hub` bumps, update this module and rerun the
//! `cache_layout_matches_hf_hub` integration test in `tests/integration.rs`.

use std::path::{Path, PathBuf};

/// The kind of repository a cache folder holds (new in v0.12.5).
///
/// The `HuggingFace` cache keeps models, datasets and Spaces side by side,
/// told apart by the plural that starts the folder name (`models--…`,
/// `datasets--…`, `spaces--…`). Only [`RepoType::Model`] is downloaded by
/// this crate; `du` lists the other two so its total is the cache's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RepoType {
    /// A model repository: `models--org--name`.
    Model,
    /// A dataset repository: `datasets--org--name`.
    Dataset,
    /// A Space: `spaces--org--name`.
    Space,
}

impl RepoType {
    /// Every repo type, in the order `du` considers them.
    pub const ALL: [Self; 3] = [Self::Model, Self::Dataset, Self::Space];

    /// The plural that starts this type's cache folder names.
    #[must_use]
    pub const fn folder_prefix(self) -> &'static str {
        match self {
            Self::Model => "models",
            Self::Dataset => "datasets",
            Self::Space => "spaces",
        }
    }

    /// The singular word for this type, as `du` prints and `--json` reports it.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Dataset => "dataset",
            Self::Space => "space",
        }
    }

    /// How a repo of this type is addressed on the command line and on the
    /// Hub: a model by its bare ID, the others behind their URL prefix
    /// (`datasets/org/name`), since a model and a dataset may share an
    /// `org/name`.
    #[must_use]
    pub fn address(self, repo_id: &str) -> String {
        match self {
            Self::Model => String::from(repo_id),
            Self::Dataset => format!("datasets/{repo_id}"),
            Self::Space => format!("spaces/{repo_id}"),
        }
    }
}

/// Splits a command-line repo argument into its type and bare ID: the inverse
/// of [`RepoType::address`].
///
/// `datasets/org/name` is a dataset and `spaces/org/name` a Space; anything
/// else is a model, returned unchanged. A prefix with nothing after it is not
/// one (`datasets/` stays a model argument and fails later, as any unknown ID
/// does).
#[must_use]
pub fn split_repo_address(arg: &str) -> (RepoType, &str) {
    for repo_type in [RepoType::Dataset, RepoType::Space] {
        let prefix = repo_type.folder_prefix();
        if let Some(rest) = arg
            .strip_prefix(prefix)
            .and_then(|r| r.strip_prefix('/'))
            .filter(|r| !r.is_empty())
        {
            return (repo_type, rest);
        }
    }
    (RepoType::Model, arg)
}

/// Repo folder name: `"models--org--name"`.
///
/// Constructed here rather than delegated: `hf-hub` 0.5 exposed the naming
/// through `Repo::folder_name()`, but the 1.0 rewrite made its equivalent
/// (`cache::storage::repo_folder_name`) `pub(crate)`, so there is no longer
/// an upstream function to call. The algorithm is unchanged across both
/// versions — the repo-type plural (`"models"`) followed by the `repo_id`
/// segments, joined by `--` — and the `cache_layout_matches_hf_hub`
/// integration test still pins it to what `hf-hub` actually writes on disk,
/// which is the guarantee that mattered.
#[must_use]
pub fn repo_folder_name(repo_id: &str) -> String {
    repo_folder_name_for(RepoType::Model, repo_id)
}

/// Repo folder name for any repo type: `"datasets--org--name"`,
/// `"spaces--org--name"`, or [`repo_folder_name`]'s `"models--org--name"`
/// (new in v0.12.5).
#[must_use]
pub fn repo_folder_name_for(repo_type: RepoType, repo_id: &str) -> String {
    // BORROW: explicit .to_owned() for &str → owned String
    let mut name = repo_type.folder_prefix().to_owned();
    for segment in repo_id.split('/') {
        name.push_str("--");
        name.push_str(segment);
    }
    name
}

/// Reconstructs a repo's type and ID from a cache folder name such as
/// `datasets--org--name`, or `None` for any other directory (the shared blob
/// store, `.locks`, a stray file) (new in v0.12.5).
///
/// The first `--` after the type prefix becomes the `/` of the ID; a
/// root-level repo (`models--gpt2`, `datasets--wikitext`) has none and keeps
/// its bare name.
#[must_use]
pub fn parse_repo_folder_name(dir_name: &str) -> Option<(RepoType, String)> {
    RepoType::ALL.into_iter().find_map(|repo_type| {
        let repo_part = dir_name
            .strip_prefix(repo_type.folder_prefix())?
            .strip_prefix("--")?;
        let repo_id = match repo_part.split_once("--") {
            Some((org, name)) => format!("{org}/{name}"),
            // BORROW: explicit .to_owned() for &str → owned String
            None => repo_part.to_owned(),
        };
        Some((repo_type, repo_id))
    })
}

/// Repo root directory: `{cache_root}/models--org--name/`.
#[must_use]
pub fn repo_dir(cache_root: &Path, repo_id: &str) -> PathBuf {
    repo_dir_for(cache_root, RepoType::Model, repo_id)
}

/// Repo root directory for any repo type:
/// `{cache_root}/datasets--org--name/` and so on (new in v0.12.5).
#[must_use]
pub fn repo_dir_for(cache_root: &Path, repo_type: RepoType, repo_id: &str) -> PathBuf {
    cache_root.join(repo_folder_name_for(repo_type, repo_id))
}

/// Snapshots directory: `{repo_dir}/snapshots/`.
#[must_use]
pub fn snapshots_dir(repo_dir: &Path) -> PathBuf {
    repo_dir.join("snapshots")
}

/// Snapshot directory for a specific commit: `{repo_dir}/snapshots/{commit_hash}/`.
#[must_use]
pub fn snapshot_dir(repo_dir: &Path, commit_hash: &str) -> PathBuf {
    snapshots_dir(repo_dir).join(commit_hash)
}

/// Pointer path: `{repo_dir}/snapshots/{commit_hash}/{filename}`.
#[must_use]
pub fn pointer_path(repo_dir: &Path, commit_hash: &str, filename: &str) -> PathBuf {
    snapshot_dir(repo_dir, commit_hash).join(filename)
}

/// Blobs directory: `{repo_dir}/blobs/`.
#[must_use]
pub fn blobs_dir(repo_dir: &Path) -> PathBuf {
    repo_dir.join("blobs")
}

/// Blob path: `{repo_dir}/blobs/{etag}`.
#[must_use]
pub fn blob_path(repo_dir: &Path, etag: &str) -> PathBuf {
    blobs_dir(repo_dir).join(etag)
}

/// Temp blob path for chunked downloads: `{repo_dir}/blobs/{etag}.chunked.part`.
///
/// Uses string concatenation rather than [`Path::with_extension`] to handle
/// etags containing periods (e.g., `"abc.def"` → `"abc.def.chunked.part"`,
/// not `"abc.chunked.part"`).
#[must_use]
pub fn temp_blob_path(repo_dir: &Path, etag: &str) -> PathBuf {
    // BORROW: explicit .to_owned() for &str → owned String for path concatenation
    let mut name = etag.to_owned();
    name.push_str(".chunked.part");
    blobs_dir(repo_dir).join(name)
}

/// Resume-state sidecar path for chunked downloads:
/// `{repo_dir}/blobs/{etag}.chunked.part.state`.
///
/// Lives next to the [`temp_blob_path`] partial and tracks per-chunk
/// completion offsets so that an interrupted download can resume on the
/// next invocation. Cleaned up on successful finalization, kept alongside
/// the partial when the download is interrupted.
///
/// Same period-handling rationale as [`temp_blob_path`]: explicit string
/// concatenation rather than [`Path::with_extension`] so that etags
/// containing periods round-trip correctly.
#[must_use]
pub fn temp_state_path(repo_dir: &Path, etag: &str) -> PathBuf {
    // BORROW: explicit .to_owned() for &str → owned String for path concatenation
    let mut name = etag.to_owned();
    name.push_str(".chunked.part.state");
    blobs_dir(repo_dir).join(name)
}

/// Refs directory: `{repo_dir}/refs/`.
#[must_use]
pub fn refs_dir(repo_dir: &Path) -> PathBuf {
    repo_dir.join("refs")
}

/// Ref file path: `{repo_dir}/refs/{revision}`.
#[must_use]
pub fn ref_path(repo_dir: &Path, revision: &str) -> PathBuf {
    refs_dir(repo_dir).join(revision)
}

/// The cache-wide shared blob store: `{cache_root}/blobs/`.
///
/// Not part of the `hf-hub` layout: `huggingface_hub` 1.32.0 and later keeps
/// each Xet file once here, at `blobs/{2 hex digits}/{xet hash}`, and makes
/// every repo's `blobs/{etag}` a relative symlink to it (wherever symlinks
/// are available). A sibling of the `models--*` directories, never one of
/// them.
#[must_use]
pub fn shared_blobs_dir(cache_root: &Path) -> PathBuf {
    cache_root.join("blobs")
}

/// Download lock path: `{cache_root}/.locks/{repo_folder}/{etag}.lock`.
///
/// `hf-hub` 1.0 and `huggingface_hub` both hold an exclusive OS lock on this
/// file while they write `blobs/{etag}.incomplete`; `cache clean-partial`
/// tests it before removing one. This crate's own chunked downloads do not
/// take it.
#[must_use]
pub fn download_lock_path(cache_root: &Path, repo_folder: &str, etag: &str) -> PathBuf {
    // BORROW: explicit .to_owned() for &str → owned String for path concatenation
    let mut name = etag.to_owned();
    name.push_str(".lock");
    download_locks_dir(cache_root, repo_folder).join(name)
}

/// Directory holding one repo's download locks:
/// `{cache_root}/.locks/{repo_folder}/` (new in v0.12.6). `cache dedupe`
/// tests every lock in it before touching the repo.
#[must_use]
pub fn download_locks_dir(cache_root: &Path, repo_folder: &str) -> PathBuf {
    cache_root.join(".locks").join(repo_folder)
}

/// Header-cache directory: `{repo_dir}/.hf-fm-header-cache/`.
///
/// An `hf-fm`-private sidecar directory, not part of the standard `hf-hub`
/// layout — holds the entries `inspect --cache-headers` and `quants --fits
/// --cache-headers` write, following the same
/// precedent as the `.hf-fm-snapshot.json` sidecar in [`crate::cache`].
#[must_use]
pub fn header_cache_dir(repo_dir: &Path) -> PathBuf {
    repo_dir.join(".hf-fm-header-cache")
}

/// Header-cache entry path:
/// `{repo_dir}/.hf-fm-header-cache/{sanitized filename}.{etag}.json`.
///
/// `filename` may contain path separators (a nested file, e.g.
/// `subdir/model.gguf`) — sanitized to a single path component by replacing
/// `/` and `\` with `__`, so the cache directory never needs subdirectories
/// of its own. Uses string concatenation rather than [`Path::with_extension`]
/// so an etag containing periods round-trips correctly, the same rationale
/// as [`temp_blob_path`] / [`temp_state_path`].
#[must_use]
pub fn header_cache_path(repo_dir: &Path, filename: &str, etag: &str) -> PathBuf {
    let mut name = filename.replace(['/', '\\'], "__");
    name.push('.');
    name.push_str(etag);
    name.push_str(".json");
    header_cache_dir(repo_dir).join(name)
}

/// Longest etag or commit hash [`is_safe_hub_name`] accepts, in bytes.
///
/// The Hub's values are 40 (git SHA-1) or 64 (SHA-256) hex digits. 200
/// leaves room for the longest suffix this crate appends to an etag
/// (`.chunked.part.state.tmp`, 23 bytes) within the 255-byte file-name limit
/// of the common filesystems.
const MAX_HUB_NAME_LEN: usize = 200;

/// Whether `name`, an etag or a commit hash the Hub sent in a response
/// header, is safe to use as one file or directory name inside the cache.
///
/// These values become `blobs/{etag}`, `blobs/{etag}.chunked.part`,
/// `.hf-fm-header-cache/….{etag}.json` and `snapshots/{commit_hash}/`, so a
/// value such as `../../x` would otherwise name a path outside the repo
/// directory. The Hub's own values are hex digests, which lets the rule be an
/// allowlist rather than a list of dangers: ASCII letters, digits, `.`, `-`
/// and `_`, starting and ending with a letter or digit (so never `.`, `..`,
/// or a trailing dot that Windows would silently drop), and at most 200
/// bytes, which leaves room for the longest suffix this crate appends
/// within the 255-byte file-name limit of the common filesystems.
#[must_use]
pub fn is_safe_hub_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    let (Some(first), Some(last)) = (bytes.first(), bytes.last()) else {
        return false;
    };
    bytes.len() <= MAX_HUB_NAME_LEN
        && first.is_ascii_alphanumeric()
        && last.is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
}

/// Whether `revision` is a full commit hash: exactly 40 hex digits, the rule
/// `hf-hub` 1.0 applies to decide the same question.
///
/// `hf-hub` and `huggingface_hub` write no `refs/<revision>` file for such a
/// revision, since it already names its snapshot directory, so a cache
/// lookup resolves it to itself ([`crate::cache::read_ref`]).
#[must_use]
pub(crate) fn is_commit_hash(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Whether `filename`, a file's path within a repo as the Hub lists it
/// (`config.json`, `subdir/model.safetensors`), stays inside the snapshot
/// directory it is joined to.
///
/// Rejects an empty name; an absolute path, which [`Path::join`] would let
/// replace the snapshot directory outright; an empty, `.` or `..` component;
/// a backslash, which is a separator on Windows; a colon, which names a
/// drive or an NTFS alternate data stream there; and a NUL. The Hub keeps
/// repos in git, which already refuses most of these, so this is defence in
/// depth for every path this crate builds from a listed filename.
///
/// Windows' reserved device names (`CON`, `NUL.txt`, …) and names ending in
/// a dot or a space, which Windows does not store as written, are not
/// refused: they cannot climb out of the directory, so they are left to the
/// filesystem.
#[must_use]
pub fn is_safe_repo_filename(filename: &str) -> bool {
    !filename.is_empty()
        && !filename.contains(['\\', ':', '\0'])
        && filename
            .split('/')
            .all(|component| !component.is_empty() && component != "." && component != "..")
}

/// [`is_safe_hub_name`] as a `Result`, for the call sites that must stop.
///
/// # Errors
///
/// Returns [`crate::FetchError::UnsafeName`] carrying `kind` (`"etag"` or
/// `"commit hash"`) when `name` is not safe.
pub(crate) fn ensure_safe_hub_name(
    kind: &'static str,
    name: &str,
) -> Result<(), crate::FetchError> {
    if is_safe_hub_name(name) {
        Ok(())
    } else {
        Err(crate::FetchError::UnsafeName {
            kind,
            // BORROW: explicit .to_owned() for &str → owned String field
            name: name.to_owned(),
        })
    }
}

/// [`is_safe_repo_filename`] as a `Result`, for the call sites that must stop.
///
/// # Errors
///
/// Returns [`crate::FetchError::UnsafeName`] when `filename` is not safe.
pub(crate) fn ensure_safe_repo_filename(filename: &str) -> Result<(), crate::FetchError> {
    if is_safe_repo_filename(filename) {
        Ok(())
    } else {
        Err(crate::FetchError::UnsafeName {
            kind: "filename",
            // BORROW: explicit .to_owned() for &str → owned String field
            name: filename.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn folder_names_carry_the_repo_type_plural() {
        assert_eq!(repo_folder_name("org/name"), "models--org--name");
        assert_eq!(
            repo_folder_name_for(RepoType::Dataset, "org/name"),
            "datasets--org--name"
        );
        assert_eq!(
            repo_folder_name_for(RepoType::Space, "org/name"),
            "spaces--org--name"
        );
        // A root-level repo has no organisation segment.
        assert_eq!(
            repo_folder_name_for(RepoType::Dataset, "wikitext"),
            "datasets--wikitext"
        );
    }

    #[test]
    fn parse_repo_folder_name_inverts_repo_folder_name_for() {
        for repo_type in RepoType::ALL {
            for repo_id in ["org/name", "gpt2", "org/a-b"] {
                let folder = repo_folder_name_for(repo_type, repo_id);
                assert_eq!(
                    parse_repo_folder_name(&folder),
                    Some((repo_type, repo_id.to_owned())),
                    "{folder}"
                );
            }
        }
    }

    #[test]
    fn parse_repo_folder_name_ignores_what_is_not_a_repo_folder() {
        for name in [
            "blobs",
            ".locks",
            "CACHEDIR.TAG",
            "models",
            "models-x",
            "modelsx--a",
        ] {
            assert_eq!(parse_repo_folder_name(name), None, "{name}");
        }
    }

    #[test]
    fn split_repo_address_reads_the_hub_url_prefix() {
        assert_eq!(
            split_repo_address("datasets/org/name"),
            (RepoType::Dataset, "org/name")
        );
        assert_eq!(
            split_repo_address("spaces/org/name"),
            (RepoType::Space, "org/name")
        );
        assert_eq!(
            split_repo_address("datasets/wikitext"),
            (RepoType::Dataset, "wikitext")
        );
        assert_eq!(
            split_repo_address("org/name"),
            (RepoType::Model, "org/name")
        );
        assert_eq!(split_repo_address("gpt2"), (RepoType::Model, "gpt2"));
        // A prefix with nothing after it, or a longer first segment, is a model argument.
        assert_eq!(
            split_repo_address("datasets/"),
            (RepoType::Model, "datasets/")
        );
        assert_eq!(
            split_repo_address("datasetsx/org"),
            (RepoType::Model, "datasetsx/org")
        );
    }

    #[test]
    fn address_is_the_inverse_of_split_repo_address() {
        for repo_type in RepoType::ALL {
            let address = repo_type.address("org/name");
            assert_eq!(split_repo_address(&address), (repo_type, "org/name"));
        }
    }

    #[test]
    fn blob_path_joins_repo_dir_and_etag() {
        let rd = Path::new("/tmp/models--x--y");
        assert_eq!(blob_path(rd, "abc123"), rd.join("blobs").join("abc123"));
    }

    #[test]
    fn temp_blob_path_preserves_periods_in_etag() {
        // Guards the specific behaviour called out in the `temp_blob_path`
        // doc comment: etags containing periods must not be truncated by
        // `Path::with_extension`. `"abc.def"` + `".chunked.part"` →
        // `"abc.def.chunked.part"`, NOT `"abc.chunked.part"`.
        let rd = Path::new("/tmp/models--x--y");
        assert_eq!(
            temp_blob_path(rd, "abc.def"),
            rd.join("blobs").join("abc.def.chunked.part")
        );
    }

    #[test]
    fn temp_state_path_lives_next_to_temp_blob() {
        let rd = Path::new("/tmp/models--x--y");
        let etag = "abc123";
        assert_eq!(
            temp_state_path(rd, etag),
            rd.join("blobs").join("abc123.chunked.part.state")
        );
        // The two helpers must agree on the directory so the sidecar
        // always sits right next to its partial.
        assert_eq!(
            temp_blob_path(rd, etag).parent(),
            temp_state_path(rd, etag).parent()
        );
    }

    #[test]
    fn temp_state_path_preserves_periods_in_etag() {
        let rd = Path::new("/tmp/models--x--y");
        assert_eq!(
            temp_state_path(rd, "abc.def"),
            rd.join("blobs").join("abc.def.chunked.part.state")
        );
    }

    #[test]
    fn header_cache_path_joins_repo_dir_filename_and_etag() {
        let rd = Path::new("/tmp/models--x--y");
        assert_eq!(
            header_cache_path(rd, "model.gguf", "abc123"),
            rd.join(".hf-fm-header-cache")
                .join("model.gguf.abc123.json")
        );
    }

    #[test]
    fn header_cache_path_sanitizes_nested_filenames() {
        let rd = Path::new("/tmp/models--x--y");
        assert_eq!(
            header_cache_path(rd, "subdir/model.gguf", "abc123"),
            rd.join(".hf-fm-header-cache")
                .join("subdir__model.gguf.abc123.json")
        );
    }

    #[test]
    fn header_cache_path_preserves_periods_in_etag() {
        let rd = Path::new("/tmp/models--x--y");
        assert_eq!(
            header_cache_path(rd, "model.gguf", "abc.def"),
            rd.join(".hf-fm-header-cache")
                .join("model.gguf.abc.def.json")
        );
    }

    #[test]
    fn hub_names_the_hub_sends_are_safe() {
        assert!(is_safe_hub_name(
            "9a3f7c1e0b2d4f6a8c0e1b3d5f7a9c1e3b5d7f9a1c3e5b7d9f1a3c5e7b9d1f3a"
        ));
        assert!(is_safe_hub_name("1110a243fdf4706b3f48f1d95db1a4f5529b4d41"));
        // Etags with periods occur in the tests above and must stay valid.
        assert!(is_safe_hub_name("abc.def"));
        assert!(is_safe_hub_name("a-b_c.d"));
        assert!(is_safe_hub_name(&"a".repeat(MAX_HUB_NAME_LEN)));
    }

    #[test]
    fn hub_names_that_could_leave_their_directory_are_refused() {
        for name in [
            "", ".", "..", "../x", "a/b", "a\\b", "/abs", "c:x", "x.", ".hidden", "-x-", "a b",
            "a\0b", "é",
        ] {
            assert!(!is_safe_hub_name(name), "{name:?} should be refused");
        }
        assert!(!is_safe_hub_name(&"a".repeat(MAX_HUB_NAME_LEN + 1)));
    }

    #[test]
    fn repo_filenames_the_hub_lists_are_safe() {
        for name in [
            "config.json",
            "model-00001-of-00002.safetensors",
            "1_Pooling/config.json",
            "layer_9/width_16k/average_l0_47/params.npz",
            ".gitattributes",
            "..data",
            "a..b",
        ] {
            assert!(is_safe_repo_filename(name), "{name:?} should be accepted");
        }
    }

    #[test]
    fn repo_filenames_that_could_leave_the_snapshot_are_refused() {
        for name in [
            "",
            "/etc/passwd",
            "../x",
            "a/../../x",
            "a/./b",
            ".",
            "..",
            "a//b",
            "a/",
            "a\\..\\b",
            "c:\\x",
            "C:x",
            "file.txt:stream",
            "a\0b",
        ] {
            assert!(!is_safe_repo_filename(name), "{name:?} should be refused");
        }
    }

    #[test]
    fn header_cache_path_lives_under_header_cache_dir() {
        let rd = Path::new("/tmp/models--x--y");
        assert_eq!(
            header_cache_path(rd, "model.gguf", "abc123").parent(),
            Some(header_cache_dir(rd).as_path())
        );
    }
}
