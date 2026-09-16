// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared atomic-write helper for the crate's on-disk JSON sidecars.
//!
//! [`crate::chunked_state::ChunkedState`]'s per-chunk progress sidecar,
//! [`crate::header_cache::HeaderCacheEntry`]'s cached header, and
//! [`crate::cache::Snapshot`]'s `--preset`/`--filter` sidecar all need the
//! same durability guarantee — a process crash mid-save must never leave a
//! half-written file a later read would choke on — so the write-tmp +
//! rename mechanics live here once instead of being duplicated per sidecar.
//! Two variants share that mechanics: [`write_atomic`] (async, `tokio::fs`)
//! for the first two, [`write_atomic_sync`] (`std::fs`) for the third,
//! which writes from a synchronous call site with no `tokio` runtime handy.

use std::path::Path;

use crate::error::FetchError;

/// Writes `contents` to `path` atomically via a sibling temp file at `tmp`,
/// then a rename onto `path`.
///
/// The rename is atomic on POSIX and effectively atomic on Windows
/// (`MoveFileEx`), so a crash between the write and the rename leaves either
/// the previous valid file or the new one intact — never a half-written
/// blend that would fail to parse on a later read. `path`'s parent
/// directory is created first (a no-op if it already exists), covering a
/// sidecar directory that may not exist yet on a first write.
///
/// Callers own JSON serialization and the choice of `tmp` path — this
/// helper only performs the write-then-rename, so a serialization failure
/// (mapped to a caller-specific [`FetchError`] variant) never reaches here.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if `path`'s parent directory cannot be
/// created.
/// Returns [`FetchError::Io`] if the write to `tmp` fails.
/// Returns [`FetchError::Io`] if the rename from `tmp` to `path` fails.
pub(crate) async fn write_atomic(
    path: &Path,
    tmp: &Path,
    contents: &[u8],
) -> Result<(), FetchError> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| FetchError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
    }
    tokio::fs::write(tmp, contents)
        .await
        .map_err(|e| FetchError::Io {
            path: tmp.to_path_buf(),
            source: e,
        })?;
    tokio::fs::rename(tmp, path)
        .await
        .map_err(|e| FetchError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
    Ok(())
}

/// Synchronous counterpart to [`write_atomic`], identical mechanics via
/// `std::fs` instead of `tokio::fs` — for a caller with no `tokio` runtime
/// handy at its call site.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if `path`'s parent directory cannot be
/// created.
/// Returns [`FetchError::Io`] if the write to `tmp` fails.
/// Returns [`FetchError::Io`] if the rename from `tmp` to `path` fails.
pub(crate) fn write_atomic_sync(
    path: &Path,
    tmp: &Path,
    contents: &[u8],
) -> Result<(), FetchError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| FetchError::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }
    std::fs::write(tmp, contents).map_err(|e| FetchError::Io {
        path: tmp.to_path_buf(),
        source: e,
    })?;
    std::fs::rename(tmp, path).map_err(|e| FetchError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[tokio::test]
    async fn write_atomic_creates_missing_parent_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nested").join("file.json");
        let tmp = path.with_extension("json.tmp");

        write_atomic(&path, &tmp, b"{}").await.expect("write");

        assert!(path.exists());
        assert!(!tmp.exists(), "tmp file must be renamed away");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{}");
    }

    #[tokio::test]
    async fn write_atomic_overwrites_an_existing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("file.json");
        let tmp = path.with_extension("json.tmp");

        write_atomic(&path, &tmp, b"first").await.expect("write");
        write_atomic(&path, &tmp, b"second").await.expect("write");

        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
    }

    #[tokio::test]
    async fn write_atomic_does_not_leave_tmp_behind() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("file.json");
        let tmp = path.with_extension("json.tmp");

        write_atomic(&path, &tmp, b"payload").await.expect("write");

        assert!(!tmp.exists());
        assert!(path.exists());
    }

    #[test]
    fn write_atomic_sync_creates_missing_parent_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nested").join("file.json");
        let tmp = path.with_extension("json.tmp");

        write_atomic_sync(&path, &tmp, b"{}").expect("write");

        assert!(path.exists());
        assert!(!tmp.exists(), "tmp file must be renamed away");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{}");
    }

    #[test]
    fn write_atomic_sync_overwrites_an_existing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("file.json");
        let tmp = path.with_extension("json.tmp");

        write_atomic_sync(&path, &tmp, b"first").expect("write");
        write_atomic_sync(&path, &tmp, b"second").expect("write");

        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
    }
}
