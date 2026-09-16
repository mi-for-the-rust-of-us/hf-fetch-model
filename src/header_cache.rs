// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in on-disk cache for parsed remote tensor-file headers.
//!
//! `inspect --cache-headers` persists the parsed header — the same
//! [`SafetensorsHeaderInfo`] every format (`.safetensors` / `.gguf` /
//! `.npz` / `.pth`) normalizes into —
//! keyed on `(repo, revision, filename, etag)`, so repeat inspection of the
//! same remote file across invocations (the natural pattern of iterative
//! narrowing over a handful of quant candidates) is free on the second and
//! third call.
//!
//! Off by default: a plain remote `inspect` never touches local disk
//! without this flag. Entries live under
//! [`cache_layout::header_cache_path`](crate::cache_layout::header_cache_path)
//! — an `hf-fm`-private sidecar directory alongside the standard `hf-hub`
//! layout, not inside it — following the same precedent as the
//! `.hf-fm-snapshot.json` sidecar in [`crate::cache`].

use std::path::Path;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::error::FetchError;
use crate::inspect::SafetensorsHeaderInfo;

/// Cache schema version embedded in every entry.
///
/// Bumped whenever the on-disk JSON shape changes incompatibly, including
/// transitively via [`SafetensorsHeaderInfo`]'s own fields. A mismatch is
/// treated as a cache miss — see [`HeaderCacheEntry::is_compatible_with`].
const SCHEMA_VERSION: u32 = 1;

/// On-disk cache entry for one parsed remote header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderCacheEntry {
    /// Cache schema version. Compared against `SCHEMA_VERSION`.
    pub schema_version: u32,
    /// The repository identifier this entry was parsed for.
    pub repo: String,
    /// The resolved revision (branch, tag, or commit SHA — `"main"` when
    /// the caller did not pin one).
    pub revision: String,
    /// The filename within the repo.
    pub filename: String,
    /// The remote etag the header was fetched against. A different etag on
    /// a later call means the upstream file changed, invalidating this entry.
    pub etag: String,
    /// When this entry was written — feeds the `Source: cached header (age:
    /// ...)` display line.
    pub cached_at: SystemTime,
    /// The parsed header.
    pub info: SafetensorsHeaderInfo,
}

impl HeaderCacheEntry {
    /// Builds a fresh entry for a header just parsed remotely.
    #[must_use]
    pub fn new(
        repo: String,
        revision: String,
        filename: String,
        etag: String,
        info: SafetensorsHeaderInfo,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            repo,
            revision,
            filename,
            etag,
            cached_at: SystemTime::now(),
            info,
        }
    }

    /// Returns `true` when this entry is still valid for the given
    /// request — every field but `cached_at`/`info` must match exactly.
    #[must_use]
    pub fn is_compatible_with(
        &self,
        repo: &str,
        revision: &str,
        filename: &str,
        etag: &str,
    ) -> bool {
        self.schema_version == SCHEMA_VERSION
            && self.repo == repo
            && self.revision == revision
            && self.filename == filename
            && self.etag == etag
    }

    /// Reads and validates the cache entry at `path` against the given
    /// request.
    ///
    /// Returns `None` on a cache miss for any reason — an absent file,
    /// unparseable JSON, or a stale/mismatched entry — never an error; the
    /// caller falls back to a normal remote fetch either way.
    pub async fn load(
        path: &Path,
        repo: &str,
        revision: &str,
        filename: &str,
        etag: &str,
    ) -> Option<Self> {
        let text = tokio::fs::read_to_string(path).await.ok()?;
        let entry: Self = serde_json::from_str(&text).ok()?;
        if entry.is_compatible_with(repo, revision, filename, etag) {
            Some(entry)
        } else {
            None
        }
    }

    /// Writes this entry to `path` atomically (write-tmp + rename), via the
    /// shared `atomic_write::write_atomic` helper — the same
    /// durability pattern `chunked_state::ChunkedState::save_atomic`
    /// uses. Creates the parent directory (`.hf-fm-header-cache/`) if it
    /// does not exist yet — the first `--cache-headers` call against a repo
    /// that was never downloaded.
    ///
    /// # Errors
    ///
    /// Returns [`FetchError::Io`] on filesystem errors during the parent
    /// directory creation, the temp write, or the rename.
    pub async fn save_atomic(&self, path: &Path) -> Result<(), FetchError> {
        let json = serde_json::to_string(self).map_err(|e| {
            FetchError::Http(format!("failed to serialize header-cache entry: {e}"))
        })?;
        let tmp = path.with_extension("json.tmp");
        crate::atomic_write::write_atomic(path, &tmp, json.as_bytes()).await
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::inspect::TensorInfo;

    fn sample_info() -> SafetensorsHeaderInfo {
        SafetensorsHeaderInfo::new(
            vec![TensorInfo {
                name: "model.embed.weight".to_owned(),
                dtype: "BF16".to_owned(),
                shape: vec![100, 100],
                data_offsets: (0, 20_000),
            }],
            None,
            64,
            Some(20_064),
            None,
        )
    }

    #[test]
    fn is_compatible_with_matches_every_key_field() {
        let entry = HeaderCacheEntry::new(
            "org/model".to_owned(),
            "main".to_owned(),
            "model.gguf".to_owned(),
            "etag-1".to_owned(),
            sample_info(),
        );
        assert!(entry.is_compatible_with("org/model", "main", "model.gguf", "etag-1"));
        assert!(!entry.is_compatible_with("org/other", "main", "model.gguf", "etag-1"));
        assert!(!entry.is_compatible_with("org/model", "v2", "model.gguf", "etag-1"));
        assert!(!entry.is_compatible_with("org/model", "main", "other.gguf", "etag-1"));
        assert!(!entry.is_compatible_with("org/model", "main", "model.gguf", "etag-2"));
    }

    #[test]
    fn is_compatible_with_rejects_schema_version_mismatch() {
        let mut entry = HeaderCacheEntry::new(
            "org/model".to_owned(),
            "main".to_owned(),
            "model.gguf".to_owned(),
            "etag-1".to_owned(),
            sample_info(),
        );
        entry.schema_version = SCHEMA_VERSION + 1;
        assert!(!entry.is_compatible_with("org/model", "main", "model.gguf", "etag-1"));
    }

    #[tokio::test]
    async fn save_then_load_round_trips() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(".hf-fm-header-cache").join("entry.json");
        let entry = HeaderCacheEntry::new(
            "org/model".to_owned(),
            "main".to_owned(),
            "model.gguf".to_owned(),
            "etag-1".to_owned(),
            sample_info(),
        );

        entry.save_atomic(&path).await.expect("save");
        let loaded = HeaderCacheEntry::load(&path, "org/model", "main", "model.gguf", "etag-1")
            .await
            .expect("load should hit");

        assert_eq!(loaded.repo, entry.repo);
        assert_eq!(loaded.etag, entry.etag);
        assert_eq!(loaded.info.tensors.len(), entry.info.tensors.len());
        assert_eq!(
            loaded.info.tensors.first().map(|t| t.name.as_str()),
            Some("model.embed.weight")
        );
    }

    #[tokio::test]
    async fn load_returns_none_for_mismatched_etag() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(".hf-fm-header-cache").join("entry.json");
        let entry = HeaderCacheEntry::new(
            "org/model".to_owned(),
            "main".to_owned(),
            "model.gguf".to_owned(),
            "etag-1".to_owned(),
            sample_info(),
        );
        entry.save_atomic(&path).await.expect("save");

        let loaded =
            HeaderCacheEntry::load(&path, "org/model", "main", "model.gguf", "etag-2").await;
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn load_returns_none_for_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(".hf-fm-header-cache").join("missing.json");
        let loaded =
            HeaderCacheEntry::load(&path, "org/model", "main", "model.gguf", "etag-1").await;
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn save_atomic_does_not_leave_tmp_behind() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(".hf-fm-header-cache").join("entry.json");
        let entry = HeaderCacheEntry::new(
            "org/model".to_owned(),
            "main".to_owned(),
            "model.gguf".to_owned(),
            "etag-1".to_owned(),
            sample_info(),
        );
        entry.save_atomic(&path).await.expect("save");
        assert!(!path.with_extension("json.tmp").exists());
        assert!(path.exists());
    }
}
