// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in on-disk cache for parsed remote tensor-file headers.
//!
//! `inspect --cache-headers` persists the parsed header — the same
//! [`SafetensorsHeaderInfo`] every format (`.safetensors` / `.gguf` /
//! `.npz` / `.pth`) normalizes into —
//! keyed on `(repo, revision, filename, etag)`, so repeat inspection of the
//! same remote file across invocations (the natural pattern of iterative
//! narrowing over a handful of quant candidates) skips the header's own
//! range requests. An entry younger than [`TRUST_WINDOW`] is used without
//! any request at all (see [`find_recent`]); an older one is used after the
//! reader's 2-request probe confirms the file's etag is unchanged. Saving
//! an entry removes the ones it supersedes (see [`prune_superseded`]), so
//! an upstream change does not leave the old one behind.
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

/// How long an entry is used without checking the file's etag again.
///
/// Within this window a hit makes no request at all, so repeat inspections
/// during one narrowing session work through a rate limit, or offline. Past
/// it, a hit first spends the reader's 2-request probe to confirm the etag
/// is unchanged, and a confirmed entry is trusted for another window. The
/// cost is that a file replaced upstream inside the window is read from its
/// old header until the window ends; published quant files rarely change.
pub const TRUST_WINDOW: std::time::Duration = std::time::Duration::from_secs(60 * 60);

/// Finds the most recently written entry in `dir` for `(repo, revision,
/// filename)` that is younger than `within`, whatever its etag.
///
/// This is the lookup a hit inside [`TRUST_WINDOW`] uses, before any
/// request has been made: the etag is not known yet, so entries are matched
/// on everything else. Entries that cannot be read, parsed or used under the
/// current schema are skipped, as are future-dated ones.
pub async fn find_recent(
    dir: &Path,
    repo: &str,
    revision: &str,
    filename: &str,
    within: std::time::Duration,
) -> Option<HeaderCacheEntry> {
    let mut entries = tokio::fs::read_dir(dir).await.ok()?;
    let mut newest: Option<HeaderCacheEntry> = None;
    // EXPLICIT: an async directory stream has no iterator chain; each entry
    // is read and parsed before it can be compared.
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "json") {
            continue;
        }
        let Ok(text) = tokio::fs::read_to_string(&path).await else {
            continue;
        };
        let Ok(candidate) = serde_json::from_str::<HeaderCacheEntry>(&text) else {
            continue;
        };
        let recent = candidate.cached_at.elapsed().is_ok_and(|age| age < within);
        if recent
            && candidate.is_compatible_with(repo, revision, filename, &candidate.etag)
            && newest
                .as_ref()
                .is_none_or(|best| candidate.cached_at > best.cached_at)
        {
            newest = Some(candidate);
        }
    }
    newest
}

/// Removes the entries in `dir` that the one just saved for `(repo,
/// revision, filename, etag)` supersedes: the same repo, revision and
/// filename under a different etag.
///
/// A lookup only ever matches the file's current etag, so once the upstream
/// file changes, the entry for its previous etag can never hit again;
/// without this, every upstream change would leave one behind for good.
/// Entries for other revisions are kept, so inspecting a pinned revision and
/// `main` in turn never evicts either. Best effort on the reading side: an
/// entry that cannot be read or parsed (another schema, a stray file) is
/// left alone. Returns how many entries were removed.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if a superseded entry cannot be removed.
pub async fn prune_superseded(
    dir: &Path,
    repo: &str,
    revision: &str,
    filename: &str,
    etag: &str,
) -> Result<usize, FetchError> {
    let Ok(mut entries) = tokio::fs::read_dir(dir).await else {
        return Ok(0);
    };
    let mut removed = 0;
    // EXPLICIT: an async directory stream has no iterator chain; each entry
    // is read and parsed before deciding whether to remove it.
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "json") {
            continue;
        }
        let Ok(text) = tokio::fs::read_to_string(&path).await else {
            continue;
        };
        let Ok(other) = serde_json::from_str::<HeaderCacheEntry>(&text) else {
            continue;
        };
        if other.repo == repo
            && other.revision == revision
            && other.filename == filename
            && other.etag != etag
        {
            tokio::fs::remove_file(&path)
                .await
                .map_err(|e| FetchError::Io { path, source: e })?;
            removed += 1;
        }
    }
    Ok(removed)
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

    /// Saves an entry for `(org/model, revision, filename, etag)` at its
    /// real cache path under `repo_dir`, returning that path.
    async fn save_entry(
        repo_dir: &Path,
        revision: &str,
        filename: &str,
        etag: &str,
    ) -> std::path::PathBuf {
        let path = crate::cache_layout::header_cache_path(repo_dir, filename, etag);
        HeaderCacheEntry::new(
            "org/model".to_owned(),
            revision.to_owned(),
            filename.to_owned(),
            etag.to_owned(),
            sample_info(),
        )
        .save_atomic(&path)
        .await
        .expect("save");
        path
    }

    #[tokio::test]
    async fn prune_superseded_removes_only_the_same_file_and_revision() {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo_dir = dir.path();
        let old = save_entry(repo_dir, "main", "model.gguf", "etag-1").await;
        let current = save_entry(repo_dir, "main", "model.gguf", "etag-2").await;
        let pinned = save_entry(repo_dir, "v2", "model.gguf", "etag-3").await;
        let other_file = save_entry(repo_dir, "main", "other.gguf", "etag-4").await;
        let cache_dir = crate::cache_layout::header_cache_dir(repo_dir);
        let stray = cache_dir.join("stray.json");
        std::fs::write(&stray, b"not an entry").expect("write stray file");

        let removed = prune_superseded(&cache_dir, "org/model", "main", "model.gguf", "etag-2")
            .await
            .expect("prune");

        assert_eq!(removed, 1);
        assert!(!old.exists(), "the superseded etag should be removed");
        assert!(current.exists(), "the entry just saved should stay");
        assert!(pinned.exists(), "another revision's entry should stay");
        assert!(other_file.exists(), "another file's entry should stay");
        assert!(stray.exists(), "an unparseable file should be left alone");
    }

    /// Saves an entry whose `cached_at` is `age` in the past.
    async fn save_aged_entry(
        repo_dir: &Path,
        revision: &str,
        filename: &str,
        etag: &str,
        age: std::time::Duration,
    ) {
        let path = crate::cache_layout::header_cache_path(repo_dir, filename, etag);
        let mut entry = HeaderCacheEntry::new(
            "org/model".to_owned(),
            revision.to_owned(),
            filename.to_owned(),
            etag.to_owned(),
            sample_info(),
        );
        entry.cached_at = SystemTime::now() - age;
        entry.save_atomic(&path).await.expect("save");
    }

    #[tokio::test]
    async fn find_recent_matches_any_etag_inside_the_window_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo_dir = dir.path();
        let cache_dir = crate::cache_layout::header_cache_dir(repo_dir);
        let minute = std::time::Duration::from_secs(60);

        // An hour-and-a-half-old entry is outside the window.
        save_aged_entry(repo_dir, "main", "model.gguf", "etag-old", 90 * minute).await;
        assert!(
            find_recent(&cache_dir, "org/model", "main", "model.gguf", TRUST_WINDOW)
                .await
                .is_none()
        );

        // Two recent entries under different etags: the newer one wins.
        save_aged_entry(repo_dir, "main", "model.gguf", "etag-a", 20 * minute).await;
        save_aged_entry(repo_dir, "main", "model.gguf", "etag-b", 5 * minute).await;
        let found = find_recent(&cache_dir, "org/model", "main", "model.gguf", TRUST_WINDOW)
            .await
            .expect("a recent entry");
        assert_eq!(found.etag, "etag-b");

        // Another revision or another file never matches.
        assert!(
            find_recent(&cache_dir, "org/model", "v2", "model.gguf", TRUST_WINDOW)
                .await
                .is_none()
        );
        assert!(
            find_recent(&cache_dir, "org/model", "main", "other.gguf", TRUST_WINDOW)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn prune_superseded_without_a_cache_dir_removes_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = crate::cache_layout::header_cache_dir(dir.path());
        let removed = prune_superseded(&cache_dir, "org/model", "main", "model.gguf", "etag-1")
            .await
            .expect("prune");
        assert_eq!(removed, 0);
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
