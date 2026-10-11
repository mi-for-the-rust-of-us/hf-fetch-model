// SPDX-License-Identifier: MIT OR Apache-2.0

//! `HuggingFace` cache directory resolution, model family scanning, disk usage,
//! and integrity verification.
//!
//! [`hf_cache_dir()`] locates the local HF cache. [`list_cached_families()`]
//! scans downloaded models and groups them by `model_type`.
//! [`cache_summary()`] provides per-repo totals of bytes on disk,
//! [`cache_repo_usage()`] returns a single repo's per-file sizes, and
//! [`verify_cache()`] re-checks `SHA256` digests of cached files against
//! `HuggingFace` LFS metadata.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::FetchError;

/// Filename of the per-repo `hf-fm` snapshot sidecar.
///
/// Lives at `{cache_root}/models--{org}--{name}/.hf-fm-snapshot.json`.
/// Written by `download` (recording the active `--preset` / `--filter` /
/// `--exclude`) and consumed by `status` (to distinguish files deliberately
/// skipped via the preset's glob list from files that are genuinely missing).
pub const SNAPSHOT_FILENAME: &str = ".hf-fm-snapshot.json";

/// Schema version of the on-disk [`Snapshot`] file. Bumped on incompatible changes.
pub const SNAPSHOT_VERSION: u32 = 1;

/// On-disk record of the arguments that produced a cached repository.
///
/// Persisted by `hf-fm download` as a small JSON file at the repository's
/// cache root, alongside `refs/`, `blobs/`, and `snapshots/`. Read back by
/// `hf-fm status` so files that don't match the recorded preset can be
/// reported as [`FileStatus::Excluded`] instead of [`FileStatus::Missing`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Schema version. Equals [`SNAPSHOT_VERSION`] for newly-written files.
    pub version: u32,
    /// Git revision at download time (resolved commit SHA or branch name).
    pub revision: String,
    /// The `--preset` value used at download time, if any. One of
    /// `"safetensors"`, `"gguf"`, `"npz"`, `"pth"`, `"config-only"`.
    pub preset: Option<String>,
    /// `--filter` glob patterns used at download time. Reserved for a later
    /// patch that adds `status --filter`; not yet consumed by `status`.
    pub filter: Vec<String>,
    /// `--exclude` glob patterns used at download time. Reserved for a later
    /// patch that adds `status --exclude`; not yet consumed by `status`.
    pub exclude: Vec<String>,
}

/// Returns the absolute path of the [`SNAPSHOT_FILENAME`] sidecar for a given
/// repository cache directory.
#[must_use]
pub fn snapshot_path(repo_dir: &Path) -> PathBuf {
    repo_dir.join(SNAPSHOT_FILENAME)
}

/// Reads the per-repo [`Snapshot`] sidecar if it exists.
///
/// A missing sidecar is not an error — older caches (downloaded before this
/// feature) simply return `Ok(None)`.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the file exists but
/// cannot be read.
/// Returns [`FetchError::InvalidArgument`]
/// if the file is present but its JSON cannot be parsed (e.g. corruption or
/// a future schema version this binary cannot understand).
pub fn read_snapshot(repo_dir: &Path) -> Result<Option<Snapshot>, FetchError> {
    let path = snapshot_path(repo_dir);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(FetchError::Io { path, source: e }),
    };
    let snapshot: Snapshot = serde_json::from_slice(&bytes).map_err(|e| {
        FetchError::InvalidArgument(format!("failed to parse snapshot {}: {e}", path.display()))
    })?;
    Ok(Some(snapshot))
}

/// Writes the [`Snapshot`] sidecar for a repository, atomically (write to a
/// `.tmp` sibling, then rename), via the shared
/// `atomic_write::write_atomic_sync` helper — the same durability
/// pattern `chunked_state::ChunkedState::save_atomic` and
/// [`crate::header_cache::HeaderCacheEntry::save_atomic`] use, synchronous
/// here since this call site has no `tokio` runtime handy.
///
/// Overwrites any previously-written sidecar — the design is intentionally
/// last-download-wins. `repo_dir`'s parent directory is created first (a
/// no-op if it already exists), matching the other two sidecars' behavior.
///
/// # Errors
///
/// Returns [`FetchError::InvalidArgument`] if `snapshot` fails to
/// serialize (would indicate a programmer bug — every field of
/// [`Snapshot`] is plain-old-data).
/// Returns [`FetchError::Io`] if `repo_dir`'s parent directory cannot be
/// created.
/// Returns [`FetchError::Io`] if the temp write fails (no write
/// permission, disk full, etc.).
/// Returns [`FetchError::Io`] if the rename fails (e.g. across
/// filesystems).
pub fn write_snapshot(repo_dir: &Path, snapshot: &Snapshot) -> Result<(), FetchError> {
    let path = snapshot_path(repo_dir);
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(snapshot)
        .map_err(|e| FetchError::InvalidArgument(format!("failed to serialize snapshot: {e}")))?;
    crate::atomic_write::write_atomic_sync(&path, &tmp, &bytes)
}

/// Reconstructs a repo ID from a `models--org--name` directory name.
///
/// Returns `None` if the directory name does not start with `models--`.
fn repo_id_from_folder_name(dir_name: &str) -> Option<String> {
    let repo_part = dir_name.strip_prefix("models--")?;

    // Reconstruct repo_id: replace first "--" with "/".
    let repo_id = match repo_part.find("--") {
        Some(pos) => {
            let (org, name_with_sep) = repo_part.split_at(pos);
            let name = name_with_sep.get(2..).unwrap_or_default();
            format!("{org}/{name}")
        }
        None => repo_part.to_string(),
    };

    Some(repo_id)
}

/// Returns the `HuggingFace` Hub cache directory.
///
/// Resolution order:
/// 1. `HF_HOME` environment variable + `/hub`
/// 2. `~/.cache/huggingface/hub/` (via [`dirs::home_dir()`])
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the home directory cannot be determined.
pub fn hf_cache_dir() -> Result<PathBuf, FetchError> {
    if let Ok(home) = std::env::var("HF_HOME") {
        let mut path = PathBuf::from(home);
        path.push("hub");
        return Ok(path);
    }

    let home = dirs::home_dir().ok_or_else(|| FetchError::Io {
        path: PathBuf::from("~"),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, "home directory not found"),
    })?;

    let mut path = home;
    path.push(".cache");
    path.push("huggingface");
    path.push("hub");
    Ok(path)
}

/// One repository inside a cached family, with optional quantization label.
///
/// Returned by [`list_cached_families`] so callers can render a quant column
/// alongside the repo ID without re-reading every snapshot's `config.json`
/// in a second pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FamilyEntry {
    /// The repository identifier (e.g., `"meta-llama/Llama-3.2-1B"`).
    pub repo_id: String,
    /// Quantization method as reported by `quantization_config.quant_method`
    /// in the cached `config.json`. Falls back to `"gguf"` when any cached
    /// file in the newest snapshot directory has a `.gguf` extension.
    /// `None` for full-precision repos.
    pub quant_method: Option<String>,
}

/// Scans the local HF cache for downloaded models and groups them by `model_type`.
///
/// Looks for `config.json` files inside model snapshot directories:
/// `<cache>/models--<org>--<name>/snapshots/*/config.json`
///
/// Returns a map from `model_type` (e.g., `"llama"`) to a sorted list of
/// [`FamilyEntry`] values, each pairing a repo ID with its quantization
/// label (if any).
///
/// Models without a `model_type` field in their `config.json` are skipped.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be read.
pub fn list_cached_families() -> Result<BTreeMap<String, Vec<FamilyEntry>>, FetchError> {
    let cache_dir = hf_cache_dir()?;

    if !cache_dir.exists() {
        return Ok(BTreeMap::new());
    }

    let entries = std::fs::read_dir(&cache_dir).map_err(|e| FetchError::Io {
        path: cache_dir.clone(),
        source: e,
    })?;

    let mut families: BTreeMap<String, Vec<FamilyEntry>> = BTreeMap::new();

    for entry in entries {
        let Ok(entry) = entry else { continue };

        let dir_name = entry.file_name();
        // BORROW: explicit .to_string_lossy() for OsString → str conversion
        let dir_str = dir_name.to_string_lossy();

        let Some(repo_id) = repo_id_from_folder_name(&dir_str) else {
            continue;
        };

        // Find the newest snapshot's config.json
        let snapshots_dir = crate::cache_layout::snapshots_dir(&entry.path());
        if !snapshots_dir.exists() {
            continue;
        }

        if let Some((model_type, quant_method)) = find_family_info_in_snapshots(&snapshots_dir) {
            families.entry(model_type).or_default().push(FamilyEntry {
                repo_id,
                quant_method,
            });
        }
    }

    // Sort repo lists within each family for stable output
    for entries in families.values_mut() {
        entries.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));
    }

    Ok(families)
}

/// Searches snapshot directories for a `config.json` containing `model_type`,
/// and reports an accompanying quantization label when one can be inferred.
///
/// Returns the first `(model_type, quant_method)` pair found. The quant
/// label comes from `quantization_config.quant_method` in `config.json`,
/// or falls back to `Some("gguf".to_owned())` when any sibling file in the
/// same snapshot directory has a `.gguf` extension. Returns `None` if no
/// snapshot yields a parseable `model_type`.
fn find_family_info_in_snapshots(
    snapshots_dir: &std::path::Path,
) -> Option<(String, Option<String>)> {
    let snapshots = std::fs::read_dir(snapshots_dir).ok()?;

    for snap_entry in snapshots {
        let Ok(snap_entry) = snap_entry else { continue };
        let snap_path = snap_entry.path();
        let config_path = snap_path.join("config.json");

        if !config_path.exists() {
            continue;
        }

        if let Some(model_type) = extract_model_type(&config_path) {
            let quant_method = extract_quant_method(&config_path)
                .or_else(|| snapshot_has_gguf(&snap_path).then(|| "gguf".to_owned())); // BORROW: explicit .to_owned()
            return Some((model_type, quant_method));
        }
    }

    None
}

/// Reads a `config.json` file and extracts the `model_type` field.
fn extract_model_type(config_path: &std::path::Path) -> Option<String> {
    let contents = std::fs::read_to_string(config_path).ok()?;
    // BORROW: explicit .as_str() instead of Deref coercion
    let value: serde_json::Value = serde_json::from_str(contents.as_str()).ok()?;
    // BORROW: explicit .as_str() on serde_json Value
    value.get("model_type")?.as_str().map(String::from)
}

/// Reads a `config.json` file and extracts the transformers-standard
/// `quantization_config.quant_method` field, if present.
///
/// Returns `None` when the file is unreadable, malformed, or contains no
/// quantization config (the repo is treated as full-precision in that case;
/// the GGUF filename fallback is applied separately by the caller).
fn extract_quant_method(config_path: &std::path::Path) -> Option<String> {
    let contents = std::fs::read_to_string(config_path).ok()?;
    // BORROW: explicit .as_str() instead of Deref coercion
    let value: serde_json::Value = serde_json::from_str(contents.as_str()).ok()?;
    // BORROW: explicit .as_str() on serde_json Value
    value
        .get("quantization_config")?
        .get("quant_method")?
        .as_str()
        .map(String::from)
}

/// Returns `true` if any file in the snapshot directory has a `.gguf` extension.
///
/// Used as a fallback quant label when `config.json` carries no
/// `quantization_config` (GGUF repos typically lack a transformers-style
/// `config.json` `quantization_config` block). Extension comparison is
/// case-insensitive via `OsStr::eq_ignore_ascii_case`.
fn snapshot_has_gguf(snapshot_dir: &std::path::Path) -> bool {
    let Ok(entries) = std::fs::read_dir(snapshot_dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            return true;
        }
    }
    false
}

/// Status of a single file in the cache.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FileStatus {
    /// File is fully downloaded (local size matches expected size, or no expected size known).
    Complete {
        /// Local file size in bytes.
        local_size: u64,
    },
    /// File exists but is smaller than expected (interrupted download),
    /// or the file's own temp file exists in `blobs/`: this crate's
    /// `<sha256>.chunked.part`, or the `<sha256>.incomplete` of `hf-hub` /
    /// `huggingface_hub` (keyed on the file's LFS `sha256` — per-file
    /// attribution).
    Partial {
        /// Local file size in bytes.
        local_size: u64,
        /// Expected file size in bytes.
        expected_size: u64,
    },
    /// File is not present in the cache.
    Missing {
        /// Expected file size in bytes (0 if unknown).
        expected_size: u64,
    },
    /// File is on the Hub but was deliberately not requested at download time
    /// (or is now filtered out by `status --preset <P>`). Distinguished from
    /// [`FileStatus::Missing`] so the user does not chase a "fix" for an
    /// intentional skip.
    Excluded {
        /// Expected file size in bytes (0 if unknown).
        expected_size: u64,
    },
}

/// Cache status report for a repository.
#[derive(Debug, Clone)]
pub struct RepoStatus {
    /// The repository identifier.
    pub repo_id: String,
    /// The resolved commit hash (if available).
    pub commit_hash: Option<String>,
    /// The cache directory for this repo.
    pub cache_path: PathBuf,
    /// Per-file status, sorted by filename.
    pub files: Vec<(String, FileStatus)>,
}

impl RepoStatus {
    /// Number of fully downloaded files.
    #[must_use]
    pub fn complete_count(&self) -> usize {
        self.files
            .iter()
            .filter(|(_, s)| matches!(s, FileStatus::Complete { .. }))
            .count()
    }

    /// Number of partially downloaded files.
    #[must_use]
    pub fn partial_count(&self) -> usize {
        self.files
            .iter()
            .filter(|(_, s)| matches!(s, FileStatus::Partial { .. }))
            .count()
    }

    /// Number of missing files.
    #[must_use]
    pub fn missing_count(&self) -> usize {
        self.files
            .iter()
            .filter(|(_, s)| matches!(s, FileStatus::Missing { .. }))
            .count()
    }

    /// Number of files deliberately excluded by the active preset / filter.
    ///
    /// Always `0` when `status` is invoked without a preset (whether from
    /// CLI or sidecar) — the `Excluded` variant requires an active filter.
    #[must_use]
    pub fn excluded_count(&self) -> usize {
        self.files
            .iter()
            .filter(|(_, s)| matches!(s, FileStatus::Excluded { .. }))
            .count()
    }
}

/// Inspects the local cache for a repository and compares against the remote file list.
///
/// When `preset_globs` is `Some`, files that do **not** match any of the
/// supplied glob patterns and are absent locally are classified as
/// [`FileStatus::Excluded`] instead of [`FileStatus::Missing`]. Files that
/// **do** match the preset and are absent are still [`FileStatus::Missing`].
/// Files that are present locally classify as `Complete` / `Partial`
/// regardless of whether they match (the user has them — by what route is
/// not status's concern).
///
/// # Arguments
///
/// * `repo_id` — The repository identifier (e.g., `"RWKV/RWKV7-Goose-World3-1.5B-HF"`).
/// * `token` — Optional authentication token.
/// * `revision` — Optional revision (defaults to `"main"`).
/// * `preset_globs` — Optional include-glob list (typically returned by
///   [`crate::config::preset_globs`]). When supplied, governs the
///   `Excluded` distinction. When `None`, no `Excluded` entries are produced.
///
/// # Notes
///
/// Partial-download detection is per-file: a file absent from the snapshot
/// directory is reported [`FileStatus::Partial`] only when its own temp file
/// exists on disk, keyed on the file's LFS `sha256`: this crate's
/// `blobs/<sha256>.chunked.part`, or the `blobs/<sha256>.incomplete` (or
/// `<sha256>.<attempt>.incomplete`) that `hf-hub` and `huggingface_hub`
/// write. `local_size` is that temp file's progress. Non-LFS files carry no
/// `sha256`, so an interrupted download of one reports
/// [`FileStatus::Missing`] until it finalizes.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the API request fails.
/// Returns [`FetchError::Io`] if the cache directory cannot be read.
/// Returns [`FetchError::InvalidPattern`]
/// if any of the supplied `preset_globs` patterns fails to compile.
pub async fn repo_status(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
    preset_globs: Option<&[&str]>,
) -> Result<RepoStatus, FetchError> {
    let revision = revision.unwrap_or("main");
    let cache_dir = hf_cache_dir()?;
    let repo_dir = crate::cache_layout::repo_dir(&cache_dir, repo_id);

    // Read commit hash from refs file if available.
    let commit_hash = read_ref(&repo_dir, revision);

    // Fetch remote file list with sizes.
    let client = crate::chunked::build_client(token)?;
    let remote_files =
        crate::repo::list_repo_files_with_metadata(repo_id, token, Some(revision), &client).await?;

    // Determine snapshot directory.
    // BORROW: explicit .as_deref() for Option<String> → Option<&str>
    let snapshot_dir = commit_hash
        .as_deref()
        .map(|hash| crate::cache_layout::snapshot_dir(&repo_dir, hash));

    // Compile preset globs once, outside the per-file loop. `compile_glob_patterns`
    // already returns `Ok(None)` for empty slices, so an empty `Some(&[])`
    // collapses to the "no filter" case below.
    let preset_globset: Option<globset::GlobSet> = if let Some(patterns) = preset_globs {
        // BORROW: explicit .to_string() — compile_glob_patterns expects &[String]
        let owned: Vec<String> = patterns.iter().map(|s| (*s).to_string()).collect();
        crate::compile_glob_patterns(&owned)?
    } else {
        None
    };

    // The `.incomplete` temp files of `hf-hub` / `huggingface_hub`, read once
    // here rather than once per listed file: a repo can list thousands.
    let incomplete = incomplete_sizes(&crate::cache_layout::blobs_dir(&repo_dir));

    // Cross-reference remote files against local state.
    let mut files: Vec<(String, FileStatus)> = Vec::with_capacity(remote_files.len());

    for remote in &remote_files {
        let expected_size = remote.size.unwrap_or(0);

        let local_path = snapshot_dir
            .as_ref()
            // A listed name that could leave the snapshot directory is never
            // joined to it; the file then reads as absent.
            // BORROW: explicit .as_str() instead of Deref coercion
            .filter(|_| crate::cache_layout::is_safe_repo_filename(remote.filename.as_str()))
            // BORROW: explicit .as_str() for path construction
            .map(|dir| dir.join(remote.filename.as_str()));

        // Size of the snapshot copy when it exists; `None` when the file is
        // absent (or no snapshot directory is known for this revision).
        let snapshot_size = local_path.as_ref().and_then(|path| {
            path.exists()
                .then(|| std::fs::metadata(path).map_or(0, |m| m.len()))
        });

        let status = if let Some(local_size) = snapshot_size {
            if expected_size > 0 && local_size < expected_size {
                FileStatus::Partial {
                    local_size,
                    expected_size,
                }
            } else {
                FileStatus::Complete { local_size }
            }
        } else if let Some(part_size) =
            // BORROW: explicit .as_deref() for Option<String> → Option<&str>
            partial_chunk_size(&repo_dir, remote.sha256.as_deref())
        {
            // This file's own chunked-download temp blob is on disk: attribute
            // exactly its byte count to this file (keyed on the LFS sha256).
            FileStatus::Partial {
                local_size: part_size,
                expected_size,
            }
        } else if let Some(&part_size) =
            remote.sha256.as_deref().and_then(|sha| incomplete.get(sha))
        {
            // An `hf-hub` / `huggingface_hub` temp file for this etag: it
            // grows as it downloads, so its length is the progress.
            FileStatus::Partial {
                local_size: part_size,
                expected_size,
            }
        } else if preset_globset
            .as_ref()
            // BORROW: explicit .as_str() — globset matches against &str
            .is_some_and(|gs| !gs.is_match(remote.filename.as_str()))
        {
            // File is absent AND the active preset deliberately excludes it.
            FileStatus::Excluded { expected_size }
        } else {
            FileStatus::Missing { expected_size }
        };

        // BORROW: explicit .clone() for owned String
        files.push((remote.filename.clone(), status));
    }

    files.sort_by(|(a, _), (b, _)| a.cmp(b));

    // BORROW: explicit .to_owned() for &str → owned String field
    Ok(RepoStatus {
        repo_id: repo_id.to_owned(),
        commit_hash,
        cache_path: repo_dir,
        files,
    })
}

/// Summary of a single cached model (local-only, no API calls).
#[derive(Debug, Clone)]
pub struct CachedModelSummary {
    /// The repository identifier (e.g., `"RWKV/RWKV7-Goose-World3-1.5B-HF"`).
    pub repo_id: String,
    /// Number of files listed across the repo's snapshots.
    pub file_count: usize,
    /// Physical bytes this repo occupies on disk: every file in its
    /// directory (`blobs/`, `snapshots/`, `refs/`, hf-fm's sidecars),
    /// counting each physical file once, plus the payloads it links to in
    /// `huggingface_hub`'s shared blob store, which live outside it.
    ///
    /// It is not the sum of the snapshot's logical file sizes. Where a
    /// snapshot entry is a full copy of its blob (always for files `hf-hub`
    /// writes on Windows, which never tries a symlink there, and for this
    /// crate's own writes whenever creating a symlink is refused), the repo
    /// holds those bytes twice, so this figure can reach double the
    /// snapshot's own size. Where the pointers are symlinks, or hard links on
    /// Unix, only the blob's bytes count (a hard link on Windows is counted
    /// twice; see `physical_id`). See `walk_repo_files` for why both
    /// directories are walked.
    ///
    /// It includes the payloads the repo links to in `huggingface_hub`'s
    /// shared blob store; see [`shared_size`](Self::shared_size). What
    /// `cache delete` frees is `total_size - shared_size`.
    pub total_size: u64,
    /// The part of [`total_size`](Self::total_size) that lives in
    /// `huggingface_hub`'s cache-wide shared blob store (`<cache>/blobs/`,
    /// see [`crate::cache_layout::shared_blobs_dir`]) rather than in the repo
    /// directory: the payloads its `blobs/<etag>` symlinks point to (new in
    /// v0.12.2). Other repos may link the same payloads, so a cache-wide total
    /// counts the store once, through [`shared_store_size`], and deleting
    /// this repo does not free them. `0` for a repo that links nothing there.
    pub shared_size: u64,
    /// Whether the repo holds a partial download's temp file: this crate's
    /// `.chunked.part`, or (since v0.12.2) an `.incomplete` left by `hf-hub`
    /// or `huggingface_hub`.
    pub has_partial: bool,
    /// Newest modification time among those temp files; `None` when
    /// `has_partial` is `false` (new in v0.12.2).
    ///
    /// `cache gc` reads it to spare a download in progress. The repo's
    /// [`last_modified`](Self::last_modified) cannot tell: it is the age of
    /// the snapshot files, so a new download into a repo whose existing
    /// files are old looked stale, and could be evicted mid-transfer.
    pub partial_modified: Option<std::time::SystemTime>,
    /// Most recent modification time among the repo's snapshot files, across
    /// every snapshot; a symlinked entry reports its blob's. A repo with no
    /// snapshot files (an interrupted first download, or only hf-fm's header
    /// cache) takes the newest among its other files instead, so it still
    /// has an age that `cache gc` can act on.
    ///
    /// `None` if no files were found or all metadata reads failed.
    pub last_modified: Option<std::time::SystemTime>,
    /// `(min, max)` size across this repo's `.gguf` files, when they are
    /// mutually-exclusive quant alternatives rather than shards of one
    /// logical file (see [`crate::discover::gguf_size_range`]). `None` when
    /// not applicable — `total_size` is already the correct figure then.
    pub gguf_size_range: Option<(u64, u64)>,
}

/// Scans the entire HF cache and returns a summary for each cached model.
///
/// This is a local-only operation (no API calls). It lists all `models--*`
/// directories and, for each, counts its snapshot files and the bytes its
/// whole directory occupies.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be read.
pub fn cache_summary() -> Result<Vec<CachedModelSummary>, FetchError> {
    let cache_dir = hf_cache_dir()?;

    if !cache_dir.exists() {
        return Ok(Vec::new());
    }

    let entries = std::fs::read_dir(&cache_dir).map_err(|e| FetchError::Io {
        path: cache_dir.clone(),
        source: e,
    })?;

    let mut summaries: Vec<CachedModelSummary> = Vec::new();

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let dir_name = entry.file_name();
        // BORROW: explicit .to_string_lossy() for OsString → str conversion
        let dir_str = dir_name.to_string_lossy();

        let Some(repo_id) = repo_id_from_folder_name(&dir_str) else {
            continue;
        };

        let repo_dir = entry.path();

        // Single walk over the repo directory for file_count/total_size/
        // last_modified and the raw filenames needed to classify `.gguf`
        // quant alternatives. `total_size` is the physical bytes of the whole
        // repo directory; `file_count` and the filenames stay the
        // snapshot's own, which is what a user means by "N files" and what
        // `gguf_size_range` needs to see extensions on.
        let RepoFileWalk {
            files: cached_files,
            total_size,
            shared_size,
            last_modified,
        } = walk_repo_files(&repo_dir);
        let file_count = cached_files.len();
        // BORROW: explicit .as_str() instead of Deref coercion
        let sized: Vec<(&str, Option<u64>)> = cached_files
            .iter()
            .map(|f| (f.filename.as_str(), Some(f.size)))
            .collect();
        let gguf_size_range = crate::discover::gguf_size_range(sized);

        // Check for partial downloads, and how recently one was written.
        let (has_partial, partial_modified) =
            scan_partials(&crate::cache_layout::blobs_dir(&repo_dir));

        // Deliberately NOT filtering out zero-byte, non-partial repos here
        // (e.g. an empty repo directory): `cache gc` and `status` both
        // consume this same list, and a repo invisible here would be
        // unreachable by bulk eviction. `du`'s own summary view filters
        // purely cosmetic zero-byte entries out at its own call site
        // instead, leaving this shared data layer complete. (A repo holding
        // only hf-fm's header cache is not zero-byte: its sidecar counts.)
        summaries.push(CachedModelSummary {
            repo_id,
            file_count,
            total_size,
            shared_size,
            has_partial,
            partial_modified,
            last_modified,
            gguf_size_range,
        });
    }

    summaries.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));

    Ok(summaries)
}

/// Returns the snapshot file count and the physical bytes on disk for a
/// single cached repo.
///
/// Avoids scanning the entire cache when only one repo's metrics are needed.
///
/// The size is [`CachedModelSummary::total_size`]'s figure: every physical
/// file in the repo's directory, counted once, plus the shared-store
/// payloads its blobs link to. Before the fix for
/// [issue #16](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues/16)
/// this counted snapshot entries only and so under-reported every repo whose
/// pointers are copies, and nearly all of every repo whose pointers are
/// symlinks. [`repo_usage`] also reports the shared part, which is what a
/// caller needs to know what deleting the repo frees.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be determined.
pub fn repo_disk_usage(repo_id: &str) -> Result<(usize, u64), FetchError> {
    let usage = repo_usage(repo_id)?;
    Ok((usage.file_count, usage.total_size))
}

/// One cached repo's footprint, from [`repo_usage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct RepoUsage {
    /// Number of files listed across the repo's snapshots.
    pub file_count: usize,
    /// Bytes on disk, as [`CachedModelSummary::total_size`] counts them.
    pub total_size: u64,
    /// The part of `total_size` in the shared blob store, as
    /// [`CachedModelSummary::shared_size`] counts it. `cache delete` frees
    /// `total_size - shared_size`.
    pub shared_size: u64,
}

/// Returns one cached repo's file count, bytes on disk, and the part of
/// those bytes in `huggingface_hub`'s shared blob store (new in v0.12.2).
///
/// The `cache delete` preview uses it: deleting a repo frees
/// `total_size - shared_size`, its links into the store being removed but
/// not the payloads they point to.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be determined.
pub fn repo_usage(repo_id: &str) -> Result<RepoUsage, FetchError> {
    let cache_dir = hf_cache_dir()?;
    let repo_dir = crate::cache_layout::repo_dir(&cache_dir, repo_id);
    let walk = walk_repo_files(&repo_dir);
    Ok(RepoUsage {
        file_count: walk.files.len(),
        total_size: walk.total_size,
        shared_size: walk.shared_size,
    })
}

/// Bytes `huggingface_hub`'s cache-wide shared blob store occupies:
/// every regular file under `<cache>/blobs/` (the payloads, their `.refs`
/// manifests and lock files), each physical file once, symlinks not followed
/// (new in v0.12.2). `0` when the cache has no store.
///
/// A cache-wide total counts each repo's own bytes
/// (`total_size - shared_size`) plus this, so a payload several repos link
/// to is counted once, and one no repo links to any more (which
/// `hf cache prune` reclaims) is counted too.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be determined.
pub fn shared_store_size() -> Result<u64, FetchError> {
    let cache_dir = hf_cache_dir()?;
    Ok(dir_bytes(&crate::cache_layout::shared_blobs_dir(
        &cache_dir,
    )))
}

/// Bytes every regular file under `dir` occupies, each physical file once,
/// symlinks not followed; `0` when `dir` does not exist.
fn dir_bytes(dir: &Path) -> u64 {
    let mut total: u64 = 0;
    let mut seen: HashSet<(u64, u64)> = HashSet::new();
    let mut newest: Option<std::time::SystemTime> = None;
    walk_other_files(dir, &mut total, &mut seen, &mut newest);
    total
}

/// Checks whether a single cached repo holds a partial download's temp
/// file: a `.chunked.part`, or an `.incomplete` left by `hf-hub` or
/// `huggingface_hub`.
///
/// Avoids scanning the entire cache when only one repo's partial status
/// is needed (e.g., for the `du <REPO>` partial-download hint).
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be determined.
pub fn repo_has_partial(repo_id: &str) -> Result<bool, FetchError> {
    let cache_dir = hf_cache_dir()?;
    let repo_dir = crate::cache_layout::repo_dir(&cache_dir, repo_id);
    let blobs_dir = crate::cache_layout::blobs_dir(&repo_dir);
    Ok(scan_partials(&blobs_dir).0)
}

/// Reads the commit hash from a refs file, if it exists.
///
/// Looks for `<repo_dir>/refs/<revision>` and returns the trimmed contents
/// (a commit hash) or `None` if the file does not exist, is empty, or holds
/// anything other than one plain name
/// ([`crate::cache_layout::is_safe_hub_name`]): every caller goes on to build
/// `snapshots/<commit_hash>/` from it.
#[must_use]
pub fn read_ref(repo_dir: &Path, revision: &str) -> Option<String> {
    let ref_path = crate::cache_layout::ref_path(repo_dir, revision);
    std::fs::read_to_string(ref_path)
        .ok()
        // BORROW: explicit .to_owned() to convert trimmed &str → owned String
        .map(|s| s.trim().to_owned())
        // BORROW: explicit .as_str() instead of Deref coercion
        .filter(|s| crate::cache_layout::is_safe_hub_name(s.as_str()))
}

/// Returns the truly-downloaded byte count of a file's own chunked-download
/// temp blob.
///
/// Looks up `{repo_dir}/blobs/<sha256>.chunked.part` — the chunked download
/// path names its temp blob after the file's etag, which for LFS files is
/// the content's `SHA256` (see [`crate::cache_layout::temp_blob_path`]) —
/// so a specific file's partial is addressable without scanning the blobs
/// directory. Returns `None` when the file has no known `sha256` (non-LFS
/// file) or no temp blob exists on disk.
///
/// The chunked download preallocates the temp blob at its full size, so the
/// blob's file length reads as `total_size` from the first byte onward.
/// True progress lives in the `.chunked.part.state` resume sidecar's
/// per-chunk `completed` offsets; this prefers their sum and falls back to
/// the blob's file length when the sidecar is absent or unparseable
/// (pre-v0.9.8 leftovers).
fn partial_chunk_size(repo_dir: &Path, sha256: Option<&str>) -> Option<u64> {
    // The sha256 comes from the Hub's listing and names a path below.
    let sha = sha256.filter(|s| crate::cache_layout::is_safe_hub_name(s))?;
    let part_path = crate::cache_layout::temp_blob_path(repo_dir, sha);
    let part_len = std::fs::metadata(part_path)
        .ok()
        .filter(std::fs::Metadata::is_file)
        .map(|m| m.len())?;

    let state_path = crate::cache_layout::temp_state_path(repo_dir, sha);
    let sidecar_sum = std::fs::read_to_string(state_path)
        .ok()
        .and_then(|text| {
            // BORROW: explicit .as_str() instead of Deref coercion
            serde_json::from_str::<crate::chunked_state::ChunkedState>(text.as_str()).ok()
        })
        .map(|state| state.chunks.iter().map(|c| c.completed).sum::<u64>());

    Some(sidecar_sum.unwrap_or(part_len))
}

/// Suffix of this crate's chunked-download temp blob:
/// `blobs/<etag>.chunked.part` (see [`crate::cache_layout::temp_blob_path`]).
const CHUNKED_PART_SUFFIX: &str = ".chunked.part";

/// Suffix of the temp file `hf-hub` and `huggingface_hub` download into:
/// `blobs/<etag>.incomplete`, or `blobs/<etag>.<8 hex digits>.incomplete`
/// since `huggingface_hub` 1.18, which gives every attempt its own name.
const INCOMPLETE_SUFFIX: &str = ".incomplete";

/// The etag an `.incomplete` temp file belongs to, or `None` when `name` is
/// not one.
///
/// Recognises both namings: `<etag>.incomplete` (`hf-hub`, and
/// `huggingface_hub` before 1.18) and `<etag>.<8 hex>.incomplete`
/// (`huggingface_hub` 1.18 and later). The Hub's etags are hex digests with
/// no period, so the 8-digit attempt suffix cannot be mistaken for part of
/// one.
fn incomplete_etag(name: &str) -> Option<&str> {
    let stem = name.strip_suffix(INCOMPLETE_SUFFIX)?;
    if let Some((etag, attempt)) = stem.rsplit_once('.')
        && attempt.len() == 8
        && attempt.bytes().all(|b| b.is_ascii_hexdigit())
        && !etag.is_empty()
    {
        return Some(etag);
    }
    (!stem.is_empty()).then_some(stem)
}

/// Whether `name` is a partial download's temp file in `blobs/`: this
/// crate's `.chunked.part`, or the `.incomplete` file `hf-hub` and
/// `huggingface_hub` leave when a download is interrupted. The chunked
/// download's `.chunked.part.state` sidecar is not one.
fn is_partial_name(name: &str) -> bool {
    name.ends_with(CHUNKED_PART_SUFFIX) || incomplete_etag(name).is_some()
}

/// Whether `blobs_dir` holds any partial download, and the newest
/// modification time among them (`None` when there are none).
///
/// `DirEntry::metadata` is right here, unlike in a snapshot walk (see
/// [`resolved_file_size`]): a temp file is always a real file its
/// downloader wrote, never a pointer.
fn scan_partials(blobs_dir: &Path) -> (bool, Option<std::time::SystemTime>) {
    let Ok(entries) = std::fs::read_dir(blobs_dir) else {
        return (false, None);
    };
    let mut any = false;
    let mut newest: Option<std::time::SystemTime> = None;
    for entry in entries {
        let Ok(entry) = entry else { continue };
        // BORROW: explicit .to_string_lossy() for OsString → str conversion
        if !is_partial_name(&entry.file_name().to_string_lossy()) {
            continue;
        }
        any = true;
        if let Ok(meta) = entry.metadata() {
            note_newest_mtime(&mut newest, &meta);
        }
    }
    (any, newest)
}

/// Bytes downloaded so far into each `.incomplete` temp file in
/// `blobs_dir`, keyed by the etag it belongs to.
///
/// These files grow as they download (unlike a `.chunked.part`, which is
/// preallocated), so a file's length is its progress. Where several attempts
/// left one each, the largest is kept.
fn incomplete_sizes(blobs_dir: &Path) -> HashMap<String, u64> {
    let mut sizes: HashMap<String, u64> = HashMap::new();
    let Ok(entries) = std::fs::read_dir(blobs_dir) else {
        return sizes;
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        // BORROW: explicit .to_string_lossy() for OsString → str conversion
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(etag) = incomplete_etag(&name) else {
            continue;
        };
        let len = entry.metadata().map_or(0, |m| m.len());
        // BORROW: explicit .to_owned() for &str → owned String key
        let slot = sizes.entry(etag.to_owned()).or_insert(0);
        *slot = (*slot).max(len);
    }
    sizes
}

/// Whether another process holds the download lock at `lock_path`.
///
/// `hf-hub` and `huggingface_hub` both hold an exclusive OS lock on
/// `<cache>/.locks/<repo folder>/<etag>.lock` while they write that etag's
/// `.incomplete` file. This opens an existing lock file, without creating
/// one, and tries to take the lock without waiting. A missing lock file means
/// no download is running; a lock that is busy, or a file that cannot be
/// opened or tested, counts as held, so the partial it guards is kept.
/// Locks a filesystem cannot provide (on some network filesystems
/// `huggingface_hub` falls back to a lock file's mere existence) are not
/// detected.
fn download_lock_held(lock_path: &Path) -> bool {
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(lock_path)
    {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return false,
        Err(_) => return true,
    };
    match file.try_lock() {
        Ok(()) => {
            // Dropping the file releases the lock too; unlocking first just
            // makes the release explicit.
            let _ = file.unlock();
            false
        }
        Err(_) => true,
    }
}

/// A partial download's temp file in a repo's `blobs/`: this crate's
/// `.chunked.part`, or the `.incomplete` file `hf-hub` and `huggingface_hub`
/// download into.
#[derive(Debug, Clone)]
pub struct PartialFile {
    /// The repository identifier (e.g., `"meta-llama/Llama-3.2-1B"`).
    pub repo_id: String,
    /// The temp file's name (e.g., `"abc123def456.chunked.part"` or
    /// `"abc123def456.incomplete"`).
    pub filename: String,
    /// Absolute path to the temp file.
    pub path: PathBuf,
    /// Size of the partial file in bytes.
    pub size: u64,
}

impl PartialFile {
    /// Returns sibling sidecar paths that should be removed alongside this
    /// partial: for a `.chunked.part`, the resume-state sidecar
    /// `{etag}.chunked.part.state` and any orphan write-tmp
    /// `{etag}.chunked.part.state.tmp` left by an interrupted atomic save.
    /// An `.incomplete` file has none.
    ///
    /// The paths are returned even when the underlying files do not exist
    /// — callers (`run_cache_clean_partial`) attempt removal best-effort.
    #[must_use]
    pub fn sidecar_paths(&self) -> Vec<PathBuf> {
        // BORROW: explicit .as_str() instead of Deref coercion
        if !self.filename.as_str().ends_with(CHUNKED_PART_SUFFIX) {
            return Vec::new();
        }
        let Some(parent) = self.path.parent() else {
            return Vec::new();
        };
        // String concat (mirrors `cache_layout::temp_state_path`'s
        // rationale): the etag may itself contain periods, so
        // `Path::with_extension` would truncate at the wrong boundary.
        // BORROW: explicit .clone() for owned String → mutated copy
        let mut state_name = self.filename.clone();
        state_name.push_str(".state");
        // BORROW: explicit .clone() for owned String → mutated copy
        let mut tmp_name = self.filename.clone();
        tmp_name.push_str(".state.tmp");
        vec![parent.join(state_name), parent.join(tmp_name)]
    }

    /// Whether a running download is still writing this file, so removing it
    /// would break that download.
    ///
    /// For an `.incomplete` file this asks the download lock that `hf-hub`
    /// and `huggingface_hub` hold while writing it,
    /// `<cache>/.locks/<repo folder>/<etag>.lock`. Where a filesystem cannot
    /// provide that lock (on some network filesystems `huggingface_hub`
    /// falls back to the lock file's mere existence), a running download is
    /// not detected. This crate's own chunked downloads take no such lock, so a
    /// `.chunked.part` always reads as not in use: make sure no `hf-fm`
    /// download is running before cleaning one.
    #[must_use]
    pub fn is_in_use(&self) -> bool {
        // BORROW: explicit .as_str() instead of Deref coercion
        let Some(etag) = incomplete_etag(self.filename.as_str()) else {
            return false;
        };
        // `<cache>/<repo folder>/blobs/<file>`: walk back up to the cache root.
        let Some(repo_dir) = self.path.parent().and_then(Path::parent) else {
            return false;
        };
        let (Some(cache_root), Some(repo_folder)) = (repo_dir.parent(), repo_dir.file_name())
        else {
            return false;
        };
        // BORROW: explicit .to_string_lossy() for OsStr → str conversion
        let lock = crate::cache_layout::download_lock_path(
            cache_root,
            &repo_folder.to_string_lossy(),
            etag,
        );
        download_lock_held(&lock)
    }
}

/// Finds every partial download's temp file in the `HuggingFace` cache:
/// this crate's `.chunked.part` files, and since v0.12.2 the `.incomplete`
/// files `hf-hub` and `huggingface_hub` leave behind.
///
/// Walks `models--*/blobs/` directories and collects partial files.
/// When `repo_filter` is `Some`, only the matching repo is scanned. Files a
/// running download is still writing are included; ask
/// [`PartialFile::is_in_use`] before removing one.
///
/// Returns an empty `Vec` if the cache directory does not exist.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be read.
pub fn find_partial_files(repo_filter: Option<&str>) -> Result<Vec<PartialFile>, FetchError> {
    let cache_dir = hf_cache_dir()?;

    if !cache_dir.exists() {
        return Ok(Vec::new());
    }

    let entries = std::fs::read_dir(&cache_dir).map_err(|e| FetchError::Io {
        // BORROW: explicit .clone() for owned PathBuf
        path: cache_dir.clone(),
        source: e,
    })?;

    let mut partials: Vec<PartialFile> = Vec::new();

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let dir_name = entry.file_name();
        // BORROW: explicit .to_string_lossy() for OsString → str conversion
        let dir_str = dir_name.to_string_lossy();

        let Some(repo_id) = repo_id_from_folder_name(&dir_str) else {
            continue;
        };

        // Skip repos that don't match the filter.
        // BORROW: explicit .as_str() instead of Deref coercion
        if let Some(filter) = repo_filter
            && repo_id.as_str() != filter
        {
            continue;
        }

        let blobs_dir = crate::cache_layout::blobs_dir(&entry.path());
        let Ok(blob_entries) = std::fs::read_dir(&blobs_dir) else {
            continue;
        };

        for blob_entry in blob_entries {
            let Ok(blob_entry) = blob_entry else { continue };
            let name = blob_entry.file_name();
            // BORROW: explicit .to_string_lossy() for OsString → str conversion
            let name_str = name.to_string_lossy();
            if is_partial_name(&name_str) {
                // Real file, never a pointer: same reasoning as
                // `scan_partials` above.
                let size = blob_entry.metadata().map_or(0, |m| m.len());
                partials.push(PartialFile {
                    // BORROW: explicit .clone() for owned String
                    repo_id: repo_id.clone(),
                    // BORROW: explicit .to_string() for Cow<str> → owned String
                    filename: name_str.to_string(),
                    path: blob_entry.path(),
                    size,
                });
            }
        }
    }

    Ok(partials)
}

/// Per-file disk usage entry within a cached repository.
#[derive(Debug, Clone)]
pub struct CacheFileUsage {
    /// Filename relative to the snapshot directory.
    pub filename: String,
    /// File size in bytes, resolved through a pointer symlink to the bytes
    /// the file actually holds rather than the length of the link itself.
    pub size: u64,
}

/// Returns per-file disk usage for a specific cached repository.
///
/// Walks the snapshot directories under
/// `<cache_dir>/models--<org>--<name>/snapshots/` and collects each file's
/// relative path and size, each size resolved through a pointer symlink to
/// the bytes the file actually holds. Results are sorted by size
/// descending.
///
/// These are the repo's *logical* files. They do not sum to
/// [`CachedModelSummary::total_size`] wherever snapshot entries are copies
/// of their blobs rather than links to them.
///
/// Returns an empty `Vec` if the repository is not cached.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be determined.
pub fn cache_repo_usage(repo_id: &str) -> Result<Vec<CacheFileUsage>, FetchError> {
    let cache_dir = hf_cache_dir()?;
    let repo_dir = crate::cache_layout::repo_dir(&cache_dir, repo_id);

    if !repo_dir.exists() {
        return Ok(Vec::new());
    }

    let mut files = walk_repo_files(&repo_dir).files;
    files.sort_by_key(|f| std::cmp::Reverse(f.size));
    Ok(files)
}

/// Size of the file at `path`, resolved through a pointer symlink.
///
/// `None` for a directory, a dangling pointer, or an unreadable path.
///
/// Prefer this to `DirEntry::metadata().len()` anywhere a `HuggingFace`
/// snapshot tree is walked. `DirEntry::metadata` is documented as equivalent
/// to `symlink_metadata`, so on the usual Unix layout, where a snapshot entry
/// is a symlink into `blobs/`, it yields the length of the link's *target
/// path* (tens of bytes) instead of the file's size. Reading sizes that way
/// is what made disk-usage reporting a rounding error on a symlinked cache:
/// see [issue #16](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues/16)
/// and [`CachedModelSummary::total_size`].
///
/// This answers "how big is this file", which is what a listing wants. It
/// deliberately says nothing about how many bytes the file *occupies*: a
/// symlink and its blob are one set of bytes under two names, so the repo
/// walk behind [`cache_summary`] keeps its own richer logic for that, needing
/// each entry's link status, mtime and physical identity as well as its size.
#[must_use]
pub fn resolved_file_size(path: &Path) -> Option<u64> {
    let meta = std::fs::metadata(path).ok()?;
    if meta.is_file() {
        Some(meta.len())
    } else {
        None
    }
}

/// Physical-file identity, used to count one file once however many
/// directory entries name it.
///
/// `Some` on Unix, where `(st_dev, st_ino)` is stable and free. `None`
/// everywhere else: Windows exposes `nFileIndex` only behind the unstable
/// `windows_by_handle` feature, and reaching it directly takes an `unsafe`
/// Win32 call that this crate's `unsafe_code = "forbid"` rules out
/// absolutely (`forbid`, unlike `deny`, cannot be lifted by an inner
/// `allow`). Nothing in the stack creates a hard link, so no layout that
/// actually occurs needs this: [`crate::chunked`]'s `symlink_or_copy` and
/// `hf-hub`'s own pointer writer each only ever symlink or copy. A hard
/// link made by hand on Windows is therefore counted twice, which is the
/// one inaccuracy left here; revisit alongside any change that starts
/// creating hard links.
#[cfg(unix)]
// This arm always returns `Some`, so `unnecessary_wraps` fires on it. The
// `Option` is still required: the signature is shared with the
// `#[cfg(not(unix))]` arm below, which can only ever return `None`, and
// unwrapping it here would push the platform split out into every call site.
// The lint cannot fire on Windows, where that arm compiles instead, which is
// how this reached CI unseen.
#[allow(clippy::unnecessary_wraps)]
fn physical_id(meta: &std::fs::Metadata) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    Some((meta.dev(), meta.ino()))
}

/// Non-Unix counterpart of [`physical_id`]; see its docs for why this is
/// always `None`.
#[cfg(not(unix))]
fn physical_id(_meta: &std::fs::Metadata) -> Option<(u64, u64)> {
    None
}

/// One cached repo's on-disk file walk: the snapshot's logical file list,
/// the repo's physical byte total, and the most recent modification time
/// seen — everything both [`cache_repo_usage`] and [`cache_summary`] need,
/// from a single pass over the repo directory.
struct RepoFileWalk {
    /// Every snapshot file's relative path + size (see [`CacheFileUsage`]),
    /// each size resolved through a pointer symlink to the bytes the file
    /// actually holds.
    files: Vec<CacheFileUsage>,
    /// Distinct physical bytes the repo directory holds: `blobs/`,
    /// `snapshots/`, and everything else in it (`refs/`, hf-fm's sidecars).
    /// Deliberately not the sum of `files`: where a snapshot entry is a
    /// *copy* of its blob (the Windows layout) both occupy their own bytes
    /// and both are counted, and where it is a symlink or a hard link only
    /// the blob is. Includes the shared-store payloads the repo's blobs link
    /// to; see `shared_size`.
    total_size: u64,
    /// The part of `total_size` that lives in `huggingface_hub`'s cache-wide
    /// shared blob store rather than in the repo directory: the payloads the
    /// repo's `blobs/<etag>` symlinks point to (see [`walk_blob_files`]).
    /// Deleting the repo does not free these bytes.
    shared_size: u64,
    /// The newest modification time across every snapshot file, or, for a
    /// repo with none, across its other files (blobs, sidecars, `refs/`).
    /// `None` only when the directory holds no readable file at all.
    last_modified: Option<std::time::SystemTime>,
}

/// Walks one cached repo, returning its snapshot file list, its physical
/// byte total and its newest mtime in a single pass.
///
/// # Why `blobs/` and `snapshots/` both, and the rest
///
/// The `HuggingFace` layout stores each file's bytes once, under
/// `blobs/<etag>`, and puts a *pointer* at `snapshots/<commit>/<filename>`.
/// That pointer is a symlink on Unix. On Windows it is usually a full copy,
/// and the repo then holds the same bytes twice: `hf-hub` copies there
/// unconditionally, never trying a symlink (its own comment says symlinks
/// "require elevated privileges"), and [`crate::chunked`]'s
/// `symlink_or_copy` tries one and falls back to `std::fs::copy` when it is
/// refused, as it is without `SeCreateSymbolicLinkPrivilege`.
///
/// Walking `snapshots/` alone, as this did until the fix for
/// [issue #16](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues/16),
/// gets *both* layouts wrong, in opposite directions and by different
/// amounts:
///
/// - Where the pointer is a **copy**, the blob is never counted, so the
///   repo reports about half the bytes it occupies.
/// - Where the pointer is a **symlink**, it was worse. The size came from
///   `DirEntry::metadata`, which `std` documents as equivalent to
///   `symlink_metadata`, so it reported the length of the link's *target
///   path* (tens of bytes) rather than the file's size, and the blob
///   holding the real bytes went uncounted as well.
///
/// So this walks `blobs/` first, recording each blob's identity, then
/// `snapshots/`, where a symlink contributes no bytes of its own and a hard
/// link to an already-counted blob contributes none either, then every
/// other file in the repo directory: `refs/`, and hf-fm's own sidecars
/// (`.hf-fm-header-cache/`, `.hf-fm-snapshot.json`). Without a shared
/// blob store (below), the result is exactly what `cache delete`'s
/// `remove_dir_all` frees.
///
/// # The shared blob store
///
/// Since 1.32.0, `huggingface_hub` keeps each Xet file once for the whole
/// cache, at `<cache>/blobs/<2 hex>/<hash>`, and makes the repo's own
/// `blobs/<etag>` a relative symlink to it (wherever symlinks are
/// available). Skipping every symlink in `blobs/` would count such a repo at
/// almost nothing. A blob symlink that resolves *into that store* is
/// therefore charged at its payload's size, in `total_size` and in
/// `shared_size`; any other symlink is still skipped, so a crafted cache
/// cannot charge a repo for bytes outside the cache. Deleting the repo
/// removes the links, not the payloads, which other repos may share:
/// `total_size - shared_size` is what deleting it frees.
fn walk_repo_files(repo_dir: &Path) -> RepoFileWalk {
    let mut files: Vec<CacheFileUsage> = Vec::new();
    let mut total_size: u64 = 0;
    let mut shared_size: u64 = 0;
    let mut last_modified: Option<std::time::SystemTime> = None;
    // Identities of the physical files already charged to `total_size`.
    let mut seen: HashSet<(u64, u64)> = HashSet::new();
    // Newest mtime outside `snapshots/`: the age of a repo that has no
    // snapshot files at all.
    let mut other_modified: Option<std::time::SystemTime> = None;
    // `huggingface_hub`'s shared blob store, a sibling of the repo directory,
    // resolved once so each blob symlink's target can be compared with it.
    let shared_store = repo_dir
        .parent()
        .map(crate::cache_layout::shared_blobs_dir)
        .and_then(|store| std::fs::canonicalize(store).ok());

    // Blobs first, so a snapshot entry that hard-links to one is recognised
    // as already counted rather than charged a second time.
    walk_blob_files(
        &crate::cache_layout::blobs_dir(repo_dir),
        shared_store.as_deref(),
        &mut BlobTotals {
            total_size: &mut total_size,
            shared_size: &mut shared_size,
            seen: &mut seen,
            newest: &mut other_modified,
        },
    );

    // A repo can legitimately hold blobs with no readable `snapshots/` (an
    // interrupted download), so an unreadable snapshots directory leaves the
    // blob bytes counted above rather than discarding them.
    if let Ok(snapshots) = std::fs::read_dir(crate::cache_layout::snapshots_dir(repo_dir)) {
        for snap_entry in snapshots {
            let Ok(snap_entry) = snap_entry else { continue };
            let snap_path = snap_entry.path();
            if !snap_path.is_dir() {
                continue;
            }
            walk_snapshot_files(
                &snap_path,
                "",
                &mut files,
                &mut total_size,
                &mut last_modified,
                &mut seen,
            );
        }
    }

    // Everything else the directory holds, so the total is what deleting
    // it frees.
    if let Ok(entries) = std::fs::read_dir(repo_dir) {
        for entry in entries {
            let Ok(entry) = entry else { continue };
            let name = entry.file_name();
            if name == "blobs" || name == "snapshots" {
                continue;
            }
            walk_other_files(
                &entry.path(),
                &mut total_size,
                &mut seen,
                &mut other_modified,
            );
        }
    }

    RepoFileWalk {
        files,
        total_size,
        shared_size,
        // A downloaded repo is aged by its snapshot files, as `cache gc`
        // documents. One with none, such as an interrupted first download or
        // a repo holding only hf-fm's header cache, would otherwise have no
        // age at all: `cache gc --older-than` could never select it, and the
        // guard that spares an active download could not recognise it.
        last_modified: last_modified.or(other_modified),
    }
}

/// The running totals [`walk_blob_files`] adds to.
struct BlobTotals<'a> {
    /// The repo's bytes on disk, shared-store payloads included.
    total_size: &'a mut u64,
    /// The part of `total_size` that lives in the shared blob store.
    shared_size: &'a mut u64,
    /// Identities of the physical files already charged.
    seen: &'a mut HashSet<(u64, u64)>,
    /// Newest mtime seen outside `snapshots/`.
    newest: &'a mut Option<std::time::SystemTime>,
}

/// Charges every regular file directly under `blobs/` to the repo's total,
/// recording each one's [`physical_id`], and every blob symlink that
/// resolves into the shared blob store (`shared_store`, canonicalised) at
/// its payload's size, to the shared total as well.
///
/// `blobs/` is flat in the `HuggingFace` layout, so this does not recurse.
/// Any other symlink is skipped rather than followed: following one would
/// let a crafted cache charge this repo for bytes that live outside it. A
/// payload two of the repo's blobs link to is charged once.
fn walk_blob_files(blobs_dir: &Path, shared_store: Option<&Path>, totals: &mut BlobTotals<'_>) {
    let Ok(entries) = std::fs::read_dir(blobs_dir) else {
        return;
    };
    // Shared payloads already charged to this repo, by resolved path:
    // `physical_id` is unavailable on Windows, where the store can exist too.
    let mut shared_seen: HashSet<PathBuf> = HashSet::new();
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let Ok(meta) = std::fs::symlink_metadata(entry.path()) else {
            continue;
        };
        if meta.is_file() {
            note_newest_mtime(totals.newest, &meta);
            if let Some(id) = physical_id(&meta)
                && !totals.seen.insert(id)
            {
                continue;
            }
            *totals.total_size = totals.total_size.saturating_add(meta.len());
            continue;
        }
        // Not a regular file: only a symlink into the shared store counts. A
        // dangling link fails to resolve and is charged nothing.
        if !meta.file_type().is_symlink() {
            continue;
        }
        let Some(store) = shared_store else { continue };
        let Ok(target) = std::fs::canonicalize(entry.path()) else {
            continue;
        };
        if !target.starts_with(store) {
            continue;
        }
        let Ok(target_meta) = std::fs::metadata(&target) else {
            continue;
        };
        if !target_meta.is_file() {
            continue;
        }
        note_newest_mtime(totals.newest, &target_meta);
        if let Some(id) = physical_id(&target_meta) {
            totals.seen.insert(id);
        }
        if !shared_seen.insert(target) {
            continue;
        }
        *totals.total_size = totals.total_size.saturating_add(target_meta.len());
        *totals.shared_size = totals.shared_size.saturating_add(target_meta.len());
    }
}

/// Charges a repo entry outside `blobs/` and `snapshots/` to `total_size`:
/// a regular file directly, a directory by walking it recursively.
///
/// These are `refs/` and hf-fm's own sidecars, which list no model file but
/// occupy bytes that `cache delete` frees. As in [`walk_blob_files`],
/// symlinks are skipped rather than followed, so nothing outside the repo is
/// charged to it, and [`physical_id`] keeps a file named twice from counting
/// twice.
fn walk_other_files(
    path: &Path,
    total_size: &mut u64,
    seen: &mut HashSet<(u64, u64)>,
    newest: &mut Option<std::time::SystemTime>,
) {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return;
    };
    if meta.is_dir() {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries {
            let Ok(entry) = entry else { continue };
            walk_other_files(&entry.path(), total_size, seen, newest);
        }
    } else if meta.is_file() {
        note_newest_mtime(newest, &meta);
        if let Some(id) = physical_id(&meta)
            && !seen.insert(id)
        {
            return;
        }
        *total_size = total_size.saturating_add(meta.len());
    }
}

/// Raises `newest` to `meta`'s modification time when that is later.
fn note_newest_mtime(newest: &mut Option<std::time::SystemTime>, meta: &std::fs::Metadata) {
    if let Ok(modified) = meta.modified()
        && newest.is_none_or(|current| modified > current)
    {
        *newest = Some(modified);
    }
}

/// Recursively walks a snapshot directory, collecting [`CacheFileUsage`]
/// entries while accumulating `total_size` and `last_modified` alongside.
///
/// The `prefix` parameter tracks the relative path from the snapshot root,
/// so that files in subdirectories get paths like `"tokenizer/vocab.json"`.
///
/// Each entry contributes its resolved size to the listing (so a pointer
/// symlink reports the bytes it points at, not the length of its target
/// path) but contributes to `total_size` only when it holds bytes of its
/// own: see [`walk_repo_files`] for the layouts this distinguishes.
fn walk_snapshot_files(
    dir: &Path,
    prefix: &str,
    files: &mut Vec<CacheFileUsage>,
    total_size: &mut u64,
    last_modified: &mut Option<std::time::SystemTime>,
    seen: &mut HashSet<(u64, u64)>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        // BORROW: explicit .to_string_lossy() for OsString -> str conversion
        let name = entry.file_name().to_string_lossy().to_string();

        if path.is_dir() {
            let child_prefix = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            walk_snapshot_files(&path, &child_prefix, files, total_size, last_modified, seen);
        } else {
            let filename = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            // `symlink_metadata` classifies the entry without following it;
            // only a symlink then costs the second, following call. This does
            // not use [`resolved_file_size`] because it needs the entry's
            // link status and mtime too, and must charge bytes by physical
            // identity rather than by name.
            let Ok(link_meta) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            let is_link = link_meta.file_type().is_symlink();
            let metadata = if is_link {
                // A broken pointer resolves to nothing: it is listed at zero
                // bytes and charged nothing.
                std::fs::metadata(&path).ok()
            } else {
                Some(link_meta)
            };
            let size = metadata.as_ref().map_or(0, std::fs::Metadata::len);
            if let Some(modified) = metadata.as_ref().and_then(|m| m.modified().ok()) {
                match *last_modified {
                    Some(current) if modified <= current => {} // EXPLICIT: current mtime is more recent, keep it
                    _ => *last_modified = Some(modified),
                }
            }
            // A symlink holds none of its own bytes; they were counted under
            // `blobs/`. A real file is charged unless it is a hard link to
            // something already counted.
            if !is_link {
                let first_sighting = match metadata.as_ref().and_then(physical_id) {
                    Some(id) => seen.insert(id),
                    None => true,
                };
                if first_sighting {
                    *total_size = total_size.saturating_add(size);
                }
            }
            files.push(CacheFileUsage { filename, size });
        }
    }
}

/// Verification status for a single cached file.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum VerifyStatus {
    /// Local `SHA256` matches the expected hash from `HuggingFace` LFS metadata.
    Ok,
    /// Local `SHA256` does not match the expected hash — the cached file is
    /// corrupted (bit rot, interrupted write, or upstream blob changed).
    Mismatch {
        /// Expected `SHA256` hex digest from `HuggingFace` LFS metadata.
        expected: String,
        /// Actual `SHA256` hex digest computed from the local file.
        actual: String,
    },
    /// File has no LFS metadata (small git-stored file); verification skipped.
    Skipped,
    /// File is absent from the local snapshot directory.
    Missing,
}

/// Result of verifying a single cached file against `HuggingFace` LFS metadata.
#[derive(Debug, Clone)]
pub struct FileVerification {
    /// Filename within the repository.
    pub filename: String,
    /// File size in bytes — local size when the file is present, otherwise
    /// the expected size from the API (or `0` when neither is known).
    pub size: u64,
    /// Verification result.
    pub status: VerifyStatus,
}

/// Streaming progress event emitted by [`verify_cache_with_progress`] so
/// callers can render per-file feedback during a long verification.
///
/// Events fire in this order:
/// 1. [`VerifyEvent::Started`] — once, after the metadata fetch completes,
///    before any per-file work begins. Carries the total file count and a
///    pre-computed maximum filename length so callers can size display
///    columns up-front.
/// 2. For each file in alphabetical order:
///    - [`VerifyEvent::FileStart`] — before the per-file `SHA256`
///      computation kicks in.
///    - [`VerifyEvent::FileComplete`] — when the per-file result is known,
///      carrying the [`VerifyStatus`] outcome.
#[non_exhaustive]
#[derive(Debug)]
pub enum VerifyEvent<'a> {
    /// Fired once at the start of the run with summary stats useful for
    /// laying out a streamed table or progress display.
    Started {
        /// Total number of files that will be verified.
        total: usize,
        /// Maximum filename length across the verification list.
        max_filename_len: usize,
    },
    /// A file is about to be verified.
    FileStart {
        /// 1-based index of this file in the verification list.
        index: usize,
        /// Total number of files in the verification list.
        total: usize,
        /// Filename within the repository.
        filename: &'a str,
        /// File size in bytes (local size when present, else expected size).
        size: u64,
        /// `true` when the file has LFS metadata (a real `SHA256` computation
        /// is about to run); `false` when the file is git-stored and will be
        /// skipped near-instantly.
        has_lfs: bool,
    },
    /// A file's verification has completed.
    FileComplete {
        /// 1-based index of this file in the verification list.
        index: usize,
        /// Total number of files in the verification list.
        total: usize,
        /// Filename within the repository.
        filename: &'a str,
        /// File size in bytes (matches the `size` from the corresponding
        /// [`VerifyEvent::FileStart`]).
        size: u64,
        /// The per-file verification result.
        status: &'a VerifyStatus,
    },
}

/// Verifies `SHA256` digests of cached files against `HuggingFace` LFS metadata.
///
/// Fetches the expected hashes from the `HuggingFace` API and, for each file
/// that has an LFS `SHA256`, reads the local cached file and compares.
///
/// Files without LFS metadata (small git-stored files such as `config.json`)
/// are reported as [`VerifyStatus::Skipped`]; files absent from the snapshot
/// directory are reported as [`VerifyStatus::Missing`]. Both are
/// non-failures — only [`VerifyStatus::Mismatch`] indicates a corrupted file.
///
/// `revision` defaults to `"main"` when `None`. Requires network access for
/// the metadata fetch; the per-file digest computation is local-only.
///
/// For long verifications (multi-GiB safetensors files), prefer
/// [`verify_cache_with_progress`] so a CLI / GUI can render a spinner or
/// progress bar while each file is hashed.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the `HuggingFace` API request fails.
/// Returns [`FetchError::Io`] when a local cached file is present but
/// cannot be read.
pub async fn verify_cache(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<Vec<FileVerification>, FetchError> {
    verify_cache_with_progress(repo_id, token, revision, |_| {}).await
}

/// Same as [`verify_cache`] but emits [`VerifyEvent`]s through `on_event`
/// so callers can render streaming progress (e.g. a spinner per file).
///
/// The callback runs on the same task as the verification — keep it short.
/// Use interior mutability ([`std::cell::Cell`], [`std::cell::RefCell`]) if
/// you need to track state across events; the closure may capture by shared
/// reference because the API requires only [`Fn`].
///
/// Files are processed in alphabetical order by filename so that streamed
/// output remains stable across runs and matches the sort order of the
/// returned [`Vec<FileVerification>`].
///
/// # Errors
///
/// Same error conditions as [`verify_cache`].
pub async fn verify_cache_with_progress<F>(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
    on_event: F,
) -> Result<Vec<FileVerification>, FetchError>
where
    F: Fn(VerifyEvent<'_>),
{
    let revision = revision.unwrap_or("main");
    let cache_dir = hf_cache_dir()?;
    let repo_dir = crate::cache_layout::repo_dir(&cache_dir, repo_id);

    let commit_hash = read_ref(&repo_dir, revision);

    let client = crate::chunked::build_client(token)?;
    let mut remote_files =
        crate::repo::list_repo_files_with_metadata(repo_id, token, Some(revision), &client).await?;

    // Sort up-front so streamed output is stable across runs and matches the
    // returned `Vec<FileVerification>`'s order.
    remote_files.sort_by(|a, b| a.filename.cmp(&b.filename));

    // BORROW: explicit .as_deref() for Option<String> → Option<&str>
    let snapshot_dir = commit_hash
        .as_deref()
        .map(|hash| crate::cache_layout::snapshot_dir(&repo_dir, hash));

    let total = remote_files.len();
    let max_filename_len = remote_files
        .iter()
        .map(|f| f.filename.len())
        .max()
        .unwrap_or(0);

    on_event(VerifyEvent::Started {
        total,
        max_filename_len,
    });

    let mut results: Vec<FileVerification> = Vec::with_capacity(total);

    for (i, remote) in remote_files.iter().enumerate() {
        let index = i + 1;
        let local_path = snapshot_dir
            .as_ref()
            // A listed name that could leave the snapshot directory is never
            // joined to it; the file then reads as absent.
            // BORROW: explicit .as_str() instead of Deref coercion
            .filter(|_| crate::cache_layout::is_safe_repo_filename(remote.filename.as_str()))
            // BORROW: explicit .as_str() for path construction
            .map(|dir| dir.join(remote.filename.as_str()));

        let exists = local_path.as_ref().is_some_and(|p| p.exists());
        let local_size = local_path
            .as_ref()
            .filter(|_| exists)
            .and_then(|p| std::fs::metadata(p).ok().map(|m| m.len()))
            .unwrap_or(0);
        let expected_size = remote.size.unwrap_or(0);
        let display_size = if exists { local_size } else { expected_size };

        let has_lfs = remote.sha256.is_some();
        on_event(VerifyEvent::FileStart {
            index,
            total,
            // BORROW: explicit .as_str() for &String → &str argument
            filename: remote.filename.as_str(),
            size: display_size,
            has_lfs,
        });

        let status = match (remote.sha256.as_deref(), local_path.as_deref(), exists) {
            (None, _, _) => VerifyStatus::Skipped,
            (Some(_), None, _) | (Some(_), Some(_), false) => VerifyStatus::Missing,
            (Some(expected), Some(path), true) => {
                // BORROW: explicit .as_str() for &String → &str argument
                match crate::checksum::verify_sha256(path, remote.filename.as_str(), expected).await
                {
                    Ok(()) => VerifyStatus::Ok,
                    Err(FetchError::Checksum {
                        expected, actual, ..
                    }) => VerifyStatus::Mismatch { expected, actual },
                    Err(e) => return Err(e),
                }
            }
        };

        on_event(VerifyEvent::FileComplete {
            index,
            total,
            // BORROW: explicit .as_str() for &String → &str argument
            filename: remote.filename.as_str(),
            size: display_size,
            status: &status,
        });

        results.push(FileVerification {
            // BORROW: explicit .clone() for owned String
            filename: remote.filename.clone(),
            size: display_size,
            status,
        });
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::panic,
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing
    )]

    use super::*;

    /// `resolved_file_size` on an ordinary file is just its length.
    #[test]
    fn resolved_file_size_reports_a_regular_file() {
        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("f.bin");
        std::fs::write(&f, vec![b'x'; 777]).unwrap();

        assert_eq!(resolved_file_size(&f), Some(777));
    }

    #[test]
    fn resolved_file_size_is_none_for_a_directory() {
        let tmp = tempfile::tempdir().unwrap();

        assert_eq!(resolved_file_size(tmp.path()), None);
    }

    #[test]
    fn resolved_file_size_is_none_for_a_missing_path() {
        let tmp = tempfile::tempdir().unwrap();

        assert_eq!(resolved_file_size(&tmp.path().join("nope")), None);
    }

    /// The case the helper exists for: a pointer symlink must report the
    /// bytes it points at, not the length of its own target path.
    #[cfg(unix)]
    #[test]
    fn resolved_file_size_follows_a_pointer_symlink() {
        let tmp = tempfile::tempdir().unwrap();
        let blob = tmp.path().join("blob");
        std::fs::write(&blob, vec![b'x'; 5000]).unwrap();
        let pointer = tmp.path().join("pointer");
        std::os::unix::fs::symlink("blob", &pointer).unwrap();

        assert_eq!(
            resolved_file_size(&pointer),
            Some(5000),
            "must resolve to the target's size, not the link's"
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolved_file_size_is_none_for_a_dangling_symlink() {
        let tmp = tempfile::tempdir().unwrap();
        let pointer = tmp.path().join("pointer");
        std::os::unix::fs::symlink("gone", &pointer).unwrap();

        assert_eq!(resolved_file_size(&pointer), None);
    }

    /// Builds a `models--org--model` repo directory holding one blob, and
    /// returns `(tempdir, repo_dir, blob_path, snapshot_dir)`.
    ///
    /// The pointer at `snapshots/<commit>/<filename>` is left for the caller
    /// to create, because *how* that pointer is made is the whole subject of
    /// the tests below: a copy, a symlink or a hard link are three real
    /// layouts that `walk_repo_files` has to account for differently.
    fn repo_with_blob(bytes: usize) -> (tempfile::TempDir, PathBuf, PathBuf, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let repo_dir = tmp.path().join("models--org--model");
        let blobs = crate::cache_layout::blobs_dir(&repo_dir);
        let snap = crate::cache_layout::snapshot_dir(&repo_dir, "c0ffee");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snap).unwrap();
        let blob = blobs.join("deadbeef");
        std::fs::write(&blob, vec![b'x'; bytes]).unwrap();
        (tmp, repo_dir, blob, snap)
    }

    /// The Windows layout: no symlink privilege, so the pointer is a full
    /// copy and the repo genuinely holds the same bytes twice.
    ///
    /// This is the case [issue #16] reported. Counting `snapshots/` alone
    /// reported 1000 bytes for a repo occupying 2000.
    ///
    /// [issue #16]: https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues/16
    #[test]
    fn walk_counts_both_a_snapshot_copy_and_its_blob() {
        let (_tmp, repo_dir, blob, snap) = repo_with_blob(1000);
        std::fs::copy(&blob, snap.join("model.bin")).unwrap();

        let walk = walk_repo_files(&repo_dir);

        assert_eq!(
            walk.total_size, 2000,
            "a blob plus a separate copy of it occupies both sets of bytes"
        );
        assert_eq!(walk.files.len(), 1, "but it is still one logical file");
        assert_eq!(walk.files[0].filename, "model.bin");
        assert_eq!(walk.files[0].size, 1000);
    }

    /// A repo with blobs but no readable `snapshots/` (an interrupted
    /// download) still occupies its blob bytes.
    #[test]
    fn walk_counts_blobs_when_there_is_no_snapshot() {
        let (_tmp, repo_dir, _blob, _snap) = repo_with_blob(4096);
        std::fs::remove_dir_all(crate::cache_layout::snapshots_dir(&repo_dir)).unwrap();

        let walk = walk_repo_files(&repo_dir);

        assert_eq!(walk.total_size, 4096);
        assert!(
            walk.files.is_empty(),
            "no snapshot means no logical files, got {:?}",
            walk.files
        );
    }

    /// An interrupted chunked download leaves its temp blob and resume
    /// sidecar in `blobs/`, and both are counted at their length. The temp
    /// blob is preallocated to the file's full length. On Windows that
    /// length is allocated, so it is what the blob occupies; on filesystems
    /// that keep the unwritten part sparse, such as ext4, the length runs
    /// ahead of the allocation until the download finishes. It is not a
    /// listed file until it is renamed into place.
    #[test]
    fn walk_counts_a_partial_temp_blob_at_its_length() {
        let (_tmp, repo_dir, blob, snap) = repo_with_blob(1000);
        std::fs::copy(&blob, snap.join("config.json")).unwrap();
        let blobs = crate::cache_layout::blobs_dir(&repo_dir);
        std::fs::write(blobs.join("cafe.chunked.part"), vec![0u8; 8192]).unwrap();
        std::fs::write(blobs.join("cafe.chunked.part.state"), b"{}").unwrap();

        let walk = walk_repo_files(&repo_dir);

        assert_eq!(walk.total_size, 1000 + 1000 + 8192 + 2);
        assert_eq!(
            walk.files.len(),
            1,
            "only the finished file is listed, got {:?}",
            walk.files
        );
    }

    /// Every caller builds `snapshots/<commit>/` from what `read_ref`
    /// returns, so a refs file that would make that path climb out of the
    /// repo reads as no ref at all.
    #[test]
    fn read_ref_refuses_a_commit_hash_that_is_not_one_plain_name() {
        let (_tmp, repo_dir, _blob, _snap) = repo_with_blob(1);
        let refs = crate::cache_layout::refs_dir(&repo_dir);
        std::fs::create_dir_all(&refs).unwrap();

        std::fs::write(
            refs.join("main"),
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41\n",
        )
        .unwrap();
        assert_eq!(
            read_ref(&repo_dir, "main").as_deref(),
            Some("1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
        );

        for bad in ["../../../escape", "a/b", "", "   ", "c:\\x"] {
            std::fs::write(refs.join("main"), bad).unwrap();
            assert_eq!(
                read_ref(&repo_dir, "main"),
                None,
                "{bad:?} should be refused"
            );
        }
    }

    /// Creates a file symlink `link` → `target`, or returns `false` where
    /// the platform refuses one (Windows without Developer Mode), so a test
    /// can skip instead of failing. Linux CI runs these tests in full.
    fn try_symlink(target: &Path, link: &Path) -> bool {
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(target, link).is_ok()
        }
        #[cfg(windows)]
        {
            std::os::windows::fs::symlink_file(target, link).is_ok()
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = (target, link);
            false
        }
    }

    /// `huggingface_hub` 1.32's layout for one repo whose file lives in the
    /// shared store: `<root>/blobs/9a/<hash>` holds the bytes,
    /// `models--org--<name>/blobs/<etag>` links to it, and the snapshot entry
    /// links to that, all with relative targets as Python writes them.
    /// Returns the repo directory, or `None` when symlinks are unavailable.
    fn repo_linking_shared_payload(root: &Path, name: &str, etags: &[&str]) -> Option<PathBuf> {
        let payload = crate::cache_layout::shared_blobs_dir(root)
            .join("9a")
            .join("payload");
        if !payload.exists() {
            std::fs::create_dir_all(payload.parent().unwrap()).unwrap();
            std::fs::write(&payload, vec![b'x'; 5000]).unwrap();
        }
        let repo_dir = root.join(format!("models--org--{name}"));
        let blobs = crate::cache_layout::blobs_dir(&repo_dir);
        let snap = crate::cache_layout::snapshot_dir(&repo_dir, "c0ffee");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snap).unwrap();
        for (i, etag) in etags.iter().enumerate() {
            if !try_symlink(Path::new("../../blobs/9a/payload"), &blobs.join(etag)) {
                return None;
            }
            let pointer = snap.join(format!("file{i}.safetensors"));
            if !try_symlink(&Path::new("../../blobs").join(etag), &pointer) {
                return None;
            }
        }
        Some(repo_dir)
    }

    #[test]
    fn walk_charges_a_blob_linked_into_the_shared_store() {
        let tmp = tempfile::tempdir().unwrap();
        let Some(repo_dir) = repo_linking_shared_payload(tmp.path(), "a", &["e1"]) else {
            eprintln!("skipped: this platform refuses symlinks");
            return;
        };
        let walk = walk_repo_files(&repo_dir);
        // Until v0.12.2 the blob symlink was skipped and the repo counted 0.
        assert_eq!(walk.total_size, 5000);
        assert_eq!(walk.shared_size, 5000);
        assert_eq!(walk.files.len(), 1);
        assert_eq!(walk.files.first().map(|f| f.size), Some(5000));
    }

    #[test]
    fn walk_charges_one_shared_payload_once_however_many_blobs_link_it() {
        let tmp = tempfile::tempdir().unwrap();
        let Some(repo_dir) = repo_linking_shared_payload(tmp.path(), "a", &["e1", "e2"]) else {
            eprintln!("skipped: this platform refuses symlinks");
            return;
        };
        let walk = walk_repo_files(&repo_dir);
        assert_eq!(walk.total_size, 5000);
        assert_eq!(walk.shared_size, 5000);
        assert_eq!(walk.files.len(), 2, "both files are listed at full size");
    }

    #[test]
    fn two_repos_each_count_a_shared_payload_and_the_store_counts_it_once() {
        let tmp = tempfile::tempdir().unwrap();
        let (Some(a), Some(b)) = (
            repo_linking_shared_payload(tmp.path(), "a", &["e1"]),
            repo_linking_shared_payload(tmp.path(), "b", &["e1"]),
        ) else {
            eprintln!("skipped: this platform refuses symlinks");
            return;
        };
        assert_eq!(walk_repo_files(&a).total_size, 5000);
        assert_eq!(walk_repo_files(&b).total_size, 5000);
        assert_eq!(
            dir_bytes(&crate::cache_layout::shared_blobs_dir(tmp.path())),
            5000
        );
    }

    #[test]
    fn walk_still_skips_a_blob_symlink_that_leaves_the_cache_or_dangles() {
        let tmp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let foreign = outside.path().join("foreign.bin");
        std::fs::write(&foreign, vec![b'x'; 7000]).unwrap();
        let repo_dir = tmp.path().join("models--org--model");
        let blobs = crate::cache_layout::blobs_dir(&repo_dir);
        std::fs::create_dir_all(&blobs).unwrap();
        // A store exists, but neither link points into it.
        std::fs::create_dir_all(crate::cache_layout::shared_blobs_dir(tmp.path())).unwrap();
        if !try_symlink(&foreign, &blobs.join("outside"))
            || !try_symlink(Path::new("../../blobs/9a/gone"), &blobs.join("dangling"))
        {
            eprintln!("skipped: this platform refuses symlinks");
            return;
        }
        let walk = walk_repo_files(&repo_dir);
        assert_eq!(walk.total_size, 0, "nothing outside the cache is charged");
        assert_eq!(walk.shared_size, 0);
    }

    #[test]
    fn incomplete_etag_reads_both_namings() {
        let sha = "9a3f7c1e0b2d4f6a8c0e1b3d5f7a9c1e3b5d7f9a1c3e5b7d9f1a3c5e7b9d1f3a";
        // hf-hub, and huggingface_hub before 1.18.
        assert_eq!(incomplete_etag(&format!("{sha}.incomplete")), Some(sha));
        // huggingface_hub 1.18 and later: one name per attempt.
        assert_eq!(
            incomplete_etag(&format!("{sha}.1a2b3c4d.incomplete")),
            Some(sha)
        );
        // An etag with a period, as in this crate's own tests, stays whole.
        assert_eq!(incomplete_etag("abc.def.incomplete"), Some("abc.def"));
        for other in [
            sha,
            ".incomplete",
            "x.chunked.part",
            "x.chunked.part.state",
            "x.incomplete.lock",
        ] {
            assert_eq!(incomplete_etag(other), None, "{other:?}");
        }
    }

    #[test]
    fn scan_partials_finds_both_kinds_but_not_the_resume_sidecar() {
        let (_tmp, repo_dir, _blob, _snap) = repo_with_blob(1);
        let blobs = crate::cache_layout::blobs_dir(&repo_dir);
        assert_eq!(scan_partials(&blobs), (false, None));

        std::fs::write(blobs.join("cafe.chunked.part.state"), b"{}").unwrap();
        assert!(!scan_partials(&blobs).0, "a sidecar alone is not a partial");

        let older = write_aged(&blobs.join("cafe.chunked.part"), 8, 2);
        let newer = write_aged(&blobs.join("beef.1a2b3c4d.incomplete"), 8, 1);
        let (any, newest) = scan_partials(&blobs);
        assert!(any);
        assert_eq!(newest, Some(newer));
        assert!(newest > Some(older));
    }

    #[test]
    fn incomplete_sizes_keeps_each_etags_largest_attempt() {
        let (_tmp, repo_dir, _blob, _snap) = repo_with_blob(1);
        let blobs = crate::cache_layout::blobs_dir(&repo_dir);
        std::fs::write(blobs.join("beef.incomplete"), vec![0u8; 10]).unwrap();
        std::fs::write(blobs.join("beef.1a2b3c4d.incomplete"), vec![0u8; 30]).unwrap();
        std::fs::write(blobs.join("cafe.chunked.part"), vec![0u8; 99]).unwrap();
        let sizes = incomplete_sizes(&blobs);
        assert_eq!(sizes.len(), 1, "only .incomplete files: {sizes:?}");
        assert_eq!(sizes.get("beef"), Some(&30));
    }

    #[test]
    fn an_incomplete_file_is_in_use_while_its_download_lock_is_held() {
        let (tmp, repo_dir, _blob, _snap) = repo_with_blob(1);
        let blobs = crate::cache_layout::blobs_dir(&repo_dir);
        let path = blobs.join("beef.incomplete");
        std::fs::write(&path, vec![0u8; 4]).unwrap();
        let partial = PartialFile {
            repo_id: "org/model".to_owned(),
            filename: "beef.incomplete".to_owned(),
            path,
            size: 4,
        };
        let lock_path =
            crate::cache_layout::download_lock_path(tmp.path(), "models--org--model", "beef");

        // No lock file: no download is running.
        assert!(!partial.is_in_use());

        std::fs::create_dir_all(lock_path.parent().unwrap()).unwrap();
        let lock = std::fs::File::create(&lock_path).unwrap();
        // A lock file nobody holds, as a finished download leaves it.
        assert!(!partial.is_in_use());

        lock.lock().unwrap();
        assert!(partial.is_in_use(), "a held lock means a running download");
        lock.unlock().unwrap();
        assert!(!partial.is_in_use());
        assert!(
            lock_path.exists(),
            "the check must not remove the lock file"
        );
    }

    #[test]
    fn a_chunked_part_is_never_in_use_and_keeps_its_sidecars() {
        let (_tmp, repo_dir, _blob, _snap) = repo_with_blob(1);
        let blobs = crate::cache_layout::blobs_dir(&repo_dir);
        let chunked = PartialFile {
            repo_id: "org/model".to_owned(),
            filename: "cafe.chunked.part".to_owned(),
            path: blobs.join("cafe.chunked.part"),
            size: 0,
        };
        assert!(!chunked.is_in_use());
        assert_eq!(chunked.sidecar_paths().len(), 2);
        let incomplete = PartialFile {
            repo_id: "org/model".to_owned(),
            filename: "beef.incomplete".to_owned(),
            path: blobs.join("beef.incomplete"),
            size: 0,
        };
        assert!(
            incomplete.sidecar_paths().is_empty(),
            "an .incomplete has no sidecar"
        );
    }

    /// A `sha256` from the Hub's listing names a temp-blob path, so one that
    /// is not a plain name is never looked up.
    #[test]
    fn partial_chunk_size_ignores_an_unsafe_sha256() {
        let (_tmp, repo_dir, _blob, _snap) = repo_with_blob(1);
        let blobs = crate::cache_layout::blobs_dir(&repo_dir);
        std::fs::write(blobs.join("cafe.chunked.part"), vec![0u8; 64]).unwrap();
        assert_eq!(partial_chunk_size(&repo_dir, Some("cafe")), Some(64));
        assert_eq!(partial_chunk_size(&repo_dir, Some("../blobs/cafe")), None);
    }

    /// Writes `bytes` at `path` with its modification time set `days_ago`.
    fn write_aged(path: &Path, bytes: usize, days_ago: u64) -> std::time::SystemTime {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, vec![b'x'; bytes]).unwrap();
        let when = std::time::SystemTime::now() - std::time::Duration::from_secs(days_ago * 86_400);
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(when)
            .unwrap();
        when
    }

    /// A repo holding only hf-fm's header cache (`inspect --cache-headers`
    /// on a repo never downloaded) has no snapshot files, so it takes its age
    /// from the entry; without one, `cache gc --older-than` never selects it.
    #[test]
    fn walk_ages_a_header_cache_only_repo_by_its_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_dir = tmp.path().join("models--org--model");
        let entry = crate::cache_layout::header_cache_path(&repo_dir, "model.gguf", "e1");
        let when = write_aged(&entry, 100, 100);

        let walk = walk_repo_files(&repo_dir);

        assert_eq!(walk.last_modified, Some(when));
        assert!(walk.files.is_empty());
    }

    /// A downloaded repo keeps its snapshot-based age, as `cache gc`
    /// documents, even when a sidecar or a blob was written later.
    #[test]
    fn walk_ages_a_downloaded_repo_by_its_snapshot_files_only() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_dir = tmp.path().join("models--org--model");
        let snap = crate::cache_layout::snapshot_dir(&repo_dir, "c0ffee");
        let snapshot_time = write_aged(&snap.join("model.bin"), 100, 100);
        write_aged(
            &crate::cache_layout::blobs_dir(&repo_dir).join("deadbeef"),
            100,
            1,
        );
        write_aged(
            &crate::cache_layout::header_cache_path(&repo_dir, "model.bin", "e1"),
            100,
            1,
        );

        let walk = walk_repo_files(&repo_dir);

        assert_eq!(walk.last_modified, Some(snapshot_time));
    }

    /// An interrupted first download, with only a `.chunked.part` temp blob
    /// and no snapshot file yet, takes its age from the temp blob, so
    /// `cache gc` can tell that it is still being written.
    #[test]
    fn walk_ages_a_first_download_in_progress_by_its_temp_blob() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_dir = tmp.path().join("models--org--model");
        let blobs = crate::cache_layout::blobs_dir(&repo_dir);
        let when = write_aged(&blobs.join("cafe.chunked.part"), 4096, 0);

        let walk = walk_repo_files(&repo_dir);

        assert_eq!(walk.last_modified, Some(when));
    }

    /// An empty repo directory is zero bytes, not an error.
    #[test]
    fn walk_of_an_empty_repo_dir_is_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_dir = tmp.path().join("models--org--model");
        std::fs::create_dir_all(&repo_dir).unwrap();

        let walk = walk_repo_files(&repo_dir);

        assert_eq!(walk.total_size, 0);
        assert!(walk.files.is_empty());
        assert!(walk.last_modified.is_none());
    }

    /// Nested snapshot files keep their relative path and resolved size.
    #[test]
    fn walk_counts_nested_snapshot_files() {
        let (_tmp, repo_dir, _blob, snap) = repo_with_blob(100);
        std::fs::create_dir_all(snap.join("tokenizer")).unwrap();
        std::fs::write(snap.join("tokenizer/vocab.json"), vec![b'v'; 50]).unwrap();

        let walk = walk_repo_files(&repo_dir);

        // 100 (blob) + 50 (the nested file, which has no blob of its own).
        assert_eq!(walk.total_size, 150);
        assert_eq!(walk.files.len(), 1);
        assert_eq!(walk.files[0].filename, "tokenizer/vocab.json");
        assert_eq!(walk.files[0].size, 50);
    }

    /// The usual Unix layout: the pointer is a symlink into `blobs/`, so the
    /// bytes exist once.
    ///
    /// Two separate regressions are pinned here. The total must not
    /// double-count (the blob's bytes are the file's bytes), and the
    /// per-file size must be the *target's* size. Before the fix the size
    /// came from `DirEntry::metadata`, which `std` documents as equivalent
    /// to `symlink_metadata`, so this file was reported at the length of the
    /// string `../../blobs/deadbeef` instead of 1000 bytes, and the blob was
    /// never counted at all.
    #[cfg(unix)]
    #[test]
    fn walk_counts_a_symlinked_pointer_once_and_resolves_its_size() {
        let (_tmp, repo_dir, _blob, snap) = repo_with_blob(1000);
        std::os::unix::fs::symlink("../../blobs/deadbeef", snap.join("model.bin")).unwrap();

        let walk = walk_repo_files(&repo_dir);

        assert_eq!(
            walk.total_size, 1000,
            "a symlink holds no bytes of its own; only the blob counts"
        );
        assert_eq!(walk.files.len(), 1);
        assert_eq!(
            walk.files[0].size, 1000,
            "the listing must show the target's size, not the link's"
        );
    }

    /// A hard-linked pointer is the same physical file as its blob, so it is
    /// counted once. Nothing in the stack creates these today; this guards
    /// the `(st_dev, st_ino)` dedup for the day something does.
    #[cfg(unix)]
    #[test]
    fn walk_counts_a_hard_linked_pointer_once() {
        let (_tmp, repo_dir, blob, snap) = repo_with_blob(2048);
        std::fs::hard_link(&blob, snap.join("model.bin")).unwrap();

        let walk = walk_repo_files(&repo_dir);

        assert_eq!(
            walk.total_size, 2048,
            "a hard link and its blob are one physical file"
        );
        assert_eq!(walk.files.len(), 1);
        assert_eq!(walk.files[0].size, 2048);
    }

    /// A pointer whose blob is gone resolves to nothing: listed at zero
    /// bytes, charged nothing, and no panic.
    #[cfg(unix)]
    #[test]
    fn walk_tolerates_a_broken_pointer() {
        let (_tmp, repo_dir, blob, snap) = repo_with_blob(1000);
        std::os::unix::fs::symlink("../../blobs/deadbeef", snap.join("model.bin")).unwrap();
        std::fs::remove_file(&blob).unwrap();

        let walk = walk_repo_files(&repo_dir);

        assert_eq!(
            walk.total_size, 0,
            "the blob is gone, so nothing is on disk"
        );
        assert_eq!(walk.files.len(), 1, "the dangling pointer is still listed");
        assert_eq!(walk.files[0].size, 0);
    }

    /// A symlink *under* `blobs/` is not followed: a crafted cache must not
    /// be able to charge this repo for bytes living outside it.
    #[cfg(unix)]
    #[test]
    fn walk_does_not_follow_a_symlink_inside_blobs() {
        let (_tmp, repo_dir, _blob, _snap) = repo_with_blob(10);
        let outside = repo_dir.parent().unwrap().join("outside.bin");
        std::fs::write(&outside, vec![b'o'; 999_999]).unwrap();
        std::os::unix::fs::symlink(
            &outside,
            crate::cache_layout::blobs_dir(&repo_dir).join("sneaky"),
        )
        .unwrap();

        let walk = walk_repo_files(&repo_dir);

        assert_eq!(
            walk.total_size, 10,
            "only the real blob counts, not the 999999 bytes outside the repo"
        );
    }

    fn sample_partial(filename: &str) -> PartialFile {
        PartialFile {
            repo_id: "org/model".to_owned(),
            filename: filename.to_owned(),
            path: PathBuf::from("/tmp/models--org--model/blobs").join(filename),
            size: 1024,
        }
    }

    #[test]
    fn sidecar_paths_returns_state_and_state_tmp() {
        let p = sample_partial("abc123.chunked.part");
        let sidecars = p.sidecar_paths();

        assert_eq!(sidecars.len(), 2);
        assert_eq!(
            sidecars[0],
            PathBuf::from("/tmp/models--org--model/blobs/abc123.chunked.part.state")
        );
        assert_eq!(
            sidecars[1],
            PathBuf::from("/tmp/models--org--model/blobs/abc123.chunked.part.state.tmp")
        );
    }

    #[test]
    fn sidecar_paths_handles_etag_with_periods() {
        // Same period-handling rationale as `cache_layout::temp_state_path`:
        // the etag may itself contain dots, so naive `Path::with_extension`
        // would chop at the wrong boundary.
        let p = sample_partial("abc.def.chunked.part");
        let sidecars = p.sidecar_paths();

        assert_eq!(
            sidecars[0],
            PathBuf::from("/tmp/models--org--model/blobs/abc.def.chunked.part.state")
        );
        assert_eq!(
            sidecars[1],
            PathBuf::from("/tmp/models--org--model/blobs/abc.def.chunked.part.state.tmp")
        );
    }

    #[test]
    fn snapshot_roundtrip_write_then_read_returns_equal_value() {
        // Use a freshly-created temp dir so the test is isolated from any
        // pre-existing cache state on the developer machine / CI runner.
        let tmp =
            std::env::temp_dir().join(format!("hf-fm-snapshot-roundtrip-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).expect("create temp dir");

        let original = Snapshot {
            version: SNAPSHOT_VERSION,
            revision: "main".to_owned(),
            preset: Some("safetensors".to_owned()),
            filter: vec!["*.json".to_owned()],
            exclude: vec!["*.md".to_owned()],
        };

        write_snapshot(&tmp, &original).expect("write_snapshot");
        let round_tripped = read_snapshot(&tmp)
            .expect("read_snapshot")
            .expect("snapshot present");

        assert_eq!(round_tripped, original);

        // Cleanup
        let _ = std::fs::remove_file(snapshot_path(&tmp));
        let _ = std::fs::remove_dir(&tmp);
    }

    #[test]
    fn snapshot_read_returns_none_when_absent() {
        let tmp =
            std::env::temp_dir().join(format!("hf-fm-snapshot-absent-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).expect("create temp dir");

        let result = read_snapshot(&tmp).expect("read_snapshot");
        assert!(result.is_none(), "expected None for absent sidecar");

        let _ = std::fs::remove_dir(&tmp);
    }

    #[test]
    fn partial_chunk_size_returns_own_chunk_size() {
        let tmp =
            std::env::temp_dir().join(format!("hf-fm-partial-own-chunk-{}", std::process::id()));
        let blobs = tmp.join("blobs");
        std::fs::create_dir_all(&blobs).expect("create blobs dir");
        let sha = "41246eed00000000000000000000000000000000000000000000000000000344";
        std::fs::write(blobs.join(format!("{sha}.chunked.part")), vec![0u8; 1234])
            .expect("write partial blob");

        assert_eq!(partial_chunk_size(&tmp, Some(sha)), Some(1234));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn partial_chunk_size_ignores_other_files_chunks() {
        // The regression this guards: a chunk belonging to ANOTHER file must
        // not be attributed to this one (the pre-v0.10.5 repo-level fallback
        // assigned the first chunk found to every absent file).
        let tmp =
            std::env::temp_dir().join(format!("hf-fm-partial-other-chunk-{}", std::process::id()));
        let blobs = tmp.join("blobs");
        std::fs::create_dir_all(&blobs).expect("create blobs dir");
        let other_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        std::fs::write(
            blobs.join(format!("{other_sha}.chunked.part")),
            vec![0u8; 999],
        )
        .expect("write partial blob");

        let queried_sha = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        assert_eq!(partial_chunk_size(&tmp, Some(queried_sha)), None);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn partial_chunk_size_none_without_sha() {
        // Non-LFS files carry no sha256; the lookup must not guess.
        let tmp = std::env::temp_dir().join(format!("hf-fm-partial-no-sha-{}", std::process::id()));
        assert_eq!(partial_chunk_size(&tmp, None), None);
    }

    #[test]
    fn partial_chunk_size_prefers_sidecar_completed_sum() {
        // The temp blob is preallocated at full size, so its file length is
        // total_size from the first byte; the truth is the sidecar's summed
        // per-chunk `completed` offsets.
        let tmp =
            std::env::temp_dir().join(format!("hf-fm-partial-sidecar-{}", std::process::id()));
        let blobs = tmp.join("blobs");
        std::fs::create_dir_all(&blobs).expect("create blobs dir");
        let sha = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        // Preallocated blob: 4096 bytes on disk, only 300 truly downloaded.
        std::fs::write(blobs.join(format!("{sha}.chunked.part")), vec![0u8; 4096])
            .expect("write partial blob");
        let sidecar = format!(
            "{{\"schema_version\":1,\"etag\":\"{sha}\",\"total_size\":4096,\
             \"connections\":2,\"chunks\":[\
             {{\"idx\":0,\"start\":0,\"end\":2047,\"completed\":100}},\
             {{\"idx\":1,\"start\":2048,\"end\":4095,\"completed\":200}}]}}"
        );
        std::fs::write(blobs.join(format!("{sha}.chunked.part.state")), sidecar)
            .expect("write sidecar");

        assert_eq!(partial_chunk_size(&tmp, Some(sha)), Some(300));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn partial_chunk_size_falls_back_to_blob_len_on_corrupt_sidecar() {
        let tmp = std::env::temp_dir().join(format!(
            "hf-fm-partial-corrupt-sidecar-{}",
            std::process::id()
        ));
        let blobs = tmp.join("blobs");
        std::fs::create_dir_all(&blobs).expect("create blobs dir");
        let sha = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
        std::fs::write(blobs.join(format!("{sha}.chunked.part")), vec![0u8; 2048])
            .expect("write partial blob");
        std::fs::write(
            blobs.join(format!("{sha}.chunked.part.state")),
            "not json at all",
        )
        .expect("write corrupt sidecar");

        assert_eq!(partial_chunk_size(&tmp, Some(sha)), Some(2048));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
