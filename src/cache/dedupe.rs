// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reclaiming the duplicated snapshot copies (new in v0.12.6,
//! [issue #21](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues/21)):
//! each snapshot file that is a full copy of its blob is replaced by a hard
//! link to that blob, so its bytes exist once.
//!
//! On Windows `hf-hub` never tries a symlink, so every file it writes is held
//! twice, as `blobs/<etag>` and as a copy at `snapshots/<commit>/<name>`.
//! [`plan_dedupe`] pairs each copy with the blobs of its length and reads no
//! file, so its figure is an upper bound. [`apply_dedupe`] compares the bytes
//! and links the pairs that are equal, reporting what it reclaimed and what
//! it skipped, with a reason for each.
//!
//! A hard link rather than a deleted blob: `hf-hub` and `huggingface_hub`
//! both read a hard-linked pointer as cached, and the blob stays in place for
//! a second revision that shares it. Hard-linked names share their bytes, so
//! editing one in place changes the other, as a symlink on Unix always did.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::{FileId, physical_id};
use crate::error::FetchError;

/// What [`plan_dedupe`] found in one cached model repo, for [`apply_dedupe`]
/// to act on.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DedupePlan {
    /// The repository identifier (e.g., `"org/model"`).
    pub repo_id: String,
    /// The repo's directory in the cache (`<cache>/models--org--model`).
    pub repo_dir: PathBuf,
    /// Snapshot files that hold bytes of their own and have a blob of the
    /// same length beside them, in path order.
    pub candidates: Vec<DedupeCandidate>,
    /// Snapshot files left alone at planning time, each with its reason.
    pub skipped: Vec<DedupeSkip>,
    /// Whether a download into the repo is in progress or was interrupted: a
    /// partial download's temp file in `blobs/`, or a download lock that
    /// another process holds. `candidates` is then empty, and
    /// [`apply_dedupe`] checks again before touching anything.
    pub download_in_progress: bool,
}

impl DedupePlan {
    /// The most [`apply_dedupe`] can reclaim: the bytes of every candidate.
    ///
    /// An upper bound, because the plan pairs by length and reads no file.
    /// It counts each copy, so a blob that two cached revisions both copy
    /// counts twice, and both copies are reclaimed; `du`'s `DUP` figure
    /// counts that blob once. The two are equal where each blob has one copy.
    #[must_use]
    pub fn reclaimable_bytes(&self) -> u64 {
        self.candidates
            .iter()
            .map(|c| c.len)
            .fold(0_u64, u64::saturating_add)
    }
}

/// A snapshot file that may be a full copy of one of the repo's blobs.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DedupeCandidate {
    /// Absolute path of the snapshot file.
    pub copy: PathBuf,
    /// Its length in bytes, which replacing it reclaims.
    pub len: u64,
    /// The regular files directly under `blobs/` with that length, in path
    /// order. The copy is linked to the first one whose bytes equal its own.
    pub blobs: Vec<PathBuf>,
}

/// A snapshot file that [`plan_dedupe`] or [`apply_dedupe`] left alone.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DedupeSkip {
    /// Absolute path of the snapshot file.
    pub path: PathBuf,
    /// Its length in bytes.
    pub len: u64,
    /// Why it was left alone.
    pub reason: DedupeSkipReason,
}

/// Why a snapshot file was not replaced by a hard link to its blob.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DedupeSkipReason {
    /// The only blobs of its length are symlinks, as `huggingface_hub`'s
    /// shared blob store makes them. A hard link to such a symlink would link
    /// the symlink itself, whose relative target does not resolve from the
    /// snapshot directory.
    SharedStore,
    /// A download into the repo is in progress, or was interrupted and left
    /// its temp file: `hf-hub` removes and rewrites a pointer while it works.
    DownloadInProgress,
    /// No blob of its length holds the same bytes, or none is left.
    Differs,
    /// It holds the same bytes as its blob, but they do not hash to the
    /// `SHA256` the blob is named by: both are corrupt (`--verify` only).
    /// Linking would lose nothing, but they are reported, not linked.
    HashMismatch,
    /// Another process has the file open or mapped.
    InUse,
    /// The volume cannot hard-link (FAT32, exFAT), or the file and its blob
    /// are on different volumes.
    NoHardLinks,
    /// Any other failure, with the system's message.
    Other(String),
}

impl std::fmt::Display for DedupeSkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SharedStore => f.write_str(
                "its blob is a symlink into huggingface_hub's shared blob store, \
                 which a hard link cannot follow",
            ),
            Self::DownloadInProgress => {
                f.write_str("a download into the repo is in progress or was interrupted")
            }
            Self::Differs => f.write_str("no blob of its length holds the same bytes"),
            Self::HashMismatch => f.write_str(
                "it equals its blob, but neither hashes to the SHA-256 the blob is named by",
            ),
            Self::InUse => f.write_str("in use by another process"),
            Self::NoHardLinks => f.write_str("the volume cannot hard-link it to its blob"),
            Self::Other(message) => f.write_str(message),
        }
    }
}

/// What [`apply_dedupe`] did to one repo.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct DedupeReport {
    /// Snapshot files replaced by a hard link to their blob.
    pub linked: usize,
    /// Bytes those files held as copies: what the volume's free space grows
    /// by, to within one cluster per file.
    pub reclaimed_bytes: u64,
    /// Candidates that were already one file with their blob when applied
    /// (the plan was stale): nothing to do.
    pub already_linked: usize,
    /// Of `linked`, those that `--verify` could not hash-check because their
    /// blob is not named by a `SHA256` (git-stored files, named by a git
    /// object hash). Byte equality was still confirmed. `0` without `verify`.
    pub unverified: usize,
    /// Bytes read to compare the copies with their blobs, both sides counted.
    pub bytes_read: u64,
    /// Candidates left alone, each with its reason. The plan's own
    /// [`DedupePlan::skipped`] entries are not repeated here.
    pub skipped: Vec<DedupeSkip>,
}

/// Plans the deduplication of one cached model repo (new in v0.12.6).
///
/// A snapshot file that is a regular file (not a link) and is not already
/// one file with a blob is a candidate when a regular file of the same length
/// sits directly under the repo's `blobs/`. Nothing is read beyond directory
/// entries and file identities, so the plan is cheap, and its
/// [`reclaimable_bytes`](DedupePlan::reclaimable_bytes) is an upper bound.
/// A repo that is not cached gives an empty plan; one with a download in
/// progress gives a plan with no candidates and
/// [`download_in_progress`](DedupePlan::download_in_progress) set.
///
/// # Errors
///
/// Returns [`FetchError::InvalidArgument`] if `repo_id` could name a path
/// outside the cache.
/// Returns [`FetchError::Io`] if the cache directory cannot be determined.
pub fn plan_dedupe(repo_id: &str) -> Result<DedupePlan, FetchError> {
    if !crate::cache_layout::is_safe_repo_filename(repo_id) {
        return Err(FetchError::InvalidArgument(format!(
            "repo ID {repo_id:?} is not a plain org/name (refusing to plan outside the cache)"
        )));
    }
    let cache_dir = super::hf_cache_dir()?;
    Ok(plan_in(
        &crate::cache_layout::repo_dir(&cache_dir, repo_id),
        repo_id,
    ))
}

/// Replaces each candidate of `plan` that holds the same bytes as one of its
/// blobs with a hard link to that blob (new in v0.12.6).
///
/// Each copy is compared with its blobs byte for byte, stopping at the first
/// difference; the first equal blob wins. With `verify`, the copy is also
/// hashed while it is compared, and a pair whose bytes do not hash to the
/// blob's 64-hex-digit name is skipped as
/// [`HashMismatch`](DedupeSkipReason::HashMismatch); a blob named otherwise is
/// linked and counted in [`DedupeReport::unverified`]. The replacement never
/// leaves the pointer missing: a link is made under a temporary name and
/// renamed over it.
///
/// Nothing is touched when a download into the repo is in progress, checked
/// again here because the plan may be old. A file that cannot be replaced is
/// skipped with its reason, and the others go on.
#[must_use]
pub fn apply_dedupe(plan: &DedupePlan, verify: bool) -> DedupeReport {
    let mut report = DedupeReport::default();
    if download_in_progress(&plan.repo_dir) {
        report.skipped = plan
            .candidates
            .iter()
            .map(|c| DedupeSkip {
                // BORROW: explicit .clone() for owned PathBuf
                path: c.copy.clone(),
                len: c.len,
                reason: DedupeSkipReason::DownloadInProgress,
            })
            .collect();
        return report;
    }
    let mut buffers = CompareBuffers::new();
    // EXPLICIT: tallies each candidate's outcome into the report as it goes
    for candidate in &plan.candidates {
        match dedupe_one(candidate, verify, &mut buffers, &mut report.bytes_read) {
            Outcome::Linked { unverified } => {
                report.linked += 1;
                report.reclaimed_bytes = report.reclaimed_bytes.saturating_add(candidate.len);
                if unverified {
                    report.unverified += 1;
                }
            }
            Outcome::AlreadyLinked => report.already_linked += 1,
            Outcome::Skipped(reason) => report.skipped.push(DedupeSkip {
                // BORROW: explicit .clone() for owned PathBuf
                path: candidate.copy.clone(),
                len: candidate.len,
                reason,
            }),
        }
    }
    report
}

/// [`plan_dedupe`] for the repo directory `repo_dir`.
fn plan_in(repo_dir: &Path, repo_id: &str) -> DedupePlan {
    let mut plan = DedupePlan {
        // BORROW: explicit .to_owned() for &str → owned String
        repo_id: repo_id.to_owned(),
        // BORROW: explicit .to_path_buf() for owned PathBuf
        repo_dir: repo_dir.to_path_buf(),
        candidates: Vec::new(),
        skipped: Vec::new(),
        download_in_progress: download_in_progress(repo_dir),
    };
    if plan.download_in_progress {
        return plan;
    }

    let blobs = read_blobs(&crate::cache_layout::blobs_dir(repo_dir));
    let mut files: Vec<PathBuf> = Vec::new();
    collect_files(&crate::cache_layout::snapshots_dir(repo_dir), &mut files);
    files.sort();

    for path in files {
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        // A link holds no bytes of its own, and an empty file has none to
        // reclaim.
        if !meta.is_file() || meta.len() == 0 {
            continue;
        }
        let len = meta.len();
        if let Some(same_len) = blobs.by_len.get(&len) {
            let id = physical_id(&path, &meta);
            // Already one file with a blob: a hard link made earlier.
            if id.is_some() && same_len.iter().any(|b| b.id == id) {
                continue;
            }
            plan.candidates.push(DedupeCandidate {
                copy: path,
                len,
                // BORROW: explicit .clone() for owned PathBuf
                blobs: same_len.iter().map(|b| b.path.clone()).collect(),
            });
        } else if blobs.linked_lens.contains(&len) {
            plan.skipped.push(DedupeSkip {
                path,
                len,
                reason: DedupeSkipReason::SharedStore,
            });
        }
        // Otherwise no blob has its length: a file stored in the snapshot
        // alone (as `download_direct` writes one), not a copy of anything.
    }
    plan
}

/// One regular file under `blobs/`.
struct BlobEntry {
    /// Absolute path of the blob.
    path: PathBuf,
    /// Its identity, where it can be read.
    id: Option<FileId>,
}

/// The blobs of one repo, by length.
struct BlobsByLen {
    /// Regular, finished blobs, in path order within each length.
    by_len: HashMap<u64, Vec<BlobEntry>>,
    /// Lengths of the files that blob symlinks resolve to.
    linked_lens: HashSet<u64>,
}

/// Reads the blobs directly under `blobs_dir`: regular files by length, and
/// the lengths that blob symlinks resolve to. Partial downloads and their
/// sidecars are left out.
fn read_blobs(blobs_dir: &Path) -> BlobsByLen {
    let mut blobs = BlobsByLen {
        by_len: HashMap::new(),
        linked_lens: HashSet::new(),
    };
    let Ok(entries) = std::fs::read_dir(blobs_dir) else {
        return blobs;
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        // BORROW: explicit .to_string_lossy() for OsString → str conversion
        if super::is_partial_bytes_name(&entry.file_name().to_string_lossy()) {
            continue;
        }
        let path = entry.path();
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if meta.is_file() {
            let id = physical_id(&path, &meta);
            blobs
                .by_len
                .entry(meta.len())
                .or_default()
                .push(BlobEntry { path, id });
        } else if meta.file_type().is_symlink()
            && let Ok(target) = std::fs::metadata(&path)
            && target.is_file()
        {
            blobs.linked_lens.insert(target.len());
        }
    }
    for same_len in blobs.by_len.values_mut() {
        same_len.sort_by(|a, b| a.path.cmp(&b.path));
    }
    blobs
}

/// Collects every non-directory entry under `dir`, recursively, without
/// following a directory link.
fn collect_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        match std::fs::symlink_metadata(&path) {
            Ok(meta) if meta.is_dir() => collect_files(&path, files),
            Ok(_) => files.push(path),
            Err(_) => {} // EXPLICIT: an entry that vanished or cannot be read is not a candidate
        }
    }
}

/// Whether a download into the repo at `repo_dir` is in progress, or was
/// interrupted: a partial download's temp file in its `blobs/` (this crate's
/// `.chunked.part`, which takes no lock, or an `.incomplete`), or one of its
/// download locks held by another process.
///
/// `hf-hub` holds the lock across the moment it removes a pointer and copies
/// the blob over it, which leaves no temp file behind, so the lock is what
/// covers it.
fn download_in_progress(repo_dir: &Path) -> bool {
    if super::scan_partials(&crate::cache_layout::blobs_dir(repo_dir)).0 {
        return true;
    }
    let (Some(cache_root), Some(folder)) = (repo_dir.parent(), repo_dir.file_name()) else {
        return false;
    };
    // BORROW: explicit .to_string_lossy() for OsStr → str conversion
    let locks_dir = crate::cache_layout::download_locks_dir(cache_root, &folder.to_string_lossy());
    let Ok(entries) = std::fs::read_dir(locks_dir) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|entry| {
        entry.path().extension().is_some_and(|ext| ext == "lock")
            && super::download_lock_held(&entry.path())
    })
}

/// What happened to one candidate.
// EXHAUSTIVE: internal outcome; this module owns and matches all variants
enum Outcome {
    /// Replaced by a hard link; `unverified` when `--verify` could not hash
    /// its blob's name.
    Linked {
        /// Whether the blob's name is not a `SHA256` under `--verify`.
        unverified: bool,
    },
    /// Already one file with its blob.
    AlreadyLinked,
    /// Left alone.
    Skipped(DedupeSkipReason),
}

/// Compares one candidate with its blobs and links it to the first equal one.
fn dedupe_one(
    candidate: &DedupeCandidate,
    verify: bool,
    buffers: &mut CompareBuffers,
    bytes_read: &mut u64,
) -> Outcome {
    let copy_meta = match std::fs::symlink_metadata(&candidate.copy) {
        Ok(meta) => meta,
        Err(e) => return Outcome::Skipped(skip_reason(&e)),
    };
    // Re-materialised as a link since the plan: nothing left to reclaim.
    if copy_meta.file_type().is_symlink() {
        return Outcome::AlreadyLinked;
    }
    if !copy_meta.is_file() || copy_meta.len() != candidate.len {
        return Outcome::Skipped(DedupeSkipReason::Differs);
    }
    let copy_id = physical_id(&candidate.copy, &copy_meta);

    // The blobs that are still regular files of the copy's length.
    let mut blobs: Vec<&Path> = Vec::with_capacity(candidate.blobs.len());
    for blob in &candidate.blobs {
        let Ok(meta) = std::fs::symlink_metadata(blob) else {
            continue;
        };
        if !meta.is_file() || meta.len() != candidate.len {
            continue;
        }
        if copy_id.is_some() && physical_id(blob, &meta) == copy_id {
            return Outcome::AlreadyLinked;
        }
        blobs.push(blob);
    }

    let mut last_error: Option<std::io::Error> = None;
    // EXPLICIT: tries each blob in turn and stops at the first equal one
    for blob in blobs {
        let mut hasher = verify.then(Sha256::new);
        match files_equal(&candidate.copy, blob, buffers, &mut hasher, bytes_read) {
            Ok(true) => {
                let mut unverified = false;
                if let Some(hasher) = hasher {
                    // BORROW: explicit .to_string_lossy() for OsStr → str conversion
                    let name = blob
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    if super::is_sha256_hex(&name) {
                        if crate::checksum::lower_hex(&hasher.finalize()) != name {
                            return Outcome::Skipped(DedupeSkipReason::HashMismatch);
                        }
                    } else {
                        unverified = true;
                    }
                }
                return match link_over(&candidate.copy, blob) {
                    Ok(LinkOutcome::Linked) => Outcome::Linked { unverified },
                    Ok(LinkOutcome::AlreadyLinked) => Outcome::AlreadyLinked,
                    Err(e) => Outcome::Skipped(skip_reason(&e)),
                };
            }
            Ok(false) => {}
            Err(e) => last_error = Some(e),
        }
    }
    Outcome::Skipped(last_error.map_or(DedupeSkipReason::Differs, |e| skip_reason(&e)))
}

/// The two read buffers a comparison streams through, allocated once per
/// [`apply_dedupe`] call.
struct CompareBuffers {
    /// Holds the next block of the copy.
    copy: Vec<u8>,
    /// Holds the next block of the blob.
    blob: Vec<u8>,
}

/// Size of each comparison read, in bytes: 1 MiB.
const COMPARE_BLOCK: usize = 1 << 20;

impl CompareBuffers {
    fn new() -> Self {
        Self {
            copy: vec![0_u8; COMPARE_BLOCK],
            blob: vec![0_u8; COMPARE_BLOCK],
        }
    }
}

/// Whether the files at `copy` and `blob` hold the same bytes, reading both
/// block by block and stopping at the first block that differs. When
/// `hasher` is `Some`, every block of the copy read before a difference is
/// fed to it. `bytes_read` grows by every byte read from either file.
fn files_equal(
    copy: &Path,
    blob: &Path,
    buffers: &mut CompareBuffers,
    hasher: &mut Option<Sha256>,
    bytes_read: &mut u64,
) -> std::io::Result<bool> {
    let mut copy_file = std::fs::File::open(copy)?;
    let mut blob_file = std::fs::File::open(blob)?;
    loop {
        let n = read_full(&mut copy_file, &mut buffers.copy)?;
        let m = read_full(&mut blob_file, &mut buffers.blob)?;
        // CAST: usize → u64, a buffer length always fits
        #[allow(clippy::as_conversions)]
        let read = (n + m) as u64;
        *bytes_read = bytes_read.saturating_add(read);
        let (Some(copy_block), Some(blob_block)) = (buffers.copy.get(..n), buffers.blob.get(..m))
        else {
            return Ok(false);
        };
        if copy_block != blob_block {
            return Ok(false);
        }
        if let Some(hasher) = hasher.as_mut() {
            hasher.update(copy_block);
        }
        if n == 0 {
            return Ok(true);
        }
    }
}

/// Reads from `file` until `buf` is full or the file ends, and returns how
/// many bytes were read: a short read is not the end of a file.
fn read_full(file: &mut std::fs::File, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    // EXPLICIT: a read may return fewer bytes than asked; loop until full or end of file
    while let Some(rest) = buf.get_mut(filled..) {
        if rest.is_empty() {
            break;
        }
        match file.read(rest) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {} // EXPLICIT: retry an interrupted read
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

/// What [`link_over`] did.
// EXHAUSTIVE: internal outcome; this crate owns and matches all variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LinkOutcome {
    /// The pointer is now a hard link to the blob.
    Linked,
    /// The pointer already was one file with the blob; nothing was changed.
    AlreadyLinked,
}

/// Replaces the file at `pointer` with a hard link to `blob`, the one routine
/// both `cache dedupe` and the download path's prevention step use.
///
/// The caller has established that the two hold the same bytes. This checks
/// their identities again immediately before linking, so a stale plan, or a
/// pointer already linked to the blob, is [`LinkOutcome::AlreadyLinked`] and
/// nothing more. The link is created beside the pointer under a temporary
/// name and renamed over it, so the pointer is never missing. A rename onto
/// another name of the same file is a no-op on Unix that leaves the temporary
/// name, so that name is removed whenever it still exists afterwards, as it
/// is when the rename fails.
///
/// # Errors
///
/// Returns the `std::io::Error` of the first step that fails: reading either
/// file's metadata, creating the link (a volume without hard links, the
/// temporary name already taken by a file that is not this blob), or the
/// rename (a pointer another process holds open).
pub(crate) fn link_over(pointer: &Path, blob: &Path) -> std::io::Result<LinkOutcome> {
    let blob_meta = std::fs::metadata(blob)?;
    let blob_id = physical_id(blob, &blob_meta);
    let pointer_meta = std::fs::metadata(pointer)?;
    if blob_id.is_some() && physical_id(pointer, &pointer_meta) == blob_id {
        return Ok(LinkOutcome::AlreadyLinked);
    }

    let Some(name) = pointer.file_name() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("pointer {} has no file name", pointer.display()),
        ));
    };
    let mut temp_name = name.to_os_string();
    temp_name.push(TEMP_LINK_SUFFIX);
    let temp = pointer.with_file_name(temp_name);
    // A leftover of an interrupted run is a link to this very blob, and is
    // replaced; anything else at that name is not ours, and `hard_link` then
    // fails with `AlreadyExists`.
    if blob_id.is_some()
        && let Ok(meta) = std::fs::metadata(&temp)
        && physical_id(&temp, &meta) == blob_id
    {
        std::fs::remove_file(&temp)?;
    }

    std::fs::hard_link(blob, &temp)?;
    let renamed = std::fs::rename(&temp, pointer);
    if std::fs::symlink_metadata(&temp).is_ok() {
        let _ = std::fs::remove_file(&temp);
    }
    renamed.map(|()| LinkOutcome::Linked)
}

/// Suffix of the temporary name [`link_over`] creates a link under, beside
/// the pointer it replaces.
const TEMP_LINK_SUFFIX: &str = ".hf-fm-link.tmp";

/// Windows system error codes `std` maps to no specific `ErrorKind`.
#[cfg(windows)]
mod win_error {
    /// `ERROR_INVALID_FUNCTION`: what FAT32 and exFAT answer to a hard link.
    pub(super) const INVALID_FUNCTION: i32 = 1;
    /// `ERROR_SHARING_VIOLATION`: another process opened the file without
    /// sharing it.
    pub(super) const SHARING_VIOLATION: i32 = 32;
    /// `ERROR_LOCK_VIOLATION`: another process locked part of the file.
    pub(super) const LOCK_VIOLATION: i32 = 33;
    /// `ERROR_NOT_SUPPORTED`: the volume does not support the request.
    pub(super) const NOT_SUPPORTED: i32 = 50;
    /// `ERROR_USER_MAPPED_FILE`: the file is memory-mapped by a process, as
    /// a model loader maps its weights.
    pub(super) const USER_MAPPED_FILE: i32 = 1224;
}

/// The skip reason for an I/O error met while comparing or linking.
fn skip_reason(e: &std::io::Error) -> DedupeSkipReason {
    #[cfg(windows)]
    {
        use win_error::{
            INVALID_FUNCTION, LOCK_VIOLATION, NOT_SUPPORTED, SHARING_VIOLATION, USER_MAPPED_FILE,
        };
        let code = e.raw_os_error();
        if matches!(
            code,
            Some(SHARING_VIOLATION | LOCK_VIOLATION | USER_MAPPED_FILE)
        ) {
            return DedupeSkipReason::InUse;
        }
        if matches!(code, Some(INVALID_FUNCTION | NOT_SUPPORTED)) {
            return DedupeSkipReason::NoHardLinks;
        }
    }
    let kind = e.kind();
    if matches!(
        kind,
        std::io::ErrorKind::ResourceBusy | std::io::ErrorKind::ExecutableFileBusy
    ) {
        DedupeSkipReason::InUse
    } else if matches!(
        kind,
        std::io::ErrorKind::CrossesDevices | std::io::ErrorKind::Unsupported
    ) {
        DedupeSkipReason::NoHardLinks
    } else {
        DedupeSkipReason::Other(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::panic,
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::as_conversions,
        clippy::cast_possible_truncation
    )]

    use super::*;

    /// A `models--org--model` repo holding one blob named `blob_name` with
    /// `bytes`; returns `(tempdir, repo_dir, blob, snapshot_dir)`.
    fn repo(blob_name: &str, bytes: &[u8]) -> (tempfile::TempDir, PathBuf, PathBuf, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let repo_dir = tmp.path().join("models--org--model");
        let blobs = crate::cache_layout::blobs_dir(&repo_dir);
        let snap = crate::cache_layout::snapshot_dir(&repo_dir, "c0ffee");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snap).unwrap();
        let blob = blobs.join(blob_name);
        std::fs::write(&blob, bytes).unwrap();
        (tmp, repo_dir, blob, snap)
    }

    /// Bytes longer than one comparison block, so the compare loop runs
    /// more than once.
    fn content() -> Vec<u8> {
        // CAST: usize → u8, deliberately wrapping into a byte pattern
        (0..COMPARE_BLOCK + 4097).map(|i| (i % 251) as u8).collect()
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        crate::checksum::lower_hex(&Sha256::digest(bytes))
    }

    fn id(path: &Path) -> Option<FileId> {
        physical_id(path, &std::fs::metadata(path).unwrap())
    }

    /// Files left under `dir` whose name carries the temporary-link suffix.
    fn temp_names(dir: &Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        collect_files(dir, &mut files);
        files
            .into_iter()
            .filter(|p| p.to_string_lossy().ends_with(TEMP_LINK_SUFFIX))
            .collect()
    }

    /// Creates a file symlink, or returns `false` where the platform refuses
    /// one (Windows without Developer Mode); CI runs these for real.
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

    #[test]
    fn replaces_a_good_copy_and_then_finds_nothing() {
        let bytes = content();
        let len = bytes.len() as u64;
        let (_tmp, repo_dir, blob, snap) = repo("deadbeef", &bytes);
        let copy = snap.join("model.bin");
        std::fs::copy(&blob, &copy).unwrap();
        assert_eq!(
            super::super::walk_repo_files(&repo_dir).duplicated_bytes,
            len
        );

        let plan = plan_in(&repo_dir, "org/model");
        assert_eq!(plan.candidates.len(), 1);
        assert_eq!(plan.reclaimable_bytes(), len);
        let report = apply_dedupe(&plan, false);

        assert_eq!(report.linked, 1);
        assert_eq!(report.reclaimed_bytes, len);
        assert!(report.skipped.is_empty(), "{:?}", report.skipped);
        assert_eq!(
            std::fs::read(&copy).unwrap(),
            bytes,
            "the pointer reads as before"
        );
        assert_eq!(id(&copy), id(&blob), "one physical file");
        let walk = super::super::walk_repo_files(&repo_dir);
        assert_eq!(walk.total_size, len, "counted once");
        assert_eq!(walk.duplicated_bytes, 0);
        let leftovers = temp_names(&snap);
        assert!(leftovers.is_empty(), "temporary names left: {leftovers:?}");

        let again = plan_in(&repo_dir, "org/model");
        assert!(again.candidates.is_empty(), "nothing left to do");
        assert_eq!(again.reclaimable_bytes(), 0);
    }

    #[test]
    fn refuses_a_copy_that_differs() {
        let (_tmp, repo_dir, _blob, snap) = repo("deadbeef", &[b'x'; 1000]);
        let copy = snap.join("model.bin");
        let mut other = vec![b'x'; 1000];
        other[999] = b'y';
        std::fs::write(&copy, &other).unwrap();

        let plan = plan_in(&repo_dir, "org/model");
        assert_eq!(plan.candidates.len(), 1, "paired by length");
        let report = apply_dedupe(&plan, false);

        assert_eq!(report.linked, 0);
        assert_eq!(report.skipped.len(), 1);
        assert_eq!(report.skipped[0].reason, DedupeSkipReason::Differs);
        assert_eq!(std::fs::read(&copy).unwrap(), other, "left untouched");
        assert_eq!(report.bytes_read, 2000, "both sides read to the end");
    }

    /// A file with no blob of its length is not a copy of anything, and a
    /// copy whose length changed after planning is not linked.
    #[test]
    fn a_length_mismatch_is_never_linked() {
        let (_tmp, repo_dir, blob, snap) = repo("deadbeef", &[b'x'; 1000]);
        std::fs::write(snap.join("own.json"), [b'x'; 999]).unwrap();
        assert!(plan_in(&repo_dir, "org/model").candidates.is_empty());

        let copy = snap.join("model.bin");
        std::fs::copy(&blob, &copy).unwrap();
        let plan = plan_in(&repo_dir, "org/model");
        assert_eq!(plan.candidates.len(), 1);
        std::fs::write(&copy, [b'x'; 1001]).unwrap();
        let report = apply_dedupe(&plan, false);

        assert_eq!(report.linked, 0);
        assert_eq!(report.skipped[0].reason, DedupeSkipReason::Differs);
    }

    /// Two cached revisions that each copy one blob: both copies go.
    #[test]
    fn links_every_copy_of_a_blob() {
        let (_tmp, repo_dir, blob, snap) = repo("deadbeef", &[b'x'; 1000]);
        std::fs::copy(&blob, snap.join("model.bin")).unwrap();
        let snap2 = crate::cache_layout::snapshot_dir(&repo_dir, "c0ffe2");
        std::fs::create_dir_all(snap2.join("sub")).unwrap();
        std::fs::copy(&blob, snap2.join("sub").join("model.bin")).unwrap();

        let plan = plan_in(&repo_dir, "org/model");
        assert_eq!(plan.reclaimable_bytes(), 2000, "each copy counts");
        let report = apply_dedupe(&plan, false);

        assert_eq!(report.linked, 2);
        assert_eq!(report.reclaimed_bytes, 2000);
        assert_eq!(super::super::walk_repo_files(&repo_dir).total_size, 1000);
    }

    /// Where two blobs share a length, the copy goes to the one it equals.
    #[test]
    fn links_a_copy_to_the_blob_it_equals() {
        let (_tmp, repo_dir, _blob, snap) = repo("aaaa", &[b'a'; 500]);
        let second = crate::cache_layout::blobs_dir(&repo_dir).join("bbbb");
        std::fs::write(&second, [b'b'; 500]).unwrap();
        let copy = snap.join("b.bin");
        std::fs::copy(&second, &copy).unwrap();

        let report = apply_dedupe(&plan_in(&repo_dir, "org/model"), false);

        assert_eq!(report.linked, 1);
        assert_eq!(id(&copy), id(&second));
    }

    /// A stale plan whose pointer was linked since is "already linked", and
    /// no temporary name is left behind.
    #[test]
    fn a_stale_plan_and_an_already_linked_pointer_are_already_linked() {
        let (_tmp, repo_dir, blob, snap) = repo("deadbeef", &[b'x'; 1000]);
        let copy = snap.join("model.bin");
        std::fs::copy(&blob, &copy).unwrap();
        let plan = plan_in(&repo_dir, "org/model");
        std::fs::remove_file(&copy).unwrap();
        std::fs::hard_link(&blob, &copy).unwrap();

        let report = apply_dedupe(&plan, false);

        assert_eq!(report.already_linked, 1);
        assert_eq!(report.linked, 0);
        assert_eq!(report.reclaimed_bytes, 0);
        let leftovers = temp_names(&snap);
        assert!(leftovers.is_empty(), "temporary names left: {leftovers:?}");
        assert_eq!(link_over(&copy, &blob).unwrap(), LinkOutcome::AlreadyLinked);
        let leftovers = temp_names(&snap);
        assert!(leftovers.is_empty(), "temporary names left: {leftovers:?}");
    }

    /// A leftover temporary link to the same blob, from an interrupted run,
    /// is replaced; a file of someone else's at that name is never removed.
    #[test]
    fn link_over_replaces_only_its_own_leftover() {
        let (_tmp, _repo_dir, blob, snap) = repo("deadbeef", &[b'x'; 100]);
        let copy = snap.join("model.bin");
        let temp = snap.join(format!("model.bin{TEMP_LINK_SUFFIX}"));
        std::fs::copy(&blob, &copy).unwrap();
        std::fs::hard_link(&blob, &temp).unwrap();
        assert_eq!(link_over(&copy, &blob).unwrap(), LinkOutcome::Linked);
        assert!(!temp.exists());
        assert_eq!(id(&copy), id(&blob));

        let copy2 = snap.join("other.bin");
        let temp2 = snap.join(format!("other.bin{TEMP_LINK_SUFFIX}"));
        std::fs::copy(&blob, &copy2).unwrap();
        std::fs::write(&temp2, b"not ours").unwrap();
        let err = link_over(&copy2, &blob).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&temp2).unwrap(), b"not ours");
        assert_ne!(id(&copy2), id(&blob), "the copy is left as it was");
    }

    #[test]
    fn a_partial_download_or_a_held_lock_blocks_the_repo() {
        let (tmp, repo_dir, blob, snap) = repo("deadbeef", &[b'x'; 1000]);
        std::fs::copy(&blob, snap.join("model.bin")).unwrap();
        let part = crate::cache_layout::blobs_dir(&repo_dir).join("cafe.chunked.part");
        std::fs::write(&part, [0_u8; 10]).unwrap();

        let plan = plan_in(&repo_dir, "org/model");
        assert!(plan.download_in_progress);
        assert!(plan.candidates.is_empty());

        std::fs::remove_file(&part).unwrap();
        let plan = plan_in(&repo_dir, "org/model");
        assert_eq!(plan.candidates.len(), 1);

        // A lock taken after planning: apply checks again and touches nothing.
        let lock_path =
            crate::cache_layout::download_lock_path(tmp.path(), "models--org--model", "beef");
        std::fs::create_dir_all(lock_path.parent().unwrap()).unwrap();
        let lock = std::fs::File::create(&lock_path).unwrap();
        lock.lock().unwrap();
        let report = apply_dedupe(&plan, false);
        assert_eq!(report.linked, 0);
        assert_eq!(
            report.skipped[0].reason,
            DedupeSkipReason::DownloadInProgress
        );

        lock.unlock().unwrap();
        assert_eq!(apply_dedupe(&plan, false).linked, 1, "a lock nobody holds");
    }

    /// `--verify`: a blob named by the `SHA256` of the shared bytes links,
    /// one named otherwise links unverified, and one whose name is the hash
    /// of other bytes (both files corrupt alike) is reported, not linked.
    #[test]
    fn verify_links_confirmed_and_unverifiable_pairs_but_not_corrupt_ones() {
        let bytes = content();
        let (_tmp, repo_dir, blob, snap) = repo(&sha256_hex(&bytes), &bytes);
        std::fs::copy(&blob, snap.join("model.bin")).unwrap();
        let blobs = crate::cache_layout::blobs_dir(&repo_dir);
        std::fs::write(blobs.join("0123abcd"), b"{\"a\": 1}").unwrap();
        std::fs::write(snap.join("config.json"), b"{\"a\": 1}").unwrap();
        let corrupt_name = sha256_hex(b"what the file should hold");
        std::fs::write(blobs.join(&corrupt_name), b"what it holds instead!!").unwrap();
        std::fs::write(snap.join("tokenizer.bin"), b"what it holds instead!!").unwrap();

        let report = apply_dedupe(&plan_in(&repo_dir, "org/model"), true);

        assert_eq!(report.linked, 2);
        assert_eq!(report.unverified, 1, "config.json's blob is git-named");
        assert_eq!(report.skipped.len(), 1);
        assert!(report.skipped[0].path.ends_with("tokenizer.bin"));
        assert_eq!(report.skipped[0].reason, DedupeSkipReason::HashMismatch);
    }

    /// The copy, symlink and hard-link layouts in one repo: only the copy is
    /// a candidate, and the symlinked pointer is left alone.
    #[test]
    fn plans_only_the_copy_among_the_three_layouts() {
        let (_tmp, repo_dir, blob, snap) = repo("deadbeef", &[b'x'; 1000]);
        std::fs::copy(&blob, snap.join("copy.bin")).unwrap();
        std::fs::hard_link(&blob, snap.join("hard.bin")).unwrap();
        let target: PathBuf = ["..", "..", "blobs", "deadbeef"].iter().collect();
        if !try_symlink(&target, &snap.join("sym.bin")) {
            eprintln!("skipped: this platform refuses symlinks");
            return;
        }

        let plan = plan_in(&repo_dir, "org/model");

        assert_eq!(plan.candidates.len(), 1);
        assert!(plan.candidates[0].copy.ends_with("copy.bin"));
        let report = apply_dedupe(&plan, false);
        assert_eq!(report.linked, 1);
        assert!(
            std::fs::symlink_metadata(snap.join("sym.bin"))
                .unwrap()
                .file_type()
                .is_symlink(),
            "a symlinked pointer is left alone"
        );
    }

    /// A blob that is a symlink into the shared store is not a candidate: a
    /// hard link to it would link the symlink itself.
    #[test]
    fn skips_a_blob_that_lives_in_the_shared_store() {
        let tmp = tempfile::tempdir().unwrap();
        let payload_dir = crate::cache_layout::shared_blobs_dir(tmp.path()).join("9a");
        std::fs::create_dir_all(&payload_dir).unwrap();
        std::fs::write(payload_dir.join("payload"), [b'x'; 700]).unwrap();
        let repo_dir = tmp.path().join("models--org--model");
        let blobs = crate::cache_layout::blobs_dir(&repo_dir);
        let snap = crate::cache_layout::snapshot_dir(&repo_dir, "c0ffee");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snap).unwrap();
        let target: PathBuf = ["..", "..", "blobs", "9a", "payload"].iter().collect();
        if !try_symlink(&target, &blobs.join("e1")) {
            eprintln!("skipped: this platform refuses symlinks");
            return;
        }
        std::fs::write(snap.join("model.bin"), [b'x'; 700]).unwrap();

        let plan = plan_in(&repo_dir, "org/model");

        assert!(plan.candidates.is_empty());
        assert_eq!(plan.skipped.len(), 1);
        assert_eq!(plan.skipped[0].reason, DedupeSkipReason::SharedStore);
    }

    /// A copy another process holds open without sharing it is skipped as in
    /// use, and left as it was. `std`'s default share mode would let the
    /// rename succeed, hence `share_mode(0)`.
    #[cfg(windows)]
    #[test]
    fn skips_a_file_held_open() {
        use std::os::windows::fs::OpenOptionsExt;
        let (_tmp, repo_dir, blob, snap) = repo("deadbeef", &[b'x'; 1000]);
        let copy = snap.join("model.bin");
        std::fs::copy(&blob, &copy).unwrap();
        let plan = plan_in(&repo_dir, "org/model");
        let held = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(&copy)
            .unwrap();

        let report = apply_dedupe(&plan, false);

        assert_eq!(report.linked, 0);
        assert_eq!(report.skipped[0].reason, DedupeSkipReason::InUse);
        let leftovers = temp_names(&snap);
        assert!(leftovers.is_empty(), "temporary names left: {leftovers:?}");
        drop(held);
        assert_ne!(id(&copy), id(&blob), "the copy is left as it was");
    }

    /// The fact the reclaim's safety rests on, against a writer that copies
    /// over a pointer without removing it first: on Windows, copying a blob
    /// onto a hard link to itself fails and leaves the blob intact.
    #[cfg(windows)]
    #[test]
    fn copying_a_blob_onto_its_hard_link_fails_and_keeps_the_blob() {
        let bytes = content();
        let (_tmp, _repo_dir, blob, snap) = repo("deadbeef", &bytes);
        let pointer = snap.join("model.bin");
        std::fs::hard_link(&blob, &pointer).unwrap();

        assert!(std::fs::copy(&blob, &pointer).is_err());
        assert_eq!(std::fs::read(&blob).unwrap(), bytes);
    }

    #[test]
    fn plan_dedupe_refuses_an_id_that_leaves_the_cache() {
        for bad in ["../escape", "org/../../x", "c:\\x", ""] {
            assert!(
                matches!(plan_dedupe(bad), Err(FetchError::InvalidArgument(_))),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn an_uncached_repo_plans_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = plan_in(&tmp.path().join("models--org--absent"), "org/absent");
        assert!(plan.candidates.is_empty());
        assert!(plan.skipped.is_empty(), "{:?}", plan.skipped);
        assert!(!plan.download_in_progress);
    }

    #[test]
    fn io_errors_map_to_their_reasons() {
        use std::io::{Error, ErrorKind};
        assert_eq!(
            skip_reason(&Error::from(ErrorKind::CrossesDevices)),
            DedupeSkipReason::NoHardLinks
        );
        assert_eq!(
            skip_reason(&Error::from(ErrorKind::ResourceBusy)),
            DedupeSkipReason::InUse
        );
        assert!(matches!(
            skip_reason(&Error::from(ErrorKind::PermissionDenied)),
            DedupeSkipReason::Other(_)
        ));
        #[cfg(windows)]
        {
            assert_eq!(
                skip_reason(&Error::from_raw_os_error(32)),
                DedupeSkipReason::InUse
            );
            assert_eq!(
                skip_reason(&Error::from_raw_os_error(1224)),
                DedupeSkipReason::InUse
            );
            assert_eq!(
                skip_reason(&Error::from_raw_os_error(1)),
                DedupeSkipReason::NoHardLinks
            );
        }
    }
}
