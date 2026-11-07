// SPDX-License-Identifier: MIT OR Apache-2.0

//! Moving one cached repo to another disk behind a link, and back (new in
//! v0.13.0).
//!
//! [`plan_move`] checks a move and counts what it will copy without writing
//! anything; [`apply_move`] performs it. The repo is copied into a staging
//! directory beside its destination, each file hashed with `SHA256` while it
//! is copied and again when read back, then the cache entry is swapped for a
//! link to the copy (a junction on Windows, which needs no privilege; a
//! symlink on Unix), the link is checked, and the original is dropped or
//! kept. [`plan_move_back`] does the reverse for a repo already behind a link:
//! its bytes come back into the cache and replace the link.
//!
//! Every intermediate state has a name a later run recognises, so an
//! interrupted move needs no journal:
//! - `<parent of the destination>/.hf-fm-move/<folder>.partial` is the
//!   staging directory. A file reaches its final name there only once its
//!   digests agree, so a later run trusts a staged file whose length and
//!   modification time equal the source's, and copies only the rest.
//! - `<cache>/.hf-fm-move/<folder>.swap` is the cache entry set aside during
//!   the swap. Found with the entry missing (or a link that does not
//!   resolve), it is put back before anything else; found with the entry in
//!   place, the drop of the original is finished.
//! - `<cache>/.hf-fm-move/<folder>` is an original kept by `keep_original`.
//!
//! Hard links inside the repo are kept as hard links (so a repo reclaimed by
//! [`apply_dedupe`](super::apply_dedupe) stays reclaimed), and links between
//! its own files are recreated with the same relative target. A link that
//! points outside the repo, as `huggingface_hub`'s cache-wide blob store
//! makes them, would dangle once the repo is elsewhere, so such a repo is
//! refused.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

use sha2::{Digest, Sha256};

use super::dedupe::{download_lock_held_for, read_full, rename_error};
use super::{FileId, physical_id, strip_verbatim_prefix};
use crate::cache_layout::RepoType;
use crate::error::FetchError;

/// Name of the directory a move keeps its intermediate and kept directories
/// in, beside the destination and in the cache root (new in v0.13.0).
///
/// Its leading dot keeps every listing of the cache from reading it as a
/// repo. `du` reports the bytes in the cache's own one
/// ([`move_leftover_bytes`]).
pub const MOVE_DIR_NAME: &str = ".hf-fm-move";

/// Suffix of the staging directory, inside [`MOVE_DIR_NAME`].
const STAGING_SUFFIX: &str = ".partial";

/// Suffix of the cache entry set aside during the swap, inside
/// [`MOVE_DIR_NAME`].
const ASIDE_SUFFIX: &str = ".swap";

/// Suffix of the link created and removed to check that the destination's
/// volume can hold one, inside [`MOVE_DIR_NAME`].
const PROBE_SUFFIX: &str = ".probe";

/// Suffix of a file being copied, beside its final name in the staging
/// directory.
const COPY_TEMP_SUFFIX: &str = ".hf-fm-copy.tmp";

/// Size of each read while copying and hashing, in bytes: 1 MiB.
const COPY_BLOCK: usize = 1 << 20;

/// What a [`MovePlan`] does.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MoveKind {
    /// The repo is a directory in the cache; it moves to the destination and
    /// leaves a link behind.
    Relocate,
    /// The repo is already behind a link; its bytes move from `from` to the
    /// destination, and the link is pointed there.
    MoveAgain {
        /// The directory the link points to now, where the bytes are.
        from: PathBuf,
    },
    /// The repo is behind a link; its bytes come back from `from` into the
    /// cache, replacing the link.
    Back {
        /// The directory the link points to now, where the bytes are.
        from: PathBuf,
    },
    /// An earlier move to this destination was interrupted after its swap:
    /// only its original is left to drop or keep, and nothing is copied.
    Finish,
}

/// What an interrupted earlier move left, which [`apply_move`] deals with
/// before anything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MoveRecovery {
    /// The cache entry was set aside and not replaced (the entry is missing,
    /// or is a link that does not resolve): it is put back, and the move
    /// starts again from the files already staged.
    RestoreAside,
    /// The swap was complete and the original was being dropped: the drop
    /// (or the keep) is finished.
    FinishDrop,
}

/// A checked move of one cached repo, from [`plan_move`] or
/// [`plan_move_back`], for [`apply_move`] to perform.
#[derive(Debug, Clone)]
#[must_use]
#[non_exhaustive]
pub struct MovePlan {
    /// The kind of repo.
    pub repo_type: RepoType,
    /// The repository identifier (e.g., `"org/model"`).
    pub repo_id: String,
    /// What the move does.
    pub kind: MoveKind,
    /// The repo's entry in the cache (`<cache>/models--org--model`), a
    /// directory or a link.
    pub entry: PathBuf,
    /// The directory the bytes are copied from: the entry itself, or the
    /// directory its link points to.
    pub source: PathBuf,
    /// The directory the bytes end in: `<DIR>/<folder>`, or the entry for
    /// [`MoveKind::Back`].
    pub destination: PathBuf,
    /// The staging directory the copy is made in, on the destination's
    /// volume.
    pub staging: PathBuf,
    /// Where the cache entry is set aside during the swap; what an
    /// interrupted move leaves behind.
    pub aside: PathBuf,
    /// Whether the destination's parent directory does not exist yet and is
    /// created by [`apply_move`].
    pub creates_parent: bool,
    /// Files in the repo, counting each hard-linked name.
    pub files: usize,
    /// Bytes the repo's files hold, each physical file once: what the copy
    /// adds to the destination's volume.
    pub bytes: u64,
    /// Of `bytes`, those an interrupted earlier run already copied and
    /// verified, which are not copied again.
    pub bytes_staged: u64,
    /// Names that are hard links to another file of the repo, recreated as
    /// hard links at the destination.
    pub hard_links: usize,
    /// Links between the repo's own files, recreated with the same target.
    pub symlinks: usize,
    /// Free space on the destination's volume, in bytes, when it can be read.
    pub available_space: Option<u64>,
    /// Whether the destination is on the source's own volume, where the move
    /// frees nothing.
    pub same_volume: bool,
    /// What an interrupted earlier move left, when it did.
    pub recovery: Option<MoveRecovery>,
    // The cache root.
    cache_dir: PathBuf,
    // The repo's cache folder name (`models--org--model`).
    folder: String,
    // The directory asked for (`None` for a move back).
    requested_dir: Option<PathBuf>,
    // The repo's tree, as walked when planned.
    entries: Vec<Entry>,
}

impl MovePlan {
    /// Bytes [`apply_move`] still has to copy: [`bytes`](Self::bytes) less
    /// [`bytes_staged`](Self::bytes_staged).
    #[must_use]
    pub const fn bytes_to_copy(&self) -> u64 {
        self.bytes.saturating_sub(self.bytes_staged)
    }
}

/// What became of the original after a move.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MoveOriginal {
    /// It was deleted.
    Dropped {
        /// The directory that was deleted.
        path: PathBuf,
    },
    /// It was kept (`keep_original`).
    Kept {
        /// Where it is.
        path: PathBuf,
    },
    /// It could not be deleted or kept as asked, and is still there: the
    /// move itself is complete, and the next move of this repo finishes the
    /// job.
    Left {
        /// What is left.
        path: PathBuf,
        /// Why, with the system's message.
        reason: String,
    },
}

/// What [`apply_move`] did.
#[derive(Debug, Clone, Default)]
#[must_use]
#[non_exhaustive]
pub struct MoveReport {
    /// Files copied and verified by this run.
    pub copied_files: usize,
    /// Bytes those files hold.
    pub copied_bytes: u64,
    /// Files an interrupted earlier run had already copied and verified,
    /// kept as they were. Their bytes are not hashed again, so a blob whose
    /// name they did not match was reported by that run only.
    pub resumed_files: usize,
    /// Bytes those files hold.
    pub resumed_bytes: u64,
    /// Names recreated as hard links to another copied file.
    pub hard_links: usize,
    /// Names that were hard links in the source but are separate copies at
    /// the destination, whose volume cannot hard-link.
    pub hard_links_copied: usize,
    /// Links between the repo's own files, recreated.
    pub symlinks: usize,
    /// Blobs whose content does not hash to the `SHA256` they are named by:
    /// the cache's own copy was already corrupt, and was moved as it is.
    /// Paths relative to the repo directory.
    pub name_mismatches: Vec<PathBuf>,
    /// What an interrupted earlier move left, and was dealt with.
    pub recovered: Option<MoveRecovery>,
    /// What became of the original; `None` when nothing was moved.
    pub original: Option<MoveOriginal>,
}

/// A step of [`apply_move_with_progress`], for a progress display.
#[derive(Debug)]
#[non_exhaustive]
pub enum MoveEvent<'a> {
    /// The copy starts.
    Started {
        /// Files in the repo, counting each hard-linked name.
        files: usize,
        /// Bytes still to copy; each is read twice (copied, then read back),
        /// so [`MoveEvent::Progress`] counts to twice this.
        bytes: u64,
    },
    /// A file is about to be copied, linked, or kept from an earlier run.
    File {
        /// 1-based position of the file.
        index: usize,
        /// Number of files.
        total: usize,
        /// Its path, relative to the repo directory.
        path: &'a Path,
        /// Its length in bytes.
        len: u64,
    },
    /// Bytes processed so far: copied, plus read back to verify.
    Progress {
        /// Bytes processed so far.
        done: u64,
        /// Bytes to process in all (twice [`MoveEvent::Started`]'s `bytes`).
        total: u64,
    },
    /// The copy is complete and verified; the cache entry is being swapped.
    Swapping,
    /// The original is being deleted.
    Dropping {
        /// The directory being deleted.
        path: &'a Path,
    },
}

/// One entry of a repo's tree, relative to its directory.
#[derive(Debug, Clone)]
struct Entry {
    // Path relative to the repo directory.
    rel: PathBuf,
    // What it is.
    kind: EntryKind,
}

/// What an [`Entry`] is.
// EXHAUSTIVE: internal tree model; this module owns and matches all variants
#[derive(Debug, Clone, PartialEq, Eq)]
enum EntryKind {
    /// A directory.
    Dir,
    /// A regular file.
    File {
        /// Its length in bytes.
        len: u64,
        /// Its modification time, when it can be read.
        modified: Option<SystemTime>,
        /// The index of an earlier entry that is the same physical file
        /// (a hard link), which the copy links to instead of copying.
        link_to: Option<usize>,
    },
    /// A symbolic link (or a Windows junction), never followed.
    Link {
        /// What it points to, as written.
        target: PathBuf,
    },
}

/// Plans moving the cached repo `repo_id` of type `repo_type` into the
/// directory `dir`, as `<dir>/<folder>`, behind a link (new in v0.13.0).
///
/// Reads only. A repo already behind a link moves again
/// ([`MoveKind::MoveAgain`]). `dir` is created by [`apply_move`] when it
/// does not exist. A destination on the source's own volume is allowed and
/// flagged in [`MovePlan::same_volume`], since the move frees nothing there.
///
/// # Errors
///
/// Returns [`FetchError::InvalidArgument`] if `repo_id` could name a path
/// outside the cache.
/// Returns [`FetchError::InvalidArgument`] if the repo is not cached, or is
/// already at that destination.
/// Returns [`FetchError::InvalidArgument`] if the repo's link points to a
/// directory not named like its cache folder, or holding neither `blobs/` nor
/// `snapshots/` (the directory a move would delete).
/// Returns [`FetchError::InvalidArgument`] if a download into the repo is in
/// progress or was interrupted.
/// Returns [`FetchError::InvalidArgument`] if a link inside the repo points
/// outside it.
/// Returns [`FetchError::InvalidArgument`] if `dir` is inside the cache or the
/// repo, if `<dir>/<folder>` already exists, or if its volume lacks the free
/// space the copy needs.
/// Returns [`FetchError::Io`] if the cache directory cannot be determined or
/// a directory cannot be read.
pub fn plan_move(repo_type: RepoType, repo_id: &str, dir: &Path) -> Result<MovePlan, FetchError> {
    plan_in(&super::hf_cache_dir()?, repo_type, repo_id, Some(dir))
}

/// Plans moving the cached repo `repo_id` of type `repo_type`, now behind a
/// link, back into the cache (new in v0.13.0).
///
/// Its bytes are copied from the directory the link points to into the
/// cache, and replace the link.
///
/// # Errors
///
/// Returns [`FetchError::InvalidArgument`] if the repo is not behind a link.
/// Returns the other errors of [`plan_move`], for the cache's volume.
pub fn plan_move_back(repo_type: RepoType, repo_id: &str) -> Result<MovePlan, FetchError> {
    plan_in(&super::hf_cache_dir()?, repo_type, repo_id, None)
}

/// Performs `plan`: copies and verifies the repo, swaps the cache entry, checks
/// it, and drops the original, or keeps it with `keep_original` (new in
/// v0.13.0).
///
/// See [`apply_move_with_progress`].
///
/// # Errors
///
/// Returns the errors of [`apply_move_with_progress`].
pub fn apply_move(plan: &MovePlan, keep_original: bool) -> Result<MoveReport, FetchError> {
    apply_move_with_progress(plan, keep_original, |_| {})
}

/// [`apply_move`], reporting each step through `on_event` (new in v0.13.0).
///
/// What an interrupted earlier move left ([`MovePlan::recovery`]) is dealt
/// with first; the move is then planned again, and must still be the move
/// `plan` describes. Before any byte is copied, a link to the destination's
/// parent is created and removed, so a volume that cannot hold one fails
/// here; a volume that cannot hard-link gets separate copies instead. The
/// source is walked again after the copy and must not have changed. Any
/// failure before the swap leaves the cache entry untouched and the staged
/// files in place for the next run; a failure during the swap, or a check
/// through the new link that does not match, puts the entry back.
///
/// With `keep_original`, an original that was a directory in the cache is
/// kept at `<cache>/.hf-fm-move/<folder>`, and one behind a link stays where
/// it is. An original that cannot be deleted is reported as
/// [`MoveOriginal::Left`], and the move still succeeds.
///
/// # Errors
///
/// Returns [`FetchError::InvalidArgument`] if the cache changed since the
/// plan into a different move, if a kept original is in the way of
/// `keep_original`, if the source changed during the copy, or if the copy did
/// not read back as copied through the new link (the swap is then undone).
/// Returns [`FetchError::Checksum`] if a copy does not read back as the bytes
/// copied.
/// Returns [`FetchError::Io`] on a filesystem failure; a cache entry that
/// cannot be swapped because a file inside it is open is reported with
/// [`std::io::ErrorKind::ResourceBusy`].
pub fn apply_move_with_progress<F>(
    plan: &MovePlan,
    keep_original: bool,
    mut on_event: F,
) -> Result<MoveReport, FetchError>
where
    F: FnMut(MoveEvent<'_>),
{
    let mut report = MoveReport {
        recovered: plan.recovery,
        ..MoveReport::default()
    };
    let kept = kept_path(&plan.cache_dir, &plan.folder);
    if keep_original
        && matches!(plan.kind, MoveKind::Relocate)
        && std::fs::symlink_metadata(&kept).is_ok()
    {
        return Err(FetchError::InvalidArgument(format!(
            "a kept original is already at {} (delete it or move without keeping the original)",
            kept.display()
        )));
    }

    match plan.recovery {
        Some(MoveRecovery::RestoreAside) => restore_aside(plan)?,
        Some(MoveRecovery::FinishDrop) => {
            report.original = Some(settle_aside(plan, keep_original, &mut on_event));
        }
        None => {}
    }
    if plan.kind == MoveKind::Finish {
        if let Some(move_dir) = plan.aside.parent() {
            let _ = std::fs::remove_dir(move_dir);
        }
        return Ok(report);
    }

    // The cache may have changed since the plan; what is done is what it is now.
    let fresh = plan_in(
        &plan.cache_dir,
        plan.repo_type,
        // BORROW: explicit .as_str() instead of Deref coercion
        plan.repo_id.as_str(),
        plan.requested_dir.as_deref(),
    )?;
    if fresh.kind != plan.kind || fresh.destination != plan.destination {
        return Err(FetchError::InvalidArgument(format!(
            "{} changed since the move was planned (plan it again)",
            plan.repo_type.address(&plan.repo_id)
        )));
    }

    let hard_links_work = prepare(&fresh)?;
    prune_staging(&fresh)?;
    copy_all(&fresh, hard_links_work, &mut report, &mut on_event)?;

    // The source must not have changed while it was copied.
    let after = walk(&fresh.source)?;
    if !same_tree(&fresh.entries, &after, true) || fresh_download_in_progress(&fresh) {
        return Err(FetchError::InvalidArgument(format!(
            "{} changed while it was copied (the copied files are kept at {}, and the next run copies only the difference)",
            fresh.repo_type.address(&fresh.repo_id),
            fresh.staging.display()
        )));
    }

    on_event(MoveEvent::Swapping);
    swap(&fresh)?;
    check_swapped(&fresh)?;
    report.original = Some(settle_aside(&fresh, keep_original, &mut on_event));
    // The move directories, once empty; one holding a kept original stays.
    for move_dir in [fresh.staging.parent(), fresh.aside.parent()]
        .into_iter()
        .flatten()
    {
        let _ = std::fs::remove_dir(move_dir);
    }
    Ok(report)
}

/// Bytes in the cache's [`MOVE_DIR_NAME`] directory: originals kept by
/// `keep_original`, and what an interrupted move left (new in v0.13.0).
///
/// No listing of the cache counts them, so `du` states them apart. Links are
/// not followed, so an original kept behind a link elsewhere is not counted.
/// `0` when there is none.
///
/// # Errors
///
/// Returns [`FetchError::Io`] if the cache directory cannot be determined.
pub fn move_leftover_bytes() -> Result<u64, FetchError> {
    Ok(super::dir_bytes(
        &super::hf_cache_dir()?.join(MOVE_DIR_NAME),
    ))
}

/// Where `keep_original` keeps an original that was a directory in the cache.
fn kept_path(cache_dir: &Path, folder: &str) -> PathBuf {
    cache_dir.join(MOVE_DIR_NAME).join(folder)
}

/// A name inside a [`MOVE_DIR_NAME`] directory: the folder name and a suffix.
fn move_dir_child(parent: &Path, folder: &str, suffix: &str) -> PathBuf {
    parent.join(MOVE_DIR_NAME).join(format!("{folder}{suffix}"))
}

/// [`plan_move`] (`dir` is `Some`) or [`plan_move_back`] (`None`) for the
/// cache at `cache_dir`.
fn plan_in(
    cache_dir: &Path,
    repo_type: RepoType,
    repo_id: &str,
    dir: Option<&Path>,
) -> Result<MovePlan, FetchError> {
    let address = repo_type.address(repo_id);
    if !crate::cache_layout::is_safe_repo_filename(repo_id) {
        return Err(FetchError::InvalidArgument(format!(
            "{address} is not a valid repo ID (refusing to move outside the cache)"
        )));
    }
    let folder = crate::cache_layout::repo_folder_name_for(repo_type, repo_id);
    let entry = crate::cache_layout::repo_dir_for(cache_dir, repo_type, repo_id);
    let aside = move_dir_child(cache_dir, &folder, ASIDE_SUFFIX);

    // What an interrupted move left: the entry set aside, and either not
    // replaced (put it back) or replaced (finish the drop).
    let entry_usable = std::fs::metadata(&entry).is_ok_and(|m| m.is_dir());
    let recovery = std::fs::symlink_metadata(&aside)
        .is_ok()
        .then_some(if entry_usable {
            MoveRecovery::FinishDrop
        } else {
            MoveRecovery::RestoreAside
        });
    if recovery == Some(MoveRecovery::RestoreAside)
        && std::fs::symlink_metadata(&entry).is_ok_and(|m| !m.file_type().is_symlink())
    {
        return Err(FetchError::InvalidArgument(format!(
            "{address}: {} is in the way of restoring {} (move it away and run again)",
            entry.display(),
            aside.display()
        )));
    }
    // The entry as it is, or as it is once the aside is put back.
    let current = if recovery == Some(MoveRecovery::RestoreAside) {
        &aside
    } else {
        &entry
    };
    let current_meta = std::fs::symlink_metadata(current).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            FetchError::InvalidArgument(format!("{address} is not cached"))
        } else {
            FetchError::Io {
                // BORROW: explicit .clone() for owned PathBuf
                path: current.clone(),
                source: e,
            }
        }
    })?;
    let current_target = if current_meta.file_type().is_symlink() {
        Some(guarded_target(current, &folder, &address)?)
    } else {
        None
    };

    let (kind, destination) = match (dir, &current_target) {
        (Some(dir), None) => (MoveKind::Relocate, resolve_dir(dir)?.join(&folder)),
        (Some(dir), Some(target)) => {
            let destination = resolve_dir(dir)?.join(&folder);
            if same_dir(&destination, target) {
                if recovery != Some(MoveRecovery::FinishDrop) {
                    return Err(FetchError::InvalidArgument(format!(
                        "{address} is already at {} (nothing to move)",
                        destination.display()
                    )));
                }
                (MoveKind::Finish, destination)
            } else {
                (
                    MoveKind::MoveAgain {
                        // BORROW: explicit .clone() for owned PathBuf
                        from: target.clone(),
                    },
                    destination,
                )
            }
        }
        (None, Some(target)) => (
            MoveKind::Back {
                // BORROW: explicit .clone() for owned PathBuf
                from: target.clone(),
            },
            // BORROW: explicit .clone() for owned PathBuf
            entry.clone(),
        ),
        (None, None) => {
            if recovery != Some(MoveRecovery::FinishDrop) {
                return Err(FetchError::InvalidArgument(format!(
                    "{address} is not behind a link (nothing to move back)"
                )));
            }
            // BORROW: explicit .clone() for owned PathBuf
            (MoveKind::Finish, entry.clone())
        }
    };

    // The directory the bytes are read from now, and the one they are read
    // from once the aside is back.
    let walk_from = current_target.clone().unwrap_or_else(|| current.clone());
    // BORROW: explicit .clone() for owned PathBuf
    let source = current_target.unwrap_or_else(|| entry.clone());
    let staging_parent = if matches!(kind, MoveKind::Back { .. }) {
        cache_dir.to_path_buf()
    } else {
        destination
            .parent()
            .map_or_else(|| cache_dir.to_path_buf(), Path::to_path_buf)
    };
    let staging = move_dir_child(&staging_parent, &folder, STAGING_SUFFIX);

    let mut plan = MovePlan {
        repo_type,
        // BORROW: explicit .to_owned() for &str → owned String
        repo_id: repo_id.to_owned(),
        kind,
        entry,
        source,
        destination,
        staging,
        aside,
        creates_parent: !staging_parent.exists(),
        files: 0,
        bytes: 0,
        bytes_staged: 0,
        hard_links: 0,
        symlinks: 0,
        available_space: None,
        same_volume: false,
        recovery,
        // BORROW: explicit .to_path_buf() for owned PathBuf
        cache_dir: cache_dir.to_path_buf(),
        folder,
        requested_dir: dir.map(Path::to_path_buf),
        entries: Vec::new(),
    };
    if plan.kind == MoveKind::Finish {
        return Ok(plan);
    }

    check_destination(&plan, &walk_from, &staging_parent)?;
    if download_in_progress_at(cache_dir, &plan.folder, &walk_from) {
        return Err(FetchError::InvalidArgument(format!(
            "a download into {address} is in progress or was interrupted \
             (a partial download, or a download lock another process holds)"
        )));
    }

    plan.entries = walk(&walk_from)?;
    let escaping: Vec<String> = plan
        .entries
        .iter()
        .filter_map(|e| match &e.kind {
            EntryKind::Link { target } if link_escapes(&e.rel, target) => {
                Some(format!("{} -> {}", e.rel.display(), target.display()))
            }
            EntryKind::Link { .. } | EntryKind::Dir | EntryKind::File { .. } => None,
        })
        .collect();
    if let Some(first) = escaping.first() {
        return Err(FetchError::InvalidArgument(format!(
            "{address} holds {} link(s) pointing outside it, which would dangle once it is moved \
             (first: {first}; huggingface_hub's shared blob store links blobs this way)",
            escaping.len()
        )));
    }

    count_entries(&mut plan);
    let existing = existing_ancestor(&staging_parent);
    plan.available_space = existing.and_then(|p| fs2::available_space(p).ok());
    plan.same_volume = match (existing.and_then(volume_id), volume_id(&walk_from)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    };
    if let Some(available) = plan.available_space
        && available < plan.bytes_to_copy()
    {
        return Err(FetchError::InvalidArgument(format!(
            "not enough free space for {address} at {} (needs {} bytes, {available} available)",
            staging_parent.display(),
            plan.bytes_to_copy()
        )));
    }
    Ok(plan)
}

/// Fills in `plan`'s counts from its walked tree, reading the staging
/// directory for what an earlier run already copied.
fn count_entries(plan: &mut MovePlan) {
    // EXPLICIT: accumulates several counts over one pass
    for e in &plan.entries {
        match &e.kind {
            EntryKind::Dir => {}
            EntryKind::Link { .. } => plan.symlinks += 1,
            EntryKind::File {
                link_to: Some(_), ..
            } => {
                plan.files += 1;
                plan.hard_links += 1;
            }
            EntryKind::File {
                len,
                modified,
                link_to: None,
            } => {
                plan.files += 1;
                plan.bytes = plan.bytes.saturating_add(*len);
                if staged_matches(&plan.staging.join(&e.rel), *len, *modified) {
                    plan.bytes_staged = plan.bytes_staged.saturating_add(*len);
                }
            }
        }
    }
}

/// Refuses a destination inside the cache or the source, and one that
/// already exists.
fn check_destination(
    plan: &MovePlan,
    walk_from: &Path,
    staging_parent: &Path,
) -> Result<(), FetchError> {
    let address = plan.repo_type.address(&plan.repo_id);
    if matches!(plan.kind, MoveKind::Back { .. }) {
        return Ok(());
    }
    if std::fs::symlink_metadata(&plan.destination).is_ok() {
        return Err(FetchError::InvalidArgument(format!(
            "{} already exists (refusing to merge {address} into it; remove it or choose another directory)",
            plan.destination.display()
        )));
    }
    let inside = |outer: &Path| {
        std::fs::canonicalize(outer)
            .map(strip_verbatim_prefix)
            .is_ok_and(|outer| staging_parent.starts_with(outer))
    };
    if inside(&plan.cache_dir) {
        return Err(FetchError::InvalidArgument(format!(
            "{} is inside the cache (choose a directory outside {})",
            staging_parent.display(),
            plan.cache_dir.display()
        )));
    }
    if inside(walk_from) {
        return Err(FetchError::InvalidArgument(format!(
            "{} is inside {address} itself (choose a directory outside it)",
            staging_parent.display()
        )));
    }
    Ok(())
}

/// Whether a download into the repo whose cache folder is `folder` is in
/// progress or was interrupted: a partial download in `source`'s `blobs/`,
/// or a download lock another process holds.
fn download_in_progress_at(cache_dir: &Path, folder: &str, source: &Path) -> bool {
    super::scan_partials(&crate::cache_layout::blobs_dir(source)).0
        || download_lock_held_for(cache_dir, folder)
}

/// [`download_in_progress_at`] for a plan's source.
fn fresh_download_in_progress(plan: &MovePlan) -> bool {
    download_in_progress_at(&plan.cache_dir, &plan.folder, &plan.source)
}

/// The real directory the link at `link` resolves to, once it has passed the
/// checks a move needs before anything there is deleted: named `folder`, and
/// holding a `blobs/` or a `snapshots/` directory (the checks `cache delete
/// --with-target` makes).
fn guarded_target(link: &Path, folder: &str, address: &str) -> Result<PathBuf, FetchError> {
    let target = std::fs::canonicalize(link)
        .map(strip_verbatim_prefix)
        .map_err(|e| {
            FetchError::InvalidArgument(format!(
                "{address} links to a target that cannot be read ({e})"
            ))
        })?;
    if target.file_name().and_then(std::ffi::OsStr::to_str) != Some(folder) {
        return Err(FetchError::InvalidArgument(format!(
            "{address} links to {} (not a directory named {folder}); refusing to move it",
            target.display()
        )));
    }
    if !crate::cache_layout::blobs_dir(&target).is_dir()
        && !crate::cache_layout::snapshots_dir(&target).is_dir()
    {
        return Err(FetchError::InvalidArgument(format!(
            "{address} links to {} (no blobs/ or snapshots/ directory inside); refusing to move it",
            target.display()
        )));
    }
    Ok(target)
}

/// `dir` as an absolute path whose existing part is canonical (links
/// resolved, no `\\?\` prefix), so it compares with the cache's own path.
fn resolve_dir(dir: &Path) -> Result<PathBuf, FetchError> {
    let io = |e: std::io::Error| FetchError::Io {
        // BORROW: explicit .to_path_buf() for owned PathBuf
        path: dir.to_path_buf(),
        source: e,
    };
    let absolute = std::path::absolute(dir).map_err(io)?;
    let existing = existing_ancestor(&absolute).ok_or_else(|| {
        FetchError::InvalidArgument(format!("{} has no existing parent", dir.display()))
    })?;
    if !existing.is_dir() {
        return Err(FetchError::InvalidArgument(format!(
            "{} is not a directory",
            existing.display()
        )));
    }
    let mut resolved = strip_verbatim_prefix(std::fs::canonicalize(existing).map_err(io)?);
    if let Ok(rest) = absolute.strip_prefix(existing) {
        resolved.push(rest);
    }
    Ok(resolved)
}

/// The deepest of `path` and its ancestors that exists.
fn existing_ancestor(path: &Path) -> Option<&Path> {
    path.ancestors().find(|p| p.exists())
}

/// Whether `a` and `b` name the same directory, comparing canonical paths.
fn same_dir(a: &Path, b: &Path) -> bool {
    let canonical = |p: &Path| std::fs::canonicalize(p).map(strip_verbatim_prefix).ok();
    match (canonical(a), canonical(b)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

/// Whether the link at `rel` (relative to the repo directory), pointing to
/// `target`, leads outside the repo: an absolute target, or one whose `..`
/// climbs above the repo directory.
fn link_escapes(rel: &Path, target: &Path) -> bool {
    if target.has_root() || target.is_absolute() {
        return true;
    }
    let mut depth = rel.parent().map_or(0, |p| {
        p.components()
            .filter(|c| matches!(c, Component::Normal(_)))
            .count()
    });
    for component in target.components() {
        match component {
            Component::Normal(_) => depth += 1,
            Component::ParentDir => match depth.checked_sub(1) {
                Some(d) => depth = d,
                None => return true,
            },
            Component::CurDir => {}
            Component::Prefix(_) | Component::RootDir => return true,
        }
    }
    false
}

/// Walks the tree under `root` (following `root` itself when it is a link,
/// and nothing below it), depth first and in name order, so a directory
/// comes before its contents and `blobs/` before `snapshots/`. A file that is
/// the same physical file as an earlier one records its index.
fn walk(root: &Path) -> Result<Vec<Entry>, FetchError> {
    let mut entries: Vec<Entry> = Vec::new();
    let mut first_by_id: HashMap<FileId, usize> = HashMap::new();
    walk_into(root, Path::new(""), &mut entries, &mut first_by_id)?;
    Ok(entries)
}

/// One directory level of [`walk`].
fn walk_into(
    root: &Path,
    rel_dir: &Path,
    entries: &mut Vec<Entry>,
    first_by_id: &mut HashMap<FileId, usize>,
) -> Result<(), FetchError> {
    let dir = root.join(rel_dir);
    let io = |path: &Path, e: std::io::Error| FetchError::Io {
        // BORROW: explicit .to_path_buf() for owned PathBuf
        path: path.to_path_buf(),
        source: e,
    };
    let mut names: Vec<std::ffi::OsString> = std::fs::read_dir(&dir)
        .map_err(|e| io(&dir, e))?
        .map(|entry| entry.map(|entry| entry.file_name()))
        .collect::<Result<_, _>>()
        .map_err(|e| io(&dir, e))?;
    names.sort();
    for name in names {
        let rel = rel_dir.join(&name);
        let path = root.join(&rel);
        let meta = std::fs::symlink_metadata(&path).map_err(|e| io(&path, e))?;
        if meta.file_type().is_symlink() {
            let target = std::fs::read_link(&path).map_err(|e| io(&path, e))?;
            entries.push(Entry {
                rel,
                kind: EntryKind::Link { target },
            });
        } else if meta.is_dir() {
            entries.push(Entry {
                // BORROW: explicit .clone() for owned PathBuf
                rel: rel.clone(),
                kind: EntryKind::Dir,
            });
            walk_into(root, &rel, entries, first_by_id)?;
        } else {
            let index = entries.len();
            let link_to = physical_id(&path, &meta).and_then(|id| {
                let first = *first_by_id.entry(id).or_insert(index);
                (first != index).then_some(first)
            });
            entries.push(Entry {
                rel,
                kind: EntryKind::File {
                    len: meta.len(),
                    modified: meta.modified().ok(),
                    link_to,
                },
            });
        }
    }
    Ok(())
}

/// Whether two walked trees hold the same entries: same paths and kinds, same
/// lengths and link targets, and with `with_times` the same modification
/// times. Which names are hard links is not compared: the copy may hold
/// separate files where its volume cannot hard-link.
fn same_tree(a: &[Entry], b: &[Entry], with_times: bool) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.rel == y.rel
                && match (&x.kind, &y.kind) {
                    (EntryKind::Dir, EntryKind::Dir) => true,
                    (EntryKind::Link { target: t1 }, EntryKind::Link { target: t2 }) => t1 == t2,
                    (
                        EntryKind::File {
                            len: l1,
                            modified: m1,
                            ..
                        },
                        EntryKind::File {
                            len: l2,
                            modified: m2,
                            ..
                        },
                    ) => l1 == l2 && (!with_times || m1 == m2),
                    (EntryKind::Dir | EntryKind::Link { .. } | EntryKind::File { .. }, _) => false,
                }
        })
}

/// Whether the staged file at `staged` can be trusted as a verified copy: a
/// regular file with the source's length and modification time, which only a
/// verified copy is given before it takes its final name.
fn staged_matches(staged: &Path, len: u64, modified: Option<SystemTime>) -> bool {
    std::fs::symlink_metadata(staged).is_ok_and(|meta| {
        meta.is_file()
            && meta.len() == len
            && modified.is_some()
            && meta.modified().ok() == modified
    })
}

/// Puts back the cache entry an interrupted move set aside, removing the link
/// that does not resolve in its place, if there is one.
fn restore_aside(plan: &MovePlan) -> Result<(), FetchError> {
    if std::fs::symlink_metadata(&plan.entry).is_ok_and(|m| m.file_type().is_symlink()) {
        remove_dir_link(&plan.entry).map_err(|e| FetchError::Io {
            // BORROW: explicit .clone() for owned PathBuf
            path: plan.entry.clone(),
            source: e,
        })?;
    }
    std::fs::rename(&plan.aside, &plan.entry).map_err(|e| FetchError::Io {
        // BORROW: explicit .clone() for owned PathBuf
        path: plan.aside.clone(),
        source: rename_error(e),
    })
}

/// Creates the destination's parent and the staging directory, and checks,
/// before any byte is copied, that the destination can hold a link and (when
/// the repo has hard links) a hard link. Returns whether hard links work.
fn prepare(plan: &MovePlan) -> Result<bool, FetchError> {
    let io = |path: &Path, e: std::io::Error| FetchError::Io {
        // BORROW: explicit .to_path_buf() for owned PathBuf
        path: path.to_path_buf(),
        source: e,
    };
    std::fs::create_dir_all(&plan.staging).map_err(|e| io(&plan.staging, e))?;

    if !matches!(plan.kind, MoveKind::Back { .. }) {
        let probe = move_dir_child(&plan.cache_dir, &plan.folder, PROBE_SUFFIX);
        if let Some(parent) = probe.parent() {
            std::fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
        }
        // A probe an interrupted run left is a link of ours; anything else at
        // that name is not, and makes the creation below fail.
        if std::fs::symlink_metadata(&probe).is_ok_and(|m| m.file_type().is_symlink()) {
            remove_dir_link(&probe).map_err(|e| io(&probe, e))?;
        }
        let target = plan.destination.parent().unwrap_or(&plan.staging);
        create_dir_link(target, &probe).map_err(|e| {
            FetchError::InvalidArgument(format!(
                "{} cannot be linked to from the cache ({e}); choose a local volume that supports {}",
                target.display(),
                if cfg!(windows) { "junctions" } else { "symlinks" }
            ))
        })?;
        remove_dir_link(&probe).map_err(|e| io(&probe, e))?;
    }

    if plan.hard_links == 0 {
        return Ok(true);
    }
    let probe_file = plan.staging.join(".hf-fm-probe");
    let probe_link = plan.staging.join(".hf-fm-probe.link");
    let _ = std::fs::remove_file(&probe_link);
    std::fs::write(&probe_file, b"").map_err(|e| io(&probe_file, e))?;
    let works = std::fs::hard_link(&probe_file, &probe_link).is_ok();
    let _ = std::fs::remove_file(&probe_link);
    let _ = std::fs::remove_file(&probe_file);
    if !works {
        // The hard-linked names become copies: they need room too.
        let extra = plan
            .entries
            .iter()
            .filter_map(|e| match e.kind {
                EntryKind::File {
                    len,
                    link_to: Some(_),
                    ..
                } => Some(len),
                EntryKind::File { .. } | EntryKind::Dir | EntryKind::Link { .. } => None,
            })
            .fold(0_u64, u64::saturating_add);
        let needed = plan.bytes_to_copy().saturating_add(extra);
        if let Some(available) = fs2::available_space(&plan.staging).ok()
            && available < needed
        {
            return Err(FetchError::InvalidArgument(format!(
                "not enough free space at {} for separate copies of the hard-linked files \
                 (its volume cannot hard-link; needs {needed} bytes, {available} available)",
                plan.staging.display()
            )));
        }
    }
    Ok(works)
}

/// Removes from the staging directory whatever the repo does not hold (files
/// it no longer has, an interrupted copy's temp files), so the staged tree
/// becomes the repo's exactly.
fn prune_staging(plan: &MovePlan) -> Result<(), FetchError> {
    let staged = walk(&plan.staging)?;
    let wanted: HashMap<&Path, &EntryKind> = plan
        .entries
        .iter()
        .map(|e| (e.rel.as_path(), &e.kind))
        .collect();
    for e in &staged {
        let keep = match (wanted.get(e.rel.as_path()), &e.kind) {
            (Some(EntryKind::Dir), EntryKind::Dir)
            | (Some(EntryKind::File { .. }), EntryKind::File { .. }) => true,
            (Some(EntryKind::Link { target: want }), EntryKind::Link { target }) => want == target,
            _ => false,
        };
        if !keep {
            let path = plan.staging.join(&e.rel);
            remove_any(&path).map_err(|err| FetchError::Io { path, source: err })?;
        }
    }
    Ok(())
}

/// Copies every entry of `plan` into its staging directory.
fn copy_all<F>(
    plan: &MovePlan,
    hard_links_work: bool,
    report: &mut MoveReport,
    on_event: &mut F,
) -> Result<(), FetchError>
where
    F: FnMut(MoveEvent<'_>),
{
    let total_files = plan.files;
    let to_copy = plan
        .entries
        .iter()
        .filter_map(|e| match e.kind {
            // A hard-linked name is linked, not copied, where links work.
            EntryKind::File {
                link_to: Some(_), ..
            } if hard_links_work => None,
            EntryKind::File { len, modified, .. }
                if !staged_matches(&plan.staging.join(&e.rel), len, modified) =>
            {
                Some(len)
            }
            EntryKind::File { .. } | EntryKind::Dir | EntryKind::Link { .. } => None,
        })
        .fold(0_u64, u64::saturating_add);
    on_event(MoveEvent::Started {
        files: total_files,
        bytes: to_copy,
    });
    let progress_total = to_copy.saturating_mul(2);
    let mut done: u64 = 0;
    let mut buf = vec![0_u8; COPY_BLOCK];
    let mut index: usize = 0;

    // EXPLICIT: carries the file index, the progress counter and the report across entries
    for e in &plan.entries {
        let src = plan.source.join(&e.rel);
        let dst = plan.staging.join(&e.rel);
        let io = |path: &Path, err: std::io::Error| FetchError::Io {
            // BORROW: explicit .to_path_buf() for owned PathBuf
            path: path.to_path_buf(),
            source: err,
        };
        match &e.kind {
            EntryKind::Dir => std::fs::create_dir_all(&dst).map_err(|err| io(&dst, err))?,
            EntryKind::Link { target } => {
                if std::fs::read_link(&dst).ok().as_ref() != Some(target) {
                    if std::fs::symlink_metadata(&dst).is_ok() {
                        remove_any(&dst).map_err(|err| io(&dst, err))?;
                    }
                    create_file_or_dir_symlink(&src, target, &dst).map_err(|err| io(&dst, err))?;
                }
                report.symlinks += 1;
            }
            EntryKind::File {
                len,
                modified,
                link_to,
            } => {
                index += 1;
                on_event(MoveEvent::File {
                    index,
                    total: total_files,
                    path: &e.rel,
                    len: *len,
                });
                let first = link_to
                    .and_then(|i| plan.entries.get(i))
                    .map(|first| plan.staging.join(&first.rel));
                if let (Some(first), true) = (first, hard_links_work) {
                    if !one_physical_file(&first, &dst) {
                        if std::fs::symlink_metadata(&dst).is_ok() {
                            remove_any(&dst).map_err(|err| io(&dst, err))?;
                        }
                        std::fs::hard_link(&first, &dst).map_err(|err| io(&dst, err))?;
                    }
                    report.hard_links += 1;
                    continue;
                }
                if link_to.is_some() {
                    report.hard_links_copied += 1;
                }
                if staged_matches(&dst, *len, *modified) {
                    report.resumed_files += 1;
                    report.resumed_bytes = report.resumed_bytes.saturating_add(*len);
                    continue;
                }
                let digest = copy_verified(&src, &dst, *modified, &mut buf, |n| {
                    done = done.saturating_add(n);
                    on_event(MoveEvent::Progress {
                        done,
                        total: progress_total,
                    });
                })?;
                report.copied_files += 1;
                report.copied_bytes = report.copied_bytes.saturating_add(*len);
                if blob_name_mismatch(&e.rel, &digest) {
                    // BORROW: explicit .clone() for owned PathBuf
                    report.name_mismatches.push(e.rel.clone());
                }
            }
        }
    }
    Ok(())
}

/// Whether the files at `a` and `b` are one physical file (two hard links),
/// neither being a link; `false` when either cannot be read or the platform
/// has no file identity.
fn one_physical_file(a: &Path, b: &Path) -> bool {
    let id = |p: &Path| {
        std::fs::symlink_metadata(p)
            .ok()
            .filter(std::fs::Metadata::is_file)
            .and_then(|meta| physical_id(p, &meta))
    };
    id(a).is_some_and(|a| Some(a) == id(b))
}

/// Whether `rel` is a blob (`blobs/<name>`) named by a `SHA256` that is not
/// `digest`.
fn blob_name_mismatch(rel: &Path, digest: &str) -> bool {
    let mut components = rel.components();
    let (Some(Component::Normal(dir)), Some(Component::Normal(name)), None) =
        (components.next(), components.next(), components.next())
    else {
        return false;
    };
    dir == "blobs"
        && name
            .to_str()
            .is_some_and(|name| super::is_sha256_hex(name) && name != digest)
}

/// Copies `src` to `dst` through a temporary name beside it, hashing the bytes
/// as they are copied; flushes the copy, reads it back and hashes it, and
/// gives it its final name, the source's modification time and permissions
/// only when the two digests agree. Returns the digest, in lowercase hex.
/// `on_bytes` is told every block copied and every block read back.
fn copy_verified<P>(
    src: &Path,
    dst: &Path,
    modified: Option<SystemTime>,
    buf: &mut [u8],
    mut on_bytes: P,
) -> Result<String, FetchError>
where
    P: FnMut(u64),
{
    let io = |path: &Path, e: std::io::Error| FetchError::Io {
        // BORROW: explicit .to_path_buf() for owned PathBuf
        path: path.to_path_buf(),
        source: e,
    };
    let Some(name) = dst.file_name() else {
        return Err(FetchError::InvalidArgument(format!(
            "{} has no file name",
            dst.display()
        )));
    };
    // BORROW: explicit .to_os_string() for &OsStr → owned OsString to append to
    let mut temp_name = name.to_os_string();
    temp_name.push(COPY_TEMP_SUFFIX);
    let temp = dst.with_file_name(temp_name);

    let mut input = std::fs::File::open(src).map_err(|e| io(src, e))?;
    let mut output = std::fs::File::create(&temp).map_err(|e| io(&temp, e))?;
    let mut hasher = Sha256::new();
    // EXPLICIT: streams blocks until the end of the file, feeding the hasher and the copy
    loop {
        let n = read_full(&mut input, buf).map_err(|e| io(src, e))?;
        let Some(block) = buf.get(..n) else { break };
        if block.is_empty() {
            break;
        }
        hasher.update(block);
        output.write_all(block).map_err(|e| io(&temp, e))?;
        // CAST: usize → u64, a buffer length always fits
        #[allow(clippy::as_conversions)]
        on_bytes(n as u64);
    }
    output.sync_all().map_err(|e| io(&temp, e))?;
    drop(output);
    let source_digest = crate::checksum::lower_hex(&hasher.finalize());

    let mut readback = std::fs::File::open(&temp).map_err(|e| io(&temp, e))?;
    let mut hasher = Sha256::new();
    // EXPLICIT: streams blocks until the end of the file, feeding the hasher
    loop {
        let n = read_full(&mut readback, buf).map_err(|e| io(&temp, e))?;
        let Some(block) = buf.get(..n) else { break };
        if block.is_empty() {
            break;
        }
        hasher.update(block);
        // CAST: usize → u64, a buffer length always fits
        #[allow(clippy::as_conversions)]
        on_bytes(n as u64);
    }
    drop(readback);
    let copy_digest = crate::checksum::lower_hex(&hasher.finalize());
    if copy_digest != source_digest {
        let _ = std::fs::remove_file(&temp);
        return Err(FetchError::Checksum {
            filename: dst.display().to_string(),
            expected: source_digest,
            actual: copy_digest,
        });
    }

    if let Some(modified) = modified {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&temp)
            .map_err(|e| io(&temp, e))?;
        file.set_modified(modified).map_err(|e| io(&temp, e))?;
    }
    std::fs::rename(&temp, dst).map_err(|e| io(dst, e))?;
    let permissions = std::fs::metadata(src)
        .map_err(|e| io(src, e))?
        .permissions();
    std::fs::set_permissions(dst, permissions).map_err(|e| io(dst, e))?;
    Ok(source_digest)
}

/// Swaps the cache entry: sets it aside, then puts the link (or, for a move
/// back, the staged directory) in its place, and gives the staging directory
/// its final name. A failure puts the entry back.
fn swap(plan: &MovePlan) -> Result<(), FetchError> {
    let io = |path: &Path, e: std::io::Error| FetchError::Io {
        // BORROW: explicit .to_path_buf() for owned PathBuf
        path: path.to_path_buf(),
        source: e,
    };
    if let Some(parent) = plan.aside.parent() {
        std::fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
    }
    std::fs::rename(&plan.entry, &plan.aside).map_err(|e| in_use_error(plan, e))?;

    let swapped = if matches!(plan.kind, MoveKind::Back { .. }) {
        std::fs::rename(&plan.staging, &plan.entry).map_err(|e| io(&plan.entry, e))
    } else {
        create_dir_link(&plan.destination, &plan.entry)
            .map_err(|e| io(&plan.entry, e))
            .and_then(|()| {
                std::fs::rename(&plan.staging, &plan.destination).map_err(|e| {
                    let _ = remove_dir_link(&plan.entry);
                    io(&plan.destination, e)
                })
            })
    };
    if let Err(e) = swapped {
        let _ = std::fs::rename(&plan.aside, &plan.entry);
        return Err(e);
    }
    Ok(())
}

/// The error for a cache entry that cannot be set aside: "in use" when the
/// system says so, which on Windows is a file open inside the directory.
fn in_use_error(plan: &MovePlan, e: std::io::Error) -> FetchError {
    let e = rename_error(e);
    if e.kind() == std::io::ErrorKind::ResourceBusy {
        return FetchError::Io {
            // BORROW: explicit .clone() for owned PathBuf
            path: plan.entry.clone(),
            source: std::io::Error::new(
                std::io::ErrorKind::ResourceBusy,
                format!(
                    "in use by another program (close it and run again; the copy at {} is kept and resumes): {e}",
                    plan.staging.display()
                ),
            ),
        };
    }
    FetchError::Io {
        // BORROW: explicit .clone() for owned PathBuf
        path: plan.entry.clone(),
        source: e,
    }
}

/// Checks the swapped entry: the link resolves to the destination, and the
/// tree read through the entry is the one copied. A mismatch undoes the swap,
/// keeping the staged copy.
fn check_swapped(plan: &MovePlan) -> Result<(), FetchError> {
    let links_right = matches!(plan.kind, MoveKind::Back { .. })
        || super::relocated_target(&plan.entry).is_some_and(|t| same_dir(&t, &plan.destination));
    let reads_right = walk(&plan.entry).is_ok_and(|tree| same_tree(&plan.entries, &tree, true));
    if links_right && reads_right {
        return Ok(());
    }
    if matches!(plan.kind, MoveKind::Back { .. }) {
        let _ = std::fs::rename(&plan.entry, &plan.staging);
    } else {
        let _ = remove_dir_link(&plan.entry);
        let _ = std::fs::rename(&plan.destination, &plan.staging);
    }
    let _ = std::fs::rename(&plan.aside, &plan.entry);
    Err(FetchError::InvalidArgument(format!(
        "{} did not read back as copied through {} (the move was undone; the copy is kept at {})",
        plan.repo_type.address(&plan.repo_id),
        plan.entry.display(),
        plan.staging.display()
    )))
}

/// Drops or keeps the original the swap set aside, and says what became of
/// it. Never fails: what cannot be done is reported as
/// [`MoveOriginal::Left`].
fn settle_aside<F>(plan: &MovePlan, keep_original: bool, on_event: &mut F) -> MoveOriginal
where
    F: FnMut(MoveEvent<'_>),
{
    let address = plan.repo_type.address(&plan.repo_id);
    let left = |path: &Path, reason: String| MoveOriginal::Left {
        // BORROW: explicit .to_path_buf() for owned PathBuf
        path: path.to_path_buf(),
        reason,
    };
    let Ok(meta) = std::fs::symlink_metadata(&plan.aside) else {
        // BORROW: explicit .to_owned() for &str → owned String
        return left(&plan.aside, "it cannot be read".to_owned());
    };

    if !meta.file_type().is_symlink() {
        // The original was a directory in the cache.
        if keep_original {
            let kept = kept_path(&plan.cache_dir, &plan.folder);
            if std::fs::symlink_metadata(&kept).is_ok() {
                return left(
                    &plan.aside,
                    format!("a kept original is already at {}", kept.display()),
                );
            }
            return match std::fs::rename(&plan.aside, &kept) {
                Ok(()) => MoveOriginal::Kept { path: kept },
                Err(e) => left(&plan.aside, e.to_string()),
            };
        }
        on_event(MoveEvent::Dropping { path: &plan.aside });
        return match std::fs::remove_dir_all(&plan.aside) {
            Ok(()) => MoveOriginal::Dropped {
                // BORROW: explicit .clone() for owned PathBuf
                path: plan.aside.clone(),
            },
            Err(e) => left(&plan.aside, e.to_string()),
        };
    }

    // The original was the directory a link pointed to.
    let target = match guarded_target(&plan.aside, &plan.folder, &address) {
        Ok(target) => target,
        Err(e) => return left(&plan.aside, e.to_string()),
    };
    // Never the directory the entry now resolves to.
    if std::fs::canonicalize(&plan.entry)
        .map(strip_verbatim_prefix)
        .is_ok_and(|now| now == target)
    {
        let _ = remove_dir_link(&plan.aside);
        return MoveOriginal::Kept { path: target };
    }
    if keep_original {
        return match remove_dir_link(&plan.aside) {
            Ok(()) => MoveOriginal::Kept { path: target },
            Err(e) => left(&plan.aside, e.to_string()),
        };
    }
    on_event(MoveEvent::Dropping { path: &target });
    if let Err(e) = std::fs::remove_dir_all(&target) {
        return left(&target, e.to_string());
    }
    match remove_dir_link(&plan.aside) {
        Ok(()) => MoveOriginal::Dropped { path: target },
        Err(e) => left(&plan.aside, e.to_string()),
    }
}

/// Creates a link at `link` to the directory `target`: a junction on Windows
/// (no privilege needed, local volumes only), a symlink on Unix.
///
/// `junction::create` makes the directory before it writes the reparse point
/// and leaves it empty when that fails, so it is removed here, unless the
/// failure was that `link` existed already.
fn create_dir_link(target: &Path, link: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        junction::create(target, link).inspect_err(|e| {
            if e.kind() != std::io::ErrorKind::AlreadyExists {
                let _ = std::fs::remove_dir(link);
            }
        })
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (target, link);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "directory links are not supported on this platform",
        ))
    }
}

/// Removes the directory link at `link` without touching what it points at:
/// as a directory on Windows (a junction or a directory symlink), as a file on
/// Unix.
fn remove_dir_link(link: &Path) -> std::io::Result<()> {
    if cfg!(windows) {
        std::fs::remove_dir(link)
    } else {
        std::fs::remove_file(link)
    }
}

/// Recreates at `dst` the link the repo holds at `src`, pointing to `target`.
/// On Windows a link to a directory must be made as one; a target that does
/// not resolve is made as a file link.
fn create_file_or_dir_symlink(src: &Path, target: &Path, dst: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        if std::fs::metadata(src).is_ok_and(|m| m.is_dir()) {
            std::os::windows::fs::symlink_dir(target, dst)
        } else {
            std::os::windows::fs::symlink_file(target, dst)
        }
    }
    #[cfg(unix)]
    {
        let _ = src;
        std::os::unix::fs::symlink(target, dst)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (src, target, dst);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "symbolic links are not supported on this platform",
        ))
    }
}

/// Removes whatever is at `path`: a directory with its contents, a link (never
/// followed), or a file.
fn remove_any(path: &Path) -> std::io::Result<()> {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    if meta.file_type().is_symlink() {
        // A file link is removed as a file everywhere; a directory link on
        // Windows only as a directory.
        return std::fs::remove_file(path).or_else(|_| std::fs::remove_dir(path));
    }
    if meta.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
}

/// An identity for the volume holding `path`: the device number on Unix; on
/// Windows the volume serial number together with the drive (or share) the
/// canonical path names. `None` where it cannot be read.
///
/// The serial number alone does not tell two volumes apart: a volume cloned
/// from another keeps its serial number, and the maintainer's `C:` and `D:`
/// share one (2026-11-07). The drive tells those apart, and the serial number
/// a volume mounted in a folder of the same drive.
fn volume_id(path: &Path) -> Option<(u64, String)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(path)
            .ok()
            .map(|meta| (meta.dev(), String::new()))
    }
    #[cfg(windows)]
    {
        let handle = winapi_util::Handle::from_path_any(path).ok()?;
        let serial = winapi_util::file::information(&handle)
            .ok()?
            .volume_serial_number();
        let drive = super::volume_label(&std::fs::canonicalize(path).ok()?)?;
        Some((serial, drive))
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = path;
        None
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

    /// Name of the fixture repo's cache folder.
    const FOLDER: &str = "models--org--model";

    /// A temp root holding `cache/` with one repo, `models--org--model`, and
    /// the path of a destination directory that does not exist yet.
    struct Fixture {
        _tmp: tempfile::TempDir,
        cache: PathBuf,
        dest: PathBuf,
        repo: PathBuf,
        blob: PathBuf,
        content: Vec<u8>,
    }

    /// Bytes longer than one copy block, so the copy loop runs more than once.
    fn content() -> Vec<u8> {
        // CAST: usize → u8, deliberately wrapping into a byte pattern
        (0..COPY_BLOCK + 4097).map(|i| (i % 251) as u8).collect()
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        crate::checksum::lower_hex(&Sha256::digest(bytes))
    }

    /// The fixture: a blob named by its `SHA256`, a snapshot copy of it, a
    /// small git-stored file, and `refs/main`.
    fn fixture() -> Fixture {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");
        let dest = tmp.path().join("dest");
        let repo = cache.join(FOLDER);
        let content = content();
        let blobs = crate::cache_layout::blobs_dir(&repo);
        let snap = crate::cache_layout::snapshot_dir(&repo, "c0ffee");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snap).unwrap();
        let blob = blobs.join(sha256_hex(&content));
        std::fs::write(&blob, &content).unwrap();
        std::fs::copy(&blob, snap.join("model.bin")).unwrap();
        std::fs::write(snap.join("config.json"), b"{}").unwrap();
        std::fs::create_dir_all(crate::cache_layout::refs_dir(&repo)).unwrap();
        std::fs::write(crate::cache_layout::ref_path(&repo, "main"), "c0ffee").unwrap();
        Fixture {
            _tmp: tmp,
            cache,
            dest,
            repo,
            blob,
            content,
        }
    }

    fn plan(f: &Fixture, dir: &Path) -> Result<MovePlan, FetchError> {
        plan_in(&f.cache, RepoType::Model, "org/model", Some(dir))
    }

    fn plan_back(f: &Fixture) -> Result<MovePlan, FetchError> {
        plan_in(&f.cache, RepoType::Model, "org/model", None)
    }

    /// Every file under `root` (following `root` itself), relative path to
    /// bytes, links read as their target.
    fn contents(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
        walk(root)
            .unwrap()
            .into_iter()
            .filter_map(|e| match e.kind {
                EntryKind::File { .. } => {
                    Some((e.rel.clone(), std::fs::read(root.join(&e.rel)).unwrap()))
                }
                EntryKind::Link { target } => {
                    Some((e.rel, target.to_string_lossy().into_owned().into_bytes()))
                }
                EntryKind::Dir => None,
            })
            .collect()
    }

    fn modified(path: &Path) -> SystemTime {
        std::fs::metadata(path).unwrap().modified().unwrap()
    }

    /// Sets the modification time of `path` to an hour ago.
    fn age(path: &Path) {
        let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        file.set_modified(SystemTime::now() - std::time::Duration::from_secs(3600))
            .unwrap();
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

    /// A relative path built from components, so it carries the native
    /// separator (Windows cannot follow a symlink target written with `/`).
    fn rel(parts: &[&str]) -> PathBuf {
        parts.iter().collect()
    }

    #[test]
    fn relocate_copies_links_and_drops_the_original() {
        let f = fixture();
        age(&f.blob);
        let before = contents(&f.repo);
        let blob_time = modified(&f.blob);

        let p = plan(&f, &f.dest).unwrap();
        assert_eq!(p.kind, MoveKind::Relocate);
        assert!(p.creates_parent);
        assert_eq!(p.files, 4);
        assert_eq!(p.bytes, 2 * f.content.len() as u64 + 2 + 6);
        assert_eq!(p.bytes_staged, 0);
        assert!(p.same_volume, "a temp dir is on one volume");
        assert_eq!(p.destination.file_name().unwrap(), FOLDER);

        let report = apply_move(&p, false).unwrap();
        assert_eq!(report.copied_files, 4);
        assert_eq!(report.resumed_files, 0);
        assert!(
            report.name_mismatches.is_empty(),
            "{:?}",
            report.name_mismatches
        );
        assert!(matches!(
            report.original,
            Some(MoveOriginal::Dropped { .. })
        ));

        let target = super::super::relocated_target(&f.repo).expect("the entry is a link");
        assert!(same_dir(&target, &p.destination));
        assert_eq!(contents(&f.repo), before, "the same files through the link");
        assert_eq!(
            contents(&p.destination),
            before,
            "the same files at the destination"
        );
        let moved_blob = p
            .destination
            .join("blobs")
            .join(f.blob.file_name().unwrap());
        assert_eq!(
            modified(&moved_blob),
            blob_time,
            "the modification time is kept"
        );
        assert!(
            !f.cache.join(MOVE_DIR_NAME).exists(),
            "nothing left in the cache"
        );
        assert!(!f.dest.join(MOVE_DIR_NAME).exists(), "no staging left");
    }

    #[test]
    fn keep_original_keeps_it_in_the_move_dir() {
        let f = fixture();
        let before = contents(&f.repo);
        let p = plan(&f, &f.dest).unwrap();
        let report = apply_move(&p, true).unwrap();
        let kept = kept_path(&f.cache, FOLDER);
        assert_eq!(
            report.original,
            Some(MoveOriginal::Kept { path: kept.clone() })
        );
        assert_eq!(contents(&kept), before);
        assert!(super::super::relocated_target(&f.repo).is_some());
        assert_eq!(
            super::super::dir_bytes(&f.cache.join(MOVE_DIR_NAME)),
            p.bytes,
            "what `du` reports apart"
        );
    }

    #[test]
    fn keep_original_refuses_when_one_is_already_kept() {
        let f = fixture();
        std::fs::create_dir_all(kept_path(&f.cache, FOLDER)).unwrap();
        let p = plan(&f, &f.dest).unwrap();
        let err = apply_move(&p, true).unwrap_err();
        assert!(err.to_string().contains("already at"), "{err}");
        assert!(!p.staging.exists(), "refused before copying");
        assert!(super::super::relocated_target(&f.repo).is_none());
    }

    #[test]
    fn a_hard_linked_pointer_stays_one_file() {
        let f = fixture();
        let pointer = crate::cache_layout::pointer_path(&f.repo, "c0ffee", "model.bin");
        std::fs::remove_file(&pointer).unwrap();
        std::fs::hard_link(&f.blob, &pointer).unwrap();

        let p = plan(&f, &f.dest).unwrap();
        assert_eq!(p.hard_links, 1);
        assert_eq!(
            p.bytes,
            f.content.len() as u64 + 2 + 6,
            "the linked bytes count once"
        );
        let report = apply_move(&p, false).unwrap();
        assert_eq!(report.hard_links, 1);
        assert_eq!(report.copied_files, 3);

        let moved_blob = p
            .destination
            .join("blobs")
            .join(f.blob.file_name().unwrap());
        let moved_pointer =
            crate::cache_layout::pointer_path(&p.destination, "c0ffee", "model.bin");
        assert!(
            one_physical_file(&moved_blob, &moved_pointer),
            "still one physical file"
        );
        assert_eq!(std::fs::read(&moved_pointer).unwrap(), f.content);
    }

    #[test]
    fn a_symlinked_pointer_is_recreated() {
        let f = fixture();
        let pointer = crate::cache_layout::pointer_path(&f.repo, "c0ffee", "model.bin");
        std::fs::remove_file(&pointer).unwrap();
        let name = f.blob.file_name().unwrap().to_str().unwrap();
        let target = rel(&["..", "..", "blobs", name]);
        if !try_symlink(&target, &pointer) {
            eprintln!("skipped: this platform refuses symlinks");
            return;
        }
        let p = plan(&f, &f.dest).unwrap();
        assert_eq!(p.symlinks, 1);
        let report = apply_move(&p, false).unwrap();
        assert_eq!(report.symlinks, 1);
        let moved = crate::cache_layout::pointer_path(&p.destination, "c0ffee", "model.bin");
        assert_eq!(std::fs::read_link(&moved).unwrap(), target);
        assert_eq!(
            std::fs::read(&moved).unwrap(),
            f.content,
            "it resolves at the destination"
        );
    }

    #[test]
    fn a_link_into_the_shared_store_is_refused() {
        let f = fixture();
        let store = crate::cache_layout::shared_blobs_dir(&f.cache).join("9a");
        std::fs::create_dir_all(&store).unwrap();
        std::fs::write(store.join("payload"), b"x").unwrap();
        let link = crate::cache_layout::blobs_dir(&f.repo).join("etag");
        if !try_symlink(&rel(&["..", "..", "blobs", "9a", "payload"]), &link) {
            eprintln!("skipped: this platform refuses symlinks");
            return;
        }
        let err = plan(&f, &f.dest).unwrap_err();
        assert!(err.to_string().contains("pointing outside"), "{err}");
    }

    /// Two drives are two volumes even when they share a serial number, as a
    /// cloned disk does (the maintainer's `C:` and `D:` both report
    /// `84AB-E302`, which made the first build call a move from `C:` to `D:`
    /// a same-volume move).
    #[cfg(windows)]
    #[test]
    fn volume_id_tells_drives_apart() {
        let (c, d) = (Path::new("C:\\"), Path::new("D:\\"));
        if !d.exists() {
            eprintln!("skipped: no D: drive");
            return;
        }
        let (Some(c_id), Some(d_id)) = (volume_id(c), volume_id(d)) else {
            panic!("both drives have an identity");
        };
        assert_ne!(c_id, d_id);
        assert_eq!(
            volume_id(Path::new("C:\\Windows")),
            Some(c_id),
            "a directory on a drive has the drive's identity"
        );
    }

    #[test]
    fn link_escapes_reads_the_target_against_the_link_s_place() {
        let snap = rel(&["snapshots", "c0ffee", "model.bin"]);
        assert!(!link_escapes(&snap, &rel(&["..", "..", "blobs", "abc"])));
        assert!(link_escapes(
            &snap,
            &rel(&["..", "..", "..", "blobs", "abc"])
        ));
        let blob = rel(&["blobs", "etag"]);
        assert!(
            link_escapes(&blob, &rel(&["..", "..", "blobs", "9a", "x"])),
            "the shared store"
        );
        assert!(!link_escapes(&blob, Path::new("other")));
        assert!(
            link_escapes(&blob, &std::env::temp_dir()),
            "an absolute target"
        );
    }

    #[test]
    fn refusals_leave_everything_as_it_was() {
        let f = fixture();
        // Not cached.
        let err = plan_in(&f.cache, RepoType::Model, "org/other", Some(&f.dest)).unwrap_err();
        assert!(err.to_string().contains("not cached"), "{err}");
        // Not behind a link, so nothing to bring back.
        let err = plan_back(&f).unwrap_err();
        assert!(err.to_string().contains("not behind a link"), "{err}");
        // A destination inside the cache.
        let err = plan(&f, &f.cache.join("elsewhere")).unwrap_err();
        assert!(err.to_string().contains("inside the cache"), "{err}");
        // A destination that exists already.
        std::fs::create_dir_all(f.dest.join(FOLDER)).unwrap();
        let err = plan(&f, &f.dest).unwrap_err();
        assert!(err.to_string().contains("already exists"), "{err}");
        std::fs::remove_dir(f.dest.join(FOLDER)).unwrap();
        // A partial download.
        let partial = crate::cache_layout::blobs_dir(&f.repo).join("abc.incomplete");
        std::fs::write(&partial, b"x").unwrap();
        let err = plan(&f, &f.dest).unwrap_err();
        assert!(err.to_string().contains("in progress"), "{err}");
        std::fs::remove_file(&partial).unwrap();
        // An unsafe ID.
        assert!(plan_in(&f.cache, RepoType::Model, "../x", Some(&f.dest)).is_err());

        assert!(super::super::relocated_target(&f.repo).is_none());
        assert!(f.repo.is_dir());
    }

    #[test]
    fn a_destination_that_appears_after_the_plan_is_refused_before_copying() {
        let f = fixture();
        let p = plan(&f, &f.dest).unwrap();
        std::fs::create_dir_all(f.dest.join(FOLDER)).unwrap();
        let err = apply_move(&p, false).unwrap_err();
        assert!(err.to_string().contains("already exists"), "{err}");
        assert!(f.repo.is_dir() && super::super::relocated_target(&f.repo).is_none());
        assert!(!p.staging.exists());
    }

    #[test]
    fn a_resumed_move_copies_only_what_is_missing() {
        let f = fixture();
        let p = plan(&f, &f.dest).unwrap();
        // An interrupted run: the blob staged and verified (its time set), a
        // `refs/main` of the same length but other content and an older time,
        // a file the repo no longer holds, and a temp file.
        let staged_blob = p.staging.join("blobs").join(f.blob.file_name().unwrap());
        std::fs::create_dir_all(staged_blob.parent().unwrap()).unwrap();
        std::fs::copy(&f.blob, &staged_blob).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&staged_blob)
            .unwrap()
            .set_modified(modified(&f.blob))
            .unwrap();
        let staged_ref = p.staging.join("refs").join("main");
        std::fs::create_dir_all(staged_ref.parent().unwrap()).unwrap();
        std::fs::write(&staged_ref, "deadbe").unwrap();
        age(&staged_ref);
        std::fs::write(p.staging.join("gone.bin"), b"old").unwrap();
        std::fs::write(p.staging.join(format!("x{COPY_TEMP_SUFFIX}")), b"half").unwrap();

        let p = plan(&f, &f.dest).unwrap();
        assert_eq!(p.bytes_staged, f.content.len() as u64, "the blob only");
        let before = contents(&f.repo);
        let report = apply_move(&p, false).unwrap();
        assert_eq!(report.resumed_files, 1);
        assert_eq!(report.resumed_bytes, f.content.len() as u64);
        assert_eq!(
            report.copied_files, 3,
            "the copy, the config and the stale ref"
        );
        assert_eq!(
            contents(&p.destination),
            before,
            "the stale ref is replaced, the extras gone"
        );
    }

    #[test]
    fn a_blob_not_matching_its_name_moves_and_is_reported() {
        let f = fixture();
        let wrong = crate::cache_layout::blobs_dir(&f.repo).join(sha256_hex(b"other bytes"));
        std::fs::write(&wrong, b"these bytes").unwrap();
        let p = plan(&f, &f.dest).unwrap();
        let report = apply_move(&p, false).unwrap();
        assert_eq!(
            report.name_mismatches,
            vec![rel(&[
                "blobs",
                wrong.file_name().unwrap().to_str().unwrap()
            ])]
        );
        assert_eq!(
            std::fs::read(p.destination.join("blobs").join(wrong.file_name().unwrap())).unwrap(),
            b"these bytes"
        );
    }

    #[test]
    fn a_relocated_repo_moves_again_and_its_old_place_is_dropped() {
        let f = fixture();
        let before = contents(&f.repo);
        let first = plan(&f, &f.dest).unwrap();
        let _ = apply_move(&first, false).unwrap();

        let second_dir = f.dest.parent().unwrap().join("dest2");
        let p = plan(&f, &second_dir).unwrap();
        assert!(
            matches!(&p.kind, MoveKind::MoveAgain { from } if same_dir(from, &first.destination)),
            "{:?}",
            p.kind
        );
        let report = apply_move(&p, false).unwrap();
        assert!(matches!(
            report.original,
            Some(MoveOriginal::Dropped { .. })
        ));
        let target = super::super::relocated_target(&f.repo).unwrap();
        assert!(same_dir(&target, &p.destination));
        assert!(!first.destination.exists(), "the old place is deleted");
        assert_eq!(contents(&f.repo), before);
    }

    #[test]
    fn moving_to_where_it_already_is_is_refused() {
        let f = fixture();
        let p = plan(&f, &f.dest).unwrap();
        let _ = apply_move(&p, false).unwrap();
        let err = plan(&f, &f.dest).unwrap_err();
        assert!(err.to_string().contains("already at"), "{err}");
    }

    #[test]
    fn back_restores_a_directory_with_the_same_files_and_times() {
        let f = fixture();
        age(&f.blob);
        let blob_time = modified(&f.blob);
        let before = contents(&f.repo);
        let away = plan(&f, &f.dest).unwrap();
        let _ = apply_move(&away, false).unwrap();

        let p = plan_back(&f).unwrap();
        assert!(matches!(p.kind, MoveKind::Back { .. }));
        let report = apply_move(&p, false).unwrap();
        assert!(matches!(
            report.original,
            Some(MoveOriginal::Dropped { .. })
        ));
        assert!(
            super::super::relocated_target(&f.repo).is_none(),
            "a directory again"
        );
        assert!(f.repo.is_dir());
        assert_eq!(contents(&f.repo), before);
        assert_eq!(
            modified(&f.blob),
            blob_time,
            "the modification time is kept"
        );
        assert!(!away.destination.exists(), "the moved copy is deleted");
        assert!(!f.cache.join(MOVE_DIR_NAME).exists());
    }

    #[test]
    fn back_with_keep_original_leaves_the_other_copy() {
        let f = fixture();
        let away = plan(&f, &f.dest).unwrap();
        let _ = apply_move(&away, false).unwrap();
        let p = plan_back(&f).unwrap();
        let report = apply_move(&p, true).unwrap();
        let Some(MoveOriginal::Kept { path }) = report.original else {
            panic!("{:?}", report.original);
        };
        assert!(same_dir(&path, &away.destination));
        assert!(f.repo.is_dir() && super::super::relocated_target(&f.repo).is_none());
        assert!(
            !f.cache.join(MOVE_DIR_NAME).exists(),
            "the aside link is removed"
        );
    }

    #[test]
    fn a_link_to_a_directory_named_otherwise_is_refused() {
        let f = fixture();
        let elsewhere = f.dest.parent().unwrap().join("not-the-folder");
        std::fs::rename(&f.repo, &elsewhere).unwrap();
        create_dir_link(&elsewhere, &f.repo).unwrap();
        for result in [plan(&f, &f.dest), plan_back(&f)] {
            let err = result.unwrap_err();
            assert!(err.to_string().contains("not a directory named"), "{err}");
        }
        assert!(elsewhere.join("blobs").is_dir(), "nothing was touched");
    }

    #[test]
    fn an_entry_set_aside_and_not_replaced_is_put_back_and_moved() {
        let f = fixture();
        let before = contents(&f.repo);
        let aside = move_dir_child(&f.cache, FOLDER, ASIDE_SUFFIX);
        std::fs::create_dir_all(aside.parent().unwrap()).unwrap();
        std::fs::rename(&f.repo, &aside).unwrap();
        // The crash came after the link was made: it does not resolve yet.
        create_dir_link(&f.dest.join(FOLDER), &f.repo).unwrap();

        let p = plan(&f, &f.dest).unwrap();
        assert_eq!(p.recovery, Some(MoveRecovery::RestoreAside));
        assert_eq!(p.kind, MoveKind::Relocate);
        let report = apply_move(&p, false).unwrap();
        assert_eq!(report.recovered, Some(MoveRecovery::RestoreAside));
        assert!(super::super::relocated_target(&f.repo).is_some());
        assert_eq!(contents(&f.repo), before);
        assert!(!aside.exists());
    }

    #[test]
    fn an_interrupted_drop_is_finished() {
        let f = fixture();
        let p = plan(&f, &f.dest).unwrap();
        let _ = apply_move(&p, true).unwrap();
        // The crash came during the drop: the original is still set aside.
        let aside = move_dir_child(&f.cache, FOLDER, ASIDE_SUFFIX);
        std::fs::rename(kept_path(&f.cache, FOLDER), &aside).unwrap();

        let again = plan(&f, &f.dest).unwrap();
        assert_eq!(again.recovery, Some(MoveRecovery::FinishDrop));
        assert_eq!(again.kind, MoveKind::Finish);
        let report = apply_move(&again, false).unwrap();
        assert!(matches!(
            report.original,
            Some(MoveOriginal::Dropped { .. })
        ));
        assert!(!aside.exists());
        assert!(!f.cache.join(MOVE_DIR_NAME).exists());
        assert!(
            super::super::relocated_target(&f.repo).is_some(),
            "the move stands"
        );
    }

    #[test]
    fn the_progress_events_add_up() {
        let f = fixture();
        let p = plan(&f, &f.dest).unwrap();
        let mut started = None;
        let mut last = (0, 0);
        let mut files = 0;
        let mut swapped = false;
        let _ = apply_move_with_progress(&p, false, |event| match event {
            MoveEvent::Started { bytes, .. } => started = Some(bytes),
            MoveEvent::File { .. } => files += 1,
            MoveEvent::Progress { done, total } => last = (done, total),
            MoveEvent::Swapping => swapped = true,
            MoveEvent::Dropping { .. } => {}
        })
        .unwrap();
        assert_eq!(started, Some(p.bytes));
        assert_eq!(
            last,
            (2 * p.bytes, 2 * p.bytes),
            "each byte copied and read back"
        );
        assert_eq!(files, 4);
        assert!(swapped);
    }
}
