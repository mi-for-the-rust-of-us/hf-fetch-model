// SPDX-License-Identifier: MIT OR Apache-2.0

//! Repository file listing via the `HuggingFace` API.
//!
//! This module provides functions to list all files in a `HuggingFace` model
//! repository, using the `hf-hub` crate's `info()` API and optionally
//! fetching extended metadata (sizes and SHA256 hashes) via a direct HTTP call.

use std::path::PathBuf;

use hf_hub::repository::{HFRepository, RepoTypeModel};
use serde::Deserialize;

use crate::error::FetchError;

/// A model repository handle paired with the revision its requests target.
///
/// `hf-hub` 0.5 baked the revision into the `Repo` value itself, so a handle
/// carried everything a call needed. In `hf-hub` 1.0 the revision moved to a
/// per-call builder argument (`.revision(...)`), which would otherwise force
/// every call site in this crate to thread it separately. Pairing the two back
/// together keeps the download pipeline's signatures as they were and gives
/// the revision exactly one place to live.
#[derive(Clone)]
pub struct ModelRepo {
    /// The underlying `hf-hub` model-repository handle (cheap to clone — it
    /// wraps an `Arc`-backed `HFClient`).
    inner: HFRepository<RepoTypeModel>,
    /// Git revision (branch, tag, or commit SHA) every request targets.
    /// `None` means the repository's default branch.
    revision: Option<String>,
}

impl ModelRepo {
    /// Wraps an `hf-hub` model repository together with its target revision.
    #[must_use]
    pub const fn new(inner: HFRepository<RepoTypeModel>, revision: Option<String>) -> Self {
        Self { inner, revision }
    }

    /// Returns the revision every request targets, or `None` for the default branch.
    #[must_use]
    pub fn revision(&self) -> Option<&str> {
        // BORROW: explicit .as_deref() for Option<String> → Option<&str>
        self.revision.as_deref()
    }

    /// Downloads one file into the `HuggingFace` cache and returns its path.
    ///
    /// # Errors
    ///
    /// Returns [`FetchError::Api`] if the download fails.
    pub async fn download_file(&self, filename: &str) -> Result<PathBuf, FetchError> {
        self.inner
            .download_file()
            // BORROW: explicit .to_owned() — the builder takes an owned String
            .filename(filename.to_owned())
            .maybe_revision(self.revision.clone())
            .send()
            .await
            .map_err(FetchError::Api)
    }
}

/// A file entry in a `HuggingFace` repository.
#[derive(Debug, Clone)]
pub struct RepoFile {
    /// The relative path of the file within the repository.
    pub filename: String,
    /// File size in bytes (if known from API metadata).
    pub size: Option<u64>,
    /// SHA256 hex digest (if the file is stored in LFS).
    pub sha256: Option<String>,
}

/// Lists all files in the given repository.
///
/// # Errors
///
/// Returns [`FetchError::Api`] if the `HuggingFace` API request fails.
/// Returns [`FetchError::RepoNotFound`] if the repository does not exist.
pub async fn list_repo_files(
    repo: &ModelRepo,
    repo_id: String,
) -> Result<Vec<RepoFile>, FetchError> {
    let info = repo
        .inner
        .info()
        .maybe_revision(repo.revision.clone())
        .send()
        .await
        .map_err(|e| {
            // `hf-hub` 1.0 reports a missing repo as a typed variant, so this
            // no longer has to sniff for "404" in the rendered message.
            if matches!(e, hf_hub::HFError::RepoNotFound { .. }) {
                FetchError::RepoNotFound { repo_id }
            } else {
                FetchError::Api(e)
            }
        })?;

    // `siblings` is `Option<Vec<_>>` in `hf-hub` 1.0 (it is absent rather than
    // empty when the info endpoint does not return a file listing).
    let files = info
        .siblings
        .unwrap_or_default()
        .into_iter()
        .map(|s| RepoFile {
            filename: s.rfilename,
            size: None,
            sha256: None,
        })
        .collect();

    Ok(files)
}

// --- Direct HF API metadata (for SHA256 and file sizes) ---

/// Raw JSON sibling entry from the `HuggingFace` API.
#[derive(Debug, Deserialize)]
struct ApiSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    lfs: Option<ApiLfs>,
}

/// LFS metadata attached to a sibling entry.
#[derive(Debug, Deserialize)]
struct ApiLfs {
    sha256: String,
    size: u64,
}

/// Raw JSON response from `GET /api/models/{repo_id}`.
#[derive(Debug, Deserialize)]
struct ApiModelInfo {
    siblings: Vec<ApiSibling>,
    /// Commit SHA of the resolved revision, when present in the API response.
    #[serde(default)]
    sha: Option<String>,
}

/// Fetches extended file metadata (sizes and SHA256 hashes) via the `HuggingFace` REST API.
///
/// This makes a direct HTTP call to `https://huggingface.co/api/models/{repo_id}?blobs=true`
/// to retrieve file sizes and LFS metadata that `hf-hub`'s `info()` does not expose.
///
/// Accepts a shared `reqwest::Client` to reuse TCP connections and TLS sessions
/// across calls. Use [`chunked::build_client`](crate::chunked::build_client) to
/// create a client with the standard connect timeout and auth headers.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the HTTP request fails.
/// Returns [`FetchError::RepoNotFound`] if the repository does not exist.
pub async fn list_repo_files_with_metadata(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
    client: &reqwest::Client,
) -> Result<Vec<RepoFile>, FetchError> {
    let (files, _commit) = list_repo_files_with_commit(repo_id, token, revision, client).await?;
    Ok(files)
}

/// Fetches file metadata **and** the resolved commit SHA of the revision.
///
/// Same HTTP call as [`list_repo_files_with_metadata`], but also returns the
/// `sha` field from the `HuggingFace` API response. Callers that need to show
/// or pin the current revision (e.g. `inspect --list`) use this variant; all
/// other callers should prefer [`list_repo_files_with_metadata`].
///
/// The commit SHA is `Option<String>` because the API has not always returned
/// it on every endpoint variant; treat it as informational.
///
/// # Errors
///
/// Returns [`FetchError::Http`] if the HTTP request fails.
/// Returns [`FetchError::RepoNotFound`] if the repository does not exist.
pub async fn list_repo_files_with_commit(
    repo_id: &str,
    token: Option<&str>,
    revision: Option<&str>,
    client: &reqwest::Client,
) -> Result<(Vec<RepoFile>, Option<String>), FetchError> {
    let mut url = format!("https://huggingface.co/api/models/{repo_id}?blobs=true");
    if let Some(rev) = revision {
        url = format!("{url}&revision={rev}");
    }

    // BORROW: explicit .as_str() instead of Deref coercion
    let mut request = client.get(url.as_str());
    if let Some(t) = token {
        request = request.bearer_auth(t);
    }

    let response = request
        .send()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        // BORROW: explicit .to_owned() for &str → owned String field
        return Err(FetchError::RepoNotFound {
            repo_id: repo_id.to_owned(),
        });
    }

    if !response.status().is_success() {
        return Err(FetchError::Http(format!(
            "HF API returned status {}",
            response.status()
        )));
    }

    let info: ApiModelInfo = response
        .json()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;

    let commit_sha = info.sha;
    let files = info
        .siblings
        .into_iter()
        .map(|s| {
            let (size, sha256) = match s.lfs {
                Some(lfs) => (Some(lfs.size), Some(lfs.sha256)),
                None => (s.size, None),
            };
            RepoFile {
                filename: s.rfilename,
                size,
                sha256,
            }
        })
        .collect();

    Ok((files, commit_sha))
}
