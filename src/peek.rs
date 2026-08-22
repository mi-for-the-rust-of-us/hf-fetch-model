// SPDX-License-Identifier: MIT OR Apache-2.0

//! Arbitrary small-file "peek" — the non-tensor sibling of [`crate::inspect`].
//!
//! [`peek`] streams a small file from a `HuggingFace` repo (`config.yaml`,
//! `README.md`, a `.gz`-compressed sidecar) over the same
//! [`crate::http_range::HttpRangeReader`] substrate the tensor-format
//! `inspect` paths use, but with **no anamnesis dispatch** — peek never
//! parses tensor headers; use [`crate::inspect`] for that. `--head`/`--tail`
//! bound the read by line or byte count; `--gunzip` transparently decodes a
//! `flate2` gzip stream; `--max` is the safety cap that keeps an accidental
//! peek of a multi-gigabyte weight file from dumping binary garbage to a
//! terminal.
//!
//! The generic, offline-testable core ([`stream_peek`]) operates on any
//! `Read + Seek` source and buffers its result fully in memory — peek's
//! whole premise is *small* files, so the [`PeekOptions::max_bytes`] cap
//! (10 MiB by default) already bounds that buffer, and buffering sidesteps
//! threading a generic `Write` sink across the `spawn_blocking` boundary
//! [`peek`] needs for the network path. The public async entry point
//! ([`peek`]) is the only piece that knows about HTTP: it opens the
//! [`crate::http_range::HttpRangeReader`], resolves the upfront (pre-fetch)
//! validation that doesn't need a single content byte, and — mirroring
//! [`crate::inspect::inspect_npz`]'s error-recovery shape — prefers the
//! reader's typed transport error over a generic wrapper on failure.

use std::io::{self, Read, Seek, SeekFrom};

use flate2::read::GzDecoder;

use crate::error::FetchError;
use crate::http_range::{HttpRangeReader, MAX_RANGE_REQUESTS, MAX_TRANSFER_BUDGET};
use crate::inspect::is_supported_tensor_file;

/// Size of one backward read window when scanning for line boundaries
/// from the end of a file (`--tail N`, lines mode). Doubles each round
/// that fails to find enough lines, up to [`PeekOptions::max_bytes`].
const TAIL_SCAN_CHUNK: u64 = 4 * 1024;

/// Forward read chunk size for [`bounded_copy`] and the gzip `Cat` buffer.
///
/// Sized to match [`crate::http_range::TAIL_PREFETCH_BYTES`] rather than
/// the transport's smaller `READAHEAD_BYTES` floor: `bounded_copy` reads
/// through [`crate::http_range::RangeReader`], which fetches one window
/// per call sized to (at least) the caller's buffer — a small fixed chunk
/// here means one HTTP range request per chunk, no window reuse, since a
/// forward-only scan never revisits a window once drained. At 4 KiB that
/// caps a `--head`/`--tail` (lines) scan at [`MAX_RANGE_REQUESTS`] × 4 KiB
/// ≈ 1 MiB — well under the 10 MiB default `--max` — before hitting
/// `"range request cap exceeded"`. 64 KiB raises that same ceiling to
/// [`MAX_RANGE_REQUESTS`] × 64 KiB ≈ 16 MiB, comfortably covering the
/// default cap without even widening [`transport_limits`]'s request count.
const READ_CHUNK: usize = 64 * 1024;

// -----------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------

/// Counting unit for [`PeekMode::Head`] / [`PeekMode::Tail`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PeekUnit {
    /// Count newline-terminated lines (POSIX `\n`).
    Lines,
    /// Count raw bytes.
    Bytes,
}

/// What to print and how much of it.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum PeekMode {
    /// `cat`-like: stream everything, bounded by [`PeekOptions::max_bytes`].
    Cat,
    /// The first `count` lines or bytes.
    Head {
        /// How many lines or bytes to read, per `unit`. Must be at least `1`.
        count: u64,
        /// Whether `count` is lines or raw bytes.
        unit: PeekUnit,
    },
    /// The last `count` lines or bytes.
    Tail {
        /// How many lines or bytes to read, per `unit`. Must be at least `1`.
        count: u64,
        /// Whether `count` is lines or raw bytes.
        unit: PeekUnit,
    },
}

/// A fully-resolved peek request.
///
/// "Resolved" means the effective `gunzip` value already folds in
/// auto-detection (a `.gz`-suffixed filename) and any `--no-gunzip`
/// override — [`resolve_mode`] and the `--gunzip`/`--no-gunzip` merge are
/// the caller's job (the `hf-fm` CLI does this in `run_peek`); this struct
/// carries only the final decision.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PeekOptions {
    /// What to read and how much of it.
    pub mode: PeekMode,
    /// Whether to transparently gzip-decode the stream before applying `mode`.
    pub gunzip: bool,
    /// Safety cap, in bytes, on content read (post-decompression when
    /// `gunzip` is set). Default `10 MiB` at the CLI layer.
    pub max_bytes: u64,
    /// The filename being peeked, for tensor-format-aware error wording
    /// (see [`peek`]'s `# Errors` section).
    pub filename: String,
}

impl PeekOptions {
    /// Constructs a new [`PeekOptions`].
    ///
    /// Since v0.11.5 the struct is `#[non_exhaustive]` — this constructor is
    /// the canonical way to build one from outside the `hf-fetch-model` lib
    /// crate.
    #[must_use]
    pub const fn new(mode: PeekMode, gunzip: bool, max_bytes: u64, filename: String) -> Self {
        Self {
            mode,
            gunzip,
            max_bytes,
            filename,
        }
    }
}

/// Outcome of a successful peek.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PeekOutcome {
    /// The content read, already gunzip-decoded if [`PeekOptions::gunzip`] was set.
    pub content: Vec<u8>,
    /// `Some(note)` when `--max` (not a natural end-of-file or a satisfied
    /// `--head`/`--tail` count) is why streaming stopped early. `None`
    /// otherwise — including the ordinary case where the file simply has
    /// fewer lines than `--head`/`--tail` requested.
    pub truncated: Option<String>,
}

// -----------------------------------------------------------------------
// Mode resolution
// -----------------------------------------------------------------------

/// Resolves raw `--head`/`--tail`/`--bytes` flags into a [`PeekMode`].
///
/// # Errors
///
/// Returns [`FetchError::InvalidArgument`] if `--head` and `--tail` are both
/// set (the `hf-fm` CLI already rejects this at clap parse time via
/// `conflicts_with`; this is a defensive check for other callers), if
/// either count is `0`, or if `bytes` is set with neither `head` nor `tail`.
pub fn resolve_mode(
    head: Option<u64>,
    tail: Option<u64>,
    bytes: bool,
) -> Result<PeekMode, FetchError> {
    let unit = if bytes {
        PeekUnit::Bytes
    } else {
        PeekUnit::Lines
    };
    match (head, tail) {
        (Some(_), Some(_)) => Err(FetchError::InvalidArgument(
            "--head and --tail are mutually exclusive".to_owned(),
        )),
        (Some(0), None) | (None, Some(0)) => Err(FetchError::InvalidArgument(
            "--head/--tail count must be at least 1".to_owned(),
        )),
        (Some(count), None) => Ok(PeekMode::Head { count, unit }),
        (None, Some(count)) => Ok(PeekMode::Tail { count, unit }),
        (None, None) if bytes => Err(FetchError::InvalidArgument(
            "--bytes requires --head or --tail".to_owned(),
        )),
        (None, None) => Ok(PeekMode::Cat),
    }
}

/// Resolves the effective gunzip decision from the explicit flags and the filename.
///
/// `no_gunzip` wins outright; otherwise gunzip is on when `gunzip` is
/// explicit or `filename` ends in `.gz` (case-insensitive).
#[must_use]
pub fn resolve_gunzip(gunzip: bool, no_gunzip: bool, filename: &str) -> bool {
    if no_gunzip {
        return false;
    }
    gunzip || filename.to_ascii_lowercase().ends_with(".gz")
}

// -----------------------------------------------------------------------
// Validation that doesn't need a content byte
// -----------------------------------------------------------------------

/// Validates `options` against `total_size` (known upfront from the
/// `HttpRangeReader` probe) before any content byte is fetched.
///
/// # Errors
///
/// Returns [`FetchError::InvalidArgument`] when:
/// - `gunzip` is set together with [`PeekMode::Tail`] (gzip is sequential;
///   tailing a compressed stream isn't supported),
/// - [`PeekMode::Tail`] with [`PeekUnit::Bytes`] requests more bytes than
///   `max_bytes` allows,
/// - [`PeekMode::Cat`] on a non-gzip file whose known `total_size` already
///   exceeds `max_bytes` (the footgun guard — rejects before streaming
///   rather than truncating raw, possibly binary, output to a terminal).
///
/// The gzip `Cat` case is **not** covered here: decompressed size isn't
/// knowable upfront, so that check happens during [`stream_peek`]'s
/// buffered decode instead.
pub fn validate_resolved(options: &PeekOptions, total_size: u64) -> Result<(), FetchError> {
    if options.gunzip && matches!(options.mode, PeekMode::Tail { .. }) {
        return Err(FetchError::InvalidArgument(format!(
            "--gunzip does not support --tail for {} (gzip is sequential); \
             run `hf-fm peek {} --gunzip --max <SIZE>` and pipe the output through `tail` instead",
            options.filename, options.filename
        )));
    }
    if let PeekMode::Tail {
        count,
        unit: PeekUnit::Bytes,
    } = options.mode
        && count > options.max_bytes
    {
        return Err(FetchError::InvalidArgument(format!(
            "--tail {count} bytes for {} exceeds --max {} (raise --max to read more)",
            options.filename,
            format_bytes_approx(options.max_bytes)
        )));
    }
    if !options.gunzip && matches!(options.mode, PeekMode::Cat) && total_size > options.max_bytes {
        let hint = if is_supported_tensor_file(options.filename.as_str()) {
            "use `hf-fm inspect` for tensor files"
        } else {
            "pass --max <SIZE> to read more of a large text file"
        };
        return Err(FetchError::InvalidArgument(format!(
            "{} is {} (exceeds --max {}); {hint}",
            options.filename,
            format_bytes_approx(total_size),
            format_bytes_approx(options.max_bytes)
        )));
    }
    Ok(())
}

// -----------------------------------------------------------------------
// Generic, offline-testable core
// -----------------------------------------------------------------------

/// Runs one peek against `reader`, returning the fully-buffered content.
///
/// Generic over any `Read + Seek` source — production code drives this
/// with an [`HttpRangeReader`] (via [`peek`]); tests drive it with an
/// in-memory `Cursor` or [`crate::http_range::RangeReader`] over an
/// in-memory fetcher, with no network involved.
///
/// # Errors
///
/// Returns an [`io::Error`] of kind [`io::ErrorKind::InvalidData`] when
/// gzip-decoded `Cat` content exceeds `options.max_bytes` before
/// end-of-file (the gzip counterpart of [`validate_resolved`]'s non-gzip
/// `Cat` check, which can't run upfront since decompressed size is
/// unknown). Other kinds propagate the source reader's own read failures.
pub fn stream_peek<R: Read + Seek>(
    reader: &mut R,
    options: &PeekOptions,
) -> io::Result<PeekOutcome> {
    match options.mode {
        PeekMode::Cat => stream_cat(reader, options),
        PeekMode::Head { count, unit } => stream_head(reader, options, count, unit),
        PeekMode::Tail {
            count,
            unit: PeekUnit::Bytes,
        } => stream_tail_bytes(reader, count),
        PeekMode::Tail {
            count,
            unit: PeekUnit::Lines,
        } => stream_tail_lines(reader, options.max_bytes, count),
    }
}

/// `PeekMode::Cat`: non-gzip content is already known to fit under
/// `max_bytes` (checked by [`validate_resolved`]), so a plain bounded copy
/// suffices. Gzip content is buffered fully before deciding whether it
/// fits — see the module docs for why full buffering is an acceptable
/// trade-off here.
fn stream_cat<R: Read + Seek>(reader: &mut R, options: &PeekOptions) -> io::Result<PeekOutcome> {
    if options.gunzip {
        let mut decoder = GzDecoder::new(&mut *reader);
        let limit = options.max_bytes.saturating_add(1);
        let mut buf = Vec::new();
        let mut chunk = vec![0u8; READ_CHUNK];
        loop {
            // CAST: usize → u64, buffer length always fits (usize <= u64::MAX)
            #[allow(clippy::as_conversions)]
            if buf.len() as u64 >= limit {
                break;
            }
            let n = decoder.read(&mut chunk)?;
            if n == 0 {
                break;
            }
            let Some(piece) = chunk.get(..n) else {
                break; // unreachable: n <= chunk.len() by Read's contract
            };
            buf.extend_from_slice(piece);
        }
        // CAST: usize → u64, buffer length always fits (usize <= u64::MAX)
        #[allow(clippy::as_conversions)]
        let written = buf.len() as u64;
        if written > options.max_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} decompressed content exceeds --max {} (raise --max to read more)",
                    options.filename,
                    format_bytes_approx(options.max_bytes)
                ),
            ));
        }
        Ok(PeekOutcome {
            content: buf,
            truncated: None,
        })
    } else {
        let mut content = Vec::new();
        reader.read_to_end(&mut content)?;
        Ok(PeekOutcome {
            content,
            truncated: None,
        })
    }
}

/// `PeekMode::Head`: forward, bounded read over the (optionally
/// gunzip-wrapped) stream.
fn stream_head<R: Read + Seek>(
    reader: &mut R,
    options: &PeekOptions,
    count: u64,
    unit: PeekUnit,
) -> io::Result<PeekOutcome> {
    let mut src: Box<dyn Read + '_> = if options.gunzip {
        Box::new(GzDecoder::new(&mut *reader))
    } else {
        Box::new(&mut *reader)
    };
    let (content, stop) = bounded_copy(&mut src, count, unit, options.max_bytes)?;
    let truncated = matches!(stop, StopReason::CapExceeded).then(|| {
        format!(
            "truncated: --head {count} {} for {} not fully read within --max {} \
             (raise --max to read more)",
            unit_label(unit),
            options.filename,
            format_bytes_approx(options.max_bytes)
        )
    });
    Ok(PeekOutcome { content, truncated })
}

/// `PeekMode::Tail { unit: Bytes }`: a single Range-from-end read. Clamps
/// to the start of the file when `count` exceeds the file's length,
/// matching POSIX `tail -c`'s behaviour on a short file.
fn stream_tail_bytes<R: Read + Seek>(reader: &mut R, count: u64) -> io::Result<PeekOutcome> {
    let end = reader.seek(SeekFrom::End(0))?;
    let start = end.saturating_sub(count);
    reader.seek(SeekFrom::Start(start))?;
    let mut content = Vec::new();
    reader.read_to_end(&mut content)?;
    Ok(PeekOutcome {
        content,
        truncated: None,
    })
}

/// `PeekMode::Tail { unit: Lines }`: a backward chunk scan, doubling the
/// scan window each round until `count` newlines are found or the scan
/// hits `max_bytes` / the start of the file.
fn stream_tail_lines<R: Read + Seek>(
    reader: &mut R,
    max_bytes: u64,
    count: u64,
) -> io::Result<PeekOutcome> {
    let end = reader.seek(SeekFrom::End(0))?;
    let mut scan = TAIL_SCAN_CHUNK.min(end).min(max_bytes);
    loop {
        let start = end.saturating_sub(scan);
        reader.seek(SeekFrom::Start(start))?;
        let read_len = end.saturating_sub(start);
        let mut buf = vec![0u8; usize::try_from(read_len).unwrap_or(usize::MAX)];
        reader.read_exact(&mut buf)?;

        // Peek's scan windows are bounded by `--max` (small-file scoped by
        // design), so the naive per-byte scan is fine — pulling in the
        // `bytecount` crate for one counting loop isn't worth it.
        #[allow(clippy::naive_bytecount)]
        let newline_count = buf.iter().filter(|&&b| b == b'\n').count();
        let newline_count_u64 = u64::try_from(newline_count).unwrap_or(u64::MAX);
        let hit_start = start == 0;
        let hit_cap = scan >= max_bytes;
        // A clean stop needs *strictly more* than `count` newlines: that
        // guarantees `positions[m - count - 1]` in `last_n_lines` is a real,
        // captured boundary, so the earliest of the last `count` lines is
        // known to start right after it. Stopping at exactly `m == count`
        // (the old `>=` condition) can't tell whether `buf`'s own start,
        // `start`, landed on a genuine line boundary or mid-line — and
        // `last_n_lines` then returns the *whole* window uncut, silently
        // splicing a partial leading line into the output. `hit_start` is
        // still clean on its own: `start == 0` is unambiguously the true
        // beginning of the file, not an arbitrary scan-window edge.
        let confident = newline_count_u64 > count || hit_start;

        if confident || hit_cap {
            let content = last_n_lines(&buf, count);
            let truncated = (!confident).then(|| {
                if newline_count_u64 < count {
                    format!(
                        "truncated: only {newline_count_u64} of {count} requested lines \
                         available within --max {} (raise --max to read more)",
                        format_bytes_approx(max_bytes)
                    )
                } else {
                    format!(
                        "truncated: found {count} requested lines but --max {} was \
                         reached before confirming the start of the earliest one \
                         (raise --max to read more)",
                        format_bytes_approx(max_bytes)
                    )
                }
            });
            return Ok(PeekOutcome { content, truncated });
        }
        scan = scan.saturating_mul(2).min(end).min(max_bytes);
    }
}

/// Extracts the last `count` newline-delimited lines from `buf` (a file
/// suffix that may itself start mid-line). Returns the whole slice when
/// `buf` contains `count` or fewer newlines.
fn last_n_lines(buf: &[u8], count: u64) -> Vec<u8> {
    let positions: Vec<usize> = buf
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| (b == b'\n').then_some(i))
        .collect();
    let m = positions.len();
    // CAST: u64 → usize, line counts requested via a CLI flag are small in practice
    #[allow(clippy::as_conversions)]
    let count_usize = usize::try_from(count).unwrap_or(usize::MAX);
    if m > count_usize && count_usize > 0 {
        let boundary_idx = m - count_usize - 1;
        let cut = positions
            .get(boundary_idx)
            .map_or(0, |p| p.saturating_add(1));
        buf.get(cut..).unwrap_or(buf).to_vec()
    } else {
        buf.to_vec()
    }
}

/// Why a bounded copy stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopReason {
    /// The requested `count` was reached exactly.
    CountSatisfied,
    /// The source hit end-of-file before `count` was reached — not a
    /// `--max`-driven truncation, just a short file.
    Eof,
    /// `max_bytes` was reached before `count` was satisfied.
    CapExceeded,
}

/// Copies from `src`, stopping once either `count` lines/bytes (per `unit`)
/// have been collected or `cap` bytes have been read, whichever comes first.
fn bounded_copy(
    src: &mut impl Read,
    count: u64,
    unit: PeekUnit,
    cap: u64,
) -> io::Result<(Vec<u8>, StopReason)> {
    let mut content = Vec::new();
    let mut lines_seen: u64 = 0;
    let mut buf = vec![0u8; READ_CHUNK];
    loop {
        // CAST: usize → u64, buffer length always fits (usize <= u64::MAX)
        #[allow(clippy::as_conversions)]
        let written = content.len() as u64;
        if written >= cap {
            return Ok((content, StopReason::CapExceeded));
        }
        // CAST: usize → u64, READ_CHUNK is a small compile-time constant
        #[allow(clippy::as_conversions)]
        let read_chunk_u64 = READ_CHUNK as u64;
        let remaining_cap =
            usize::try_from(cap.saturating_sub(written).min(read_chunk_u64)).unwrap_or(READ_CHUNK);
        let Some(target) = buf.get_mut(..remaining_cap) else {
            return Ok((content, StopReason::CapExceeded)); // unreachable: remaining_cap <= buf.len()
        };
        let n = src.read(target)?;
        if n == 0 {
            return Ok((content, StopReason::Eof));
        }
        let Some(chunk) = buf.get(..n) else {
            return Ok((content, StopReason::CapExceeded)); // unreachable: n <= target.len()
        };

        match unit {
            PeekUnit::Bytes => {
                // CAST: u64 → usize, remaining count for a bounded read is small in practice
                #[allow(clippy::as_conversions)]
                let remaining = usize::try_from(count.saturating_sub(written)).unwrap_or(n);
                let take = remaining.min(n);
                let Some(piece) = chunk.get(..take) else {
                    return Ok((content, StopReason::CapExceeded)); // unreachable: take <= n
                };
                content.extend_from_slice(piece);
                // CAST: usize → u64, taken length always fits (usize <= u64::MAX)
                #[allow(clippy::as_conversions)]
                let new_written = written.saturating_add(take as u64);
                if new_written >= count {
                    return Ok((content, StopReason::CountSatisfied));
                }
            }
            PeekUnit::Lines => {
                let mut remaining = chunk;
                loop {
                    if let Some(nl) = remaining.iter().position(|&b| b == b'\n') {
                        let Some((line, rest)) = remaining.split_at_checked(nl + 1) else {
                            break; // unreachable: nl + 1 <= remaining.len()
                        };
                        content.extend_from_slice(line);
                        lines_seen += 1;
                        if lines_seen >= count {
                            return Ok((content, StopReason::CountSatisfied));
                        }
                        remaining = rest;
                    } else {
                        content.extend_from_slice(remaining);
                        break;
                    }
                }
            }
        }
    }
}

/// `"lines"` / `"bytes"`, for truncation-note wording.
const fn unit_label(unit: PeekUnit) -> &'static str {
    match unit {
        PeekUnit::Lines => "lines",
        PeekUnit::Bytes => "bytes",
    }
}

/// Human-readable byte count for peek's own error/truncation wording.
///
/// A private duplicate of the binary-only `format_size` helper
/// (`src/format.rs`, `#[path]`-included per-binary, not part of the public
/// library surface) — kept small and local rather than promoting
/// `format_size` into the library API for one module's error strings.
///
/// **Must stay byte-for-byte identical to `format_size`'s buckets and
/// precision** (including the `< 1000 MiB` / `< 1000 GiB` flip thresholds,
/// not `< 1024`, and the `.1`-vs-`.2` decimal-place split) — the same
/// quantity should render the same way regardless of which `hf-fm`
/// subcommand's message the user is reading. An earlier version of this
/// function drifted (`.2` KiB precision instead of `.1`, pure-1024
/// thresholds instead of 1000, and no `TiB` tier at all), caught by review
/// before release; `format_bytes_approx_matches_format_size_test_cases`
/// below cross-checks the two against the same sample values as
/// `src/format.rs`'s own test suite so a future drift fails loudly.
fn format_bytes_approx(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * 1024 * 1024;
    const TIB: u64 = 1024 * GIB;

    if bytes >= 1000 * GIB {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only size scalar
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let val = bytes as f64 / TIB as f64;
        format!("{val:.2} TiB")
    } else if bytes >= 1000 * MIB {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only size scalar
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let val = bytes as f64 / GIB as f64;
        format!("{val:.2} GiB")
    } else if bytes >= MIB {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only size scalar
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let val = bytes as f64 / MIB as f64;
        format!("{val:.2} MiB")
    } else if bytes >= KIB {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only size scalar
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let val = bytes as f64 / KIB as f64;
        format!("{val:.1} KiB")
    } else {
        format!("{bytes} B")
    }
}

/// Sizes the transport's own safety budgets ([`MAX_RANGE_REQUESTS`],
/// [`MAX_TRANSFER_BUDGET`]) to `max_bytes` (`PeekOptions::max_bytes`).
///
/// [`HttpRangeReader::open`]'s defaults are tuned for archive-header
/// parsing (`inspect`'s handful-of-requests, well-under-1-MiB case) —
/// peek's own `--max` can be, and by default already is, far larger.
/// Returns `(max_requests, max_transfer_bytes)`, each floored at the
/// transport's own default so a small/default `--max` keeps exactly
/// today's safety margins:
///
/// - `max_transfer_bytes` doubles `max_bytes` — headroom for
///   [`stream_tail_lines`]'s backward doubling scan (each round re-fetches
///   its whole window, not just the newly-scanned prefix — see that
///   function's doc comment) and the gzip `Cat` path's `max_bytes + 1`
///   buffering probe.
/// - `max_requests` budgets one request per [`READ_CHUNK`] of forward scan
///   ([`bounded_copy`]'s per-chunk fetch pattern, the dominant cost for
///   `--head` / `--tail` lines mode), plus a fixed slack for
///   [`stream_tail_lines`]'s handful of doubling rounds and the transport's
///   own probe/tail-prefetch requests.
fn transport_limits(max_bytes: u64) -> (u32, u64) {
    let max_transfer_bytes = max_bytes.saturating_mul(2).max(MAX_TRANSFER_BUDGET);
    // CAST: usize → u64, READ_CHUNK is a small compile-time constant
    #[allow(clippy::as_conversions)]
    let read_chunk_u64 = READ_CHUNK as u64;
    let by_chunk = (max_bytes / read_chunk_u64).saturating_add(32);
    let max_requests = u32::try_from(by_chunk)
        .unwrap_or(u32::MAX)
        .max(MAX_RANGE_REQUESTS);
    (max_requests, max_transfer_bytes)
}

// -----------------------------------------------------------------------
// Remote entry point
// -----------------------------------------------------------------------

/// Peeks `filename` in `repo_id` at `revision` (default `main`) over HTTP
/// Range requests. Remote-only by design — the cached equivalent is
/// `cat $(hf-fm cache path <repo>)/<file>` (`Get-Content` on `PowerShell`);
/// see the `hf-fm peek --help` text for why a `--cached` flag would
/// duplicate that pattern with worse ergonomics.
///
/// Opens an [`HttpRangeReader`] (one probe request, resolving `total_size`
/// for free), runs [`validate_resolved`] against it, then hands the reader
/// to a blocking thread running [`stream_peek`] — mirroring
/// [`crate::inspect::inspect_npz`]'s shape: on failure, the reader's typed
/// transport error is preferred over a generic wrapper, so a gated repo's
/// `401`/`403` stays recognisable to the `hf-fm` CLI's gated-repo diagnosis.
///
/// # Errors
///
/// Returns [`FetchError::InvalidArgument`] if [`validate_resolved`] rejects
/// `options` against the probed file size, or if gzip-decoded `Cat`
/// content exceeds `options.max_bytes` (see [`stream_peek`]).
/// Returns [`FetchError::Http`] if the probe or a range request fails,
/// including gated repos (`returned status 401/403`, upgraded by the
/// `hf-fm` CLI into a gated-repo diagnosis).
pub async fn peek(
    repo_id: &str,
    filename: &str,
    token: Option<&str>,
    revision: Option<&str>,
    options: PeekOptions,
) -> Result<PeekOutcome, FetchError> {
    let (max_requests, max_transfer_bytes) = transport_limits(options.max_bytes);
    let mut reader = HttpRangeReader::open_with_limits(
        repo_id,
        revision,
        filename,
        token,
        max_requests,
        max_transfer_bytes,
    )
    .await?;
    validate_resolved(&options, reader.total_size())?;

    let (result, transport_error) = tokio::task::spawn_blocking(move || {
        let outcome = stream_peek(&mut reader, &options);
        (outcome, reader.take_last_error())
    })
    .await
    .map_err(|e| FetchError::Http(format!("failed to join peek task: {e}")))?;

    match result {
        Ok(outcome) => Ok(outcome),
        Err(io_err) if io_err.kind() == io::ErrorKind::InvalidData => {
            Err(FetchError::InvalidArgument(io_err.to_string()))
        }
        Err(io_err) => Err(transport_error
            .unwrap_or_else(|| FetchError::Http(format!("failed to peek {filename}: {io_err}")))),
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

    use std::io::Cursor;

    use super::*;

    fn opts(mode: PeekMode, gunzip: bool, max_bytes: u64) -> PeekOptions {
        PeekOptions::new(mode, gunzip, max_bytes, "test.txt".to_owned())
    }

    fn gzip(data: &[u8]) -> Vec<u8> {
        use std::io::Write as _;
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(data).unwrap();
        enc.finish().unwrap()
    }

    // ---------- format_bytes_approx ----------

    #[test]
    fn format_bytes_approx_matches_format_size_test_cases() {
        // Same sample values as `src/format.rs`'s own `format_size` test
        // suite, asserted against the identical expected strings — a
        // regression guard against the two formatters drifting apart
        // (they did once; see the doc comment on `format_bytes_approx`).
        assert_eq!(format_bytes_approx(0), "0 B");
        assert_eq!(format_bytes_approx(1), "1 B");
        assert_eq!(format_bytes_approx(1023), "1023 B");
        assert_eq!(format_bytes_approx(1024), "1.0 KiB");
        assert_eq!(format_bytes_approx(1536), "1.5 KiB");
        assert_eq!(format_bytes_approx(1024 * 1024), "1.00 MiB");
        assert_eq!(format_bytes_approx(10 * 1024 * 1024), "10.00 MiB");
        assert_eq!(format_bytes_approx(999 * 1024 * 1024), "999.00 MiB");
        let gib = 1u64 << 30;
        assert_eq!(format_bytes_approx(gib), "1.00 GiB");
        assert_eq!(format_bytes_approx(999 * gib), "999.00 GiB");
        let tib = 1024 * gib;
        assert_eq!(format_bytes_approx(tib), "1.00 TiB");
    }

    // ---------- transport_limits ----------

    #[test]
    fn transport_limits_floors_at_the_transport_defaults() {
        // A small/default `--max` must not shrink the transport's own
        // safety margins below what `inspect` already relies on.
        let (max_requests, max_transfer_bytes) = transport_limits(1024);
        assert_eq!(max_requests, MAX_RANGE_REQUESTS);
        assert_eq!(max_transfer_bytes, MAX_TRANSFER_BUDGET);
    }

    #[test]
    fn transport_limits_scales_past_the_defaults_for_a_large_max() {
        let large = 200 * 1024 * 1024; // 200 MiB
        let (max_requests, max_transfer_bytes) = transport_limits(large);
        assert!(
            max_transfer_bytes >= 2 * large,
            "must cover the doubled-headroom formula, got {max_transfer_bytes}"
        );
        assert!(
            max_requests > MAX_RANGE_REQUESTS,
            "a 200 MiB --max needs more than the {MAX_RANGE_REQUESTS}-request \
             default at {READ_CHUNK}-byte chunks, got {max_requests}"
        );
    }

    // ---------- resolve_mode / resolve_gunzip ----------

    #[test]
    fn resolve_mode_covers_the_flag_matrix() {
        assert!(matches!(
            resolve_mode(None, None, false).unwrap(),
            PeekMode::Cat
        ));
        assert!(matches!(
            resolve_mode(Some(5), None, false).unwrap(),
            PeekMode::Head {
                count: 5,
                unit: PeekUnit::Lines
            }
        ));
        assert!(matches!(
            resolve_mode(None, Some(5), true).unwrap(),
            PeekMode::Tail {
                count: 5,
                unit: PeekUnit::Bytes
            }
        ));
        assert!(resolve_mode(Some(1), Some(1), false).is_err());
        assert!(resolve_mode(Some(0), None, false).is_err());
        assert!(resolve_mode(None, None, true).is_err());
    }

    #[test]
    fn resolve_gunzip_prefers_no_gunzip_then_explicit_then_extension() {
        assert!(!resolve_gunzip(false, true, "data.json.gz"));
        assert!(resolve_gunzip(true, false, "data.json"));
        assert!(resolve_gunzip(false, false, "index.JSON.GZ"));
        assert!(!resolve_gunzip(false, false, "config.yaml"));
    }

    // ---------- validate_resolved ----------

    #[test]
    fn validate_resolved_rejects_oversized_non_gzip_cat_upfront() {
        let o = opts(PeekMode::Cat, false, 10);
        let err = validate_resolved(&o, 11).unwrap_err();
        assert!(matches!(err, FetchError::InvalidArgument(_)));
    }

    #[test]
    fn validate_resolved_names_inspect_for_tensor_files() {
        let o = PeekOptions::new(PeekMode::Cat, false, 10, "model.safetensors".to_owned());
        let err = validate_resolved(&o, 11).unwrap_err();
        assert!(err.to_string().contains("hf-fm inspect"), "{err}");
    }

    #[test]
    fn validate_resolved_rejects_gunzip_with_tail() {
        let o = opts(
            PeekMode::Tail {
                count: 5,
                unit: PeekUnit::Lines,
            },
            true,
            1024,
        );
        let err = validate_resolved(&o, 100).unwrap_err();
        assert!(err.to_string().contains("--tail"), "{err}");
    }

    #[test]
    fn validate_resolved_rejects_tail_bytes_over_max() {
        let o = opts(
            PeekMode::Tail {
                count: 100,
                unit: PeekUnit::Bytes,
            },
            false,
            10,
        );
        let err = validate_resolved(&o, 1000).unwrap_err();
        assert!(err.to_string().contains("--max"), "{err}");
    }

    #[test]
    fn validate_resolved_allows_cat_under_cap() {
        let o = opts(PeekMode::Cat, false, 10);
        validate_resolved(&o, 10).unwrap();
    }

    // ---------- stream_peek: Cat ----------

    #[test]
    fn cat_streams_everything_under_cap() {
        let data = b"hello world\n".to_vec();
        let mut r = Cursor::new(data.clone());
        let o = opts(PeekMode::Cat, false, 1024);
        let out = stream_peek(&mut r, &o).unwrap();
        assert_eq!(out.content, data);
        assert!(out.truncated.is_none());
    }

    #[test]
    fn cat_gunzip_round_trips_and_rejects_over_cap() {
        let data = b"line one\nline two\nline three\n".to_vec();
        let compressed = gzip(&data);
        let mut r = Cursor::new(compressed.clone());
        let o = opts(PeekMode::Cat, true, 1024);
        let out = stream_peek(&mut r, &o).unwrap();
        assert_eq!(out.content, data);

        let mut r2 = Cursor::new(compressed);
        let tiny = opts(PeekMode::Cat, true, 4);
        let err = stream_peek(&mut r2, &tiny).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    // ---------- stream_peek: Head ----------

    #[test]
    fn head_lines_stops_at_the_nth_newline() {
        let data = b"a\nb\nc\nd\n".to_vec();
        let mut r = Cursor::new(data);
        let o = opts(
            PeekMode::Head {
                count: 2,
                unit: PeekUnit::Lines,
            },
            false,
            1024,
        );
        let out = stream_peek(&mut r, &o).unwrap();
        assert_eq!(out.content, b"a\nb\n");
        assert!(out.truncated.is_none());
    }

    #[test]
    fn head_lines_short_file_is_not_truncated() {
        let data = b"only one line\n".to_vec();
        let mut r = Cursor::new(data.clone());
        let o = opts(
            PeekMode::Head {
                count: 5,
                unit: PeekUnit::Lines,
            },
            false,
            1024,
        );
        let out = stream_peek(&mut r, &o).unwrap();
        assert_eq!(out.content, data);
        assert!(
            out.truncated.is_none(),
            "short file is not a --max truncation"
        );
    }

    #[test]
    fn head_bytes_exact_count() {
        let data = b"abcdefgh".to_vec();
        let mut r = Cursor::new(data);
        let o = opts(
            PeekMode::Head {
                count: 3,
                unit: PeekUnit::Bytes,
            },
            false,
            1024,
        );
        let out = stream_peek(&mut r, &o).unwrap();
        assert_eq!(out.content, b"abc");
    }

    #[test]
    fn head_cap_exceeded_before_count_is_truncated() {
        // No newlines within the 4-byte cap, count never satisfied.
        let data = b"no_newlines_at_all_here".to_vec();
        let mut r = Cursor::new(data);
        let o = opts(
            PeekMode::Head {
                count: 1,
                unit: PeekUnit::Lines,
            },
            false,
            4,
        );
        let out = stream_peek(&mut r, &o).unwrap();
        assert_eq!(out.content.len(), 4);
        assert!(out.truncated.is_some());
    }

    // ---------- stream_peek: Tail bytes ----------

    #[test]
    fn tail_bytes_reads_exactly_the_last_n() {
        let data = b"0123456789".to_vec();
        let mut r = Cursor::new(data);
        let out = stream_tail_bytes(&mut r, 4).unwrap();
        assert_eq!(out.content, b"6789");
    }

    #[test]
    fn tail_bytes_clamps_when_file_is_shorter_than_requested() {
        let data = b"hi".to_vec();
        let mut r = Cursor::new(data.clone());
        let out = stream_tail_bytes(&mut r, 100).unwrap();
        assert_eq!(out.content, data);
    }

    // ---------- stream_peek: Tail lines ----------

    #[test]
    fn tail_lines_returns_the_last_n_lines() {
        let data = b"a\nb\nc\nd\ne\n".to_vec();
        let mut r = Cursor::new(data);
        let out = stream_tail_lines(&mut r, 1024, 2).unwrap();
        assert_eq!(out.content, b"d\ne\n");
        assert!(out.truncated.is_none());
    }

    #[test]
    fn tail_lines_exact_newline_count_at_window_edge_does_not_splice_a_partial_line() {
        // Regression test for a real bug found via code review + an offline
        // reproduction (not live): the first backward-scan window can land
        // mid-line and still contain exactly `count` newlines — the old
        // `newline_count >= count` stop condition treated that as "done",
        // but `last_n_lines` only trims when `m > count` (strict), so it
        // returned the *whole*, partially-mid-line window instead of just
        // the last two lines. Verified to fail without the fix (returned
        // the 4096-byte first-round window instead of the correct 5006
        // bytes) by temporarily reverting the `confident` check.
        let line0 = format!("{}\n", "A".repeat(10_000)); // 10 001 bytes
        let line1 = format!("{}\n", "Q".repeat(5_000)); // 5 001 bytes
        let line2 = "last\n"; // 5 bytes
        let data = format!("{line0}{line1}{line2}").into_bytes();
        // The first 4 KiB scan window lands inside line1 and happens to
        // contain exactly 2 newlines (line1's own + line2's) — the exact
        // ambiguous boundary this test targets.
        let mut r = Cursor::new(data);
        let out = stream_tail_lines(&mut r, 1024 * 1024, 2).unwrap();
        assert_eq!(
            out.content,
            format!("{line1}{line2}").into_bytes(),
            "must be exactly the last two complete lines, not a partial-line splice"
        );
        assert!(
            out.truncated.is_none(),
            "the scan grew far enough to confirm the boundary; not a --max truncation"
        );
    }

    #[test]
    fn tail_lines_grows_the_scan_window_past_one_chunk() {
        // Force at least two doubling rounds: > TAIL_SCAN_CHUNK bytes of
        // filler before the requested lines.
        let filler = "x".repeat(6000);
        let data = format!("{filler}\nlast one\nlast two\n").into_bytes();
        let mut r = Cursor::new(data);
        let out = stream_tail_lines(&mut r, 1024 * 1024, 2).unwrap();
        assert_eq!(out.content, b"last one\nlast two\n");
    }

    #[test]
    fn tail_lines_short_file_returns_everything_untruncated() {
        let data = b"only\ntwo\n".to_vec();
        let mut r = Cursor::new(data.clone());
        let out = stream_tail_lines(&mut r, 1024, 10).unwrap();
        assert_eq!(out.content, data);
        assert!(out.truncated.is_none());
    }

    #[test]
    fn tail_lines_bounded_by_max_is_truncated() {
        let filler = "y".repeat(6000);
        let data = format!("{filler}\nlast\n").into_bytes();
        let mut r = Cursor::new(data);
        // Cap small enough that the scan can never reach the leading filler's start.
        let out = stream_tail_lines(&mut r, 128, 5).unwrap();
        assert!(out.truncated.is_some());
    }

    #[test]
    fn tail_lines_empty_file() {
        let mut r = Cursor::new(Vec::<u8>::new());
        let out = stream_tail_lines(&mut r, 1024, 3).unwrap();
        assert!(out.content.is_empty());
        assert!(out.truncated.is_none());
    }

    // ---------- last_n_lines ----------

    #[test]
    fn last_n_lines_pure_helper() {
        assert_eq!(last_n_lines(b"a\nb\nc\n", 2), b"b\nc\n");
        assert_eq!(last_n_lines(b"a\nb\nc\n", 10), b"a\nb\nc\n");
        assert_eq!(last_n_lines(b"no newline here", 1), b"no newline here");
    }

    // ---------- end-to-end over the shared RangeReader substrate ----------

    #[test]
    fn head_over_range_reader_reads_only_the_needed_window() {
        use crate::http_range::{RangeFetcher, RangeReader};

        struct InMemory {
            data: Vec<u8>,
        }
        impl RangeFetcher for InMemory {
            fn fetch(&mut self, start: u64, end_inclusive: u64) -> Result<Vec<u8>, FetchError> {
                let s = usize::try_from(start).unwrap();
                let e = usize::try_from(end_inclusive).unwrap();
                self.data
                    .get(s..=e)
                    .map(<[u8]>::to_vec)
                    .ok_or_else(|| FetchError::Http("bad range".to_owned()))
            }
            fn total_size(&self) -> u64 {
                u64::try_from(self.data.len()).unwrap()
            }
        }

        let mut body = "first\nsecond\n".as_bytes().to_vec();
        body.extend(std::iter::repeat_n(b'z', 200 * 1024)); // large tail never needed
        let mut reader = RangeReader::new(InMemory { data: body });
        let o = opts(
            PeekMode::Head {
                count: 2,
                unit: PeekUnit::Lines,
            },
            false,
            1024,
        );
        let out = stream_peek(&mut reader, &o).unwrap();
        assert_eq!(out.content, b"first\nsecond\n");
        assert!(
            reader.stats().bytes_fetched < 200 * 1024,
            "head must not fetch the large unused tail"
        );
    }
}
