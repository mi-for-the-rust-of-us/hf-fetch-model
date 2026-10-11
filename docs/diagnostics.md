# Download Diagnostics

<!-- Last updated: 2026-10-11, hf-fm v0.12.2 (real capture; every event the library emits) -->

hf-fetch-model emits structured `tracing` events while it downloads: `debug` for each step it takes, `warn` for the problems that change how a download runs. All of them come from the download path; `inspect`, `du`, `cache` and the other subcommands emit none.

## CLI

Use the `--verbose` / `-v` flag on either of the two commands that download, the default command and `download-file`:

```sh
hf-fm google/gemma-2-2b-it -v
hf-fm download-file mntss/clt-gemma-2-2b-426k W_enc_0.safetensors -v
```

The events go to stderr, filtered to `hf_fetch_model=debug`. A `RUST_LOG` variable, when set, takes the place of that filter.

## Library

The library only emits the events. To see them, install a subscriber, for example `tracing-subscriber` with its `env-filter` feature:

```rust
// Via RUST_LOG environment variable
// RUST_LOG=hf_fetch_model=debug cargo run

// Or programmatically
tracing_subscriber::fmt()
    .with_env_filter("hf_fetch_model=debug")
    .init();
```

The default `fmt` output also names each event's target (`hf_fetch_model::download`); the CLI turns targets off, which is why they are absent below.

## Example output

A real run on 2026-10-11, into an empty `--output-dir` (a cached repo would stop after the cache check instead). `--chunk-threshold-mib 50` sends the 86 MiB `model.safetensors` down the multi-connection path; left to itself, the plan kept every file of this repo on a single connection.

```sh
hf-fm sentence-transformers/all-MiniLM-L6-v2 --filter model.safetensors --filter config.json --chunk-threshold-mib 50 -v
```

```
2026-10-11T07:44:28.110701Z DEBUG listing repository files repo_id=sentence-transformers/all-MiniLM-L6-v2
2026-10-11T07:44:28.292871Z DEBUG fetching extended metadata (checksums=true, chunk_threshold=52428800 bytes)
2026-10-11T07:44:28.456673Z DEBUG metadata fetch succeeded files_with_size=30 total_files=30
2026-10-11T07:44:28.457021Z DEBUG merged plan-recommended settings concurrency=4 connections_per_file=8 chunk_threshold=52428800
2026-10-11T07:44:28.460585Z DEBUG download settings (after plan optimization) total_files=2 concurrency=4
2026-10-11T07:44:28.464532Z DEBUG single-connection download (below chunk threshold) filename=config.json size_mib=0
2026-10-11T07:44:28.464538Z DEBUG chunked download (multi-connection) filename=model.safetensors size_mib=86 connections=8
2026-10-11T07:44:28.769447Z DEBUG download complete filename=config.json elapsed_secs=0.3 throughput_mbps=0.0
2026-10-11T07:44:32.681089Z DEBUG download complete filename=model.safetensors elapsed_secs=4.2 throughput_mbps=172.4
2026-10-11T07:44:32.681462Z DEBUG download complete files_downloaded=2
```

`throughput_mbps` is in megabits per second; the summary the CLI prints on stdout (`86.66 MiB in 4.8s (18.1 MiB/s)`) is in mebibytes.

## What to look for

Every event, in the order a download meets them:

| Message | Level | Meaning |
|---------|-------|---------|
| "listing repository files" | debug | The remote file list, fetched even when everything is cached: it is the only way to know whether the cache is complete for this filter |
| "all filtered files resolved from local cache (no download needed)" | debug | Every selected file is cached; nothing else follows |
| "cache fast-path declined: filtered file absent on disk" | debug | A selected file is missing, so the download goes ahead |
| "fetching extended metadata (…)" | debug | Requests each file's size and SHA256, needed for checksum verification and to choose chunked downloads |
| "skipping metadata fetch (checksums disabled, chunk_threshold=MAX)" | debug | Neither needs it, so the request is skipped |
| "metadata fetch succeeded" | debug | `files_with_size` of `total_files` have a known size |
| "metadata fetch failed; chunked downloads disabled for this run" | warn | Sizes unknown: every file uses a single connection |
| "merged plan-recommended settings" | debug | The plan's concurrency, connections per file and chunk threshold, applied to every setting not given explicitly |
| "download settings (after plan optimization)" | debug | The number of files to download and the concurrency actually used |
| "insufficient disk space" / "disk space is tight after download" | warn | Also printed as a `warning:` line on stderr, with or without `-v` |
| "could not check available disk space" | debug | The free-space check failed; the download goes ahead without it |
| "file resolved from local cache" | debug | This file is already cached and is not downloaded again |
| "chunked download (multi-connection)" | debug | The file is at or above `chunk_threshold`: parallel HTTP Range connections |
| "single-connection download (below chunk threshold)" | debug | The file is below `chunk_threshold` |
| "single-connection download" with a `reason` | debug | `file size unknown (metadata missing)`, or `chunked downloads disabled` |
| "download complete" (per file) | debug | `elapsed_secs` and `throughput_mbps`, useful for comparing single and chunked |
| "download complete (size unknown)" | debug | The same without a throughput, since the size is unknown |
| "download failed" | debug | A file failed, with its `error`; a repo download then returns `FetchError::PartialDownload`, listing every failed file |
| "download complete" (`files_downloaded`) | debug | End of a successful run, with the number of files downloaded |
