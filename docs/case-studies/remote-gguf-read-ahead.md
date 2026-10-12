# Remote GGUF inspect: from 201 range requests to 18

*How a 64 KiB read buffer made a large GGUF header cost one HTTP request per 64 KiB, what growing read-ahead changed in v0.12.3, and the before-and-after numbers on eleven real files.*

**Source:** internal, found during the v0.12.3 consistency pass · fix: [`bd2aa25`](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/commit/bd2aa25) (growing read-ahead)

---

## The symptom

Remote `inspect` prints what each read cost on its `Source:` line. For the largest GGUF headers it was alarming:

```
$ hf-fm inspect ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf     # v0.12.2
  Source:   remote (201 range requests, 12.44 MiB fetched)
```

201 requests out of the 256 a remote read is allowed, and 74 seconds of wall clock to read a header. `unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF` took 202. A tokenizer about a quarter larger would have failed outright with `range request cap exceeded`.

## The diagnostic path

**Count requests per MiB, not requests.** Across every GGUF the docs cite, the `Source:` line showed about 16 requests per MiB of header, plus 2 (SmolLM2: 30 requests for 1.75 MiB; Qwen3.5-4B: 170 for 10.5 MiB). Sixteen per MiB is one per 64 KiB.

**Find the 64 KiB.** The GGUF parser (anamnesis) wraps the reader it is given in a 64 KiB `BufReader`, and hf-fm's `RangeReader` fetched exactly what each buffer refill asked for. A GGUF header is mostly its tokenizer's vocabulary (token strings, token types, BPE merges) read straight through, so every 64 KiB of vocabulary was one HTTP round trip, at about 0.36 s each: the request through `/resolve` and its redirect to a freshly signed CDN URL.

**Model it before changing it.** An offline harness wrote synthetic GGUF files of realistic tokenizer sizes and ran anamnesis's real parser over hf-fm's real `RangeReader`, through an in-memory transport that counts every fetch. Calibrated against the real thing (the SmolLM2-shaped file reproduced 30 requests and 1.75 MiB exactly), it put the cap at a header of about 15.9 MiB and compared the candidate fixes on the same files: a window that doubles while reads stay sequential, up to 1 MiB (7 to 21 requests across all cases, under 1 MiB read past the header); the same up to 4 MiB (fewer requests, up to about 3 MiB of weight data read); a fixed 1 MiB window (every read pays 1 MiB, even a tiny one); or raising the request cap alone (every inspect stays slow).

**The fix.** `RangeReader` now doubles its window whenever a read continues exactly where the last window ended, from the caller's size up to `MAX_READAHEAD_BYTES` (1 MiB), and starts again at the caller's size after any seek. Growth never spends budget a fixed window would not have: near the 32 MiB byte budget it falls back to the caller's size. Formats that seek (`.npz`, `.pth`) keep their small windows, and a safetensors header, read in one call, is untouched.

## Before and after

Measured on 2026-10-12 from the maintainer's machine (Windows 11, a home connection). **Before** is hf-fm 0.12.2 from crates.io; **after** is a release build of `bd2aa25`. Each run is `hf-fm inspect <repo> <file> --limit 1` with a fresh, empty `HF_HOME`, so every read is remote, and authenticated through `HF_TOKEN_PATH`. Three passes, before and after interleaved file by file; the time is the median wall clock of the whole command. Request and byte counts were identical in all three passes.

| Format | File | Fetched, before → after | Requests, before → after | Time, before → after | Speed-up |
|---|---|---|---|---|---|
| GGUF | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` (Q4_K_M) | 768.0 KiB → 960.0 KiB | 14 → 6 | 5.5 s → 2.1 s | **2.6×** |
| GGUF | `bartowski/SmolLM2-135M-Instruct-GGUF` (Q2_K) | 1.75 MiB → 1.94 MiB | 30 → 7 | 11.2 s → 2.3 s | **4.8×** |
| GGUF | `bartowski/zed-industries_zeta-2-GGUF` (Q4_K_M) | 6.00 MiB → 6.94 MiB | 98 → 12 | 35.3 s → 4.5 s | **7.9×** |
| GGUF | `unsloth/gemma-3-4b-it-GGUF` (Q4_K_M) | 6.25 MiB → 6.94 MiB | 102 → 12 | 36.2 s → 4.3 s | **8.4×** |
| GGUF | `bartowski/Llama-3.2-1B-Instruct-GGUF` (Q4_K_M) | 7.50 MiB → 7.94 MiB | 122 → 13 | 43.4 s → 4.7 s | **9.3×** |
| GGUF | `unsloth/Qwen3.5-4B-GGUF` (UD-IQ2_XXS) | 10.50 MiB → 10.94 MiB | 170 → 16 | 61.6 s → 5.9 s | **10.4×** |
| GGUF | `ggml-org/gpt-oss-20b-GGUF` (MXFP4) | 12.44 MiB → 12.94 MiB | 201 → 18 | 74.1 s → 6.8 s | **11.0×** |
| GGUF | `unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF` (IQ4_XS, shard 1) | 12.50 MiB → 12.94 MiB | 202 → 18 | 76.0 s → 6.5 s | **11.7×** |
| PTH | `RWKV/RWKV7-Goose-World-PTH` (0.1B, 364 MiB) | 113.7 KiB → 132.0 KiB | 12 → 9 | 4.3 s → 3.1 s | 1.4× |
| NPZ | `google/gemma-scope-2b-pt-transcoders` (`params.npz`, 288 MiB) | 84.0 KiB → 84.0 KiB | 8 → 8 | 3.2 s → 2.7 s | unchanged |
| safetensors | `EleutherAI/pythia-1.4b` (`model.safetensors`) | 41.5 KiB → 41.5 KiB | 4 → 4 | 1.4 s → 1.3 s | unchanged |

Across the eight GGUF files: **939 requests become 102, and 343 seconds become 37**. The speed-up grows with the header, because the request count now grows with its logarithm up to the 1 MiB window and linearly only beyond it. The price is at most one window read past the header: the "after" column fetches under 1 MiB more (0.94 MiB at most here), against up to 64 KiB before. The `.pth` row improves too, since its pickle stream is also read straight through; `.npz` and safetensors, which seek or read their header in one call, take exactly the requests they took before.

Run-to-run spread was small: under 10% for every "before" median and most "after" ones. The one outlier was a single 12.2 s run of Llama-4-Scout's "after", against 6.4 and 6.5 s for the other two.

## Where the limit is now

With requests no longer the constraint, the 32 MiB byte budget binds instead. Offline, with synthetic BPE tokenizers (tokens of 9 bytes, their types, merges of 11 bytes; the fixture of the tests below) read through the same parser and reader:

| Header | Requests | Fetched | Outcome |
|---|---|---|---|
| 12.11 MiB (gpt-oss-20b's shape: 201,088 tokens, 446,189 merges) | 18 | 12.94 MiB | inspects, the real file's figures exactly |
| 30.52 MiB | 36 | 30.94 MiB | inspects |
| 31.66 MiB | 37 | 31.94 MiB | inspects |
| 32.04 MiB | 38 | 32.00 MiB | `range transfer budget exceeded` |
| 33.57 MiB | 38 | 32.00 MiB | `range transfer budget exceeded`; inspects with a 64 MiB budget |

So a header up to about 31.9 MiB inspects, two and a half times the largest measured. Those cases are permanent tests in `src/http_range.rs` (`gguf_with_a_gpt_oss_sized_tokenizer_inspects_in_a_few_requests`, `gguf_header_just_under_the_transfer_budget_still_inspects`, `gguf_header_over_the_transfer_budget_fails_and_a_larger_budget_reads_it`), and `tests/cli.rs` checks the real `gpt-oss-20b` file live (`inspect_remote_gguf_with_a_large_tokenizer_takes_a_few_requests`). For a file past the limit, the [FAQ](../FAQ.md#which-models-cant-be-inspected-remotely) gives the two ways through: download it and use `inspect --cached`, or, from Rust, open it with `HttpRangeReader::open_with_limits` and a larger budget.

## Reproduce one row

In PowerShell, with hf-fm installed:

```powershell
$env:HF_TOKEN_PATH = "$HOME\.cache\huggingface\token"   # authenticated, without reading the token
$env:HF_HOME = (New-Item -ItemType Directory (Join-Path $env:TEMP ([guid]::NewGuid()))).FullName
Measure-Command { hf-fm inspect ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf --limit 1 | Select-String 'Source:' | Out-Host }
```

A fresh `HF_HOME` matters: a file already in your cache reads `Source: cached` and makes no request at all.

## The transferable workflow

**Divide a remote read's request count by the bytes it fetched.** A steady ratio (here 16 requests per MiB) means a reader refilling a fixed-size buffer one round trip at a time, and the buffer size is 1 MiB divided by that number. The `Source:` line puts both figures on screen for every remote inspect.
