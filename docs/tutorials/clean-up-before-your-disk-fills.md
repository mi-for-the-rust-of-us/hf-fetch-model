# Clean up before your disk fills

*See what the HuggingFace cache holds, decide what to keep, and reclaim the rest — with a dry-run before anything is deleted.*

*~2,830 words · about 11 min read*

<!-- Last updated: 2026-11-01, hf-fm v0.12.6 -->

<!--
STYLE CONVENTIONS for editing this tutorial — keep growth consistent.

1. Tone: match the FAQ and the inspect tutorial. Conversational, address
   the reader as "you", short paragraphs over bullet lists where prose
   works.
2. Reproducibility: unlike the inspect tutorial, there is no revision to
   pin — the cache is machine-local. Output blocks come from capture
   sessions on the same machine, dated below. The du / gc blocks are from 2026-10-10
   (816 GiB, 75 repos), recaptured for v0.12.2, whose du counts bytes on
   disk, everything in each repo's directory (issue #16); the 2026-06-11
   captures they replace counted snapshot files only. The status / delete /
   verify / path blocks are from 2026-06-11, and the partial-download
   blocks (status PARTIAL, clean-partial, resume) from a 2026-06-12
   follow-up that staged an interrupted download (note 3). Of the 2026-06
   blocks, only the delete preview shows a figure v0.12.2 changes: its
   276.5 KiB now reads 276.7 KiB, the repo's refs/ being counted. It is
   kept rather than re-run, so that nothing is deleted or prompted on the
   real cache. The du ● marker block those sessions produced was dropped
   rather than kept with a pre-fix size. The reader's numbers WILL differ;
   the column shapes and legends must not.

   v0.12.5 (2026-10-31) recaptured, from the same machine's cache after a
   cleanup (763 GiB, 72 repos, 21 holding duplicated copies; the 2026-10-10
   figures above describe the state before it), every du and gc block, the
   du <repo> drill-down, and the cache delete / DUP / --verify-duplicates
   additions. The status / delete-prompt / verify / path blocks are older
   and unchanged. The 2026-10-10 narrative numbers (816 GiB, 75 repos) were
   replaced, not mixed with the new ones.

   v0.12.6 (2026-11-01) adds the cache dedupe walkthrough, captured from
   a --release build of main on the same cache: the dry run, then the real
   run (with --yes) of the very preview shown, whose result block starts
   after that repeated preview, then a du of the reclaimed cache (609 GiB),
   trimmed after its first rows and its totals. The v0.12.5 du / gc / du <repo>
   blocks above it still describe the cache before dedupe, on purpose; only
   their pointer lines were updated to the text v0.12.6 prints for that same
   state (the footer note, the gc line, the du <repo> hint), since v0.12.5's
   pointed to issue #21, which this release resolves.
3. Safety: every destructive command appears with --dry-run, or with its
   confirmation prompt visible and answered `n`. Never paste an output
   that shows an actual deletion the reader did not see previewed first.
   The one exception is cache dedupe, which deletes no model and loses no
   byte: its real run is shown, but only after its --dry-run preview, and
   the reader is told to run the preview first.
   The partial-download captures (status PARTIAL rows, clean-partial dry
   run, the resume run) were produced by deliberately interrupting a
   `--preset safetensors` download of the ungated mirror
   NousResearch/Meta-Llama-3.1-8B (chosen so no license is needed to
   reproduce) and re-capturing as it resumed. The clean-partial dry run is
   framed as a counterfactual ("had you abandoned it") because in the real
   session the download was resumed, not cleaned — keep that framing so the
   narrative timeline stays honest.
4. Output blocks: paste exact output, do not paraphrase. Trim only when a
   block runs longer than ~20 lines and the trimmed lines are
   representative repetition (note the trim with `…`).
5. Length budget: under 300 lines total, including embedded outputs.
   Update the word count + reading-time line at the top whenever the
   prose changes non-trivially (250 wpm).
6. Word count = total words in this file excluding code blocks and
   HTML comments. Reading time = word count / 250, rounded to the
   nearest minute, minimum 1.
-->

The second tutorial in the v0.10 docs effort. If a step is confusing or an output looks different from yours, please open an issue on [GitHub](https://github.com/mi-for-the-rust-of-us/hf-fetch-model/issues) — the tutorial improves as real questions land.

## Contents

- [Why the cache fills silently](#why-the-cache-fills-silently)
- [The 30-second answer](#the-30-second-answer)
- [Seeing: the `du` family](#seeing-the-du-family)
- [Diagnosing one repo: `status`](#diagnosing-one-repo-status)
- [Reclaiming the copies, keeping every model: `cache dedupe`](#reclaiming-the-copies-keeping-every-model-cache-dedupe)
- [Acting: `delete`, `clean-partial`, `gc`](#acting-delete-clean-partial-gc)
- [Trust, but verify: `cache verify`](#trust-but-verify-cache-verify)
- [Scripting: `cache path`](#scripting-cache-path)
- [What you've learned](#what-youve-learned)

## Why the cache fills silently

Every `hf-fm <repo>` — and every Python `huggingface_hub` download, since both tools share the same cache — adds a `models--org--name/` directory under `~/.cache/huggingface/hub/` that persists until you delete it. There is no expiry, no size limit, no garbage collection. You discover the problem the day the disk is full, facing dozens of directories whose `blobs/` + `refs/` + `snapshots/` internals were never meant to be cleaned by hand: delete the wrong piece and you orphan refs or break the snapshot symlinks for a model you still use.

hf-fm splits the job into two verbs, and that split is the whole mental model of this tutorial: **`du` for seeing, `cache` for acting** — the same separation Docker draws between `docker system df` and `docker system prune`. Everything in the `du` family is read-only; everything under `cache` that deletes asks first, or accepts `--dry-run`.

## The 30-second answer

Two commands. The first shows where the bytes are; the second previews what an age-based sweep would reclaim. Nothing is deleted.

```sh
hf-fm du
```

```
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub

    #                      SIZE         DUP  REPO                                                     FILES
    1                159.05 GiB           -  mntss/clt-gemma-2-2b-2.5M                                   53
    2                 78.92 GiB   38.30 GiB  bluelightai/clt-qwen3-1.7b-base-20k                         88
    3                 46.09 GiB   22.42 GiB  bluelightai/clt-qwen3-0.6b-base-20k                         88
    4                 34.54 GiB           -  datasets/TheQweaker/rented_boxes                          1952  ●
    5                 32.39 GiB   15.60 GiB  bluelightai-dev/clt-Qwen3-0.6B-Base-16k-test                87
…
   71                     607 B           -  EleutherAI/pythia-70m                                        1
   72                     141 B           -  sentence-transformers/all-MiniLM-L6-v2                       0
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
  763.22 GiB  total (72 repos, 2839 files)
  By volume: C: 762.71 GiB · D: 525.44 MiB
  154.16 GiB of 763.22 GiB are duplicated copies (21 repos)
  Note: a duplicated copy is a snapshot file stored again beside its blob, which is how hf-hub writes on Windows. `du <repo>` shows each repo's share; `--verify-duplicates` hashes the copies to confirm them. `hf-fm cache dedupe` replaces them with hard links to their blobs, keeping every model.
  ● = partial downloads
  → = relocated: the repo's directory is a link to where its bytes are
  Note: a size range means that repo's cached `.gguf` files are mutually exclusive quant alternatives rather than shards of one file — you likely only need one of them (see `du <repo>` for the exact file sizes). The total above still reflects real bytes on disk across every cached file.
```

```sh
hf-fm cache gc --older-than 180 --dry-run
```

```
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub

note: 1 relocated repo skipped (the bytes are behind a link, not in this cache); `hf-fm cache delete` removes one
Will remove:
  Qwen/Qwen2.5-Coder-7B-Instruct                       14.19 GiB  9 months ago
  google/codegemma-7b-it                               15.92 GiB  9 months ago
  codellama/CodeLlama-7b-hf                            12.55 GiB  8 months ago
  …
  mntss/clt-gemma-2-2b-2.5M                           159.05 GiB  8 months ago
  …
  google/gemma-4-E2B-it                                19.12 GiB  6 months ago

Skipped (relocated behind a link; evicting them frees nothing here):
  gpt2  525.44 MiB

Cache: 728.67 GiB → 282.79 GiB (free 445.88 GiB)
  154.16 GiB of 763.22 GiB are duplicated copies (21 repos)
  gc selects whole repos by age and budget and does not deduplicate: removing models to recover space that deduplication would recover without losing any is what this figure warns against. `hf-fm du` names the copies per repo. `hf-fm cache dedupe` reclaims them and keeps every model.
Note: 3 dataset or Space repos (34.55 GiB) not counted above; gc evicts models only.
```

That preview says: every model untouched for six months, removed in one stroke, frees 445.88 GiB of the 728.67 GiB of models, about three fifths. Pick the threshold to fit your own cache: on this one, 90 days would have taken 702.54 GiB, nearly all of it, since most of it has sat idle for three months or more. Read the lines under `Cache:` before you trust the figure. On Windows the cache holds a second copy of many files (154.16 GiB here, which `du` names, below), and eviction removes whole models: it would delete models to recover space that deduplication recovers without losing any. Run [`cache dedupe`](#reclaiming-the-copies-keeping-every-model-cache-dedupe) first. And `gc` selects among models only, so the 34.55 GiB of datasets that `du` lists are not in its 728.67 GiB; the last line says so. The `note:` above the list is a repo, `gpt2`, that was moved to another disk behind a link: evicting it would free nothing here, so it is skipped. When the list looks right, re-run without `--dry-run` and answer the prompt. The rest of the tutorial is what to check before you trust that list.

## Seeing: the `du` family

`du` is one command with progressive flags. The `#` column is not decoration: every index it prints is accepted wherever a repo ID is (`du 15`, `cache delete 15`, `cache verify 15`), so you never type `models--microsoft--Phi-3.5-mini-instruct` or even `microsoft/Phi-3.5-mini-instruct` by hand. A `●` after a row's file count flags a repo with an interrupted download, and a `● = partial downloads` line under the total explains it; row 4 above carries one, a dataset whose download never finished, and the next section interrupts a model download on purpose. Since v0.12.5 `du` lists datasets and Spaces too, so its total is the cache's: row 4 is `datasets/TheQweaker/rented_boxes`, 34.54 GiB that the listing used to leave out, addressed behind the Hub's `datasets/` prefix (`du 4` works as well). The `DUP` column is explained below. While a repo carries the marker, its size includes the in-flight shards at the full size each one is preallocated to, which on Windows is what they occupy on disk, but its file count leaves them out until they finish. `status`, below, is what shows their true progress.

`--age` adds the question GC will ask — *when did I last touch this?*

```sh
hf-fm du --age
```

```
    #                      SIZE         DUP  REPO                                                     FILES  AGE            
    1                159.05 GiB           -  mntss/clt-gemma-2-2b-2.5M                                   53  8 months ago   
    2                 78.92 GiB   38.30 GiB  bluelightai/clt-qwen3-1.7b-base-20k                         88  5 months ago   
    3                 46.09 GiB   22.42 GiB  bluelightai/clt-qwen3-0.6b-base-20k                         88  5 months ago   
    4                 34.54 GiB           -  datasets/TheQweaker/rented_boxes                          1952  2 days ago       ●
    5                 32.39 GiB   15.60 GiB  bluelightai-dev/clt-Qwen3-0.6B-Base-16k-test                87  5 months ago   
…
```

Row 1 is the classic case this tutorial exists for: a 159 GiB set of cross-layer transcoders, fetched eight months ago and untouched since. Row 2 is half its size and three months younger, so the six-month sweep above keeps it.

**The `DUP` column** is new in v0.12.5, and on Windows it can be the biggest saving in the cache. It appears only when some repo holds duplicated copies: a file stored once as its blob and again as a full copy in `snapshots/`, which is how `hf-hub` writes where it cannot make a symlink. Row 1 has none (a dash): its 159 GiB are in `snapshots/` alone. Row 2 has 38.30 GiB, half its size, stored twice. `--sort dup` puts the repos with the most copies first, and `#` stays each repo's rank by size, so `du <N>` still names what you see:

```sh
hf-fm du --sort dup
```

```
    #                      SIZE         DUP  REPO                                                     FILES
    2                 78.92 GiB   38.30 GiB  bluelightai/clt-qwen3-1.7b-base-20k                         88
    3                 46.09 GiB   22.42 GiB  bluelightai/clt-qwen3-0.6b-base-20k                         88
    6                 32.26 GiB   16.13 GiB  mntss/transcoder-Llama-3.2-1B                               17
    5                 32.39 GiB   15.60 GiB  bluelightai-dev/clt-Qwen3-0.6B-Base-16k-test                87
    …
```

The footer under the total adds up the column, `154.16 GiB of 763.22 GiB are duplicated copies (21 repos)`: a fifth of this cache, which no amount of sorting by `SIZE` shows. Detection is by length (a snapshot file the same length as a blob in the same repo, that is not a link to it), so `--verify-duplicates` settles it by hashing each copy against its blob's name, which for an LFS file is the SHA-256 of its content. It reads every copy once, so scope it to a repo:

```sh
hf-fm du google/gemma-scope-2b-pt-res --verify-duplicates
```

```
  google/gemma-scope-2b-pt-res:

    #        SIZE  FILE
    1  288.13 MiB  layer_0/width_16k/average_l0_105/params.npz
    2   72.04 MiB  embedding/width_4k/average_l0_44/params.npz
  ────────────────────────────────────────────────────────────
  360.18 MiB  listed above (2 files)
  432.22 MiB  total on disk

  duplicated copies 72.04 MiB · orphan blobs 0 B · partial downloads 0 B · bookkeeping 40 B

  Verified 1 duplicated copy (72.04 MiB): identical to their blob.
```

(That repo's 288 MiB file has no copy; the 72 MiB file is the duplicate, and `Verified` shows it holds its blob's bytes.) Reclaiming them without deleting a model is [`cache dedupe`](#reclaiming-the-copies-keeping-every-model-cache-dedupe), below. Since v0.12.6 `du` also recognises a hard link on Windows, so a pointer hard-linked to its blob, by `cache dedupe` or by hand, is counted once.

Drill into one repo by its index:

```sh
hf-fm du 15
```

```
  microsoft/Phi-3.5-mini-instruct:

    #       SIZE  FILE
    1   4.63 GiB  model-00001-of-00002.safetensors
    2   2.49 GiB  model-00002-of-00002.safetensors
    …
   20      195 B  generation_config.json
  ────────────────────────────────────────────────
   7.12 GiB  listed above (20 files)
  14.24 GiB  total on disk

  duplicated copies 7.12 GiB · orphan blobs 0 B · partial downloads 0 B · bookkeeping 140 B
  `hf-fm cache dedupe microsoft/Phi-3.5-mini-instruct` replaces the duplicated copies with hard links to their blobs.
```

The two totals differ, and not by rounding. The files as listed add up to 7.12 GiB, but the repo occupies 14.24 GiB, because each file is stored twice: as its blob in `blobs/`, and as a full copy under `snapshots/` where Linux and macOS would put a symlink. That is how hf-fm's downloads landed on Windows before v0.12.6, where creating a symlink needs Developer Mode or an elevated shell; since then they hard-link LFS files to their blob instead, and `cache dedupe` reclaims the copies already there. The last line says it in numbers: all 7.12 GiB of the difference is duplicated copies (the same 7.12 GiB as in row 15's `DUP`), and the rest, 140 B, is bookkeeping (`refs/` and hf-fm's header cache). Together with the listed bytes the four figures add up to `total on disk`; a repo with an interrupted download adds a `partial downloads` figure, and a blob that nothing points at an `orphan blobs` one. The listing shows what the model is made of; `total on disk` is what deleting it frees, and it is the figure the whole-cache `du`, `cache gc` and `cache delete` all use. Where the two agree, as they do with symlinked entries on Linux and macOS, `du` prints a single `total` line.

And `du --tree` renders the whole cache as one structural view — repos as branches, files as leaves sorted by size — which is where pathological layouts jump out (here, per-layer decoder shards stepping down from 10.97 GiB):

```sh
hf-fm du --tree
```

```
  ├── mntss/clt-gemma-2-2b-2.5M  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .    159.05 GiB  (53 files)
  │   ├── W_dec_0.safetensors                                                          10.97 GiB
  │   ├── W_dec_1.safetensors                                                          10.55 GiB
  │   ├── W_dec_2.safetensors                                                          10.13 GiB
  …
```

## Diagnosing one repo: `status`

`du` tells you a repo's weight; `status` tells you its health — it fetches the remote file list and classifies every file:

```sh
hf-fm status microsoft/Phi-3.5-mini-instruct
```

```
microsoft/Phi-3.5-mini-instruct (main @ 2fe192450127e6a83f7441aef6e3ca586c338b77)
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub\models--microsoft--Phi-3.5-mini-instruct

  .gitattributes                      1.5 KiB  complete
  CODE_OF_CONDUCT.md                    453 B  complete
  …
  model-00001-of-00002.safetensors   4.63 GiB  complete
  model-00002-of-00002.safetensors   2.49 GiB  complete
  …
  tokenizer_config.json               3.9 KiB  complete

20/20 complete, 0 partial, 0 missing
```

Four states can appear, and the all-green run above shows only one. Start a download and interrupt it — close the laptop, drop the connection, hit Ctrl-C — then ask `status` what survived. Here is an ungated Llama-3.1-8B mirror, fetched with `--preset safetensors` and interrupted partway through its four shards:

```sh
hf-fm status NousResearch/Meta-Llama-3.1-8B --preset safetensors
```

```
NousResearch/Meta-Llama-3.1-8B (main @ 1f47e50cdbe801ad8a5174156ec3a0655108fb9f)
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub\models--NousResearch--Meta-Llama-3.1-8B

  .gitattributes                      1.5 KiB  excluded
  LICENSE                             7.4 KiB  excluded
  config.json                           826 B  complete
  generation_config.json                185 B  complete
  model-00001-of-00004.safetensors   1.38 GiB / 4.63 GiB    PARTIAL
  model-00002-of-00004.safetensors   1.27 GiB / 4.66 GiB    PARTIAL
  model-00003-of-00004.safetensors   1.36 GiB / 4.58 GiB    PARTIAL
  model-00004-of-00004.safetensors   1.09 GiB  complete
  model.safetensors.index.json       23.4 KiB  MISSING
  original/consolidated.00.pth      14.96 GiB  excluded
  …
  tokenizer.json                     8.66 MiB  MISSING

3/17 complete, 3 partial, 5 missing, 6 excluded
```

All four at once. `complete` and `MISSING` mean what they say. Each `PARTIAL` row shows that file's *own* downloaded bytes against its full size — `1.38 GiB / 4.63 GiB`, a different figure per row, not one repo-level number stamped on every line (a v0.10.5 fix: the count is read from the resume sidecar, so it is true progress, not the preallocated blob's reserved size). A mid-download `status` is therefore a live per-file progress report. `excluded` marks files the `--preset` deliberately skipped — here the `original/consolidated.00.pth` PyTorch checkpoint and the docs — telling "I chose not to fetch this" apart from "this download is incomplete". (Drop `--preset` and those same files report `MISSING`; the preset is what lets `status` know the difference.)

One thing to unlearn: a `.chunked.part` file is **not** garbage. It is resume state — run the same download again and it continues from those exact bytes instead of restarting (v0.9.8):

```sh
hf-fm NousResearch/Meta-Llama-3.1-8B --preset safetensors
```

```
  Disk: 13.88 GiB to fetch, 2088.86 GiB available (2074.98 GiB after download)
[hf-fm] model-00003-of-00004.safetensors: 1.36 GiB/4.58 GiB (29%)
[hf-fm] model-00001-of-00004.safetensors: 1.38 GiB/4.63 GiB (29%)
[hf-fm] model-00002-of-00004.safetensors: 1.27 GiB/4.66 GiB (27%)
  …
Downloaded to: …/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50c…
  14.97 GiB in 205.7s (74.5 MiB/s)
```

The shards pick up at 27–29% — exactly where `status` left them above — not 0%. The interrupted bytes counted. (The final `14.97 GiB` line is hf-fm reporting the file's full size, not the bytes fetched this run; the ~4 GiB already on disk is why it finished as fast as it did.) Clean partials only when you have decided *not* to resume.

## Reclaiming the copies, keeping every model: `cache dedupe`

On Windows, the cheapest gigabytes in the cache are the duplicated copies: deleting nothing gets them back. Since v0.12.6 `cache dedupe` replaces each snapshot file that holds the same bytes as its blob with a hard link to that blob, so the file is still there, and still read by hf-fm, `hf-hub` and Python's `huggingface_hub`, but its bytes exist once. Preview it first:

```sh
hf-fm cache dedupe --dry-run
```

```
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub

  up to   1.11 GiB  Qwen/Qwen3-0.6B-Base  (1 file)
  up to   9.69 GiB  bartowski/Mistral-7B-Instruct-v0.3-GGUF  (5 files)
…
  up to  38.30 GiB  bluelightai/clt-qwen3-1.7b-base-20k  (54 files)
…
  up to   7.60 GiB  google/gemma-scope-2b-pt-transcoders  (27 files)
…
  up to   7.12 GiB  microsoft/Phi-3.5-mini-instruct  (2 files)
…

  Up to 155.00 GiB in 231 files across 21 repos. Each file is compared with its blob before it is replaced.
  Dry run: nothing was compared or changed.
```

The preview reads nothing: it pairs every snapshot file with the blobs of its length, hence "up to". The real run compares each pair byte for byte before it touches anything, and links only the pairs that are equal. Without `--dry-run` it shows the same preview and asks once (this run used `--yes`, which skips the question), and then, while it reports each repo on stderr as it starts:

```
…
  Reclaimed 154.16 GiB: 228 files replaced by hard links to their blobs.
  Compared 309.23 GiB in 1 min 34 s (3.29 GiB/s).
  Left alone: 3 files (864.40 MiB):
    google/gemma-scope-2b-pt-transcoders  50eec2f25c60545a9a74c1c3a26a0afdd0b4b872\layer_0\width_16k\average_l0_76\params.npz: no blob of its length holds the same bytes
    google/gemma-scope-2b-pt-transcoders  50eec2f25c60545a9a74c1c3a26a0afdd0b4b872\layer_12\width_16k\average_l0_6\params.npz: no blob of its length holds the same bytes
    google/gemma-scope-2b-pt-transcoders  50eec2f25c60545a9a74c1c3a26a0afdd0b4b872\layer_25\width_16k\average_l0_41\params.npz: no blob of its length holds the same bytes
```

That is exactly the 154.16 GiB the `DUP` footer named, and `C:` gained it, to within a few hundred KiB of cluster slack. The 0.84 GiB the preview over-counted is the three files left alone: that repo holds 27 files of one length but only 24 blobs, so three of its files are stored in the snapshot alone and are not copies of anything; the comparison found that out, and says so. `du` afterwards:

```
    #                      SIZE  REPO                                                     FILES
    1                159.05 GiB  mntss/clt-gemma-2-2b-2.5M                                   53
    2                 40.62 GiB  bluelightai/clt-qwen3-1.7b-base-20k                         88
    3                 34.54 GiB  datasets/TheQweaker/rented_boxes                          1952  ●
…
  ─────────────────────────────────────────────────────────────────────────────────────────────
  609.07 GiB  total (72 repos, 2839 files)
  By volume: C: 608.55 GiB · D: 525.44 MiB
…
```

The `DUP` column is gone, row 2 is half the size it was, and `du microsoft/Phi-3.5-mini-instruct` now prints a single `7.12 GiB  total (20 files)`. A second run finds the same three files, compares them again (a fraction of a second; they differ early) and reports `Nothing reclaimed`. Name repos (or `du` indexes) to limit it, add `--verify` to also check each copy against the SHA-256 its blob is named by, and note that a repo with a download in progress, or one interrupted and left to resume, is skipped whole. Hard-linked files share their bytes, so editing one in place would change the other; no download tool does that. From v0.12.6 on, new downloads on Windows do not stay stored twice for LFS files: the chunked path makes a hard link where a symlink is refused, and the copy `hf-hub` writes is linked to its blob once the download is done.

## Acting: `delete`, `clean-partial`, `gc`

Three commands, in increasing blast radius. All preview before acting.

**One repo:** `cache delete` takes a repo ID or a `du` index, shows what it is about to remove, and asks:

```sh
hf-fm cache delete julien-c/dummy-unknown
```

```
  julien-c/dummy-unknown  (276.5 KiB, 8 files)
  Delete? [y/N] n
  Aborted.
```

It removes the repo's entire `models--…` directory — blobs, refs, snapshots — so nothing is left half-alive. Pass `--yes` to skip the prompt in scripts.

**Several at once, and a dry run.** Since v0.12.5 `cache delete` takes any number of repos, and `--dry-run` says what would go and what it frees without deleting or prompting. Every argument is resolved before anything is removed, so one that does not exist deletes nothing, and the list is de-duplicated (an index and an ID can name the same repo):

```sh
hf-fm cache delete julien-c/dummy-unknown EleutherAI/pythia-70m --dry-run
```

```
  julien-c/dummy-unknown  (276.7 KiB, 8 files)
  EleutherAI/pythia-70m  (607 B, 1 file)
  Dry run: would delete 2 repos, freeing 277.2 KiB. Nothing was deleted.
```

Mind the indexes. They are the `#` of the last `du`, and they shift after each deletion, so a list of indexes read off one `du` and deleted one call at a time would remove the wrong repos from the second call on. In one call they are all resolved against the same listing, and each prints the repo it stands for, `--yes` or not (`index 3 is org/name`). A dataset or Space is deleted behind its prefix: `cache delete datasets/org/name`.

**A repo moved to another disk.** To free `C:` without a second cache, a repo can be copied to another disk and a junction (or symlink) left at its old path; `du` follows it and marks the row `→ D:\hf-cache\models--gpt2`, and the footer says what each disk holds (`By volume: C: 762.71 GiB · D: 525.44 MiB`). Deleting such a repo removes the link and leaves the bytes, and says so:

```sh
hf-fm cache delete gpt2 --dry-run
```

```
  gpt2  (525.44 MiB, 7 files)  → D:\hf-cache\models--gpt2
  Only the link is removed: its 525.44 MiB stay at D:\hf-cache\models--gpt2. `--with-target` deletes them too.
  Dry run: would delete 1 repo, freeing 0 B. Nothing was deleted.
```

`--with-target` deletes the bytes behind the link as well, only from a directory named like the repo's cache folder, and it always asks: `--yes` does not answer it.

**Leftover partials:** suppose you had *abandoned* that interrupted Llama download instead of resuming it. `cache clean-partial` sweeps the leftover `.chunked.part` files (and their resume sidecars) across the whole cache, with the same `--dry-run` / prompt / `--yes` controls. Run the dry run first:

```sh
hf-fm cache clean-partial --dry-run
```

```
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub

Would remove 3 files (13.87 GiB):
  NousResearch/Meta-Llama-3.1-8B: c28b25e7…chunked.part  (4.66 GiB)
  NousResearch/Meta-Llama-3.1-8B: d8e9504d…chunked.part  (4.58 GiB)
  NousResearch/Meta-Llama-3.1-8B: f8b9704a…chunked.part  (4.63 GiB)
```

The reclaimed figure is real disk on Windows: a chunked download preallocates each temp blob at its *full* size, so a half-finished 4.66 GiB shard occupies 4.66 GiB on disk even though `status` (reading the sidecar) correctly reports only the bytes truly transferred. Pass a repo ID to scope the sweep to one model (`cache clean-partial NousResearch/Meta-Llama-3.1-8B`); omit it to clean everything. Either way, heed the lesson above — a partial is resume state, so clean only the downloads you have given up on. An empty cache simply reports `No partial downloads found.`

**The sweep:** `cache gc` evicts by either of the two budgets you actually think in. `--older-than <DAYS>` answers *"what haven't I used lately?"* (the 30-second answer above). `--max-size <SIZE>` answers *"how much space can I spare?"* — it removes oldest-first until the cache fits the target. Both combine, and `--except` protects repos you want kept regardless of age:

```sh
hf-fm cache gc --max-size 400GiB --except bluelightai/clt-qwen3-1.7b-base-20k --dry-run
```

```
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub

note: 1 relocated repo skipped (the bytes are behind a link, not in this cache); `hf-fm cache delete` removes one
Will remove:
  Qwen/Qwen2.5-Coder-7B-Instruct           14.19 GiB  9 months ago
  google/codegemma-7b-it                   15.92 GiB  9 months ago
  …
  mntss/clt-gemma-2-2b-2.5M               159.05 GiB  8 months ago
  …
  nvidia/Llama-3.1-8B-Instruct-FP8         12.98 GiB  7 months ago

Skipped (relocated behind a link; evicting them frees nothing here):
  gpt2  525.44 MiB

Protected by --except:
  bluelightai/clt-qwen3-1.7b-base-20k

Cache: 728.67 GiB → 390.12 GiB (free 338.55 GiB)
  154.16 GiB of 763.22 GiB are duplicated copies (21 repos)
  gc selects whole repos by age and budget and does not deduplicate: removing models to recover space that deduplication would recover without losing any is what this figure warns against. `hf-fm du` names the copies per repo. `hf-fm cache dedupe` reclaims them and keeps every model.
Note: 3 dataset or Space repos (34.55 GiB) not counted above; gc evicts models only.
```

Two caveats worth knowing before you trust GC. "Last accessed" is approximated by the newest modification time among the repo's snapshot files (or, for a repo that has none, such as one you only inspected with `--cache-headers`, among its other files) — the HF cache layout does not record true access times, so a repo you *read* daily but never re-download looks old; protect it with `--except`. And repos with a partial download modified within the last hour are skipped automatically, so GC never races an `hf-fm` download running in another shell.

## Trust, but verify: `cache verify`

After a crash, a flaky disk, or before archiving a cache you plan to rely on offline, re-check the bytes against HuggingFace's LFS metadata (needs network for the expected hashes — the files themselves are read locally):

```sh
hf-fm cache verify bartowski/Qwen2.5-0.5B-Instruct-GGUF
```

```
bartowski/Qwen2.5-0.5B-Instruct-GGUF (main @ 41ba88dbac95fed2528c92514c131d73eb5a174b)
Cache: C:\Users\Eric JACOPIN\.cache\huggingface\hub

  — .gitattributes                         3.1 KiB  no LFS hash
  ✓ Qwen2.5-0.5B-Instruct-IQ2_M.gguf    313.37 MiB  SHA256 OK
  ! Qwen2.5-0.5B-Instruct-IQ3_M.gguf    326.87 MiB  MISSING
  …

27 files: 1 SHA256 OK, 0 mismatch, 3 skipped, 23 missing
```

Read the legend, not the row count: `✓` is a re-hashed match, `✗ SHA256 MISMATCH` is real corruption (delete the repo and re-download; `verify` also exits non-zero), `—` marks small non-LFS files that have no upstream hash to check, and `!` `MISSING` simply means never downloaded — entirely normal in a quant repo like this one, where you fetched one variant out of 27 files. The one number that should always be 0 is `mismatch`.

## Scripting: `cache path`

Where visibility meets the rest of your toolchain: `cache path` prints a repo's snapshot directory and exits non-zero if the repo is not cached.

```sh
hf-fm cache path google/gemma-2-2b-it
```

```
C:\Users\Eric JACOPIN\.cache\huggingface\hub\models--google--gemma-2-2b-it\snapshots\299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
```

```sh
cd $(hf-fm cache path google/gemma-2-2b-it)        # bash / zsh
cd (hf-fm cache path google/gemma-2-2b-it)         # PowerShell
```

## What you've learned

| Question | Command |
|----------|---------|
| Where are the bytes? | `du`, `du --age`, `du --tree` (models, datasets and Spaces) |
| What is stored twice (Windows)? | `du --sort dup`, `du <N>`, `du --verify-duplicates` |
| Get the copies' space back, keeping every model | `cache dedupe` (`--dry-run` first; `--verify` to also hash) |
| What's inside repo #N? | `du <N>` |
| Is this repo healthy? | `status <repo>`, `cache verify <repo\|N>` |
| Remove repos | `cache delete <repo\|N>...` (`--dry-run`; a repo behind a link loses the link, `--with-target` the bytes) |
| Remove interrupted-download leftovers | `cache clean-partial` |
| Reclaim space in bulk | `cache gc --older-than D` / `--max-size S` (+ `--except`, always `--dry-run` first) |
| Hand the path to another tool | `cache path <repo>` |

One model in one sentence: `du` sees, `cache` acts — and every acting command previews or prompts before a byte is deleted.

For the other half of cache discipline — deciding what is worth downloading *before* it lands on your disk — see the companion tutorial, [Inspect before you download](inspect-before-downloading.md). The [FAQ's cache section](../FAQ.md#cache-location-and-management) and the [CLI reference](../cli-reference.md#cache-commands) cover every flag mentioned here.
