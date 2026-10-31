# Multi-GPU Roadmap

**Date:** October 31, 2026
**Status:** Proposed; waits on a second GPU in the maintainer's machine
**Context:** `inspect --check-gpu` reads one device and gives one verdict. This roadmap covers the release that widens it to every device and to a sweep of context lengths, plus the per-process and gated-repo features of the same family. It was a section of the [cache-management roadmap](cache-management-roadmap.md), which is the release plan for v0.9.2 to v0.12.x; it moved here on 2026-10-31, when that file passed 900 lines and its next three releases (v0.12.4 to v0.12.6) were not about GPUs.

---

## v0.13.0 — Multi-GPU + multi-context `--check-gpu` matrix

| Feature | Scope |
|---------|-------|
| `inspect --check-gpu all` | Multi-GPU verdict. Iterates over `hypomnesis::device_count()`, prints one `GPU N:` line per visible device with its own free/used numbers, and picks the device with the most free VRAM for the `Fit:` summary line. The `all` token is parsed via clap's `value_parser`: a bare `--check-gpu` keeps the v0.10.1 single-device default, `--check-gpu N` keeps single-device targeting, `--check-gpu all` enables the iteration. JSON path gains a `devices: [...]` array under `gpu_check` and the existing `device` key becomes an alias for the auto-picked entry. Closes the "I have a 5060 Ti and a 5050 in the same box" case without breaking the v0.10.1 single-device default. |
| `inspect --check-gpu --context-sweep` | Multi-context fit-profile sweep. Prints a table of `Total = W + KV [+ Mamba]` and Fit verdicts at a representative set of context lengths (e.g. 4 K / 16 K / 64 K / 128 K) instead of a single context number. Composes with `--check-gpu N` and `--check-gpu all`, so the matrix can be one-GPU-many-contexts or many-GPUs-many-contexts. Reuses v0.10.4's KV + Mamba budgeting math at each cell. Natural companion to `--check-gpu all` — both are *"show me a matrix of verdicts"* — which is why they ship together. |
| `inspect --check-gpu --processes` (hypomnesis `0.2.2` `PDH` backend) | Itemize the device's `used:` VRAM by process, so the verdict answers not just *"will it fit?"* but *"what do I need to close first?"*. The single-line `free: X, used: Y` becomes a short per-process breakdown (`ollama.exe 4.2 GiB`, `Code.exe 0.8 GiB`, …). Built on `hypomnesis::gpu_processes()`, whose Windows/`WDDM` per-process path landed in **hypomnesis 0.2.2** via the new `PDH` backend (reads the same `VidMm` data as Task Manager's *Dedicated GPU memory* column). hf-fm already carries the `pdh` feature for it (see Hypomnesis below), so nothing new compiles in; what the feature adds is the first call to `gpu_processes()`. It turns the consumer-GPU "I have 2.76 GiB used and don't know by what" gap into an answer. |
| `inspect --check-gpu` gated-repo estimated verdict | When the header Range request hits a gated repo's 401/403, fall back to the **public** file listing (gated repos' metadata — sizes, SHA256s — is public; only content is gated) and emit an *estimated* weights-only verdict from the summed `.safetensors` file sizes (file size ≈ weights + a few-KiB header; on Llama-3.1-8B the estimate and the header-precise figure are both 14.96 GiB). Output labels the provenance explicitly: `Fit (estimated from public file sizes — header-precise verdict needs the accepted license): ✗ short by 2.2 GiB`. `--context` cannot compose (a gated repo's `config.json` content is itself gated), so the estimate stays weights-only and says so. **Dogfooding origin (2026-06-12):** the "why must I accept a license before knowing whether the model even fits my GPU?" complaint while provisioning the Goodfire geometric-calculator replication; the session workaround — inspect a byte-identical ungated mirror, verified by matching LFS SHA256s — is documented in the FAQ's gated-model entry. This feature makes the workaround unnecessary for the weights question. Placed here because it extends the same `--check-gpu` verdict family, but small enough to pull forward into a patch release if demand surfaces: the public listing is already fetched by `list-files`, and the 401/403 detection hook shipped in v0.10.5. |

**Hypomnesis.** hf-fm pins `hypomnesis = "0.2.15"`, an optional dependency of the `cli` feature with `default-features = false` and the `nvml`, `nvidia-smi-fallback`, `dxgi`, `pdh` and `metal` features. The pin's history: it stayed at `0.2.1` through the first v0.10.x releases, because `--check-gpu` reads only `device_info` and 0.2.2's headline, the Windows `PDH` per-process backend, would have compiled unused; v0.10.5 moved it to `0.2.3` for macOS and Apple Silicon (`MTLDevice.recommendedMaxWorkingSetSize` is exactly the `device_info` figure `--check-gpu` reads), which turned "no GPU detected" into a real fit verdict on a Mac; and it has followed upstream since, to `0.2.15` in the v0.12.4 line. What this release would newly call is `device_count()` (for `all`) and `gpu_processes()` (for `--processes`); hf-fm calls neither today, and `pdh` is carried for the second at no cost, since it shares the `windows` crate that `dxgi` already pulls.

`hypomnesis` 0.2.15 gives both a firmer footing, without removing the need for hardware here: when `NVML` loads but cannot start, `device_info` and `Snapshot::all` now return the cause instead of reading as zero GPUs, `process_name` names a PID, and `hmn ps` shows which GPU each process is on. Upstream checked that on two multi-GPU Linux nodes (8× V100 and 2× H100 NVL under Slurm) and pins it in CI with a fake `NVML`. That is `hypomnesis` exercised on several devices, not hf-fm, so the hardware prerequisite below stands.

**Theme: broaden `--check-gpu` from a single verdict to a matrix.** v0.10.1 introduced single-device, weights-only `--check-gpu`; v0.10.4 added KV-cache and hybrid-Mamba budgeting at a user-supplied context. v0.13.0 closes out the family: instead of one device × one context giving one verdict, the user can ask for all devices × a context-length sweep and read off a table. Deferred to v0.13.0 because both features want real multi-GPU hardware to validate (the iteration ordering, the auto-pick heuristic, and the per-GPU sweep all need an actual second device on the box).

**Why v0.13 and not v0.11.x or any v0.12.x.** The v0.11 line is dedicated to remote-inspect framework work (`HttpRangeReader` + format coverage) — stitching multi-GPU into v0.11 would dilute that focus. v0.12.0 was then claimed by backlog cleaning (in the [release plan](cache-management-roadmap.md)) rather than left idle while waiting on hardware, since none of that phase's six items need a second GPU either — and v0.12.1's quant-aware fit planning follows the same reasoning, surfaced by dogfooding after the v0.12.0 scope was already fixed. v0.12.2's disk-accounting follow-through needed no second GPU either, and v0.12.3's inspect API ergonomics need none. That keeps each boundary clean: *v0.11 = format reach, v0.12.0 = backlog cleaning, v0.12.1 = quant/fit-planning reach, v0.12.2 = honest disk accounting, v0.12.3 = inspect API ergonomics, v0.12.4 = the TLS advisories, v0.12.5 = `du` names the duplicates, v0.12.6 = `cache dedupe`, v0.13 = GPU reach* — and means v0.13.0 can slip freely against hardware availability without blocking anything else.

**Hardware prerequisite.** Maintainer's box gains a second GPU (5050 8GiB alongside the 5060 Ti 16GiB) before this lands — the multi-GPU iteration ordering, the auto-pick heuristic, and the per-device sweep numbers all want a real second device to be exercised against, not a mock.

---

### Reference — RTX 5050 8 GB in PCIEX2 on the B550 GAMING X V2

> Compatibility analysis verifying that the **Hardware prerequisite** above is feasible: can a second NVIDIA GPU (specifically an RTX 5050 8 GB) be added to the maintainer's existing system (B550 GAMING X V2 + RTX 5060 Ti 16 GB) in the PCIEX2 slot? Conducted 2026-05-19; source: GIGABYTE *B550 GAMING X V2 User's Manual* rev. 1301, §1-2 *Product Specifications* (p. 6), §1-5 *Installing an Expansion Card* (p. 10), and motherboard layout (p. 4).

**Verdict.** The slot will physically and electrically accept the RTX 5050 8 GB, with three qualifications: PCIe 3.0 ×2 wiring (one-sixteenth the card's native bandwidth), physical clearance below the primary GPU must be verified, and the slot delivers only 75 W (rest from PSU).

**Slot spec — verbatim from the manual** (§1-2, p. 6):

> *"1 x PCI Express x16 slot (PCIEX2), integrated in the Chipset: Supporting PCIe 3.0 x2 mode"*

| Attribute | Value | Implication |
|---|---|---|
| Physical connector | x16 (full-length) | Any PCIe card width (x1/x2/x4/x8/x16) fits mechanically. |
| Electrical lanes | x2 (only 2 of 16 wired) | Card runs in compatibility mode at 2-lane width regardless of native width. |
| PCIe generation | 3.0 | Card auto-negotiates down from its native generation (PCIe 5.0 on Blackwell). |
| Lane source | Chipset (AMD B550), not CPU | Routes via the B550 chipset → CPU link (PCIe 4.0 ×4 on Ryzen 5000), shared with M.2 SB, SATA, other chipset slots/USB. |

**Compatibility — three dimensions.**

1. *Electrical / PCIe protocol — fully compatible.* PCIe is backward-compatible across generations and width-down-negotiable. RTX 5050 (PCIe 5.0 ×8 native) in this slot runs as PCIe 3.0 ×2 effective ≈ 1.97 GB/s peak unidirectional. Nothing in the manual prohibits or restricts which GPU goes here.

2. *Physical clearance — verify against the primary GPU.* Motherboard layout (manual p. 4) shows slot order top-to-bottom: `PCIEX16 → PCIEX1_1 → PCIEX1_2 → PCIEX2 → PCIEX1_3`. PCIEX2 sits ~3 slot pitches (≈ 6 cm) below PCIEX16. The risk is the existing 5060 Ti's cooler shroud overhanging the PCIEX2 area:

   | 5060 Ti slot width | Vertical occupation | Clearance to PCIEX2 |
   |---|---|---|
   | 2 slots | ~40 mm | ~20 mm headroom ✓ |
   | 2.5 slots | ~50 mm | ~10 mm headroom ✓ |
   | 3 slots | ~60 mm | Marginal — verify with calipers |
   | 3.5+ slots | ~70+ mm | Blocked ✗ |

   RTX 5050 8 GB is almost universally a 2-slot card, so it fits *into* PCIEX2 without issue; the only open question is whether the primary 5060 Ti's cooler overhangs it.

3. *Power.* The slot supplies the standard PCIe 75 W. RTX 5050 8 GB TDP ≈ 130 W → needs an external 8-pin (or 6+2) PCIe power cable from the PSU. The 850 W PSU has ample cables.

**Performance implications for the v0.13.0 pipeline-parallel use case.** For PP-style serving where the second GPU receives intermediate activations across a layer boundary:

| Operation | Data per event | Time at PCIe 3.0 ×2 (~2 GB/s) | Acceptable? |
|---|---|---|---|
| One-time model load (~6 GiB worth of weights to Cuda(1)) | 6 GiB | ~3 s | ✓ once per session |
| Prefill: per-layer-boundary activation transfer (seq=2048, hidden=4096, BF16) | 16 MiB | ~8 ms | ✓ |
| Incremental decode: per-layer-boundary transfer (seq=1) | 8 KiB | ~4 µs | ✓ trivial |
| KV cache transfer (KV lives on each layer's device — no cross-device transfer required) | n/a | n/a | n/a |

For autoregressive inference the PCIe 3.0 ×2 wiring is fine. Prefill on long sequences incurs ~8 ms per layer boundary (≈ 16 ms per token in an 18/14 split with both forward boundaries); decode is bandwidth-trivial.

**Two things the manual doesn't address but are worth knowing.**

1. *Chipset-link contention.* All chipset PCIe traffic shares one PCIe 4.0 ×4 link to the CPU (≈ 8 GB/s aggregate). Heavy concurrent M.2 SSD reads + GPU on PCIEX2 + USB traffic could in principle contend; for inference workloads this won't bite — 2 GB/s GPU + ≈ 1 GB/s M.2 reads at startup is well under the ceiling.
2. *No BIOS configuration required.* §1 and §2 of the manual mention no disabling or sharing of PCIEX2 lanes. The slot is hardwired ×2 from the chipset regardless of which other slots are populated. Both GPUs will appear in `nvidia-smi` and in `hypomnesis::device_count()` automatically — which is exactly what `inspect --check-gpu all` needs.

**Conclusion.** Hardware plan is viable for the v0.13.0 work. Only material risk before purchase is **physical clearance with the existing 5060 Ti's cooler** — measure first. PSU, power delivery, BIOS, and chipset routing are all fine.
