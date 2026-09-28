# RTX 5090 Llama H6.5 corrected-placement viability pilot v3

Status: frozen before candidate selection or live evaluation. This supersedes
the v2 memory admission rule without erasing either stopped predecessor. This
is a regulated development pilot, not primary confirmation.

## Memory-accounting correction

The v2 run completed every calibration worker cleanly, but its controller
incorrectly treated `vram_cap_gb` as a whole-process device-memory ceiling.
That setting is a PyTorch caching-allocator limit. NVIDIA-SMI additionally
observes CUDA context, driver, and non-PyTorch allocations. In the v2 decoded-
RAM calibration, the allocator peak was 7.3166 decimal GB while the measured
whole-process delta was 10.2016 GB. This was not an OOM and did not select or
time an H6.5 candidate.

V3 freezes three distinct, equally applied quantities:

- **8.0 decimal GB logical residency-planning budget**;
- **10.0 decimal GB PyTorch caching-allocator cap**; and
- **12.0 decimal GB whole-process NVIDIA-SMI admission ceiling**.

The 12 GB admission ceiling was selected before candidate selection or live
evaluation from the completed v2 calibration peak, leaving 1.7984 GB of
headroom. It is not a claim that the experiment uses 8 GB total VRAM. Every
paper table must report both the 8 GB residency budget and each method's
measured whole-process peak.

## Reused development calibration

V3 reuses the completed v2 calibration rather than rerunning it. Reuse is
permitted because its runtime settings already were exactly 8 GB logical,
10 GB allocator, 16 GB RAM, 0.5 GB planning safety, decoder-table reuse,
4,194,304-element slices, fixed prefetch depth 2, per-blob reads, exact full
head, one untimed warm-up token, and one timed token. All five calibration
rows had zero worker exits, successful cache drops, observable clean thermal
windows, exact configuration capture, and retained traces. Only the mistaken
10 GB whole-process controller comparison failed.

Calibration prompts remain disjoint from evaluation:

1. disk traces: `summary-photosynthesis`, `logic-glippets`, `copy-nonce`;
2. decoded-RAM trace: `json-status`, `code-squares`.

The source v2 run is immutable evidence under artifact ID
`h65-fixed-pilot-llama70b-8v16r-logical-10vram-physical-rtx5090-20260910-v1`.
V3 records SHA-256 hashes for the imported traces, worker results, manifest,
H2D calibration, source commit, protocol, and private controller before use.

## Candidate selection

Corrected placement-only search uses seed `202609093`, 256 iterations, and
requires nonnegative predicted improvement on every training and held-out
trace. The first two disk traces train; the third is held out and may only
accept or reject. Full search is evaluated offline with placement's selected
candidate as an incumbent; it is not a second live arm. Plans and all hashes
are frozen before live evaluation.

Stop before timing if placement is physically identical to traffic, fails
replay, violates the logical budget, or full returns a worse offline objective
than its placement incumbent.

## Live paired pilot

| Block | Prompt | Adjacent order |
|---|---|---|
| 0 | `fact-gold` | traffic, H6.5 |
| 1 | `arithmetic-17x6` | H6.5, traffic |
| 2 | `code-square` | traffic, H6.5 |
| 3 | `retrieval-7319` | H6.5, traffic |

Each arm runs in a fresh process with the frozen runtime settings. All eight
cells require zero exits, exact paired token IDs, successful cache drops, a
completed warm-up, clean observable thermal windows, exact plan/tier
fingerprints, expected read counters, allocator enforcement, and measured
whole-process memory no greater than 12 GB. No automatic retry, optional
stopping, prompt replacement, outlier removal, or limit change is permitted.

Primary pilot estimand: geometric mean of the four paired ratios
`traffic_seconds / h65_seconds`, accompanied by every ratio and win count.
Advance only if all gates pass, geometric speedup is at least 1.03x, and H6.5
wins at least three of four blocks. Report separately whether every eligible
block improves by the frozen 1% deployment threshold.

No external framework, additional model, budget sweep, multi-token run, or
confirmation is automatically launched by this protocol.
