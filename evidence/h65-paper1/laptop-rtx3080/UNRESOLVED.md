# Unresolved: Qwen3-14B / Gemma 2 27B RTX 3080 evidence (paper Table 9, Figure 7-8)

This needs Oney's confirmation before it gets published. Written during the
2026-09-27 arXiv-readiness audit; nothing here has been guessed into the evidence
folder.

## What the paper reports

Table 9 (Section 5.4), same-engine comparison on an RTX 3080 Laptop GPU:

| Method | Qwen request (s) | Qwen peak GPU (GB) | Gemma request (s) | Gemma peak GPU (GB) |
|---|---:|---:|---:|---:|
| traffic control | 14.87 | 4.136 | 41.58 | 4.171 |
| H6.5 | 14.21 | 3.611 | 36.84 | 3.660 |

Plus a 32-token Qwen boundary study (Section 5.5, Table 10) and the external-package
matrices behind Figure 7.

## Why the source file is not yet in `evidence/`

The private evidence bundle (`paper1-5090-bundle-20260911/corpus-3080/`) is the
*legacy H0-H18 corpus* -- the same data already public in this repository's
`results/` folder -- not an H6.5-specific laptop campaign. The bundle's own README
says plainly: "There are no Gemma runs on the 3080: breadth there is Qwen3-14B and
Mistral-Small-24B." That statement is about the bundle's own `corpus-3080/` snapshot
(dated through 2026-08-26); it does not match the paper, which reports Gemma 2 27B
IT on the RTX 3080 (System A) as a full second study, D9/D10.

## What a numeric search found instead

Scanning `scripts/local/paper1/output/` (gitignored, in the working tree used for
this audit, not in the bundle) for values close to the four numbers above turned up
two later-dated directories whose numbers are close but not exact matches:

- `paper-h65-gemma-practical-20260909-r2b/matched-ttft/gemma2-27b-h65-matched-ttft-practical-20260909-r2b-1tok.json`
  and the sibling `.../planner/gemma2-27b-h65-pilot-practical-20260909-r2b.json` --
  contain a `decode_seconds`/`seconds_per_token` value of about 36.84, matching the
  paper's Gemma H6.5 request time to three significant figures.
- `paper-h65-gemma-practical-20260909-r2b/external-ttft/...` -- contains a
  `wall_seconds` value of about 41.56-41.59, close to but not an exact match for the
  paper's Gemma traffic-control request time of 41.58.
- No file with an equally strong match for the Qwen row (14.87 / 14.21 / 4.136 /
  3.611 together) was found. Several files matched one number in isolation (e.g. a
  `p1-e7.2-figure8-scaling/qwen3-14b.json` with `seconds_per_token: 14.213`), but
  those files' *other* fields (peak VRAM 2.234 GB, not 3.611 GB) rule them out --
  they are an older, unrelated legacy-H6 scaling study that happens to share a
  similar latency number.
- At least four other `paper-h65-gemma-practical-*` and `paper-h65-qwen*-practical-*`
  directories exist with different date/revision suffixes (`20260902`, `20260903`,
  `20260903-r2-vram27`, `20260909-r2b`), consistent with multiple superseded
  attempts. Nothing in this audit's scope (no access to a run log or validity ledger
  for these specific directories) can tell which one is the frozen, paper-eligible
  run without the author's own memory of which attempt is final.

## What is needed to close this out

1. Confirm (or correct) that `paper-h65-gemma-practical-20260909-r2b` and its Qwen
   sibling are the final, paper-eligible D6-D10 runs.
2. If so, treat them the same way D1/D2/D4 were treated above: copy the frozen
   summary artifact (not the full trace/log tree) into
   `evidence/h65-paper1/laptop-rtx3080/`, record its hash in `MANIFEST.sha256`, and
   extend `verify_llama_confirmations.py` (or a sibling script) to recompute Table 9
   and Table 10 the same way.
3. Either update the bundle's own README to note the newer laptop campaign exists
   outside its snapshot, or fold the D6-D10 artifacts into a future bundle revision,
   so the "what's public" story is consistent in one place.
4. Write the missing protocol docs for D6-D10 and the 32-token boundary study
   (`docs/PAPER1_5090_*` currently only covers the Llama/RTX 5090 side).
