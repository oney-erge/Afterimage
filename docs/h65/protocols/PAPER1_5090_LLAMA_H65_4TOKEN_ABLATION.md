# RTX 5090 Llama H6.5 four-token representation ablation

Status: frozen on 2026-09-10 while the independent one-token confirmation's
first cell was still running. This is a secondary validation dataset, not a
replacement for the primary confirmation.

## Fixed purpose

The primary experiment measures cold one-token latency. This compact follow-up
checks whether its direction persists over four generated tokens and separates
the novel placement contribution from H6.5's optional compressed-RAM
representation selection.

Three immutable plans are compared:

1. traffic-density placement, SHA-256
   `926b0adb51275de35473a76ac3fc7dc7d82c58d3669027118f5bd433bb585563`;
2. H6.5 placement-only endpoint plan, SHA-256
   `a31a0d4bbe2309998514c70182a04bce2024c2a385dc99bfd1c0585f0aa3ebb4`;
3. full H6.5 endpoint/representation plan, SHA-256
   `99d0869e6be6b0e2e55249a3a2a189d8ef24889581b951163dd39e0bbe762f67`.

Both H6.5 plans keep the exact output head in decoded VRAM. The full plan's
frozen tier fingerprint is
`bffe4333c117fdbf4c854745c859c4b40b2b731a029797bd3c05a5294ce17ab1`.
No plan is rebuilt or selected from this experiment.

## Fixed execution

- Model: `meta-llama/Llama-3.3-70B-Instruct`, exact existing store.
- Logical budget: 8.0 decimal GB VRAM and 16.0 decimal GB RAM, with the same
  0.5 GB planner safety allowance.
- PyTorch caching-allocator cap: 10.0 decimal GB; whole-process NVIDIA-SMI
  ceiling: 12.0 decimal GB.
- One untimed warm-up token followed by exactly four timed tokens.
- Fresh process, successful cold-cache drop, and clean measured thermal/power
  window for every cell.
- Ten-minute cap per cell and 60-minute total cap.

Three reserved cases form three blocks with balanced method position:

| Block | Case | Order |
|---:|---|---|
| 0 | `confirm-explain-tides` | traffic, placement-only, full |
| 1 | `confirm-code-deduplicate` | placement-only, full, traffic |
| 2 | `confirm-retrieval-quartz` | full, traffic, placement-only |

The primary secondary contrast is traffic versus placement-only. Full versus
traffic and full versus placement-only are ablations. Report every block,
geometric speedup for all three paired contrasts, generated-token identity,
and whole-process peak VRAM. There is no retry, optional stopping, outlier
deletion, prompt replacement, or budget change. The run may start only after
the one-token confirmation completes all validity gates; its outcome is not
used to change this frozen matrix.
