# Execution Plan — Build, Deploy, Test End to End

Written against **this machine's actual hardware**, measured 2026-08-17, not a
hypothetical rig. Supersedes the target-hardware section of
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) #0.

---

## 0. What you already have

| | Measured | vs. the plan's target rig |
|---|---|---|
| GPU | RTX 3080 Laptop, **8192 MiB VRAM**, driver 596.49 / CUDA 13.2 | **matches** (target was 8 GB consumer GeForce) |
| VRAM actually free | ~6.7 GB (Windows desktop holds ~1.5 GB) | **tighter than planned** — see §1.5 |
| Host RAM | **39.4 GB** | 2.5x the 16 GB target — a *problem*, see §1.3 |
| Fast NVMe | **Samsung 980 PRO 2TB** (Gen4), D: with 1.17 TB free | matches on paper — **but not reachable at speed yet**, see A.5 |
| Slow NVMe | Intel SSDPEKNU010TZ 1TB (QLC Gen3) | currently hosts the WSL VHDX; also a free bandwidth ablation |
| **Measured storage** | **2.0 GB/s** sustained O_DIRECT (native ext4 on the Intel drive) | below target; fixable by relocating the VHDX (A.5) |
| OS | Windows 11 + **WSL2 Ubuntu-24.04 already installed**, GPU passthrough verified live | WSL2 enables everything in Stage A |
| PyTorch | 2.11.0 **+cpu** (Windows), absent in WSL | must be installed, see A.2 |

Verified working on 2026-08-17, not assumed: `/dev/dxg` present, `nvidia-smi`
inside WSL reports the RTX 3080 with 8192 MiB, passwordless sudo, and
`/proc/sys/vm/drop_caches` writable. Scripts: `scripts/verify_rig.sh`,
`scripts/verify_storage.sh`, `scripts/verify_odirect.py`.

**Bottom line: total cloud spend for the critical path is $0.** The only
optional paid step is §3.3, and it is a convenience, not a requirement.

Three hardware facts change the plan materially:

1. **39 GB of RAM defeats the entire NVMe benchmark** unless constrained. A
   16 GB model will sit entirely in the page cache and you will measure RAM
   speed while reporting NVMe numbers — the exact failure IMPLEMENTATION_PLAN
   #4.1 calls "the one that invalidates most disk-offload benchmarks."
2. **39 GB of RAM also means the honest *product* configuration for this
   laptop is RAM offload, not NVMe** (see §6.1). The NVMe path is the
   *research* configuration. Do not conflate them.
3. **Two NVMe drives at different generations** is a natural experiment most
   papers cannot run: the same method, same machine, same model, only storage
   bandwidth varying. Take it.

---

## Stage A — Make the machine ready (half a day, $0)

### A.1 Work inside WSL2, not Windows

Everything benchmark-related runs in WSL2 Ubuntu-24.04. Reasons, all load-bearing:

- `/proc/sys/vm/drop_caches` works (real Linux kernel) — Windows has no equivalent
- root is available for cache dropping and cgroups
- `.wslconfig` can cap RAM to simulate a 16 GB machine
- `bench/cachectl.py::is_cache_control_available()` returns True, so the
  harness will actually certify NVMe numbers instead of refusing to

```bash
wsl -d Ubuntu-24.04
```

### A.2 Install CUDA PyTorch

The current install is CPU-only. Inside WSL2 — **do not install an NVIDIA
driver inside WSL**, the Windows driver is passed through:

```bash
python3 -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True NVIDIA GeForce RTX 3080 Laptop GPU`. If False, update the
Windows-side driver and confirm `/dev/dxg` exists in WSL.

Then the project deps:

```bash
cd /mnt/c/for\ fun/Afterimage
pip install -e ".[models,dev]"
pytest tests/ -q          # expect 47 passed
```

### A.3 Cap WSL2 RAM to simulate the target machine

Create/edit `C:\Users\oneye\.wslconfig` (Windows side), then `wsl --shutdown`:

```ini
[wsl2]
memory=14GB
swap=0
processors=8
```

This does double duty: it creates the honest "16 GB consumer laptop" scenario
the project targets, **and** it makes it physically impossible for the page
cache to hold a 16 GB model. Verify:

```bash
free -g                    # total should show ~14
```

Keep a second profile at `memory=32GB` for the RAM-offload product
configuration (§6.1). Switching requires `wsl --shutdown`, so batch your runs.

### A.4 MEASURED: `drop_caches` alone is NOT sufficient on this machine

This was tested, not assumed, on 2026-08-17. Run it yourself:

```bash
bash scripts/verify_rig.sh          # preconditions
bash scripts/verify_storage.sh      # the storage truth table
python3 scripts/verify_odirect.py   # O_DIRECT correctness
```

**Measured results (same file, same drive, `dd bs=1M`):**

| Path | 2 GB after `drop_caches` | 2 GB warm | 24 GB O_DIRECT | 24 GB buffered |
|---|---|---|---|---|
| WSL native ext4 | 3.5 GB/s | **14.8 GB/s** | **2.0 GB/s** | 4.3 GB/s |
| DrvFs `/mnt/d` (Samsung) | 372 MB/s | 373 MB/s | 380 MB/s | — |
| DrvFs `/mnt/c` (Intel) | 389 MB/s | 372 MB/s | 381 MB/s | — |

Three conclusions, all of which change the plan:

1. **`drop_caches` does not give you a cold read here.** It drops *Linux's*
   page cache, but the WSL filesystem is a VHDX file that **Windows** also
   caches, and Linux cannot reach that. The 24 GB buffered read still hit
   4.3 GB/s against a true 2.0 GB/s device — a **2x inflation**, and the
   small-file warm read inflated **7x**. A benchmark trusting the buffered
   path would have been fiction.

2. **O_DIRECT is mandatory, and it works.** It bypasses the page cache at the
   syscall level so no cache above or below can defeat it.
   `runtime/directio.py` implements it; `TieredStore(direct_io=True)` uses it;
   `store.io_mode_report()` refuses to call a buffered run a storage
   measurement. Verified byte-identical to buffered reads on both filesystems.

3. **Never put the weight store on `/mnt/c` or `/mnt/d`.** DrvFs is a hard
   ~380 MB/s wall — **5x slower than the same machine's native ext4 and ~17x
   slower than the Samsung's rated speed** — and O_DIRECT does not rescue it.
   Benchmarking there would make everything look catastrophically I/O bound
   for reasons that have nothing to do with the method.

### A.5 Put the weight store on native ext4 — and move the VHDX

The WSL VHDX is at:

```
C:\Users\oneye\AppData\Local\wsl\{db706872-72db-402d-825e-5bc2fa2ef0d2}\ext4.vhdx
```

That is on **C:, the Intel SSDPEKNU010TZ** (QLC Gen3) — not the Samsung. It
is why sustained O_DIRECT measures 2.0 GB/s rather than the Samsung's rated
~7 GB/s.

Weight store goes on native ext4, never DrvFs:

```bash
mkdir -p ~/afterimage/{models,nvme_store,results}
```

**To get true Gen4 bandwidth, relocate the distro to the Samsung (D:):**

```powershell
wsl --shutdown
wsl --export Ubuntu-24.04 D:\wsl_backup\ubuntu.tar
wsl --unregister Ubuntu-24.04
wsl --import Ubuntu-24.04 D:\wsl\Ubuntu-24.04 D:\wsl_backup\ubuntu.tar --version 2
```

Then re-run `scripts/verify_storage.sh` and confirm sustained O_DIRECT rises
toward 5–7 GB/s. **Do this before Stage E, not after** — every stored number
depends on it, and re-running the matrix is expensive.

If you skip it, that is defensible, but then report the storage tier honestly
as "~2 GB/s Gen3-class NVMe," never as "Samsung 980 PRO Gen4."

### A.6 Free the VRAM Windows is holding (optional, ~1.5 GB)

The desktop occupies ~1.5 GB of the 8 GB. If the laptop has a MUX switch or
Optimus setting (ASUS Armoury Crate is present), set the display to the AMD
integrated GPU so the 3080 is compute-only. This moves usable VRAM from
~6.7 GB to ~7.9 GB, which is a >15% change in the residency budget and will
move every result. **Decide once and keep it fixed for all runs** — record
which mode was used in every result file.

---

## Stage B — Phase 0 on a proxy model (1 day, $0)

Purpose: get the *shape* of the answer fast, and shake out the probe code on a
real transformer before spending a long run on 27B.

**Model:** Qwen3-1.7B (fits in VRAM comfortably, real architecture, real
tokenizer).

### B.1 Adapt the probe for real-model scale

`probe/spectra.py` currently takes a full activation matrix `X`. At real scale
that does not fit: 2000 tokens x 5376 dims x 434 linear ops is ~18 GB of
captured activations. Two required changes:

- **Subsample layers.** Measure 8 layers spread across depth (early, middle,
  late), not all 62. Rank behaviour varies smoothly with depth; 8 points
  characterise the curve.
- **Stream, don't store.** Use `runtime/basis.OnlineBasis` to build the basis
  incrementally per layer, and accumulate only the scalar error statistics.
  The class already does this — it was built for the runtime, and it is the
  right tool here too.

### B.2 Run the four workloads

Per IMPLEMENTATION_PLAN #2.3, and they are expected to differ a lot:

| Workload | Expectation |
|---|---|
| Focused code (one file, one language) | best case — narrowest subspace |
| Multi-turn chat | moderate drift |
| Long-form prose (2000+ tokens) | moderate |
| Adversarial topic-switch (every 50 tokens) | worst case — motivates clustering |

Report each separately. **Do not average them.** A method that only works on
focused code is still useful; averaging hides that.

### B.3 What you are looking for

The deliverable is one table per layer: functional error `‖W x⊥‖/‖W x‖` vs.
rank, **measured closed-loop** (`probe/closed_loop.py`), alongside the
variance-captured curve. The gap between the two curves is the
rogue-dimension effect and is expected to be large.

---

## Stage C — Phase 0 on the real target (1–2 days, $0)

### C.1 Run 27B on CPU, not GPU

Counter-intuitive but correct: Phase 0 is *measurement*, not decoding speed.
Gemma-3-27B at Q4 is ~16 GB and **fits in your 39 GB of RAM**. A CPU forward
pass is slow (~1–2 tok/s) but you only need a few thousand tokens of
activations, and you avoid the entire offloading problem while measuring.

For this stage only, run WSL with the **32 GB** profile from A.3.

Overnight run, free, no cloud account needed.

### C.2 Optional paid shortcut

If CPU throughput is too painful, one A100-40GB hour on RunPod/Vast/Lambda
(single-digit dollars) loads 27B entirely in VRAM and finishes the capture in
minutes. This is a convenience purchase, not a requirement. Prices move —
check current rates rather than trusting a number written here.

### C.3 THE DECISION GATE

This is the point the whole project pivots on.

| Outcome | Action |
|---|---|
| Functional error < 1e-3 at rank ≤ 256 | Proceed to Stage D. The bet is live. |
| Only holds at rank ≥ 1024 | Proceed, but reset expectations to ~2x and say so publicly |
| No usable rank below 0.5·d | **Stop. Go to §6.1.** The Kolmogorov barrier is confirmed |

**Publish the measurement either way.** Within-session activation rank on a
27B model is unpublished, and the negative result is as useful as the positive
one. This is the one deliverable that is guaranteed regardless of outcome.

---

## Stage D — Build the real runtime (2–3 weeks)

Only if C.3 passes. Ordered by what the current codebase is missing
(IMPLEMENTATION_STATUS.md "What is NOT implemented").

### D.1 Real model integration (highest risk, do first)

Replace `testing/toy_lm.py` with a real HF model. The work:

- Wire `probe/hooks.py` to a real `AutoModelForCausalLM`
- Serialize real weights into `runtime/layout.py`'s plane ladder on D:
- Replace `AfterimageLayer` into the real model's `nn.Linear` slots via module
  substitution (the pattern in `probe/closed_loop.py::TruncatedLinear`
  already does exactly this, transactionally)
- Real KV cache handling — **entirely absent** from the toy path and a genuine
  design problem, since the KV cache competes with the sketch for the same
  6.7 GB

### D.2 CUDA streaming

Replace the thread-based overlap in `runtime/streamer.py` with
`torch.cuda.Stream` + pinned staging buffers. Remember there is **no
GPUDirect Storage on GeForce** — the path is NVMe → pinned host buffer →
PCIe → VRAM, and the pinned buffer costs host RAM you have capped at 14 GB.

### D.3 Tree speculation

Extend `runtime/verify.py` from a linear chain to a SpecExec-style tree with a
tree attention mask. The chain proves correctness; the tree is what delivers
the acceptance lengths SubSpec/SpecExec report. Without it you will not
reproduce the ~9x baseline in Stage E.2 and every later comparison is
uninterpretable.

### D.4 Clustered subspaces

Only if Stage B's adversarial workload showed a single global basis failing.
HYPOTHESIS #3.3.

### D.5 Sensitivity calibration

`GlobalController` currently defaults every layer's `s_ℓ` to 1.0. The
water-filling argument for a single global λ *requires* real per-layer
logit sensitivities. Calibrate against the real model.

---

## Stage E — Test end to end

### E.1 Fix the confound before measuring anything

Three configurations must be distinguished, and conflating them is the
easiest way to produce a meaningless result:

| Config | RAM cap | Model location | I/O mode | What it measures |
|---|---|---|---|---|
| **R** (product) | 32 GB | host RAM | n/a | fastest this laptop can go |
| **N** (research) | 14 GB | native ext4, Samsung after A.5 | **O_DIRECT** | the bandwidth-starved claim |
| **N-slow** | 14 GB | native ext4, Intel Gen3 | **O_DIRECT** | bandwidth sensitivity |

**Every run in N and N-slow must assert `store.direct_io_effective is True`
before its numbers are recorded.** The measurements in A.4 show buffered
reads inflating storage bandwidth by 2–7x on this exact machine; the assert
is what stops that from silently entering a result table. `bench/harness.py`
should refuse to aggregate a trial whose store reports otherwise.

### E.2 Reproduce the known art FIRST

Before any Afterimage number is trustworthy, reproduce **SubSpec-class ~9x
over AirLLM at 8 GB** using residency + tree speculation alone (config C in
the test matrix). If you cannot reproduce published results with published
methods, your engine is wrong and every subsequent number is noise.

This is a gate, not a formality.

### E.3 The test matrix

Run `bench/harness.py` over rows A–H from IMPLEMENTATION_PLAN #9, N=5,
interleaved and shuffled, median + IQR, unstable configs re-run not averaged.

**The two numbers that decide the project:**

- **D vs B** — the cache's contribution in isolation. Threshold ≥1.3x.
- **E vs C** — does the cache survive contact with speculation? Threshold ≥1.15x.

If E vs C < 1.15x, the cache is redundant with speculation. That is the
central scientific risk in HYPOTHESIS #4, and the honest response is to
publish the Phase 0 measurement and ship config C as the product.

### E.4 Laptop-specific benchmark hazards

Your rig has failure modes a desktop does not:

| Hazard | Control |
|---|---|
| **85 W power cap + thermal throttling.** A laptop 3080 will downclock over a long run and silently bias whichever config runs later | Interleave and shuffle (harness already does); log `nvidia-smi --query-gpu=clocks.sm,temperature.gpu` per run; discard runs where SM clock drops >15% |
| Windows desktop reclaiming VRAM mid-run | Fix A.6 mode once; assert free VRAM at run start |
| DrvFs (`/mnt/d`) overhead vs native ext4 | Pick one, document it, never mix across configs |
| Battery vs AC power profile | Always AC, high-performance profile, verify |

### E.5 Quality, not just speed

Speed with degraded output is worthless. Per IMPLEMENTATION_PLAN #10.3:
token-identity rate vs. an exact reference (≥99.9%), WikiText-2 perplexity,
GSM8K subset, HumanEval subset, and the **long-session drift test** — 4000
tokens, compare quality of the first 500 against the last 500. That last one
is unique to this method: the cache changes as the session runs, so quality
could drift in either direction and nobody else needs to test for it.

---

## Stage F — Deploy

### F.1 Ship an OpenAI-compatible endpoint

LocalDeploy "can also work with llama.cpp and loopback OpenAI-compatible
runtimes." That is the integration contract. Wrap the engine in a small
FastAPI server exposing `/v1/chat/completions` with streaming, and Afterimage
becomes a runtime LocalDeploy can drive.

### F.2 Let LocalDeploy do the benchmarking UI

LocalDeploy already "runs repeatable local benchmarks and compares accuracy,
latency, throughput, and memory use" and "exports benchmark reports."

**Do not rebuild any of that.** Register Afterimage as a loopback runtime and
you get comparison against Ollama and llama.cpp on the same machine, with a
UI and exportable reports, for free. `bench/harness.py` stays for the
byte-accounting metric (GB/accepted token) that LocalDeploy does not track,
since that is the project's primary metric and is invisible to a
throughput-oriented harness.

### F.3 Packaging

- `pip install afterimage` with `[gpu]` extra
- Dockerfile with CUDA base for the reproducible-benchmark path
- Pin the exact Q4 checkpoint by hash — IMPLEMENTATION_PLAN #4.2 requires
  identical weights across every configuration compared

---

## 6. Fallbacks, stated in advance

### 6.1 If Stage C fails the gate

**Ship the RAM-offload product.** With 39 GB of RAM, a 27B Q4 fits entirely
in host memory, and residency + tree speculation over PCIe (~20 GB/s) is a
genuinely good 27B-on-8GB experience on this laptop — roughly 3x the
bandwidth of the Gen4 NVMe path. It is published, reproducible engineering
with no novelty claim, and it delivers the original goal.

The NVMe path only matters for machines with less RAM than the model. That is
a real and common case, and worth supporting — but it is not *your* machine's
best configuration, and pretending otherwise would be dishonest benchmarking.

### 6.2 If the runtime build stalls

The Phase 0 measurement stands alone as a publishable artifact. It requires
only Stages A–C — about three days of work and zero dollars.

---

## 7. Sequence summary

| Stage | Time | Cost | Gate |
|---|---|---|---|
| A — machine ready | 0.5 day | $0 | `pytest` 47/47 on CUDA; cache-drop verified |
| B — Phase 0 proxy | 1 day | $0 | probe runs on a real transformer |
| C — Phase 0 on 27B | 1–2 days | $0 | **DECISION GATE** |
| D — real runtime | 2–3 weeks | $0 | reproduce ~9x baseline |
| E — test matrix | 1 week | $0 | D vs B ≥1.3x; E vs C ≥1.15x |
| F — deploy | 3 days | $0 | LocalDeploy drives it |

**Three days and zero dollars reaches the decision that governs the other six
weeks.** Start at A.1 tonight.
