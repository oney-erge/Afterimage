# Software environment behind the H6.5 paper's numbers

The normal installers (`run.sh`, `run.ps1`) install the newest compatible
packages, which is right for using Afterimage but not for reproducing a
measurement. This page records what the paper's runs actually used.

## System B: RTX 5090 (every Llama-3.3-70B study)

Recorded by the run itself: the `environment` block of
[`llama-rtx5090/external-comparison/status.json`](llama-rtx5090/external-comparison/status.json),
which ran on the same machine, at the same commit (`637bbd55`), on the same day
as D2 and D4. The D1, D2, and D4 artifacts record the source commit and a hash of
every source file, but not package versions, so this is the closest recorded
source.

| Item | Version |
|---|---|
| GPU / driver / CUDA | RTX 5090 (34.19 GB) / 610.88 / 13.0 |
| OS | WSL2, Linux 6.18.33.2 (`Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`) |
| CPU / host RAM | AMD Ryzen 9 9950X3D, 32 logical CPUs / 50.5 GB |
| Python | 3.12.13 |
| torch | 2.13.0+cu130 |
| transformers | 5.12.1 |
| accelerate | 1.14.0 |
| safetensors | 0.8.0 |
| numpy | 2.5.2 |
| airllm / deepspeed / dfloat11 (baselines) | 3.3.0 / 0.19.6 / 0.5.0 |

To install exactly this set (Linux or WSL2, Python 3.12):

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[gpu,server,bench]" -c evidence/h65-paper1/constraints-rtx5090.txt
```

[`constraints-rtx5090.txt`](constraints-rtx5090.txt) was checked to resolve
together with this repository's own dependencies on Linux
(`uv pip compile ... --python-platform x86_64-unknown-linux-gnu`); it has not
been installed and run end to end outside the original machine.

## System A: RTX 3080 Laptop (Qwen3-14B, Gemma 2 27B)

Recorded by every laptop study itself (the `environment` block of each file in
[`laptop-rtx3080/`](laptop-rtx3080/README.md)), and checked by
`verify_paper_evidence.py`:

| Item | Version |
|---|---|
| GPU / driver / CUDA | RTX 3080 Laptop GPU (8.59 GB) / 596.49 / 12.4 |
| OS | WSL2, Linux 6.18.33.2 |
| CPU | AMD Ryzen 9 5900HS |
| Python | 3.12.3 |
| torch | 2.6.0+cu124 |
| transformers / accelerate | 5.12.1 / 1.14.0 |
| safetensors / numpy | 0.8.0 / 2.5.2 |
| airllm / deepspeed / dfloat11 (baselines) | 3.2.0 / 0.19.5 / 0.5.0 |

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[gpu,server,bench]" -c evidence/h65-paper1/constraints-rtx3080.txt
```

[`constraints-rtx3080.txt`](constraints-rtx3080.txt) was checked the same way as
the RTX 5090 file.

## Why this matters for the numbers

H6.5's placement decision does not depend on these versions (it is offline
replay over recorded traces; see
[`llama-rtx5090/plan-rederivation/`](llama-rtx5090/plan-rederivation/README.md)).
The measured latencies and peak-memory numbers do: a different torch build,
allocator, or driver can move them. Compare against a rerun only after matching
this environment, and report the environment you ran with.
