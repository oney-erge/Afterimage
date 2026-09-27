---
name: H6.5 reproduction report
about: You ran the paper's evidence or reproduced (or failed to reproduce) an H6.5 number
labels: h6.5
---

**Which number or claim**
Point at the exact paper table/figure, or the row in
[`evidence/h65-paper1/README.md`](../../evidence/h65-paper1/README.md).

**What you ran**
```
python evidence/h65-paper1/verify_llama_confirmations.py
# or: afterimage research h65-plan ...
# or: your own command
```

**What you got**
Paste the script's output, or your own measured numbers, in full.

**What the paper/evidence says**
The published value and its source file.

**Environment**
- `afterimage --version`:
- GPU / driver / CUDA version:
- `afterimage doctor` output (paste it):
- If you ran a live GPU comparison: WSL2 or native, cold page cache
  confirmed (`cache_regime` in your result JSON)?

**Anything else**
Hardware, OS, or protocol differences from
[`docs/H65.md`](../../docs/H65.md)'s protocol table that might explain a gap.
