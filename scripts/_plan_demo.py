import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from afterimage.runtime.vram_planner import plan_from_manifest
man = json.loads(pathlib.Path("/root/afterimage/store_14b/manifest.json").read_text())
print("Qwen3-14B  (29.54 GB bf16 -> 19.81 GB compressed)")
print("AirLLM measured: 1.57 GB VRAM, 32.23 s/token, 29.54 GB read/token")
print("Ours measured  : 5.10 GB VRAM, 24.93 s/token, 17.74 GB read/token")
print()
print("%-9s %-9s %-11s %-12s %s" % ("budget","feasible","resident","stream/tok","note"))
for gb in [1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]:
    p = plan_from_manifest(man, budget_gb=gb)
    note = "" if p.feasible else p.reason.split("needed")[0].strip()
    print("%-9.1f %-9s %-11s %-12s %s" % (
        gb, p.feasible,
        ("%.2f GB" % p.resident_gb) if p.feasible else "-",
        ("%.2f GB" % p.streamed_gb_per_token) if p.feasible else "-", note))
