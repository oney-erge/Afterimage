import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from afterimage.runtime.vram_planner import plan_from_manifest
man = json.loads(pathlib.Path("/root/afterimage/store_14b/manifest.json").read_text())
print("Qwen3-14B  (29.54 GB bf16 -> 20.33 GB compressed, 1.453x)")
print("AirLLM measured: 1.57 GB VRAM, 28.49 s/token, 29.54 GB read/token")
print("Ours measured  : 3.55 GB VRAM, 17.38 s/token, 17.74 GB read/token")
print()
print("%-9s %-6s %-9s %-11s %-11s %-12s %s" % (
    "vram_gb", "ram_gb", "feasible", "vram tier", "ram tier", "disk/tok", "note"))
for vram_gb, ram_gb in [(1.5, 0), (2.0, 0), (2.5, 0), (3.0, 0), (4.0, 0),
                        (4.0, 4.0), (4.0, 8.0), (6.0, 0), (6.0, 8.0), (8.0, 0)]:
    p = plan_from_manifest(man, vram_budget_gb=vram_gb, ram_budget_gb=ram_gb)
    note = "" if p.feasible else p.reason.split("needed")[0].strip()
    print("%-9.1f %-6.1f %-9s %-11s %-11s %-12s %s" % (
        vram_gb, ram_gb, p.feasible,
        ("%.2f GB" % p.vram_gb) if p.feasible else "-",
        ("%.2f GB" % p.ram_gb) if p.feasible else "-",
        ("%.2f GB" % p.disk_gb_per_token) if p.feasible else "-", note))
