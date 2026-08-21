import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
from afterimage.runtime.config import EngineConfig
from afterimage.runtime.streaming_engine import StreamingLosslessModel, load_draft_model

TARGET = "Qwen/Qwen3-14B"
DRAFT = "Qwen/Qwen3-0.6B"
STORE = "/root/afterimage/store_14b"
PROMPT = "The capital of France is"


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    draft = load_draft_model(DRAFT, device="cuda")

    cfg = EngineConfig(vram_cap_gb=6.0, io_prefetch_depth=2)
    with StreamingLosslessModel(TARGET, STORE, device="cuda", config=cfg) as sm:
        for temp in [1.0, 0.7, 0.3, 0.05]:
            gen = torch.Generator(device="cuda").manual_seed(0)
            sm.stats.reset()
            t0 = time.perf_counter()
            seq = sm.generate_speculative(ids, max_new_tokens=16, draft_model=draft,
                                          k=8, temperature=temp, generator=gen)
            wall = time.perf_counter() - t0
            n_out = seq.shape[1] - ids.shape[1]
            accept = sm.stats.spec_accepted_tokens / max(1, sm.stats.spec_sweeps * 8)
            print("temp=%.2f  %d tok in %.1fs (%.2f s/tok)  sweeps=%d  tok/sweep=%.2f  accept=%.1f%%"
                  % (temp, n_out, wall, wall / n_out, sm.stats.spec_sweeps,
                     n_out / max(1, sm.stats.spec_sweeps), 100 * accept), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
