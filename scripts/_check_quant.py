"""Is a ladder checkpoint really quantized? Per FAI-LIF: distinct output levels, grid step, share of
saturated outputs; then the CE on the first windows of the eval tail with the LIFs as trained and
with every LIF forced to act_quant="none" (FP activations).

Run from the repo root (CUDA, first 32 windows of the eval tail, eval mode):

    uv run python scripts/_check_quant.py rung5_a4_s0 rung5_a4_s1
"""
import sys
import torch, torch.nn.functional as F
from spiking_ternary_lm.eval.perplexity import protocol_eval_corpus
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV
from spiking_ternary_lm.neurons.fai_lif import FAILIF
from spiking_ternary_lm.training.data import tokenize_tinystories

dev = torch.device("cuda")
ids = torch.from_numpy(protocol_eval_corpus(tokenize_tinystories(tokenizer_id="RWKV/rwkv-4-169m-pile")).ids.astype("int64"))
SEQ, B = 128, 32
win = torch.stack([ids[b * SEQ: b * SEQ + SEQ + 1] for b in range(B)]).to(dev)


def ce(m):
    m.reset_state()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m(win[:, :-1])
    return float(F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), win[:, 1:].reshape(-1)))


for run in sys.argv[1:]:
    ck = torch.load(f"checkpoints/ladder/{run}/final.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = SpikingTernaryRWKV(SpikingTernaryRWKVConfig(**cfg)); m.load_state_dict(ck["model"]); m.to(dev).eval()
    lifs = [(n, mod) for n, mod in m.named_modules() if isinstance(mod, FAILIF)]
    out = {}
    hooks = [mod.register_forward_hook(lambda mod, i, o, n=n: out.__setitem__(n, o.detach().float())) for n, mod in lifs]
    base = ce(m)
    for h in hooks:
        h.remove()
    print(f"=== {run}  act_quant {cfg.get('act_quant')} bits {cfg.get('bits')} ffn hidden {cfg.get('bits_ffn_hidden')}  CE {base:.4f}")
    worst = 0
    for n, mod in lifs:
        o = out[n]
        if mod.act_quant == "lsq" and int(mod.lsq_initialized) == 1:
            s, zp = mod.lsq_scale_and_zero_point()
        else:
            s, zp = mod.observer.get_scale_and_zero_point()
        q = o / s.to(o.dtype) + zp.to(o.dtype)
        off_grid = float((q - q.round()).abs().max())
        levels = int(torch.unique(q.round()).numel())
        sat = float(((q.round() == mod.observer.qmax) | (q.round() == mod.observer.qmin)).float().mean())
        worst = max(worst, off_grid)
        if n.startswith(("blocks.0.", "blocks.11.")):
            print(f"  {n:28s} act_quant {mod.act_quant:6s} grid [{mod.observer.qmin},{mod.observer.qmax}] step {float(s):.4g} "
                  f"levels used {levels:3d}  saturated {sat:.2e}  max |q - round(q)| {off_grid:.1e}")
    print(f"  all {len(lifs)} LIFs: max |q - round(q)| {worst:.1e}; act_quant set: {sorted({mod.act_quant for _, mod in lifs})}")
    modes = {n: mod.act_quant for n, mod in lifs}
    for _, mod in lifs:
        mod.act_quant = "none"
    print(f"  CE with FP activations (same weights): {ce(m):.4f}")
    for n, mod in lifs:
        mod.act_quant = modes[n]
    del m; torch.cuda.empty_cache()
