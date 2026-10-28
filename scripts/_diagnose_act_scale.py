"""Diagnostic: activation scale and backward profile of ladder checkpoints.

Train mode (observers update, as in training), bf16 autocast, CE only, the
first 16 windows of 128 tokens of the eval tail. Prints the gradient norm of
the residual stream at each block output, then for blocks 11, 1 and 0 the
gradient norm, absmax and RMS of every time-mix / channel-mix sub-module
output, and the LIF grid steps of blocks 0 and 1. Arguments: run names under
``checkpoints/ladder`` (``final.pt``).

    uv run python scripts/_diagnose_act_scale.py rung2c_lr3e-3_s0 rung3c_a8_s1
"""
import sys
import torch, torch.nn.functional as F
from spiking_ternary_lm.eval.perplexity import protocol_eval_corpus
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV
from spiking_ternary_lm.training.data import tokenize_tinystories

dev = torch.device("cuda")
ids = torch.from_numpy(protocol_eval_corpus(tokenize_tinystories(tokenizer_id="RWKV/rwkv-4-169m-pile")).ids.astype("int64"))
SEQ, B = 128, 16
INNER = ["ln1", "ln2", "time_mix.W_k", "time_mix.lif_k", "time_mix.W_v", "time_mix.lif_v", "time_mix.W_r",
         "time_mix.lif_out", "time_mix.W_o", "time_mix", "channel_mix.W_k", "channel_mix.lif_k",
         "channel_mix.W_v", "channel_mix.lif_v", "channel_mix.W_r", "channel_mix"]

for run in sys.argv[1:]:
    ck = torch.load(f"checkpoints/ladder/{run}/final.pt", map_location="cpu", weights_only=False)
    m = SpikingTernaryRWKV(SpikingTernaryRWKVConfig(**ck["config"])); m.load_state_dict(ck["model"]); m.to(dev).train()
    g, f = {}, {}
    def keep(name):
        def fwd(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            if isinstance(x, torch.Tensor):
                f[name + ".in_absmax"] = float(x.detach().float().abs().max())
            f[name + ".out_absmax"] = float(out.detach().float().abs().max())
            f[name + ".out_rms"] = float(out.detach().float().pow(2).mean().sqrt())
            if out.requires_grad:
                out.register_hook(lambda gr, n=name: g.__setitem__(n, float(gr.float().norm())))
        return fwd
    hooks = []
    for i, blk in enumerate(m.blocks):
        hooks.append(blk.register_forward_hook(keep(f"b{i}")))
        if i in (0, 1, 11):
            for s in INNER:
                hooks.append(blk.get_submodule(s).register_forward_hook(keep(f"b{i}.{s}")))
    win = torch.stack([ids[b * SEQ: b * SEQ + SEQ + 1] for b in range(B)]).to(dev)
    m.reset_state(); m.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m(win[:, :-1])
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), win[:, 1:].reshape(-1))
    loss.backward()
    print(f"=== {run}  loss {float(loss):.3f}")
    print("  residual grad at block output:", " ".join(f"{i}:{g.get(f'b{i}', float('nan')):.2g}" for i in range(len(m.blocks))))
    for i in (11, 1, 0):
        print(f"  block {i}")
        for s in INNER:
            n = f"b{i}.{s}"
            print(f"    {s:20s} |dL/dout| {g.get(n, float('nan')):10.3g}   out absmax {f.get(n + '.out_absmax', float('nan')):9.3g} rms {f.get(n + '.out_rms', float('nan')):9.3g}")
    for i in (0, 1):
        tm = m.blocks[i].time_mix
        print(f"  block {i} lif scales:", {k: f"{float(getattr(tm, k).observer.get_scale_and_zero_point()[0]):.3g}" for k in ("lif_k", "lif_v", "lif_out")},
              {k: f"{float(getattr(m.blocks[i].channel_mix, k).observer.get_scale_and_zero_point()[0]):.3g}" for k in ("lif_k", "lif_v")})
        print(f"  block {i} ln1 w absmax {float(m.blocks[i].ln1.weight.abs().max()):.3g}, time_decay range [{float(tm.time_decay.min()):.3g}, {float(tm.time_decay.max()):.3g}], time_first range [{float(tm.time_first.min()):.3g}, {float(tm.time_first.max()):.3g}]")
    for h in hooks: h.remove()
    del m; torch.cuda.empty_cache()
