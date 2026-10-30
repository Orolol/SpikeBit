"""Learned leak beta of stateful FAI-LIFs (DECISIONS #24 point 7), CPU, from the checkpoints.

Per LIF site pooled over the blocks: mean, min, max of beta = sigmoid(beta_logit), and the
share of neurons with beta < 0.1 and beta > 0.5.

    uv run python scripts/_beta_stats.py rung4b_a4_s0 rung4c_a4_s0
"""
import sys
import torch

for run in sys.argv[1:]:
    sd = torch.load(f"checkpoints/ladder/{run}/final.pt", map_location="cpu", weights_only=False)["model"]
    sites = {}
    for k, v in sd.items():
        if k.endswith("beta_logit"):  # blocks.<i>.<mix>.<lif>.beta_logit
            sites.setdefault(".".join(k.split(".")[2:4]), []).append(torch.sigmoid(v.float()))
    print(f"=== {run}")
    for site, vs in sites.items():
        b = torch.cat(vs)
        print(f"  {site:20s} n {b.numel():6d}  mean {b.mean():.3f}  min {b.min():.3f}  max {b.max():.3f}  "
              f"<0.1 {(b < 0.1).float().mean():.3f}  >0.5 {(b > 0.5).float().mean():.3f}")
    b = torch.cat([torch.cat(vs) for vs in sites.values()])
    print(f"  {'all':20s} n {b.numel():6d}  mean {b.mean():.3f}  min {b.min():.3f}  max {b.max():.3f}  "
          f"<0.1 {(b < 0.1).float().mean():.3f}  >0.5 {(b > 0.5).float().mean():.3f}")
