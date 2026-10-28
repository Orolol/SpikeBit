"""Quick diagnostic: BitLinear + ReLU (no FAI-LIF) on MNIST.

If this hits >97%, the bottleneck is FAI-LIF, not BitLinear.
If this also plateaus around 94%, the ternary weights themselves are the cap.

Not part of the main sanity gate — diagnostic only.
"""
from __future__ import annotations

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "scripts")
from sanity_mnist import _HFMNIST  # noqa: E402

from spiking_ternary_lm.layers.norms import RMSNorm  # noqa: E402
from spiking_ternary_lm.quantization.bitlinear import BitLinear  # noqa: E402


class BitMLP(nn.Module):
    def __init__(self, hidden=512, group_size=16):
        super().__init__()
        self.norm1 = RMSNorm(784)
        self.fc1 = BitLinear(784, hidden, group_size=group_size, weight_dtype=torch.float32)
        self.norm2 = RMSNorm(hidden)
        self.fc2 = BitLinear(hidden, 10, group_size=group_size, weight_dtype=torch.float32)

    def forward(self, x):
        h = F.relu(self.fc1(self.norm1(x)))
        return self.fc2(self.norm2(h))


def main():
    device = torch.device("cuda")
    train = _HFMNIST(split="train", cache_dir="./data")
    test = _HFMNIST(split="test", cache_dir="./data")
    tl = DataLoader(train, batch_size=256, shuffle=True, num_workers=2, drop_last=True)
    vl = DataLoader(test, batch_size=512, shuffle=False, num_workers=2)
    model = BitMLP(hidden=512).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, total_steps=len(tl) * 10, pct_start=0.1
    )
    t0 = time.time()
    for ep in range(10):
        model.train()
        for x, y in tl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x = x.to(device)
                y = y.to(device)
                p = model(x).argmax(-1)
                correct += (p == y).sum().item()
                total += y.numel()
        print(f"ep {ep} acc {correct/total:.4f} loss {loss.item():.3f} wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
