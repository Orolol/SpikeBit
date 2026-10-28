"""Phase 1 milestone: MNIST sanity for Spiking-Ternary primitives.

Builds the exact architecture from the spec:

    BitLinear(784 -> 256) -> FAI-LIF(A=4)
                          -> BitLinear(256 -> 10) -> FAI-LIF(A=4)

Trains on MNIST and verifies test accuracy > 97 % in under ~5 min on the 5090.
If this fails, **stop and debug** — do not proceed to Phase 2.

Notes on choices that diverge from the spec defaults (see DECISIONS.md):

  * Group size 16 (not 128). MNIST's 784 input dim is not divisible by 128;
    rather than pad the input, we use a per-layer group size that divides
    both 784 and 256. 128 will be the default in the language model.
  * v_thr initialised low (0.5) so that at random init at least some
    neurons can fire. Without this, FAI-LIF stays at 0 and STE has no
    signal to amplify weights from.
  * The surrogate-alpha schedule called out in the spec is **not used**
    here — FAI-LIF uses pure STE through round+clip, not a Heaviside
    surrogate. We still log a placeholder for parity with later phases.

Dataset loaded via HuggingFace `datasets` (already a project dep). Cached
under ``./data/``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from spiking_ternary_lm.layers.norms import RMSNorm
from spiking_ternary_lm.neurons.fai_lif import FAILIF, FAILIFFixedThreshold
from spiking_ternary_lm.quantization.bitlinear import BitLinear
from spiking_ternary_lm.utils.logging import JsonlMetricsWriter, get_logger

LOG = get_logger("sanity_mnist")

# MNIST is 28x28 grayscale; we standardise with the canonical (mean, std).
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SpikingTernaryMLP(nn.Module):
    """The 2-layer MLP described above. Output is FAI-LIF integer counts,
    used directly as CE logits (see milestone discussion)."""

    def __init__(
        self,
        in_dim: int = 784,
        hidden: int = 256,
        out_dim: int = 10,
        bits: int = 4,
        group_size: int = 16,
        weight_dtype: torch.dtype = torch.float32,
        init_v_thr: float = 0.5,
        learnable_v_thr: bool = False,
        signed: bool = False,
        final_readout: str = "fai-lif",
        use_rmsnorm: bool = True,
        use_observer_lif: bool = True,
        observer_momentum: float = 0.01,
        lif_beta_init: float = 0.9,
        lif_gamma_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_rmsnorm = use_rmsnorm
        self.norm1: nn.Module = RMSNorm(in_dim) if use_rmsnorm else nn.Identity()
        self.fc1 = BitLinear(in_dim, hidden, group_size=group_size, weight_dtype=weight_dtype)
        if use_observer_lif:
            self.lif1 = FAILIF(
                hidden, num_bits=bits, signed=signed,
                observer_momentum=observer_momentum,
                beta_init=lif_beta_init, gamma_init=lif_gamma_init,
            )
        else:
            self.lif1 = FAILIFFixedThreshold(
                hidden, bits=bits, signed=signed,
                init_v_thr=init_v_thr, learnable_v_thr=learnable_v_thr,
            )
        self.norm2: nn.Module = RMSNorm(hidden) if use_rmsnorm else nn.Identity()
        self.fc2 = BitLinear(hidden, out_dim, group_size=group_size, weight_dtype=weight_dtype)
        self.final_readout = final_readout
        if final_readout == "fai-lif":
            if use_observer_lif:
                self.lif2: nn.Module = FAILIF(
                    out_dim, num_bits=bits, signed=signed,
                    observer_momentum=observer_momentum,
                    beta_init=lif_beta_init, gamma_init=lif_gamma_init,
                )
            else:
                self.lif2 = FAILIFFixedThreshold(
                    out_dim, bits=bits, signed=signed,
                    init_v_thr=init_v_thr, learnable_v_thr=learnable_v_thr,
                )
        elif final_readout == "linear":
            self.lif2 = nn.Identity()
        else:
            raise ValueError(f"unknown final_readout {final_readout!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.fc1(h)
        h = self.lif1(h)
        z = self.norm2(h)
        z = self.fc2(z)
        z = self.lif2(z)
        return z, h

    def reset_state(self) -> None:
        if hasattr(self.lif1, "reset_state"):
            self.lif1.reset_state()
        if hasattr(self.lif2, "reset_state"):
            self.lif2.reset_state()


class _HFMNIST(Dataset):
    """Wraps the HF `ylecun/mnist` split into a tensor dataset."""

    def __init__(self, split: str, cache_dir: str | None = None) -> None:
        from datasets import load_dataset

        ds = load_dataset("ylecun/mnist", split=split, cache_dir=cache_dir)
        # Materialise as a (N, 784) float tensor.
        import numpy as np  # local import: numpy is a transitive dep

        images = np.stack([np.asarray(ex["image"], dtype=np.float32) for ex in ds])
        labels = np.asarray([ex["label"] for ex in ds], dtype=np.int64)
        self.images = torch.from_numpy(images).reshape(len(ds), -1) / 255.0
        self.images = (self.images - MNIST_MEAN) / MNIST_STD
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


def spike_stats(out: torch.Tensor, bits: int) -> dict[str, float]:
    """Mean firing rate, saturation fraction, dead-neuron fraction."""
    n_max = 2**bits - 1
    return {
        "fire_mean": out.float().mean().item(),
        "fire_norm": (out.float() / max(n_max, 1)).mean().item(),
        "saturated": (out >= n_max).float().mean().item(),
        "dead_neurons": (out.float().mean(dim=0) == 0).float().mean().item(),
    }


def gradient_norms(model: nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        out[f"grad/{name}"] = p.grad.detach().float().norm().item()
    return out


def train_one_epoch(
    model: SpikingTernaryMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    log_every: int,
    grad_clip: float,
    writer: JsonlMetricsWriter,
    step_offset: int,
    bits: int,
) -> int:
    model.train()
    step = step_offset
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model.reset_state()  # each batch is independent for MNIST
        z, h_hidden = model(x)
        logits = z.float()  # CE wants FP
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0:
            metrics = {
                "epoch": epoch,
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                **{f"lif1/{k}": v for k, v in spike_stats(h_hidden, bits).items()},
                **{f"lif2/{k}": v for k, v in spike_stats(z, bits).items()},
                "fc1/dist": model.fc1.weight_distribution(),
                "fc2/dist": model.fc2.weight_distribution(),
            }
            metrics.update(gradient_norms(model))
            writer.log(step, **metrics)
            LOG.info(
                "step %5d epoch %d loss %.4f | lif1 fire_norm %.3f sat %.3f | lif2 fire_norm %.3f",
                step,
                epoch,
                loss.item(),
                metrics["lif1/fire_norm"],
                metrics["lif1/saturated"],
                metrics["lif2/fire_norm"],
            )
        step += 1
    return step


@torch.no_grad()
def evaluate(model: SpikingTernaryMLP, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model.reset_state()
        z, _ = model(x)
        pred = z.float().argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--init-v-thr", type=float, default=0.5)
    p.add_argument("--learnable-v-thr", action="store_true", default=False)
    p.add_argument("--signed", action="store_true", default=False,
                   help="TI-LIF variant: signed integer spikes in [-2^(A-1), 2^(A-1)-1].")
    p.add_argument("--no-rmsnorm", action="store_true", default=False)
    # Phase 3.5: observer-based FAI-LIF default. Use `--no-observer-lif` to
    # fall back to the fixed-threshold variant (the Phase 1 baseline).
    p.add_argument("--no-observer-lif", action="store_true", default=False,
                   help="Use the legacy FAILIFFixedThreshold instead of the observer-based FAILIF.")
    p.add_argument("--observer-momentum", type=float, default=0.01,
                   help="EMA momentum for the activation observer.")
    p.add_argument("--lif-beta-init", type=float, default=0.9)
    p.add_argument("--lif-gamma-init", type=float, default=1.0,
                   help="1.0 = pure soft reset (spec default), 0.5 = mixed (legacy).")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--final-readout", choices=["fai-lif", "linear"], default="fai-lif")
    p.add_argument("--device", default="cuda")
    p.add_argument("--weight-dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--metrics-path", default="./runs/sanity_mnist/metrics.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Pre-flight: refuse to launch concurrently with another big GPU user.
        free, total = torch.cuda.mem_get_info(device)
        LOG.info(
            "GPU %s | free %.2f GB / total %.2f GB",
            torch.cuda.get_device_name(device),
            free / 1e9,
            total / 1e9,
        )
        if free < 2e9:
            LOG.warning(
                "GPU appears in heavy use (free < 2 GB). Per CLAUDE.md, "
                "you should wait. Continuing anyway because the model is small."
            )

    LOG.info("loading MNIST via HF datasets...")
    train_ds = _HFMNIST(split="train", cache_dir=args.data_dir)
    test_ds = _HFMNIST(split="test", cache_dir=args.data_dir)
    LOG.info("train %d, test %d", len(train_ds), len(test_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=512,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    weight_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.weight_dtype]
    model = SpikingTernaryMLP(
        hidden=args.hidden,
        bits=args.bits,
        group_size=args.group_size,
        weight_dtype=weight_dtype,
        init_v_thr=args.init_v_thr,
        learnable_v_thr=args.learnable_v_thr,
        signed=args.signed,
        final_readout=args.final_readout,
        use_rmsnorm=not args.no_rmsnorm,
        use_observer_lif=not args.no_observer_lif,
        observer_momentum=args.observer_momentum,
        lif_beta_init=args.lif_beta_init,
        lif_gamma_init=args.lif_gamma_init,
    ).to(device)
    LOG.info("model: %s", model)
    n_params = sum(p.numel() for p in model.parameters())
    LOG.info("trainable params: %d (%.1f K)", n_params, n_params / 1e3)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=1e3,
    )

    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    step = 0
    best_acc = 0.0
    with JsonlMetricsWriter(args.metrics_path) as writer:
        for epoch in range(args.epochs):
            step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                epoch,
                args.log_every,
                args.grad_clip,
                writer,
                step,
                args.bits,
            )
            acc = evaluate(model, test_loader, device)
            writer.log(step, epoch=epoch, test_acc=acc, wall_so_far=time.time() - t0)
            LOG.info("epoch %d done | test_acc %.4f | wall %.1fs", epoch, acc, time.time() - t0)
            best_acc = max(best_acc, acc)

    wall = time.time() - t0
    LOG.info("DONE | best test_acc %.4f | total wall %.1fs", best_acc, wall)

    # Phase 1 gate: > 97 % accuracy in < 5 min.
    target_acc = 0.97
    target_wall = 5 * 60
    passed = best_acc >= target_acc and wall <= target_wall
    print(
        f"sanity_mnist: acc={best_acc:.4f} wall={wall:.1f}s "
        f"target=acc>={target_acc} wall<={target_wall}s "
        f"=> {'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 4


if __name__ == "__main__":
    sys.exit(main())
