"""Phase 3.5 Step 2: activation-range calibration of the spiking-ternary
student before Stage A fine-tune.

Loads the converted RWKV-4-169M student checkpoint, enables calibration
mode on every observer-based :class:`FAILIF` (forward returns the FP
membrane potential, the observer's running min/max are EMA-updated), then
streams a few TinyStories batches through the network. After N batches,
disables calibration and saves a fresh checkpoint whose state dict now
carries populated ``observer.running_min/max/num_updates`` buffers.

Two overrides vs. the saved config:
  * ``spiking_r`` is forced to **False** per DEC-8 (the FP-receptance
    dérogation that unblocked Stage A pol #2).
  * ``observer_momentum`` defaults to the value in
    :class:`SpikingTernaryRWKVConfig` if missing in the saved config.

Inputs are detached after every forward so the autograd graph does not
accumulate across batches. The model is in ``train()`` mode (otherwise
observers would not update) but we wrap everything in ``torch.no_grad()``
since we have nothing to differentiate.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV
from spiking_ternary_lm.neurons.fai_lif import FAILIF, set_calibrating
from spiking_ternary_lm.training.data import build_train_eval_loaders, tokenize_tinystories
from spiking_ternary_lm.utils.logging import get_logger

LOG = get_logger("calibrate_rwkv")


def _override_config(saved: dict, bits: int | None = None) -> SpikingTernaryRWKVConfig:
    """Build the config we will use for the run, applying Phase 3.5 overrides."""
    cfg_kwargs = dict(saved)
    cfg_kwargs["spiking_r"] = False
    cfg_kwargs.setdefault("observer_momentum", 0.01)
    if bits is not None:
        cfg_kwargs["bits"] = int(bits)
    return SpikingTernaryRWKVConfig(**cfg_kwargs)


def _observer_summary(model: SpikingTernaryRWKV) -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for name, m in model.named_modules():
        if isinstance(m, FAILIF):
            rows.append(
                (
                    name,
                    {
                        "running_min": float(m.observer.running_min),
                        "running_max": float(m.observer.running_max),
                        "num_updates": int(m.observer.num_updates),
                    },
                )
            )
    return rows


@torch.no_grad()
def calibrate(
    init_ckpt: str,
    out_ckpt: str,
    n_batches: int,
    seq_len: int,
    batch_size: int,
    tokenizer_id: str,
    max_examples: int | None,
    device_str: str,
    seed: int,
    bits: int | None = None,
) -> int:
    torch.manual_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        free_gb = torch.cuda.mem_get_info(device)[0] / 1e9
        LOG.info("GPU %s | %.2f GB free", torch.cuda.get_device_name(device), free_gb)

    # --- Load + rebuild model -----------------------------------------------
    LOG.info("loading init checkpoint: %s", init_ckpt)
    ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    cfg = _override_config(ckpt["config"], bits=bits)
    LOG.info(
        "config: spiking_r=%s observer_momentum=%s bits=%d hidden=%d n_layers=%d",
        cfg.spiking_r, cfg.observer_momentum, cfg.bits, cfg.hidden_size, cfg.n_layers,
    )
    model = SpikingTernaryRWKV(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    LOG.info(
        "state_dict load: missing=%d unexpected=%d (lif_r entries from spiking_r=True go to unexpected)",
        len(missing), len(unexpected),
    )
    n_unexp_lifr = sum(1 for k in unexpected if ".lif_r." in k)
    n_missing_obs = sum(1 for k in missing if ".observer." in k)
    LOG.info(
        "  unexpected lif_r=%d (expected: %d×%d=%d)",
        n_unexp_lifr, cfg.n_layers, 2, cfg.n_layers * 2,
    )
    LOG.info("  missing observer buffers=%d", n_missing_obs)

    model.to(device)
    n_lifs = set_calibrating(model, True)
    LOG.info("enabled calibration on %d FAI-LIF instances", n_lifs)
    model.train()  # required so MovingAverageMinMaxObserver.update_stats runs
    model.reset_state()

    # --- Data ----------------------------------------------------------------
    LOG.info("tokenizing TinyStories (cached after first run)...")
    corpus = tokenize_tinystories(tokenizer_id=tokenizer_id, max_examples=max_examples)
    LOG.info("corpus tokens: %.2f M", len(corpus.ids) / 1e6)
    train_loader, _ = build_train_eval_loaders(
        corpus, seq_len=seq_len, batch_size=batch_size,
        eval_fraction=0.01, shuffle_train=False,
    )

    # --- Calibration pass ----------------------------------------------------
    LOG.info("running calibration for %d batches (seq_len=%d batch=%d)", n_batches, seq_len, batch_size)
    train_iter = iter(train_loader)
    for step in range(n_batches):
        try:
            x, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, _ = next(train_iter)
        x = x.to(device, non_blocking=True)
        model.detach_state()
        _ = model(x)
        if (step + 1) % 10 == 0 or step == 0:
            sample = next(
                (
                    (n, float(m.observer.running_min), float(m.observer.running_max),
                     int(m.observer.num_updates))
                    for n, m in model.named_modules() if isinstance(m, FAILIF)
                ),
                None,
            )
            if sample is not None:
                n, mn, mx, nu = sample
                LOG.info(
                    "  step %3d | %s: running=[%.4f, %.4f] updates=%d",
                    step + 1, n, mn, mx, nu,
                )

    set_calibrating(model, False)
    model.eval()

    # --- Sanity check on observer state -------------------------------------
    rows = _observer_summary(model)
    bad = [r for r in rows if r[1]["num_updates"] == 0]
    if bad and n_batches > 0:
        LOG.error("some observers were never updated (%d of %d):", len(bad), len(rows))
        for n, s in bad[:5]:
            LOG.error("  %s: %s", n, s)
        return 2
    if n_batches == 0:
        LOG.info(
            "n_batches=0: skipping observer warm-up; observers will self-initialise "
            "on the first batch of fine-tuning (update_stats first-call branch)."
        )
    LOG.info("all %d observers updated; range summary:", len(rows))
    # Print a digest: per-block min, max.
    mn_vals = [s["running_min"] for _, s in rows]
    mx_vals = [s["running_max"] for _, s in rows]
    LOG.info(
        "  running_min: median=%.4f  min=%.4f  max=%.4f",
        sorted(mn_vals)[len(mn_vals) // 2], min(mn_vals), max(mn_vals),
    )
    LOG.info(
        "  running_max: median=%.4f  min=%.4f  max=%.4f",
        sorted(mx_vals)[len(mx_vals) // 2], min(mx_vals), max(mx_vals),
    )
    # Detail: first 4 and last 4 modules.
    for n, s in rows[:4] + rows[-4:]:
        LOG.info(
            "    %-48s running=[%+8.4f, %+8.4f] n=%d",
            n, s["running_min"], s["running_max"], s["num_updates"],
        )

    # --- Save ----------------------------------------------------------------
    out_path = Path(out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "calibration": {
                "n_batches": n_batches,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "tokenizer_id": tokenizer_id,
                "n_observers": len(rows),
                "source_checkpoint": init_ckpt,
            },
        },
        out_path,
    )
    LOG.info("saved calibrated checkpoint to %s", out_path)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-checkpoint", type=str, default="checkpoints/stage_a_init.pt")
    p.add_argument("--out-checkpoint", type=str, default="checkpoints/stage_a_init_calibrated.pt")
    p.add_argument("--n-batches", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--tokenizer-id", type=str, default="RWKV/rwkv-4-169m-pile")
    p.add_argument("--max-examples", type=int, default=2000,
                   help="TinyStories examples for the calibration corpus (small is fine).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bits", type=int, default=None,
                   help="Override config.bits (activation bit-width). E.g. 8 for a8.")
    return p.parse_args()


def main() -> int:
    ns = _parse_args()
    return calibrate(
        init_ckpt=ns.init_checkpoint,
        out_ckpt=ns.out_checkpoint,
        n_batches=ns.n_batches,
        seq_len=ns.seq_len,
        batch_size=ns.batch_size,
        tokenizer_id=ns.tokenizer_id,
        max_examples=ns.max_examples,
        device_str=ns.device,
        seed=ns.seed,
        bits=ns.bits,
    )


if __name__ == "__main__":
    sys.exit(main())
