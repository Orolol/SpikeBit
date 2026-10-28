"""Phase 6: side-by-side eval of Stage A and Stage C checkpoints.

Loads both, runs the eval suite on each, prints a comparison table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from spiking_ternary_lm.eval.perplexity import compute_perplexity
from spiking_ternary_lm.eval.spike_stats import measure
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV
from spiking_ternary_lm.training.data import TokenizedCorpus, tokenize_tinystories
from spiking_ternary_lm.utils.logging import get_logger

LOG = get_logger("compare")


def _load(path: str, device: torch.device) -> tuple[SpikingTernaryRWKV, SpikingTernaryRWKVConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = SpikingTernaryRWKVConfig(**ckpt["config"])
    m = SpikingTernaryRWKV(cfg)
    m.load_state_dict(ckpt["model"], strict=False)
    return m.to(device).eval(), cfg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-a", required=True)
    p.add_argument("--stage-c", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tokenizer-id", default="RWKV/rwkv-4-169m-pile")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=20_000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOG.info("loading Stage A: %s", args.stage_a)
    a, cfg_a = _load(args.stage_a, device)
    LOG.info("loading Stage C: %s", args.stage_c)
    c, cfg_c = _load(args.stage_c, device)

    corpus = tokenize_tinystories(tokenizer_id=args.tokenizer_id)
    n = len(corpus.ids)
    eval_corpus = TokenizedCorpus(
        ids=corpus.ids[int(n * 0.99):], vocab_size=corpus.vocab_size,
        eos_token_id=corpus.eos_token_id,
    )

    def run(name, model, cfg):
        ppl = compute_perplexity(
            model, eval_corpus, seq_len=args.seq_len, batch_size=args.batch_size,
            device=device, max_tokens=args.max_tokens, vocab_size=cfg.vocab_size,
        )
        # Spike stats on a small dummy batch.
        ids = torch.tensor(corpus.ids[:64], dtype=torch.long, device=device).unsqueeze(0)
        stats = measure(model, ids)
        sat = sum(s["saturated"] for s in stats.values()) / max(len(stats), 1)
        dead = sum(s["dead"] for s in stats.values()) / max(len(stats), 1)
        fire = sum(s["fire_norm"] for s in stats.values()) / max(len(stats), 1)
        return {"name": name, "loss": ppl["loss"], "ppl": ppl["perplexity"],
                "fire_norm_mean": fire, "saturated_mean": sat, "dead_mean": dead,
                "n_params": sum(p.numel() for p in model.parameters())}

    res_a = run("Stage A", a, cfg_a)
    res_c = run("Stage C", c, cfg_c)

    print()
    print(f"{'metric':22} {'Stage A':>14} {'Stage C':>14}")
    print("-" * 54)
    for k in ("n_params", "loss", "ppl", "fire_norm_mean", "saturated_mean", "dead_mean"):
        print(f"{k:22} {res_a[k]:>14.4f} {res_c[k]:>14.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"stage_a": res_a, "stage_c": res_c}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
