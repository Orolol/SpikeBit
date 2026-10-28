"""Phase 4 driver: evaluate a Stage A (or Stage C) checkpoint.

Runs:
  * Held-out TinyStories perplexity (always)
  * Spike statistics on a small batch (always)
  * Generation samples (always)
  * Zero-shot PIQA accuracy (optional, --benchmarks)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from spiking_ternary_lm.eval.benchmarks import BENCHMARKS
from spiking_ternary_lm.eval.generation import generate
from spiking_ternary_lm.eval.perplexity import compute_perplexity
from spiking_ternary_lm.eval.spike_stats import measure
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV
from spiking_ternary_lm.training.data import tokenize_tinystories
from spiking_ternary_lm.utils.logging import get_logger

LOG = get_logger("evaluate")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer-id", default="RWKV/rwkv-4-169m-pile")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ppl-seq-len", type=int, default=128)
    p.add_argument("--ppl-batch", type=int, default=2)
    p.add_argument("--ppl-max-tokens", type=int, default=20_000)
    p.add_argument("--prompt", default="Once upon a time,")
    p.add_argument("--gen-tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--benchmarks", action="store_true",
                   help="Also run PIQA zero-shot (slow).")
    p.add_argument("--piqa-max", type=int, default=100)
    p.add_argument("--out", default=None,
                   help="Optional JSON file to dump all metrics.")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOG.info("loading %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = SpikingTernaryRWKVConfig(**ckpt["config"])
    model = SpikingTernaryRWKV(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    LOG.info("model: %.1f M params", model.num_parameters() / 1e6)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)

    metrics: dict[str, object] = {}

    LOG.info("computing held-out TinyStories perplexity...")
    corpus = tokenize_tinystories(tokenizer_id=args.tokenizer_id)
    # Use the last 1% as held-out, same as training split logic.
    n = len(corpus.ids)
    eval_tail = corpus.ids[int(n * 0.99):]
    from spiking_ternary_lm.training.data import TokenizedCorpus
    eval_corpus = TokenizedCorpus(
        ids=eval_tail, vocab_size=corpus.vocab_size, eos_token_id=corpus.eos_token_id
    )
    ppl = compute_perplexity(
        model, eval_corpus, seq_len=args.ppl_seq_len, batch_size=args.ppl_batch,
        device=device, max_tokens=args.ppl_max_tokens, vocab_size=cfg.vocab_size,
    )
    LOG.info("ppl: %.2f (loss %.4f, n_tokens %d)",
             ppl["perplexity"], ppl["loss"], ppl["n_tokens"])
    metrics["perplexity"] = ppl

    LOG.info("spike stats on a small batch...")
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    stats = measure(model, ids)
    sample = list(stats.items())[:3]
    for n_, s in sample:
        LOG.info("  %s: fire_norm %.3f sat %.3f dead %.3f",
                 n_, s["fire_norm"], s["saturated"], s["dead"])
    metrics["spike_stats"] = stats

    LOG.info("generation from prompt: %r", args.prompt)
    out = generate(
        model, ids, max_new_tokens=args.gen_tokens,
        temperature=args.temperature, top_k=args.top_k,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    LOG.info("generated:\n  %s", text)
    metrics["generation"] = text

    if args.benchmarks:
        LOG.info("running PIQA (max %d examples)...", args.piqa_max)
        res = BENCHMARKS["piqa"](
            model, tok, device, cfg.vocab_size, max_examples=args.piqa_max
        )
        LOG.info("PIQA: %.3f acc (n=%d)", res.accuracy, res.n_examples)
        metrics["piqa"] = {"accuracy": res.accuracy, "n_examples": res.n_examples}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        LOG.info("wrote %s", args.out)

    print(f"evaluate: PPL={ppl['perplexity']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
