"""Evaluate the pre-trained RWKV-4 teacher PPL on the held-out TinyStories
tail — the baseline our spiking-ternary student has to track within 95 %.

Uses the same eval tail as ``scripts/evaluate.py`` (the last 1 % of the
packed corpus) so the numbers compare apples-to-apples.
"""
from __future__ import annotations

import argparse
import math
import sys

import torch
import torch.nn.functional as F

from spiking_ternary_lm.training.data import (
    PackedTokenDataset,
    TokenizedCorpus,
    tokenize_tinystories,
)
from spiking_ternary_lm.utils.logging import get_logger

LOG = get_logger("eval_teacher")


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="RWKV/rwkv-4-169m-pile")
    p.add_argument("--tokenizer-id", default="RWKV/rwkv-4-169m-pile")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=20_000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    LOG.info("loading teacher %s ...", args.model_id)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, dtype=torch.float32
    )
    model.eval().to(device)
    vocab_size = int(model.config.vocab_size)
    LOG.info("teacher: vocab=%d", vocab_size)

    LOG.info("tokenizing TinyStories (cached)...")
    corpus = tokenize_tinystories(tokenizer_id=args.tokenizer_id)
    n = len(corpus.ids)
    eval_tail = corpus.ids[int(n * 0.99):]
    LOG.info("eval tokens available: %d (using up to %d)", len(eval_tail), args.max_tokens)
    eval_corpus = TokenizedCorpus(
        ids=eval_tail, vocab_size=corpus.vocab_size, eos_token_id=corpus.eos_token_id
    )
    ds = PackedTokenDataset(eval_corpus, seq_len=args.seq_len, shuffle_chunks=False)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    total_loss = 0.0
    total_tokens = 0
    state = None
    LOG.info("running forward (seq_len=%d batch=%d)...", args.seq_len, args.batch)
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        # The HF RwkvForCausalLM supports a recurrent ``state`` arg; keep
        # it None across windows so each window starts fresh (matches our
        # PPL definition where we report a per-window perplexity).
        out = model(input_ids=x, use_cache=False)
        logits = out.logits
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1), reduction="sum"
        )
        total_loss += float(loss)
        total_tokens += y.numel()
        if total_tokens >= args.max_tokens:
            break

    mean_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(mean_loss)
    LOG.info("teacher TinyStories PPL: %.2f  (loss %.4f, n_tokens %d)",
             ppl, mean_loss, total_tokens)
    print(f"teacher: PPL={ppl:.2f} loss={mean_loss:.4f} n_tokens={total_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
