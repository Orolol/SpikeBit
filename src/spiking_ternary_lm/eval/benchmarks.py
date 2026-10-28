"""Zero-shot multiple-choice benchmarks (NSLLM table 1 set).

The NSLLM paper reports zero-shot accuracy on Winogrande, ARC-E, ARC-C,
OpenBookQA, PIQA, and HellaSwag. The standard scoring scheme is:
    1. Form one continuation per candidate completion.
    2. Compute the model's total log-prob of each candidate.
    3. Pick the highest. Normalise by completion-token count for ARC.

We provide a thin scoring loop that loads each dataset from HF and
computes accuracy. Datasets are not bundled — they download on first
run.

Status: implemented for PIQA (the smallest / simplest), with stubs for
the others to keep CI fast. Full eval will be wired in during Phase 4
once Stage A fine-tuning produces a real checkpoint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class BenchmarkResult:
    name: str
    accuracy: float
    n_examples: int


@torch.no_grad()
def _score_continuation(
    model: nn.Module,
    tokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
    vocab_size: int,
    length_normalised: bool = False,
) -> float:
    """Return the model's log-prob of ``continuation`` given ``prompt``.

    With ``length_normalised=True`` returns the per-token average (ARC
    convention); otherwise the sum (Winogrande / PIQA convention).
    """
    full = prompt + continuation
    full_ids = tokenizer(full, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    p_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= p_len:
        return 0.0  # empty continuation
    if hasattr(model, "reset_state"):
        model.reset_state()
    logits = model(full_ids)  # (1, T, V)
    cont_logits = logits[0, p_len - 1 : -1]  # predict each continuation token
    cont_targets = full_ids[0, p_len:]
    log_probs = F.log_softmax(cont_logits.float(), dim=-1)
    tok_log_probs = log_probs.gather(-1, cont_targets.unsqueeze(-1)).squeeze(-1)
    score = float(tok_log_probs.sum())
    if length_normalised:
        score /= max(tok_log_probs.numel(), 1)
    return score


def evaluate_piqa(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    vocab_size: int,
    max_examples: int | None = 100,
) -> BenchmarkResult:
    """Two-choice physical-commonsense QA, validation split."""
    from datasets import load_dataset
    ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    correct = 0
    for ex in ds:
        goal = ex["goal"] + " "
        s1 = _score_continuation(model, tokenizer, goal, ex["sol1"], device, vocab_size)
        s2 = _score_continuation(model, tokenizer, goal, ex["sol2"], device, vocab_size)
        pred = 0 if s1 >= s2 else 1
        if pred == ex["label"]:
            correct += 1
    return BenchmarkResult(name="piqa", accuracy=correct / len(ds), n_examples=len(ds))


# Stubs for the other benchmarks — same scoring pattern, different data layout.
# To be wired up when Stage A is producing checkpoints worth scoring.
BENCHMARKS: dict[str, Callable] = {
    "piqa": evaluate_piqa,
}
