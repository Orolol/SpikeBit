"""Data pipeline for Stage A / Stage C training.

Loads TinyStories (or any HF text dataset) and packs it into fixed-length
token sequences using the RWKV-4-Pile tokenizer. Output is a simple
``IterableDataset`` of ``(input_ids, labels)`` tensors of shape
``(seq_len,)``; the trainer stacks them into ``(B, T)`` batches.

We tokenize lazily in a background thread (HF datasets' streaming map)
to avoid materialising the full corpus in memory. For TinyStories (the
default) the full BPE-tokenized corpus fits in a few GB so we
*pre-tokenize* the entire dataset to a packed buffer and reuse it across
epochs.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset


@dataclass
class TokenizedCorpus:
    """A flat 1-D array of token IDs."""

    ids: np.ndarray  # dtype uint16 or int32 (uint16 if vocab fits in 2 bytes)
    vocab_size: int
    eos_token_id: int


def _make_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def tokenize_tinystories(
    tokenizer_id: str = "RWKV/rwkv-4-169m-pile",
    cache_path: str | Path = "./data/tinystories_packed.npz",
    split: str = "train",
    max_examples: int | None = None,
    text_field: str = "text",
) -> TokenizedCorpus:
    """Tokenize the entire TinyStories split and cache it as packed tokens.

    On second call, simply loads the cached file. The cache key includes
    the tokenizer id + split + max_examples so changes invalidate it.
    """
    cache_path = Path(cache_path)
    cache_key = f"{tokenizer_id.replace('/', '_')}_{split}_{max_examples}"
    final_cache = cache_path.with_name(f"{cache_path.stem}_{cache_key}.npz")
    if final_cache.exists():
        npz = np.load(final_cache, allow_pickle=False)
        return TokenizedCorpus(
            ids=npz["ids"], vocab_size=int(npz["vocab_size"][0]), eos_token_id=int(npz["eos"][0])
        )

    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    tok = _make_tokenizer(tokenizer_id)
    vocab_size = tok.vocab_size
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0

    chunks: list[np.ndarray] = []
    total = 0
    # Tokenize in batches for speed.
    BATCH = 1000
    for i in range(0, len(ds), BATCH):
        texts = ds[i : i + BATCH][text_field]
        out = tok(texts, add_special_tokens=False)["input_ids"]
        for ids in out:
            chunks.append(np.asarray(ids, dtype=np.int32))
            chunks.append(np.asarray([eos], dtype=np.int32))
            total += len(ids) + 1

    flat = np.concatenate(chunks)
    # Pick the narrowest int type that fits.
    if vocab_size < 2**16:
        flat = flat.astype(np.uint16)

    final_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        final_cache,
        ids=flat,
        vocab_size=np.array([vocab_size], dtype=np.int32),
        eos=np.array([eos], dtype=np.int32),
    )
    return TokenizedCorpus(ids=flat, vocab_size=vocab_size, eos_token_id=eos)


class PackedTokenDataset(IterableDataset):
    """Stream fixed-length token windows from a packed corpus.

    Each yielded item is ``(input_ids, labels)`` of shape ``(seq_len,)``
    where ``labels[i] = input_ids[i+1]`` (standard LM next-token target),
    so the trainer just does ``F.cross_entropy(logits[:, :-1], labels)``.

    For Stage A we keep streaming order across batches so the LIF /
    WKV state continuity from the spec (no reset between sequences) is
    meaningful at the per-worker level. The trainer is responsible for
    calling `model.reset_state` once at the start of training and
    `model.detach_state` between training windows.
    """

    def __init__(
        self,
        corpus: TokenizedCorpus,
        seq_len: int = 256,
        start_offset: int = 0,
        stride: int | None = None,
        shuffle_chunks: bool = False,
        seed: int = 0,
        contiguous_rows_batch: int = 0,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.seq_len = int(seq_len)
        self.stride = int(stride) if stride is not None else self.seq_len
        self.start_offset = int(start_offset)
        self.shuffle_chunks = bool(shuffle_chunks)
        self.seed = int(seed)
        # > 0: emit chunks so that, batched by this size, row b of batch n+1
        # is the text continuation of row b of batch n (B contiguous
        # streams). 0: plain sequential order (row b of batch n+1 is B chunks
        # further down the corpus, not a continuation).
        self.contiguous_rows_batch = int(contiguous_rows_batch)
        if self.seq_len < 2:
            raise ValueError("seq_len must be >= 2")

    def __len__(self) -> int:
        # Lower bound — the iterator may yield fewer if the corpus is short.
        n_tokens = len(self.corpus.ids) - self.start_offset
        return max(0, (n_tokens - 1) // self.stride)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        ids = self.corpus.ids
        n = len(ids)
        # +1 because we need one extra token for the label of the last position.
        starts = list(range(self.start_offset, n - self.seq_len - 1, self.stride))
        if self.shuffle_chunks:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(starts)
        elif self.contiguous_rows_batch > 1:
            B = self.contiguous_rows_batch
            per_row = len(starts) // B
            # Row b owns chunks [b*per_row, (b+1)*per_row); interleave so
            # the DataLoader's batch n is (chunk n of row 0, ..., of row B-1).
            # The seed shifts every row by seed/8 of its region, so seeds 0-7
            # read disjoint text for runs up to per_row/8 steps (DECISIONS
            # #15 seeds; seed 0 keeps the historical layout).
            shift = (self.seed * per_row) // 8
            starts = [
                starts[b * per_row + (k + shift) % per_row]
                for k in range(per_row) for b in range(B)
            ]
        for s in starts:
            window = ids[s : s + self.seq_len + 1].astype(np.int64)
            x = torch.from_numpy(window[:-1])
            y = torch.from_numpy(window[1:])
            yield x, y


def build_train_eval_loaders(
    corpus: TokenizedCorpus,
    seq_len: int,
    batch_size: int,
    eval_fraction: float = 0.01,
    shuffle_train: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    contiguous_rows: bool = False,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Split the packed corpus into a train and eval region by token offset.

    No randomness in the split itself: the eval portion is the tail of
    the corpus. The train side may shuffle chunks within itself.
    """
    n = len(corpus.ids)
    eval_tokens = max(seq_len * 4, int(n * eval_fraction))
    train_end = n - eval_tokens

    train_corpus = TokenizedCorpus(
        ids=corpus.ids[:train_end], vocab_size=corpus.vocab_size, eos_token_id=corpus.eos_token_id
    )
    eval_corpus = TokenizedCorpus(
        ids=corpus.ids[train_end:], vocab_size=corpus.vocab_size, eos_token_id=corpus.eos_token_id
    )

    train_ds = PackedTokenDataset(
        train_corpus, seq_len=seq_len, shuffle_chunks=shuffle_train, seed=seed,
        contiguous_rows_batch=batch_size if (contiguous_rows and not shuffle_train) else 0,
    )
    eval_ds = PackedTokenDataset(eval_corpus, seq_len=seq_len, shuffle_chunks=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader
