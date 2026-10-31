"""Batch-row continuity of the packed stream (DEC-2 streaming, review 2026-10-28)."""
import numpy as np
import torch

from spiking_ternary_lm.training.data import PackedTokenDataset, TokenizedCorpus, wikitext_articles


def _corpus(n=1000):
    return TokenizedCorpus(ids=np.arange(n, dtype=np.int64), vocab_size=n, eos_token_id=0)


def test_contiguous_rows_make_each_row_a_continuation():
    B, T = 4, 8
    ds = PackedTokenDataset(_corpus(), seq_len=T, contiguous_rows_batch=B)
    loader = torch.utils.data.DataLoader(ds, batch_size=B)
    prev = None
    for n, (x, _y) in enumerate(loader):
        if prev is not None:
            # Row b of batch n+1 starts right after row b of batch n ended.
            assert torch.equal(x[:, 0], prev[:, -1] + 1), (n, x[:, 0], prev[:, -1])
        prev = x
        if n == 5:
            break


def test_default_order_is_sequential_across_the_batch():
    B, T = 4, 8
    ds = PackedTokenDataset(_corpus(), seq_len=T)
    x, _ = next(iter(torch.utils.data.DataLoader(ds, batch_size=B)))
    assert torch.equal(x[:, 0], torch.arange(0, B * T, T))


def test_wikitext_articles_split_on_level1_headings():
    """DECISIONS #24 tier 3: a `` = Title = `` row starts an article; `` = = Sub = = `` does not."""
    rows = [" = A = \n", "", " text a \n", " = = Sub = = \n", " more a \n", " = B = \n", " text b \n"]
    assert wikitext_articles(rows) == [" = A = \n text a \n = = Sub = = \n more a \n",
                                       " = B = \n text b \n"]
