"""The spike engine reproduces the integer engine bit for bit on a whole model (DECISIONS #24)."""
import torch

from spiking_ternary_lm.eval import integer_engine
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV


def test_spike_engine_logits_equal_integer_engine_logits():
    torch.manual_seed(0)
    # Rung 5's flags (DEC-22) at a toy size; widths are multiples of the 128-input groups.
    cfg = SpikingTernaryRWKVConfig(vocab_size=64, hidden_size=128, n_layers=2, ffn_dim=256,
                                   act_quant="lsq", act_grid="faithful", act_output="dequant",
                                   act_ste="clipped", lif_stateless=True, act_quant_inputs=True,
                                   ffn_relu_sq=True)
    m = SpikingTernaryRWKV(cfg)
    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        m.train(); m.reset_state(); m(x)  # LSQ step sizes initialise on the first batch
        m.eval()
        logits = {}
        for name, spiking in (("integer", False), ("spikes", True)):
            handles, stats = integer_engine.install(m, integer_engine.PRODUCER, spiking)
            m.reset_state()
            logits[name] = m(x)
            for h in handles:
                h.remove()
            assert stats["max_input_err"] < 1e-5  # each consumer's input is s * q
            assert stats["max_acc"] <= 128 * 15  # a group sum fits a 12-bit accumulator
            assert stats["max_acc"] > 0 and stats["steps"] == ({8, 15} if spiking else set())
    assert torch.isfinite(logits["integer"]).all()
    assert torch.equal(logits["spikes"], logits["integer"])
