"""End-to-end check of the spike reading (DECISIONS #24): logits of the whole model with every
spike-native BitLinear run as an integer engine, and as a spike engine.

Three runs on the first windows of the eval tail (CPU, float32, fresh state per window):

  * ``float``: the model as trained (each BitLinear a float matmul on ``s * q``);
  * ``integer``: each of the 84 spike-native consumers accumulates ``W q`` per weight group in
    integers (exact in float32 below 2**24), then applies ``s * alpha_g`` once and sums the
    groups, as ``BitLinear.forward_integer`` does;
  * ``spikes``: the same, with the integer accumulations taken over the ±1 / {0, 1} spikes of
    ``q``, step by step, and summed before the same scaling.

``integer`` and ``spikes`` must give bit-identical logits: the spike reading of an integer
engine is exact. Both differ from ``float`` by float rounding (another summation order), which
a downstream LIF can turn into a different count; the script reports that difference too.

    CUDA_VISIBLE_DEVICES= uv run python scripts/spike_logits_check.py rung5_a4_s0

``N_WIN`` (environment, default 16) sets the number of 128-token windows; ``CORPUS`` as in
spike_accounting.py.
"""
import json, os, sys
import torch
import torch.nn.functional as F
from spiking_ternary_lm.eval.perplexity import protocol_eval_corpus
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV
from spiking_ternary_lm.neurons.binary_lif import decompose_integer_to_binary, decompose_signed_to_spikes
from spiking_ternary_lm.training.data import load_corpus

torch.set_grad_enabled(False)
SEQ = 128
N_WIN = int(os.environ.get("N_WIN", 16))
CORPUS = os.environ.get("CORPUS", "tinystories")
# Consumer BitLinear -> the LIF that produces its input (rung 5, DEC-22).
PRODUCER = {"time_mix.W_r": "time_mix.lif_xr", "time_mix.W_k": "time_mix.lif_xk",
            "time_mix.W_v": "time_mix.lif_xv", "time_mix.W_o": "time_mix.lif_out",
            "channel_mix.W_k": "channel_mix.lif_xk", "channel_mix.W_v": "channel_mix.lif_k",
            "channel_mix.W_r": "channel_mix.lif_xr"}

ids = torch.from_numpy(protocol_eval_corpus(load_corpus(CORPUS)).ids.astype("int64"))
windows = torch.stack([ids[w * SEQ:(w + 1) * SEQ] for w in range(N_WIN)])


def integer_consumer(module, lif, spiking, stats):
    """Forward hook: the consumer's output from exact integer accumulations of its LIF's counts."""
    ternary, alpha = module.pack_weights()  # (out, in) int8, (out, G) fp32
    t = ternary.view(module.out_features, module.num_groups, module.group_size).to(torch.float32)

    def hook(mod, inputs, output):
        x, q = inputs[0], lif.last_spike
        s, zp = lif.lsq_scale_and_zero_point()
        assert float(zp.abs().max()) == 0.0 and mod.bias is None
        stats["max_input_err"] = max(stats["max_input_err"], float((x - s * q).abs().max()))
        group = lambda v: v.to(torch.float32).reshape(*v.shape[:-1], mod.num_groups, mod.group_size)
        if spiking:
            spikes = (decompose_signed_to_spikes(q, bits=lif.num_bits) if lif.observer.qmin < 0
                      else decompose_integer_to_binary(q, bits=lif.num_bits))
            stats["steps"].add(spikes.shape[0])
            acc = sum(torch.einsum("...gk,ogk->...og", group(sp), t) for sp in spikes)
        else:
            acc = torch.einsum("...gk,ogk->...og", group(q), t)
        stats["max_acc"] = max(stats["max_acc"], float(acc.abs().max()))
        return (acc * (s.to(torch.float32) * alpha)).sum(-1).to(x.dtype)
    return hook


def run_model(m, spiking=None):
    stats = {"max_input_err": 0.0, "max_acc": 0.0, "steps": set()}
    handles = [] if spiking is None else [
        blk.get_submodule(cons).register_forward_hook(
            integer_consumer(blk.get_submodule(cons), blk.get_submodule(prod), spiking, stats))
        for blk in m.blocks for cons, prod in PRODUCER.items()]
    m.reset_state()
    logits = m(windows).float()
    for h in handles:
        h.remove()
    return logits, stats


for run in sys.argv[1:]:
    ck = torch.load(f"checkpoints/ladder/{run}/final.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    assert cfg.get("act_quant_inputs"), f"{run}: every projection input must be a LIF output (rung 5)"
    m = SpikingTernaryRWKV(SpikingTernaryRWKVConfig(**cfg)); m.load_state_dict(ck["model"]); m.eval()
    ref, _ = run_model(m)
    integer, st_i = run_model(m, spiking=False)
    spikes, st_s = run_model(m, spiking=True)
    assert st_s["max_acc"] < 2 ** 24, "integer sums no longer exact in float32"

    targets = windows[:, 1:].reshape(-1)
    ce = lambda lg: float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), targets))
    vs_float = (integer - ref).abs()
    out = {"run": run, "corpus": CORPUS, "windows": N_WIN, "seq": SEQ, "device": "cpu",
           "consumers": 7 * len(m.blocks), "virtual_steps": sorted(st_s["steps"]),
           "max_input_err_vs_s_times_q": max(st_i["max_input_err"], st_s["max_input_err"]),
           "max_abs_integer_sum": st_s["max_acc"],
           "spikes_vs_integer_bit_identical": bool(torch.equal(spikes, integer)),
           "spikes_vs_integer_max_abs_diff": float((spikes - integer).abs().max()),
           "integer_vs_float": {"max_abs_logit_diff": float(vs_float.max()),
                                "mean_abs_logit_diff": float(vs_float.mean()),
                                "max_abs_logit": float(ref.abs().max()),
                                "argmax_agreement": float((integer.argmax(-1) == ref.argmax(-1)).float().mean())},
           "ce": {"float": ce(ref), "integer": ce(integer), "spikes": ce(spikes)}}
    print(json.dumps(out), flush=True)
    os.makedirs("runs/accounting", exist_ok=True)
    with open(f"runs/accounting/logits_check_{run}.json", "w") as f:
        json.dump(out, f, indent=1)
