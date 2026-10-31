"""Integer and spike engines for the spike-native BitLinears (DECISIONS #24).

A forward hook replaces a ternary consumer's float matmul on ``s * q`` by integer accumulations
of its LIF's counts ``q``, one per weight group (exact in float32 below 2**24), then applies
``s * alpha_g`` once and sums the groups, as ``BitLinear.forward_integer`` does. With
``spiking=True`` the accumulations are taken over the ±1 / {0, 1} spikes of ``q``, step by
step, and summed before the same scaling. Downstream LIFs then see the engine's values, so
their counts are the engine's own trace rather than the float graph's.
"""
import torch

from ..neurons.binary_lif import decompose_integer_to_binary, decompose_signed_to_spikes

# Consumer BitLinear -> the LIF that produces its input, per block (rung 5, DEC-22). Rung 3
# keeps only the two neuron-site consumers.
PRODUCER = {"time_mix.W_r": "time_mix.lif_xr", "time_mix.W_k": "time_mix.lif_xk",
            "time_mix.W_v": "time_mix.lif_xv", "time_mix.W_o": "time_mix.lif_out",
            "channel_mix.W_k": "channel_mix.lif_xk", "channel_mix.W_v": "channel_mix.lif_k",
            "channel_mix.W_r": "channel_mix.lif_xr"}


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


def install(model, producer, spiking=False):
    """Hook every block's consumers in ``producer`` (consumer -> LIF); returns (handles, stats).

    ``stats`` collects the largest |x - s q| at a consumer input, the largest |integer sum|
    and the spike steps seen. Remove the handles to get the float graph back.
    """
    stats = {"max_input_err": 0.0, "max_acc": 0.0, "steps": set()}
    handles = [blk.get_submodule(cons).register_forward_hook(
                   integer_consumer(blk.get_submodule(cons), blk.get_submodule(prod), spiking, stats))
               for blk in model.blocks for cons, prod in producer.items()]
    return handles, stats
