"""Diagnostic: stateless (cold) vs streaming (warm) eval of ladder checkpoints.

Cold = DECISIONS #15 protocol (fresh state per 128-token window). Warm = the
eval tail split into 16 contiguous streams, state carried across windows.
Also reports the CE by position inside the window. First 1 M tokens of the
eval tail. Arguments: run names under ``checkpoints/ladder`` (``final.pt``),
or ``teacher`` for the unquantized converted teacher. For a run with stateful
LIFs (rung 4b), also the learned beta per LIF site and the cold eval with beta
forced to 0 (membrane ablation).

    uv run python scripts/_diagnose_state_regime.py teacher rung1_lr3e-5_s0
"""
import json, math, sys
import torch, torch.nn.functional as F
from spiking_ternary_lm.conversion.rwkv_loader import load_rwkv4_teacher
from spiking_ternary_lm.conversion.swap_layers import load_teacher_into_student
from spiking_ternary_lm.eval.perplexity import protocol_eval_corpus
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV
from spiking_ternary_lm.neurons.fai_lif import FAILIF, set_stateless
from spiking_ternary_lm.training.data import tokenize_tinystories

torch.set_grad_enabled(False)
dev = torch.device("cuda")
SEQ, B, NTOK = 128, 16, 1_000_000
ids = torch.from_numpy(protocol_eval_corpus(tokenize_tinystories(tokenizer_id="RWKV/rwkv-4-169m-pile")).ids.astype("int64"))
n_win = NTOK // SEQ // B * B
per_row = n_win // B

def load(path):
    if path == "teacher":
        info = load_rwkv4_teacher("RWKV/rwkv-4-169m-pile")
        m = SpikingTernaryRWKV(SpikingTernaryRWKVConfig(vocab_size=info.vocab_size, hidden_size=info.hidden_size,
            n_layers=info.n_layers, ffn_dim=info.ffn_dim, act_quant="none", weight_quant="none", ffn_relu_sq=True))
        load_teacher_into_student(info, m, strict=True)
    else:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = SpikingTernaryRWKV(SpikingTernaryRWKVConfig(**ck["config"])); m.load_state_dict(ck["model"])
    return m.eval().to(dev)

def run(m, warm):
    # Row b streams windows b*per_row ... (b+1)*per_row-1 in order.
    pos_loss = torch.zeros(SEQ, dtype=torch.float64); count = 0
    m.reset_state()
    for k in range(per_row):
        starts = [(b * per_row + k) * SEQ for b in range(B)]
        win = torch.stack([ids[s:s + SEQ + 1] for s in starts]).to(dev)
        x, y = win[:, :-1], win[:, 1:]
        if not warm:
            m.reset_state()
        logits = m(x).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").view(B, SEQ)
        pos_loss += ce.sum(0).double().cpu(); count += B
    pos = pos_loss / count
    return {"ppl": math.exp(float(pos.mean())), "ce_pos0_8": float(pos[:8].mean()),
            "ce_pos8_32": float(pos[8:32].mean()), "ce_pos32_128": float(pos[32:].mean())}

out = {}
for name in sys.argv[1:]:
    path = name if name == "teacher" else f"checkpoints/ladder/{name}/final.pt"
    m = load(path)
    betas = {}
    for n, mod in m.named_modules():
        if isinstance(mod, FAILIF) and not mod.stateless:
            site = n.split(".", 2)[2]
            betas.setdefault(site, []).append(mod.beta.detach().float().cpu())
    beta_stats = {k: {"mean": float(torch.cat(v).mean()), "min": float(torch.cat(v).min()),
                      "max": float(torch.cat(v).max())} for k, v in betas.items()}
    out[name] = {"cold": run(m, False), "warm": run(m, True), "beta": beta_stats}
    if beta_stats:
        set_stateless(m)
        out[name]["cold_beta0"] = run(m, False)
    print(name, json.dumps(out[name]), flush=True)
    del m; torch.cuda.empty_cache()
