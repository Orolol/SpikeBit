"""DECISIONS #15 rung 0 on a tiny random RWKV-4: with quantization off and
the teacher's ``relu²`` kept, the converted student is the teacher."""
from __future__ import annotations

import torch

from spiking_ternary_lm.conversion.rwkv_loader import TeacherInfo
from spiking_ternary_lm.conversion.swap_layers import load_teacher_into_student
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV
from spiking_ternary_lm.neurons.fai_lif import calibrate_observers


def _tiny_teacher() -> tuple[torch.nn.Module, TeacherInfo]:
    from transformers import RwkvConfig, RwkvForCausalLM

    torch.manual_seed(0)
    cfg = RwkvConfig(
        vocab_size=256, hidden_size=128, num_hidden_layers=2,
        attention_hidden_size=128, intermediate_size=512, context_length=64,
    )
    model = RwkvForCausalLM(cfg).eval()
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    info = TeacherInfo(
        model_id="tiny-random", state_dict=sd, n_layers=2,
        hidden_size=128, ffn_dim=512, vocab_size=256,
    )
    return model, info


def _student(info: TeacherInfo, relu_sq: bool) -> SpikingTernaryRWKV:
    cfg = SpikingTernaryRWKVConfig(
        vocab_size=info.vocab_size, hidden_size=info.hidden_size,
        n_layers=info.n_layers, ffn_dim=info.ffn_dim, group_size=32,
        act_quant="none", weight_quant="none", ffn_relu_sq=relu_sq,
    )
    student = SpikingTernaryRWKV(cfg)
    report = load_teacher_into_student(info, student, strict=True)
    assert not report.skipped_teacher
    return student.eval()


@torch.no_grad()
def test_unquantized_student_reproduces_teacher_logits():
    teacher, info = _tiny_teacher()
    ids = torch.randint(0, 256, (3, 24), generator=torch.Generator().manual_seed(1))
    ref = teacher(input_ids=ids, use_cache=False).logits

    faithful = _student(info, relu_sq=True)
    faithful.reset_state()
    assert torch.allclose(faithful(ids), ref, rtol=1e-4, atol=1e-4)

    # The legacy layout (FAI-LIF in place of relu²) is a different function.
    legacy = _student(info, relu_sq=False)
    legacy.reset_state()
    assert (legacy(ids) - ref).abs().max() > 1e-2


@torch.no_grad()
def test_faithful_8bit_activations_stay_close_to_teacher():
    """Rung 3 wiring: 8-bit MinMax activations with the faithful grids and
    dequantized outputs keep the converted student near the teacher; the
    legacy integer-output layout does not."""
    teacher, info = _tiny_teacher()
    ids = torch.randint(0, 256, (3, 24), generator=torch.Generator().manual_seed(1))
    ref = teacher(input_ids=ids, use_cache=False).logits

    def run(**over):
        cfg = SpikingTernaryRWKVConfig(
            vocab_size=256, hidden_size=128, n_layers=2, ffn_dim=512, group_size=32,
            weight_quant="none", bits=8, **over,
        )
        student = SpikingTernaryRWKV(cfg)
        load_teacher_into_student(info, student, strict=True)
        calibrate_observers(student, [ids])
        student.eval(); student.reset_state()
        return float((student(ids) - ref).norm() / ref.norm())

    faithful_cfg = dict(ffn_relu_sq=True, act_grid="faithful", act_output="dequant", lif_stateless=True)
    faithful = run(**faithful_cfg)
    legacy = run()
    assert faithful < 0.05
    assert legacy > 10 * faithful
    # LSQ starts from the same calibrated grid as MinMax.
    assert abs(run(**faithful_cfg, act_quant="lsq") - faithful) < 1e-3
