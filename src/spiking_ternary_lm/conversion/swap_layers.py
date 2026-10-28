"""Map HuggingFace RWKV-4 weights onto our SpikingTernaryRWKV.

The spec describes a separate "swap" step that replaces ``nn.Linear``
with :class:`BitLinear` and activations with :class:`FAILIF`. In our
codebase the swap is *structural*: the student model already uses
BitLinear + FAI-LIF by construction. What's left is the **weight
assignment**: the teacher's FP weights become the BitLinear shadow
weights, and the teacher's RWKV recurrent parameters (``time_decay``,
``time_first``, ``time_mix_*``) are copied verbatim.

The newly-added LIF parameters (``beta_logit``, ``gamma_logit``,
``v_thr``) have no teacher equivalent and keep their default init.

Sign conventions: see DECISIONS.md #8. The HF
``rwkv.blocks.{i}.attention.time_decay`` matches our
``blocks.{i}.time_mix.time_decay`` 1:1 — the BlinkDL parameterisation
where the per-step log-magnitude shift is ``-exp(time_decay)`` (always
negative) is what both implementations use.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from spiking_ternary_lm.conversion.rwkv_loader import TeacherInfo
from spiking_ternary_lm.model.model import SpikingTernaryRWKV


# Static name map between HF RwkvForCausalLM and SpikingTernaryRWKV.
# `{i}` is substituted per layer.
HF_TO_STUDENT: dict[str, str] = {
    "rwkv.embeddings.weight": "embed.weight",
    "rwkv.blocks.0.pre_ln.weight": "ln_in.weight",
    "rwkv.blocks.0.pre_ln.bias": "ln_in.bias",
    "rwkv.ln_out.weight": "ln_out.weight",
    "rwkv.ln_out.bias": "ln_out.bias",
    "head.weight": "lm_head.weight",
}

# Per-layer suffixes — substitute `{i}` for the layer index.
HF_LAYER_TO_STUDENT: dict[str, str] = {
    "rwkv.blocks.{i}.ln1.weight": "blocks.{i}.ln1.weight",
    "rwkv.blocks.{i}.ln1.bias": "blocks.{i}.ln1.bias",
    "rwkv.blocks.{i}.ln2.weight": "blocks.{i}.ln2.weight",
    "rwkv.blocks.{i}.ln2.bias": "blocks.{i}.ln2.bias",
    # Time-mix (attention)
    "rwkv.blocks.{i}.attention.time_decay": "blocks.{i}.time_mix.time_decay",
    "rwkv.blocks.{i}.attention.time_first": "blocks.{i}.time_mix.time_first",
    "rwkv.blocks.{i}.attention.time_mix_key": "blocks.{i}.time_mix.mu_k",
    "rwkv.blocks.{i}.attention.time_mix_value": "blocks.{i}.time_mix.mu_v",
    "rwkv.blocks.{i}.attention.time_mix_receptance": "blocks.{i}.time_mix.mu_r",
    "rwkv.blocks.{i}.attention.key.weight": "blocks.{i}.time_mix.W_k.shadow_weight",
    "rwkv.blocks.{i}.attention.value.weight": "blocks.{i}.time_mix.W_v.shadow_weight",
    "rwkv.blocks.{i}.attention.receptance.weight": "blocks.{i}.time_mix.W_r.shadow_weight",
    "rwkv.blocks.{i}.attention.output.weight": "blocks.{i}.time_mix.W_o.shadow_weight",
    # Channel-mix (FFN)
    "rwkv.blocks.{i}.feed_forward.time_mix_key": "blocks.{i}.channel_mix.mu_k",
    "rwkv.blocks.{i}.feed_forward.time_mix_receptance": "blocks.{i}.channel_mix.mu_r",
    "rwkv.blocks.{i}.feed_forward.key.weight": "blocks.{i}.channel_mix.W_k.shadow_weight",
    "rwkv.blocks.{i}.feed_forward.value.weight": "blocks.{i}.channel_mix.W_v.shadow_weight",
    "rwkv.blocks.{i}.feed_forward.receptance.weight": "blocks.{i}.channel_mix.W_r.shadow_weight",
}


@dataclass
class ConversionReport:
    assigned: list[str]      # student keys that were assigned from teacher
    skipped_teacher: list[str]  # teacher keys we did NOT use (warn about)
    kept_student: list[str]  # student-only keys that kept default init (LIF, etc.)


def convert_state_dict(teacher: TeacherInfo, n_layers: int) -> tuple[dict[str, Tensor], list[str]]:
    """Translate the teacher state-dict into a student-key dict.

    Returns ``(mapped, unmapped_teacher_keys)``. ``mapped`` contains
    student-key → tensor entries. Tensors are CPU FP32 (caller casts).
    """
    src = teacher.state_dict
    out: dict[str, Tensor] = {}
    used: set[str] = set()

    for hf_key, st_key in HF_TO_STUDENT.items():
        if hf_key in src:
            out[st_key] = src[hf_key]
            used.add(hf_key)

    for i in range(n_layers):
        for hf_pat, st_pat in HF_LAYER_TO_STUDENT.items():
            hf_key = hf_pat.format(i=i)
            st_key = st_pat.format(i=i)
            if hf_key in src:
                t = src[hf_key]
                # HF stores time_mix_* with shape (1, 1, hidden). We use (hidden,).
                if t.dim() == 3 and t.shape[0] == 1 and t.shape[1] == 1:
                    t = t.squeeze(0).squeeze(0)
                out[st_key] = t
                used.add(hf_key)

    unmapped = sorted(set(src) - used)
    return out, unmapped


def load_teacher_into_student(
    teacher: TeacherInfo,
    student: SpikingTernaryRWKV,
    strict: bool = False,
) -> ConversionReport:
    """Assign teacher weights into a freshly-built SpikingTernaryRWKV.

    The student is mutated in place. Tensors are cast to the dtype of
    the corresponding student parameter (so BitLinear shadow weights end
    up in ``weight_dtype``, time-mix params in FP32, etc.).

    If ``strict`` and any student key targeted by the static map is
    missing or has a shape mismatch, this raises ``RuntimeError``.
    Otherwise it logs into the returned :class:`ConversionReport` and
    continues.
    """
    mapped, unmapped_hf = convert_state_dict(teacher, n_layers=student.config.n_layers)

    student_state = dict(student.state_dict())
    assigned: list[str] = []
    skipped_teacher: list[str] = list(unmapped_hf)

    # Apply mapped tensors.
    with torch.no_grad():
        for st_key, tensor in mapped.items():
            if st_key not in student_state:
                if strict:
                    raise RuntimeError(f"target missing in student: {st_key}")
                continue
            student_param = student_state[st_key]
            if tuple(tensor.shape) != tuple(student_param.shape):
                if strict:
                    raise RuntimeError(
                        f"shape mismatch on {st_key}: teacher {tuple(tensor.shape)} "
                        f"vs student {tuple(student_param.shape)}"
                    )
                continue
            target_dtype = student_param.dtype
            student_param.copy_(tensor.to(device=student_param.device, dtype=target_dtype))
            assigned.append(st_key)

    # Student keys we did NOT assign (LIF beta_logit / gamma_logit / v_thr,
    # and any state-buffer entries) — those keep their default init.
    student_keys_kept = sorted(set(student_state) - set(assigned))

    return ConversionReport(
        assigned=sorted(assigned),
        skipped_teacher=skipped_teacher,
        kept_student=student_keys_kept,
    )
