"""Tests for the conversion layer — uses a synthetic teacher state-dict
so we don't depend on a 350 MB HF download in CI."""
from __future__ import annotations

import torch

from spiking_ternary_lm.conversion.rwkv_loader import TeacherInfo
from spiking_ternary_lm.conversion.swap_layers import (
    HF_LAYER_TO_STUDENT,
    HF_TO_STUDENT,
    convert_state_dict,
    load_teacher_into_student,
)
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV


def _make_fake_teacher(hidden=128, ffn=512, vocab=256, layers=2) -> TeacherInfo:
    sd: dict[str, torch.Tensor] = {
        "rwkv.embeddings.weight": torch.randn(vocab, hidden),
        "rwkv.blocks.0.pre_ln.weight": torch.ones(hidden),
        "rwkv.blocks.0.pre_ln.bias": torch.zeros(hidden),
        "rwkv.ln_out.weight": torch.ones(hidden),
        "rwkv.ln_out.bias": torch.zeros(hidden),
        "head.weight": torch.randn(vocab, hidden),
    }
    for i in range(layers):
        sd.update({
            f"rwkv.blocks.{i}.ln1.weight": torch.ones(hidden),
            f"rwkv.blocks.{i}.ln1.bias": torch.zeros(hidden),
            f"rwkv.blocks.{i}.ln2.weight": torch.ones(hidden),
            f"rwkv.blocks.{i}.ln2.bias": torch.zeros(hidden),
            f"rwkv.blocks.{i}.attention.time_decay": torch.randn(hidden),
            f"rwkv.blocks.{i}.attention.time_first": torch.randn(hidden),
            f"rwkv.blocks.{i}.attention.time_mix_key": torch.rand(1, 1, hidden),
            f"rwkv.blocks.{i}.attention.time_mix_value": torch.rand(1, 1, hidden),
            f"rwkv.blocks.{i}.attention.time_mix_receptance": torch.rand(1, 1, hidden),
            f"rwkv.blocks.{i}.attention.key.weight": torch.randn(hidden, hidden) * 0.02,
            f"rwkv.blocks.{i}.attention.value.weight": torch.randn(hidden, hidden) * 0.02,
            f"rwkv.blocks.{i}.attention.receptance.weight": torch.randn(hidden, hidden) * 0.02,
            f"rwkv.blocks.{i}.attention.output.weight": torch.randn(hidden, hidden) * 0.02,
            f"rwkv.blocks.{i}.feed_forward.time_mix_key": torch.rand(1, 1, hidden),
            f"rwkv.blocks.{i}.feed_forward.time_mix_receptance": torch.rand(1, 1, hidden),
            f"rwkv.blocks.{i}.feed_forward.key.weight": torch.randn(ffn, hidden) * 0.02,
            f"rwkv.blocks.{i}.feed_forward.value.weight": torch.randn(hidden, ffn) * 0.02,
            f"rwkv.blocks.{i}.feed_forward.receptance.weight": torch.randn(hidden, hidden) * 0.02,
        })
    return TeacherInfo(
        model_id="fake", state_dict=sd, n_layers=layers, hidden_size=hidden,
        ffn_dim=ffn, vocab_size=vocab,
    )


def test_convert_state_dict_returns_all_layer_params():
    teacher = _make_fake_teacher(hidden=128, ffn=512, vocab=256, layers=2)
    mapped, unmapped = convert_state_dict(teacher, n_layers=2)
    # Global keys.
    assert "embed.weight" in mapped
    assert "ln_in.weight" in mapped
    assert "lm_head.weight" in mapped
    # Per-layer keys.
    for i in range(2):
        for st_pat in HF_LAYER_TO_STUDENT.values():
            assert st_pat.format(i=i) in mapped
    # Nothing was left unmapped from the fake teacher.
    assert unmapped == []


def test_load_into_student_assigns_and_runs():
    teacher = _make_fake_teacher(hidden=128, ffn=512, vocab=256, layers=2)
    cfg = SpikingTernaryRWKVConfig.tiny()  # matches the fake teacher shape
    student = SpikingTernaryRWKV(cfg)
    report = load_teacher_into_student(teacher, student)
    # All static-mapped keys should be assigned.
    assert "embed.weight" in report.assigned
    assert "lm_head.weight" in report.assigned
    # FAI-LIF parameters keep their default init (no teacher equivalent).
    kept_set = set(report.kept_student)
    assert any("beta_logit" in k for k in kept_set)
    assert any("gamma_logit" in k for k in kept_set)
    # Forward must still run.
    student.reset_state()
    ids = torch.randint(0, cfg.vocab_size, (2, 4), dtype=torch.long)
    logits = student(ids)
    assert logits.shape == (2, 4, cfg.vocab_size)


def test_time_mix_shape_squeezing():
    """HF stores time_mix_* as (1,1,H); we use (H,)."""
    teacher = _make_fake_teacher()
    mapped, _ = convert_state_dict(teacher, n_layers=2)
    assert mapped["blocks.0.time_mix.mu_k"].shape == (128,)
    assert mapped["blocks.0.channel_mix.mu_r"].shape == (128,)


def test_shape_mismatch_strict_raises():
    teacher = _make_fake_teacher(hidden=128)
    # Build a student with a different hidden size — assignment should fail in strict mode.
    cfg = SpikingTernaryRWKVConfig.tiny()
    cfg.hidden_size = 64
    cfg.ffn_dim = 256
    cfg.group_size = 16
    student = SpikingTernaryRWKV(cfg)
    try:
        load_teacher_into_student(teacher, student, strict=True)
    except RuntimeError as e:
        assert "shape mismatch" in str(e) or "missing" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
