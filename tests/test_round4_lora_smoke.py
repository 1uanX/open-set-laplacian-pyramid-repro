from __future__ import annotations

from pathlib import Path

import pytest
import torch

from repro_model import ModelConfig, PRRPADModel
from round4_lora_utils import (
    ROUND4_FIXED_SEEDS,
    LoRALinear,
    Round4Settings,
    apply_round4_scaffolding,
    iter_lora_modules,
    measure_inference_latency_ms,
    merge_lora_adapters,
    parameter_report,
    unmerge_lora_adapters,
)


def _model() -> PRRPADModel:
    torch.manual_seed(20260414)
    return PRRPADModel(ModelConfig(num_classes=4, input_dim=6, hidden_dim=16, embed_dim=8, latent_dim=5))


def _full_lora_settings(**overrides: object) -> Round4Settings:
    payload = {
        "enabled": True,
        "method_variant": "full_lora",
        "seed": 20260414,
        "adapter_rank": 2,
        "adapter_alpha": 4.0,
        "adapter_dropout": 0.0,
    }
    payload.update(overrides)
    return Round4Settings.from_dict(payload)


def test_round4_config_files_parse_and_preserve_fixed_values() -> None:
    config_dir = Path("configs/round4_contract_v1")
    expected = {
        "p003_baseline.json",
        "full_lora_rank8.json",
        "frozen_no_adapter.json",
        "full_finetune.json",
        "lora_rank2.json",
        "lora_rank4.json",
        "lora_rank8.json",
        "lora_rank16.json",
        "no_preservation_lock_rank8.json",
        "merged_lora_rank8.json",
        "unmerged_lora_rank8.json",
    }
    parsed = {path.name: Round4Settings.from_json(path) for path in config_dir.glob("*.json")}
    assert expected <= set(parsed)
    assert all(settings.contract_version == "round4_v1" for settings in parsed.values())
    assert all(settings.fixed_seeds == ROUND4_FIXED_SEEDS for settings in parsed.values())
    assert all(settings.expected_experiments == ("exp4", "exp5", "exp6") for settings in parsed.values())
    assert parsed["full_lora_rank8.json"].adapter_rank == 8
    assert parsed["full_lora_rank8.json"].preservation_lock is True
    assert parsed["no_preservation_lock_rank8.json"].preservation_lock is False


def test_round4_rejects_unfixed_seed_and_unsupported_experiment() -> None:
    with pytest.raises(ValueError, match="fixed contract seeds"):
        Round4Settings.from_dict({"enabled": True, "method_variant": "p003_baseline", "seed": 1})
    settings = Round4Settings.from_dict({"enabled": True, "method_variant": "p003_baseline", "seed": 20260414})
    with pytest.raises(ValueError, match="exp4, exp5, exp6"):
        settings.validate_selected_experiments(["exp1"])


def test_lora_insertion_targets_backbone_and_embedding_heads_only() -> None:
    model = _model()
    model.eval()
    features = torch.randn(3, 6)
    before = model.forward_unknown(features)

    report = apply_round4_scaffolding(model, _full_lora_settings())
    after = model.forward_unknown(features)
    lora_names = [name for name, _module in iter_lora_modules(model)]

    assert report["adapter_report"]["inserted_count"] == 12
    assert len(lora_names) == 12
    assert all(name.startswith(("backbone.", "scale1.", "scale2.", "scale3.")) for name in lora_names)
    assert not any(name.startswith(("head1.", "head2.", "head3.", "generator.", "discriminator.")) for name in lora_names)
    assert torch.allclose(after["probs"], before["probs"], atol=1e-6)
    assert torch.allclose(after["open_score"], before["open_score"], atol=1e-6)


def test_full_lora_freezes_base_and_boundary_heads() -> None:
    model = _model()
    apply_round4_scaffolding(model, _full_lora_settings())
    trainable_names = parameter_report(model)["trainable_parameter_names"]
    assert trainable_names
    assert all(".lora_A" in name or ".lora_B" in name for name in trainable_names)
    assert all(not parameter.requires_grad for parameter in model.head1.parameters())
    assert all(not parameter.requires_grad for parameter in model.head2.parameters())
    assert all(not parameter.requires_grad for parameter in model.head3.parameters())


def test_no_preservation_lock_unfreezes_boundary_heads_without_changing_inference_code() -> None:
    model = _model()
    apply_round4_scaffolding(model, _full_lora_settings(method_variant="no_preservation_lock", preservation_lock=False))
    trainable_names = parameter_report(model)["trainable_parameter_names"]
    assert any(name.startswith(("head1.", "head2.", "head3.")) for name in trainable_names)
    assert model.open_score.__func__ is PRRPADModel.open_score
    assert model.forward_unknown.__func__ is PRRPADModel.forward_unknown


def test_merge_and_unmerge_keep_forward_outputs_stable() -> None:
    model = _model()
    apply_round4_scaffolding(model, _full_lora_settings())
    for _name, module in iter_lora_modules(model):
        assert isinstance(module, LoRALinear)
        module.lora_B.data.normal_(mean=0.0, std=0.01)
    model.eval()
    features = torch.randn(4, 6)
    unmerged = model.forward_unknown(features)["probs"]
    merged_names = merge_lora_adapters(model)
    merged = model.forward_unknown(features)["probs"]
    unmerged_names = unmerge_lora_adapters(model)
    restored = model.forward_unknown(features)["probs"]
    assert len(merged_names) == 12
    assert merged_names == unmerged_names
    assert torch.allclose(merged, unmerged, atol=1e-5)
    assert torch.allclose(restored, unmerged, atol=1e-5)


def test_resource_payload_and_latency_hook_are_available_on_synthetic_input() -> None:
    model = _model()
    apply_round4_scaffolding(model, _full_lora_settings())
    payload = parameter_report(model)
    latency = measure_inference_latency_ms(
        model,
        torch.randn(2, 6),
        torch.device("cpu"),
        warmup_steps=0,
        repeats=1,
    )
    assert payload["total_parameters"] > payload["trainable_parameters"] > 0
    assert payload["adapter_parameters"] == payload["trainable_parameters"]
    assert latency >= 0.0
