from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bcrps_utils import (
    BCRPSSettings,
    REQUIRED_FAILURE_RECORD_FIELDS,
    collect_reciprocal_point_tensors,
    compute_bcrps_loss,
    write_reciprocal_point_diagnostics,
)
from repro_model import ModelConfig, PRRPADModel
from run_repro import _validate_frozen_pair_source, _validate_protocol_lock_runtime, _write_failure_run_record


def _model() -> PRRPADModel:
    torch.manual_seed(17)
    return PRRPADModel(ModelConfig(num_classes=4, input_dim=6, hidden_dim=16, embed_dim=8, latent_dim=5))


def _valid_protocol_lock_payload(output_dir: Path | str, frozen_pair_manifest: Path | str | None = None) -> dict:
    return {
        "strict_protocol_lock": True,
        "enable_bcrps": True,
        "lambda": 0.001,
        "margin": 1.0,
        "use_normalized_distance": True,
        "use_warmup": True,
        "warmup_fraction": 0.2,
        "ablation_mode": "full_bcrps",
        "logging_diagnostics": True,
        "run_id": "strict_smoke",
        "frozen_pair_manifest": str(frozen_pair_manifest) if frozen_pair_manifest is not None else None,
        "protocol_lock": {
            "contract_source": r"D:\keyan\ai_research_workflow_base\03_ideas\research_contract_v1.md",
            "experiment_plan_source": r"D:\keyan\ai_research_workflow_base\05_experiments\experiment_plan.md",
            "planned_run_manifest_source": r"D:\keyan\ai_research_workflow_base\05_experiments\planned_run_manifest.md",
            "run_id": "strict_smoke",
            "output_dir": str(output_dir),
            "dataset_split": {
                "dataset_root": (
                    r"D:\keyan\research-units-pipeline-skills-main\workspaces"
                    r"\open-set-laplacian-pyramid-repro\artifacts\datasets\pr_rpad_v1"
                ),
                "split_identity": "pr_rpad_v1_mixed_80_10_10",
                "train": r"exp*\mixed\train",
                "val": r"exp*\mixed\val",
                "test": r"exp*\mixed\test",
            },
            "known_unknown_split": {
                "exp4": {"known": [1, 3, 4], "unknown": [2]},
                "exp5": {"known": [1, 4, 5], "unknown": [2, 3]},
                "exp6": {"known": [1, 4, 5, 7], "unknown": [2, 3, 6]},
            },
            "test_set": r"exp*\mixed\test",
            "metric": {
                "primary": "OSCR",
                "primary_scope": ["exp4", "exp5", "exp6"],
                "guardrails": ["ACC_known", "ACC", "MR", "MP", "MIOU"],
            },
            "threshold": "unchanged_p003_open_set_threshold_sweep",
            "inference_protocol": "unchanged_p003_forward_unknown_open_score_evaluate_open_set",
            "frozen_similar_class_pair_source": str(frozen_pair_manifest) if frozen_pair_manifest is not None else None,
            "no_test_set_tuning": True,
            "no_undeclared_extra_data": True,
            "failure_record": {
                "enabled": True,
                "required_fields": sorted(REQUIRED_FAILURE_RECORD_FIELDS),
            },
            "diagnostics": {
                "reciprocal_point_distance": True,
                "similar_class_confusion": True,
                "oscr": True,
                "acc_known": True,
                "primary_metric": True,
                "runtime_cost": True,
            },
        },
    }


def _write_protocol_lock_pair_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "experiments": {
                    "exp4": {
                        "pairs": [
                            {"pair_rank": 1, "class_i": 0, "class_j": 1, "mean_normalized_distance": 0.5}
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _remove_nested(payload: dict, dotted_key: str) -> dict:
    mutated = json.loads(json.dumps(payload))
    parts = dotted_key.split(".")
    cursor = mutated
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor.pop(parts[-1])
    return mutated


def test_reciprocal_point_tensors_have_expected_shape() -> None:
    model = _model()
    points = collect_reciprocal_point_tensors(model)
    assert len(points) == 3
    assert all(tuple(item.shape) == (4, 8) for item in points)


def test_bcrps_loss_is_scalar_finite_and_respects_warmup() -> None:
    model = _model()
    settings = BCRPSSettings.from_dict(
        {
            "enable_bcrps": True,
            "lambda": 0.001,
            "margin": 1.0,
            "use_normalized_distance": True,
            "use_warmup": True,
            "warmup_fraction": 0.2,
            "ablation_mode": "full_bcrps",
        }
    )
    warmup_result = compute_bcrps_loss(collect_reciprocal_point_tensors(model), settings, iteration=1, total_iterations=10)
    active_result = compute_bcrps_loss(collect_reciprocal_point_tensors(model), settings, iteration=3, total_iterations=10)
    assert warmup_result.loss.ndim == 0
    assert torch.isfinite(warmup_result.unweighted_loss)
    assert warmup_result.in_warmup is True
    assert warmup_result.loss.item() == 0.0
    assert active_result.active is True
    assert torch.isfinite(active_result.loss)
    model.zero_grad(set_to_none=True)
    active_result.loss.backward()
    assert all(point.grad is not None for point in model.reciprocal_point_tensors())


def test_lambda_zero_keeps_baseline_loss_value() -> None:
    model = _model()
    features = torch.randn(5, 6)
    labels = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
    baseline_loss = model.forward_known(features, labels)["loss"]
    settings = BCRPSSettings.from_dict(
        {
            "enable_bcrps": True,
            "lambda": 0.0,
            "margin": 1.0,
            "use_normalized_distance": True,
            "use_warmup": False,
            "warmup_fraction": 0.0,
            "ablation_mode": "lambda_zero",
        }
    )
    bcrps_result = compute_bcrps_loss(collect_reciprocal_point_tensors(model), settings, iteration=1, total_iterations=10)
    combined_loss = baseline_loss + bcrps_result.loss
    assert torch.allclose(combined_loss, baseline_loss)
    assert bcrps_result.unweighted_loss.item() >= 0.0


def test_mean_anchor_control_is_finite() -> None:
    model = _model()
    settings = BCRPSSettings.from_dict(
        {
            "enable_bcrps": True,
            "lambda": 0.001,
            "margin": 1.0,
            "use_normalized_distance": True,
            "use_warmup": False,
            "warmup_fraction": 0.0,
            "ablation_mode": "mean_anchor_control",
        }
    )
    result = compute_bcrps_loss(collect_reciprocal_point_tensors(model), settings, iteration=1, total_iterations=10)
    assert result.active is True
    assert torch.isfinite(result.loss)


def test_contract_config_parses() -> None:
    config_path = Path("configs/bcrps_contract_v1/full_bcrps.json")
    settings = BCRPSSettings.from_json(config_path)
    assert settings.enable_bcrps is True
    assert settings.lambda_weight == 0.001
    assert settings.margin == 1.0
    assert settings.use_normalized_distance is True
    assert settings.warmup_fraction == 0.2
    assert settings.frozen_pair_manifest


def test_valid_protocol_lock_passes_validation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "frozen_pair_manifest.json"
    _write_protocol_lock_pair_manifest(manifest_path)
    output_dir = tmp_path / "strict_output"
    settings = BCRPSSettings.from_dict(_valid_protocol_lock_payload(output_dir, manifest_path))
    protocol_payload = _validate_protocol_lock_runtime(
        Path.cwd(),
        settings,
        settings,
        run_id="strict_smoke",
        output_dir=output_dir,
        selected_experiments=["exp4"],
        seed_label_value="legacy",
        timestamp="202604250000",
    )
    frozen_source = _validate_frozen_pair_source(Path.cwd(), settings, run_id="strict_smoke", experiment_keys=["exp4"])
    assert settings.strict_protocol_lock is True
    assert protocol_payload is not None
    assert protocol_payload["run_id"] == "strict_smoke"
    assert frozen_source is not None
    assert frozen_source["experiments"]["exp4"]["pair_count"] == 1


def test_missing_protocol_lock_fails() -> None:
    payload = _valid_protocol_lock_payload("strict_output")
    payload.pop("protocol_lock")
    with pytest.raises(ValueError, match="protocol_lock is required"):
        BCRPSSettings.from_dict(payload)


def test_fixed_protocol_alias_passes_validation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "frozen_pair_manifest.json"
    _write_protocol_lock_pair_manifest(manifest_path)
    payload = _valid_protocol_lock_payload(tmp_path / "strict_output", manifest_path)
    payload["fixed_protocol"] = payload.pop("protocol_lock")
    settings = BCRPSSettings.from_dict(payload)
    assert settings.protocol_lock is not None
    assert settings.protocol_lock.run_id == "strict_smoke"


@pytest.mark.parametrize(
    ("missing_key", "message"),
    [
        ("protocol_lock.dataset_split", "dataset_split"),
        ("protocol_lock.known_unknown_split", "known_unknown_split"),
        ("protocol_lock.metric", "metric"),
        ("protocol_lock.threshold", "threshold"),
        ("protocol_lock.inference_protocol", "inference_protocol"),
        ("protocol_lock.frozen_similar_class_pair_source", "frozen_similar_class_pair_source"),
        ("protocol_lock.output_dir", "output_dir"),
    ],
)
def test_missing_required_protocol_lock_fields_fail(missing_key: str, message: str) -> None:
    payload = _remove_nested(_valid_protocol_lock_payload("strict_output"), missing_key)
    with pytest.raises(ValueError, match=message):
        BCRPSSettings.from_dict(payload)


@pytest.mark.parametrize(
    ("dotted_key", "message"),
    [
        ("protocol_lock.failure_record.enabled", "failure_record.enabled must be true"),
        ("protocol_lock.no_test_set_tuning", "no_test_set_tuning must be true"),
        ("protocol_lock.no_undeclared_extra_data", "no_undeclared_extra_data must be true"),
    ],
)
def test_protocol_lock_required_true_flags_fail_when_false(dotted_key: str, message: str) -> None:
    payload = json.loads(json.dumps(_valid_protocol_lock_payload("strict_output")))
    cursor = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = False
    with pytest.raises(ValueError, match=message):
        BCRPSSettings.from_dict(payload)


def test_non_baseline_protocol_lock_null_frozen_pair_source_fails_fast(tmp_path: Path) -> None:
    output_dir = tmp_path / "strict_output"
    with pytest.raises(ValueError, match="frozen_pair_manifest is required"):
        BCRPSSettings.from_dict(_valid_protocol_lock_payload(output_dir, frozen_pair_manifest=None))


def test_protocol_lock_unknown_field_fails() -> None:
    payload = _valid_protocol_lock_payload("strict_output")
    payload["protocol_lock"]["unsupported_field"] = True
    with pytest.raises(ValueError, match="Unknown protocol_lock keys"):
        BCRPSSettings.from_dict(payload)


def test_protocol_lock_run_id_mismatch_fails(tmp_path: Path) -> None:
    manifest_path = tmp_path / "frozen_pair_manifest.json"
    _write_protocol_lock_pair_manifest(manifest_path)
    output_dir = tmp_path / "strict_output"
    settings = BCRPSSettings.from_dict(_valid_protocol_lock_payload(output_dir, manifest_path))
    with pytest.raises(ValueError, match="run_id"):
        _validate_protocol_lock_runtime(
            Path.cwd(),
            settings,
            settings,
            run_id="different_run",
            output_dir=output_dir,
            selected_experiments=["exp4"],
            seed_label_value="legacy",
            timestamp="202604250000",
        )


def test_protocol_lock_output_dir_mismatch_fails(tmp_path: Path) -> None:
    manifest_path = tmp_path / "frozen_pair_manifest.json"
    _write_protocol_lock_pair_manifest(manifest_path)
    settings = BCRPSSettings.from_dict(_valid_protocol_lock_payload(tmp_path / "strict_output", manifest_path))
    with pytest.raises(ValueError, match="output_dir"):
        _validate_protocol_lock_runtime(
            Path.cwd(),
            settings,
            settings,
            run_id="strict_smoke",
            output_dir=tmp_path / "other_output",
            selected_experiments=["exp4"],
            seed_label_value="legacy",
            timestamp="202604250000",
        )


def test_non_baseline_diagnostics_require_frozen_pair_manifest() -> None:
    with pytest.raises(ValueError, match="frozen_pair_manifest is required"):
        BCRPSSettings.from_dict(
            {
                "enable_bcrps": True,
                "lambda": 0.001,
                "margin": 1.0,
                "use_normalized_distance": True,
                "use_warmup": True,
                "warmup_fraction": 0.2,
                "ablation_mode": "full_bcrps",
                "logging_diagnostics": True,
            }
        )


def test_frozen_pair_source_validation_requires_existing_manifest(tmp_path: Path) -> None:
    settings = BCRPSSettings.from_dict(
        {
            "enable_bcrps": True,
            "lambda": 0.001,
            "margin": 1.0,
            "use_normalized_distance": True,
            "use_warmup": True,
            "warmup_fraction": 0.2,
            "ablation_mode": "full_bcrps",
            "logging_diagnostics": True,
            "frozen_pair_manifest": str(tmp_path / "missing_manifest.json"),
        }
    )
    with pytest.raises(FileNotFoundError, match="Frozen pair manifest not found"):
        _validate_frozen_pair_source(tmp_path, settings, run_id="smoke", experiment_keys=["exp4"])


def test_valid_frozen_pair_source_logs_path_count_and_hash(tmp_path: Path) -> None:
    manifest_path = tmp_path / "frozen_pair_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": {
                    "exp4": {
                        "pairs": [
                            {"pair_rank": 1, "class_i": 0, "class_j": 1, "mean_normalized_distance": 0.5}
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    settings = BCRPSSettings.from_dict(
        {
            "enable_bcrps": True,
            "lambda": 0.001,
            "margin": 1.0,
            "use_normalized_distance": True,
            "use_warmup": True,
            "warmup_fraction": 0.2,
            "ablation_mode": "full_bcrps",
            "logging_diagnostics": True,
            "frozen_pair_manifest": str(manifest_path),
        }
    )
    source = _validate_frozen_pair_source(tmp_path, settings, run_id="smoke", experiment_keys=["exp4"])
    assert source is not None
    assert source["source_manifest"] == str(manifest_path)
    assert source["source_manifest_sha256"]
    assert source["experiments"]["exp4"]["pair_count"] == 1
    assert source["experiments"]["exp4"]["pair_hash_sha256"]


def test_diagnostic_files_are_created(tmp_path: Path) -> None:
    model = _model()
    outputs = write_reciprocal_point_diagnostics(
        model,
        tmp_path,
        run_id="smoke",
        experiment="exp4",
        stage="smoke",
        iteration=1,
    )
    expected = {
        "rp_distance_stats",
        "rp_nearest_neighbors",
        "rp_nearest_pairs",
        "rp_distance_histogram_normalized",
        "rp_distance_histogram_raw",
    }
    assert expected == set(outputs)
    for path in outputs.values():
        assert Path(path).exists()
        assert Path(path).read_text(encoding="utf-8").count("\n") >= 1


def test_failure_record_contains_required_fields(tmp_path: Path) -> None:
    settings = BCRPSSettings.from_dict({"ablation_mode": "baseline"})
    try:
        raise RuntimeError("simulated smoke failure")
    except RuntimeError as exc:
        record_path = _write_failure_run_record(
            tmp_path,
            workspace=Path.cwd(),
            run_id="smoke_failed",
            stage="smoke_stage",
            seed=20260417,
            config_path=None,
            settings=settings,
            exc=exc,
            reports=[],
        )
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    required = {
        "run_id",
        "status",
        "timestamp",
        "stage",
        "seed",
        "config",
        "git_commit",
        "output_dir",
        "error_type",
        "short_error_message",
        "traceback_tail",
    }
    assert required <= set(payload)
    assert payload["run_id"] == "smoke_failed"
    assert payload["status"] == "failed"
    assert payload["stage"] == "smoke_stage"
    assert payload["seed"] == 20260417
    assert payload["error_type"] == "RuntimeError"
    assert "simulated smoke failure" in payload["short_error_message"]
    assert payload["traceback_tail"]
    assert payload["abnormal_events"]
