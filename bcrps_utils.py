from __future__ import annotations

import csv
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


ALLOWED_ABLATION_MODES = {
    "baseline",
    "full_bcrps",
    "lambda_zero",
    "lambda_sweep",
    "margin_sweep",
    "raw_distance",
    "no_warmup",
    "mean_anchor_control",
}

FROZEN_PAIR_REQUIRED_MODES = ALLOWED_ABLATION_MODES - {"baseline"}

EXPECTED_PROTOCOL_DATASET_ROOT = (
    r"D:\keyan\research-units-pipeline-skills-main\workspaces"
    r"\open-set-laplacian-pyramid-repro\artifacts\datasets\pr_rpad_v1"
)
EXPECTED_PROTOCOL_DATASET_SPLIT = {
    "split_identity": "pr_rpad_v1_mixed_80_10_10",
    "train": r"exp*\mixed\train",
    "val": r"exp*\mixed\val",
    "test": r"exp*\mixed\test",
}
EXPECTED_PROTOCOL_KNOWN_UNKNOWN = {
    "exp4": {"known": [1, 3, 4], "unknown": [2]},
    "exp5": {"known": [1, 4, 5], "unknown": [2, 3]},
    "exp6": {"known": [1, 4, 5, 7], "unknown": [2, 3, 6]},
}
EXPECTED_PROTOCOL_PRIMARY_SCOPE = {"exp4", "exp5", "exp6"}
EXPECTED_PROTOCOL_GUARDRAILS = {"ACC_known", "ACC", "MR", "MP", "MIOU"}
EXPECTED_PROTOCOL_THRESHOLD = "unchanged_p003_open_set_threshold_sweep"
EXPECTED_PROTOCOL_INFERENCE = "unchanged_p003_forward_unknown_open_score_evaluate_open_set"
EXPECTED_PROTOCOL_TEST_SET = r"exp*\mixed\test"

PROTOCOL_LOCK_ALLOWED_KEYS = {
    "contract_source",
    "experiment_plan_source",
    "planned_run_manifest_source",
    "run_id",
    "output_dir",
    "dataset_split",
    "known_unknown_split",
    "test_set",
    "metric",
    "threshold",
    "inference_protocol",
    "frozen_similar_class_pair_source",
    "no_test_set_tuning",
    "no_undeclared_extra_data",
    "failure_record",
    "diagnostics",
}
DATASET_SPLIT_ALLOWED_KEYS = {"dataset_root", "split_identity", "train", "val", "test"}
METRIC_ALLOWED_KEYS = {"primary", "primary_scope", "guardrails"}
FAILURE_RECORD_ALLOWED_KEYS = {"enabled", "required_fields"}
DIAGNOSTICS_ALLOWED_KEYS = {
    "reciprocal_point_distance",
    "similar_class_confusion",
    "oscr",
    "acc_known",
    "primary_metric",
    "runtime_cost",
}
REQUIRED_FAILURE_RECORD_FIELDS = {
    "run_id",
    "status",
    "timestamp",
    "stage",
    "seed",
    "config_path",
    "config",
    "git_commit",
    "output_dir",
    "error_type",
    "short_error_message",
    "traceback_tail",
    "partial_metrics",
    "abnormal_events",
}


def _unknown_keys(payload: dict[str, Any], allowed: set[str], prefix: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {prefix} keys: {unknown}")


def _require_dict(payload: dict[str, Any], key: str, prefix: str) -> dict[str, Any]:
    if key not in payload:
        raise ValueError(f"{prefix}.{key} is required.")
    value = payload[key]
    if not isinstance(value, dict):
        raise ValueError(f"{prefix}.{key} must be an object.")
    return value


def _require_non_empty_str(payload: dict[str, Any], key: str, prefix: str) -> str:
    if key not in payload:
        raise ValueError(f"{prefix}.{key} is required.")
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix}.{key} must be a non-empty string.")
    return value


def _normalize_protocol_path(value: str) -> str:
    return str(Path(value)).replace("/", "\\").rstrip("\\").casefold()


def _normalize_protocol_fragment(value: str) -> str:
    return value.replace("/", "\\").rstrip("\\").casefold()


def _as_int_list(value: Any, prefix: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{prefix} must be a list.")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix} must contain integer class ids.") from exc


def _as_str_set(value: Any, prefix: str) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{prefix} must be a list of strings.")
    return set(value)


def _validate_dataset_split(payload: dict[str, Any]) -> None:
    _unknown_keys(payload, DATASET_SPLIT_ALLOWED_KEYS, "protocol_lock.dataset_split")
    dataset_root = _require_non_empty_str(payload, "dataset_root", "protocol_lock.dataset_split")
    if _normalize_protocol_path(dataset_root) != _normalize_protocol_path(EXPECTED_PROTOCOL_DATASET_ROOT):
        raise ValueError("protocol_lock.dataset_split.dataset_root must match the fixed P003 dataset root.")
    split_identity = _require_non_empty_str(payload, "split_identity", "protocol_lock.dataset_split")
    if split_identity != EXPECTED_PROTOCOL_DATASET_SPLIT["split_identity"]:
        raise ValueError("protocol_lock.dataset_split.split_identity must match the fixed P003 split identity.")
    for key in ("train", "val", "test"):
        value = _require_non_empty_str(payload, key, "protocol_lock.dataset_split")
        if _normalize_protocol_fragment(value) != _normalize_protocol_fragment(EXPECTED_PROTOCOL_DATASET_SPLIT[key]):
            raise ValueError(f"protocol_lock.dataset_split.{key} must match the fixed P003 {key} path.")


def _validate_known_unknown_split(payload: dict[str, Any]) -> None:
    expected_keys = set(EXPECTED_PROTOCOL_KNOWN_UNKNOWN)
    missing = sorted(expected_keys - set(payload))
    extra = sorted(set(payload) - expected_keys)
    if missing:
        raise ValueError(f"protocol_lock.known_unknown_split is missing: {missing}")
    if extra:
        raise ValueError(f"protocol_lock.known_unknown_split has unsupported experiments: {extra}")
    for experiment, expected in EXPECTED_PROTOCOL_KNOWN_UNKNOWN.items():
        value = payload[experiment]
        if not isinstance(value, dict):
            raise ValueError(f"protocol_lock.known_unknown_split.{experiment} must be an object.")
        _unknown_keys(value, {"known", "unknown"}, f"protocol_lock.known_unknown_split.{experiment}")
        known = _as_int_list(value.get("known"), f"protocol_lock.known_unknown_split.{experiment}.known")
        unknown = _as_int_list(value.get("unknown"), f"protocol_lock.known_unknown_split.{experiment}.unknown")
        if known != expected["known"] or unknown != expected["unknown"]:
            raise ValueError(f"protocol_lock.known_unknown_split.{experiment} must match the fixed P003 class split.")


def _validate_metric(payload: dict[str, Any]) -> None:
    _unknown_keys(payload, METRIC_ALLOWED_KEYS, "protocol_lock.metric")
    primary = _require_non_empty_str(payload, "primary", "protocol_lock.metric")
    if primary != "OSCR":
        raise ValueError("protocol_lock.metric.primary must be OSCR.")
    primary_scope = _as_str_set(payload.get("primary_scope"), "protocol_lock.metric.primary_scope")
    if primary_scope != EXPECTED_PROTOCOL_PRIMARY_SCOPE:
        raise ValueError("protocol_lock.metric.primary_scope must be exp4, exp5, and exp6.")
    guardrails = _as_str_set(payload.get("guardrails"), "protocol_lock.metric.guardrails")
    if guardrails != EXPECTED_PROTOCOL_GUARDRAILS:
        raise ValueError("protocol_lock.metric.guardrails must match the fixed P003 guardrail metrics.")


def _validate_failure_record(payload: dict[str, Any]) -> tuple[str, ...]:
    _unknown_keys(payload, FAILURE_RECORD_ALLOWED_KEYS, "protocol_lock.failure_record")
    if payload.get("enabled") is not True:
        raise ValueError("protocol_lock.failure_record.enabled must be true.")
    required_fields = payload.get("required_fields")
    if not isinstance(required_fields, list) or not all(isinstance(item, str) for item in required_fields):
        raise ValueError("protocol_lock.failure_record.required_fields must be a list of strings.")
    missing = sorted(REQUIRED_FAILURE_RECORD_FIELDS - set(required_fields))
    if missing:
        raise ValueError(f"protocol_lock.failure_record.required_fields is missing: {missing}")
    return tuple(required_fields)


def _validate_diagnostics(payload: dict[str, Any]) -> None:
    _unknown_keys(payload, DIAGNOSTICS_ALLOWED_KEYS, "protocol_lock.diagnostics")
    missing = sorted(DIAGNOSTICS_ALLOWED_KEYS - set(payload))
    if missing:
        raise ValueError(f"protocol_lock.diagnostics is missing: {missing}")
    disabled = sorted(key for key in DIAGNOSTICS_ALLOWED_KEYS if payload.get(key) is not True)
    if disabled:
        raise ValueError(f"protocol_lock.diagnostics entries must be true: {disabled}")


@dataclass(frozen=True)
class ProtocolLock:
    payload: dict[str, Any]
    run_id: str
    output_dir: str
    frozen_similar_class_pair_source: str | None
    failure_record_required_fields: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ProtocolLock":
        if not isinstance(payload, dict):
            raise ValueError("protocol_lock must be an object.")
        payload = dict(payload)
        _unknown_keys(payload, PROTOCOL_LOCK_ALLOWED_KEYS, "protocol_lock")
        for key in (
            "dataset_split",
            "known_unknown_split",
            "metric",
            "threshold",
            "inference_protocol",
            "frozen_similar_class_pair_source",
            "output_dir",
            "run_id",
            "failure_record",
            "diagnostics",
            "no_test_set_tuning",
            "no_undeclared_extra_data",
        ):
            if key not in payload:
                raise ValueError(f"protocol_lock.{key} is required.")

        for key in ("contract_source", "experiment_plan_source", "planned_run_manifest_source"):
            if key in payload and (not isinstance(payload[key], str) or not payload[key].strip()):
                raise ValueError(f"protocol_lock.{key} must be a non-empty string when provided.")

        run_id = _require_non_empty_str(payload, "run_id", "protocol_lock")
        output_dir = _require_non_empty_str(payload, "output_dir", "protocol_lock")
        _validate_dataset_split(_require_dict(payload, "dataset_split", "protocol_lock"))
        _validate_known_unknown_split(_require_dict(payload, "known_unknown_split", "protocol_lock"))
        _validate_metric(_require_dict(payload, "metric", "protocol_lock"))
        threshold = _require_non_empty_str(payload, "threshold", "protocol_lock")
        if threshold != EXPECTED_PROTOCOL_THRESHOLD:
            raise ValueError("protocol_lock.threshold must declare the unchanged P003 threshold protocol.")
        inference_protocol = _require_non_empty_str(payload, "inference_protocol", "protocol_lock")
        if inference_protocol != EXPECTED_PROTOCOL_INFERENCE:
            raise ValueError("protocol_lock.inference_protocol must declare the unchanged P003 inference protocol.")
        if "test_set" in payload:
            test_set = _require_non_empty_str(payload, "test_set", "protocol_lock")
            if _normalize_protocol_fragment(test_set) != _normalize_protocol_fragment(EXPECTED_PROTOCOL_TEST_SET):
                raise ValueError("protocol_lock.test_set must match the fixed P003 test path.")
        if payload.get("no_test_set_tuning") is not True:
            raise ValueError("protocol_lock.no_test_set_tuning must be true.")
        if payload.get("no_undeclared_extra_data") is not True:
            raise ValueError("protocol_lock.no_undeclared_extra_data must be true.")
        frozen_source = payload.get("frozen_similar_class_pair_source")
        if frozen_source is not None and (not isinstance(frozen_source, str) or not frozen_source.strip()):
            raise ValueError("protocol_lock.frozen_similar_class_pair_source must be a non-empty string or null.")
        required_failure_fields = _validate_failure_record(_require_dict(payload, "failure_record", "protocol_lock"))
        _validate_diagnostics(_require_dict(payload, "diagnostics", "protocol_lock"))
        return cls(
            payload=payload,
            run_id=run_id,
            output_dir=output_dir,
            frozen_similar_class_pair_source=frozen_source,
            failure_record_required_fields=required_failure_fields,
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.payload))

    def validate_failure_record_payload(self, payload: dict[str, Any]) -> None:
        missing = sorted(set(self.failure_record_required_fields) - set(payload))
        if missing:
            raise ValueError(f"failure record payload is missing required fields: {missing}")


@dataclass(frozen=True)
class BCRPSSettings:
    enable_bcrps: bool = False
    lambda_weight: float = 1e-3
    margin: float = 1.0
    use_normalized_distance: bool = True
    use_warmup: bool = True
    warmup_fraction: float = 0.2
    seed: int | None = None
    ablation_mode: str = "baseline"
    logging_diagnostics: bool = False
    diagnostic_output_dir: str | None = None
    run_id: str | None = None
    frozen_pair_manifest: str | None = None
    strict_protocol_lock: bool = False
    protocol_lock: ProtocolLock | None = None

    def requires_frozen_pair_manifest(self) -> bool:
        return self.ablation_mode in FROZEN_PAIR_REQUIRED_MODES and self.logging_diagnostics

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "BCRPSSettings":
        payload = dict(payload or {})
        if "lambda" in payload:
            payload["lambda_weight"] = payload.pop("lambda")
        if "fixed_protocol" in payload:
            if "protocol_lock" in payload:
                raise ValueError("Use only one of protocol_lock or fixed_protocol.")
            payload["protocol_lock"] = payload.pop("fixed_protocol")
        if "protocol_lock" in payload and payload["protocol_lock"] is not None:
            payload["protocol_lock"] = ProtocolLock.from_dict(payload["protocol_lock"])
            if (
                payload.get("frozen_pair_manifest") is None
                and payload["protocol_lock"].frozen_similar_class_pair_source is not None
            ):
                payload["frozen_pair_manifest"] = payload["protocol_lock"].frozen_similar_class_pair_source
        valid_fields = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - valid_fields)
        if unknown:
            raise ValueError(f"Unknown BCRPS config keys: {unknown}")
        settings = cls(**payload)
        settings.validate()
        return settings

    @classmethod
    def from_json(cls, path: Path | str | None) -> "BCRPSSettings":
        if path is None:
            return cls()
        config_path = Path(path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def with_overrides(self, **overrides: Any) -> "BCRPSSettings":
        payload = self.to_dict()
        for key, value in overrides.items():
            if value is not None:
                payload["lambda" if key == "lambda_weight" else key] = value
        return BCRPSSettings.from_dict(payload)

    def validate(self) -> None:
        if not isinstance(self.strict_protocol_lock, bool):
            raise ValueError("strict_protocol_lock must be a boolean.")
        if self.strict_protocol_lock and self.protocol_lock is None:
            raise ValueError("protocol_lock is required when strict_protocol_lock is true.")
        if self.protocol_lock is not None and not isinstance(self.protocol_lock, ProtocolLock):
            raise ValueError("protocol_lock must be parsed as a ProtocolLock.")
        if self.ablation_mode not in ALLOWED_ABLATION_MODES:
            raise ValueError(f"Unsupported ablation_mode: {self.ablation_mode}")
        if self.lambda_weight < 0:
            raise ValueError("BCRPS lambda must be non-negative.")
        if self.margin <= 0:
            raise ValueError("BCRPS margin must be positive.")
        if not 0.0 <= self.warmup_fraction <= 1.0:
            raise ValueError("BCRPS warmup_fraction must be in [0, 1].")
        if self.requires_frozen_pair_manifest() and not self.frozen_pair_manifest:
            raise ValueError(
                "frozen_pair_manifest is required for non-baseline BCRPS diagnostics; "
                "baseline validation-selected pairs must be supplied explicitly."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_bcrps": self.enable_bcrps,
            "lambda": self.lambda_weight,
            "margin": self.margin,
            "use_normalized_distance": self.use_normalized_distance,
            "use_warmup": self.use_warmup,
            "warmup_fraction": self.warmup_fraction,
            "seed": self.seed,
            "ablation_mode": self.ablation_mode,
            "logging_diagnostics": self.logging_diagnostics,
            "diagnostic_output_dir": self.diagnostic_output_dir,
            "run_id": self.run_id,
            "frozen_pair_manifest": self.frozen_pair_manifest,
            "strict_protocol_lock": self.strict_protocol_lock,
            "protocol_lock": self.protocol_lock.to_dict() if self.protocol_lock else None,
        }

    def validate_failure_record_payload(self, payload: dict[str, Any]) -> None:
        if self.strict_protocol_lock and self.protocol_lock is not None:
            self.protocol_lock.validate_failure_record_payload(payload)


@dataclass(frozen=True)
class BCRPSLossResult:
    loss: torch.Tensor
    unweighted_loss: torch.Tensor
    active: bool
    in_warmup: bool
    effective_lambda: float
    per_head_losses: tuple[torch.Tensor, ...]

    def log_dict(self) -> dict[str, float | bool]:
        return {
            "bcrps_loss": float(self.loss.detach().cpu().item()),
            "bcrps_unweighted_loss": float(self.unweighted_loss.detach().cpu().item()),
            "bcrps_active": self.active,
            "bcrps_in_warmup": self.in_warmup,
            "bcrps_effective_lambda": self.effective_lambda,
        }


def collect_reciprocal_point_tensors(model: Any) -> list[torch.Tensor]:
    if hasattr(model, "reciprocal_point_tensors"):
        tensors = list(model.reciprocal_point_tensors())
    else:
        tensors = [
            model.head1.reciprocal_points,
            model.head2.reciprocal_points,
            model.head3.reciprocal_points,
        ]
    if not tensors:
        raise RuntimeError("No reciprocal point tensors found.")
    return tensors


def compute_bcrps_loss(
    reciprocal_points: list[torch.Tensor],
    settings: BCRPSSettings,
    *,
    iteration: int,
    total_iterations: int,
) -> BCRPSLossResult:
    zero = reciprocal_points[0].sum() * 0.0
    per_head_losses: list[torch.Tensor] = []
    for points in reciprocal_points:
        if settings.ablation_mode == "mean_anchor_control":
            distances = _mean_anchor_distances(points)
        else:
            working = _l2_normalize(points) if settings.use_normalized_distance else points
            distances, _pairs = _pairwise_distances(working)
        if distances.numel() == 0:
            per_head_losses.append(zero)
        else:
            per_head_losses.append(F.relu(settings.margin - distances).pow(2).mean())
    unweighted_loss = torch.stack(per_head_losses).mean() if per_head_losses else zero

    warmup_steps = int(total_iterations * settings.warmup_fraction)
    in_warmup = bool(settings.enable_bcrps and settings.use_warmup and warmup_steps > 0 and iteration <= warmup_steps)
    active = bool(settings.enable_bcrps and not in_warmup)
    effective_lambda = float(settings.lambda_weight if active else 0.0)
    return BCRPSLossResult(
        loss=unweighted_loss * effective_lambda,
        unweighted_loss=unweighted_loss,
        active=active,
        in_warmup=in_warmup,
        effective_lambda=effective_lambda,
        per_head_losses=tuple(per_head_losses),
    )


def nearest_normalized_pairs(model: Any, top_k: int) -> list[dict[str, float | int]]:
    return _average_pair_rows(collect_reciprocal_point_tensors(model), top_k=top_k)


def write_reciprocal_point_diagnostics(
    model: Any,
    diagnostics_dir: Path | str,
    *,
    run_id: str,
    experiment: str,
    stage: str,
    iteration: int | None,
    histogram_bins: int = 10,
) -> dict[str, str]:
    diagnostics_path = Path(diagnostics_dir)
    diagnostics_path.mkdir(parents=True, exist_ok=True)
    points_by_head = collect_reciprocal_point_tensors(model)
    stats_rows = _distance_stats_rows(points_by_head, run_id, experiment, stage, iteration)
    nearest_rows = _nearest_neighbor_rows(points_by_head, run_id, experiment, stage, iteration)
    pair_rows = _nearest_pair_rows(points_by_head, run_id, experiment, stage, iteration)
    norm_hist_rows = _histogram_rows(
        points_by_head,
        run_id,
        experiment,
        stage,
        iteration,
        mode="normalized",
        bins=histogram_bins,
    )
    raw_hist_rows = _histogram_rows(
        points_by_head,
        run_id,
        experiment,
        stage,
        iteration,
        mode="raw",
        bins=histogram_bins,
    )

    outputs = {
        "rp_distance_stats": diagnostics_path / "rp_distance_stats.csv",
        "rp_nearest_neighbors": diagnostics_path / "rp_nearest_neighbors.csv",
        "rp_nearest_pairs": diagnostics_path / "rp_nearest_pairs.csv",
        "rp_distance_histogram_normalized": diagnostics_path / "rp_distance_histogram_normalized.csv",
        "rp_distance_histogram_raw": diagnostics_path / "rp_distance_histogram_raw.csv",
    }
    _append_csv(outputs["rp_distance_stats"], stats_rows, _STATS_FIELDS)
    _append_csv(outputs["rp_nearest_neighbors"], nearest_rows, _NEAREST_FIELDS)
    _append_csv(outputs["rp_nearest_pairs"], pair_rows, _PAIR_FIELDS)
    _append_csv(outputs["rp_distance_histogram_normalized"], norm_hist_rows, _HIST_FIELDS)
    _append_csv(outputs["rp_distance_histogram_raw"], raw_hist_rows, _HIST_FIELDS)
    return {key: str(value) for key, value in outputs.items()}


def _l2_normalize(points: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return points / points.norm(dim=1, keepdim=True).clamp_min(eps)


def _pairwise_distances(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_classes = points.shape[0]
    if num_classes < 2:
        empty_dist = points.new_empty((0,))
        empty_pairs = torch.empty((2, 0), dtype=torch.long, device=points.device)
        return empty_dist, empty_pairs
    pairs = torch.triu_indices(num_classes, num_classes, offset=1, device=points.device)
    distances = torch.linalg.vector_norm(points[pairs[0]] - points[pairs[1]], dim=1)
    return distances, pairs


def _mean_anchor_distances(points: torch.Tensor) -> torch.Tensor:
    normalized = _l2_normalize(points)
    anchor = _l2_normalize(normalized.mean(dim=0, keepdim=True))
    return torch.linalg.vector_norm(normalized - anchor, dim=1)


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _stat_dict(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"min": float("nan"), "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    values_cpu = values.detach().float().cpu()
    return {
        "min": float(values_cpu.min().item()),
        "mean": float(values_cpu.mean().item()),
        "median": float(values_cpu.median().item()),
        "std": float(values_cpu.std(unbiased=False).item()),
    }


def _distance_stats_rows(
    points_by_head: list[torch.Tensor],
    run_id: str,
    experiment: str,
    stage: str,
    iteration: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for head_idx, points in enumerate(points_by_head, start=1):
        raw_distances, _ = _pairwise_distances(points)
        normalized_distances, _ = _pairwise_distances(_l2_normalize(points))
        raw_nearest = _nearest_distances(points)
        normalized_nearest = _nearest_distances(_l2_normalize(points))
        raw_stats = _stat_dict(raw_distances)
        normalized_stats = _stat_dict(normalized_distances)
        raw_nearest_stats = _stat_dict(raw_nearest)
        normalized_nearest_stats = _stat_dict(normalized_nearest)
        rows.append(
            {
                "run_id": run_id,
                "experiment": experiment,
                "stage": stage,
                "iteration": iteration,
                "head": f"head{head_idx}",
                "raw_min": raw_stats["min"],
                "raw_mean": raw_stats["mean"],
                "raw_median": raw_stats["median"],
                "raw_std": raw_stats["std"],
                "normalized_min": normalized_stats["min"],
                "normalized_mean": normalized_stats["mean"],
                "normalized_median": normalized_stats["median"],
                "normalized_std": normalized_stats["std"],
                "raw_nearest_min": raw_nearest_stats["min"],
                "raw_nearest_mean": raw_nearest_stats["mean"],
                "raw_nearest_median": raw_nearest_stats["median"],
                "raw_nearest_std": raw_nearest_stats["std"],
                "normalized_nearest_min": normalized_nearest_stats["min"],
                "normalized_nearest_mean": normalized_nearest_stats["mean"],
                "normalized_nearest_median": normalized_nearest_stats["median"],
                "normalized_nearest_std": normalized_nearest_stats["std"],
            }
        )
    if rows:
        aggregate = {
            "run_id": run_id,
            "experiment": experiment,
            "stage": stage,
            "iteration": iteration,
            "head": "aggregate",
        }
        for key in _STAT_VALUE_FIELDS:
            aggregate[key] = _finite_mean([float(row[key]) for row in rows])
        rows.append(aggregate)
    return rows


def _nearest_distances(points: torch.Tensor) -> torch.Tensor:
    num_classes = points.shape[0]
    if num_classes < 2:
        return points.new_empty((0,))
    distance_matrix = torch.cdist(points, points)
    distance_matrix.fill_diagonal_(float("inf"))
    distances, _nearest = distance_matrix.min(dim=1)
    return distances


def _nearest_neighbor_rows(
    points_by_head: list[torch.Tensor],
    run_id: str,
    experiment: str,
    stage: str,
    iteration: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for head_idx, points in enumerate(points_by_head, start=1):
        for mode, working in (("raw", points), ("normalized", _l2_normalize(points))):
            num_classes = working.shape[0]
            if num_classes < 2:
                continue
            distance_matrix = torch.cdist(working, working)
            distance_matrix.fill_diagonal_(float("inf"))
            distances, nearest = distance_matrix.min(dim=1)
            for class_idx in range(num_classes):
                rows.append(
                    {
                        "run_id": run_id,
                        "experiment": experiment,
                        "stage": stage,
                        "iteration": iteration,
                        "head": f"head{head_idx}",
                        "mode": mode,
                        "class_index": class_idx,
                        "nearest_class": int(nearest[class_idx].detach().cpu().item()),
                        "nearest_distance": float(distances[class_idx].detach().cpu().item()),
                    }
                )
    return rows


def _average_pair_rows(points_by_head: list[torch.Tensor], top_k: int | None = None) -> list[dict[str, float | int]]:
    pair_values: dict[tuple[int, int], list[float]] = {}
    for points in points_by_head:
        distances, pairs = _pairwise_distances(_l2_normalize(points))
        for idx in range(distances.numel()):
            pair = (int(pairs[0, idx].item()), int(pairs[1, idx].item()))
            pair_values.setdefault(pair, []).append(float(distances[idx].detach().cpu().item()))
    rows = [
        {
            "class_i": pair[0],
            "class_j": pair[1],
            "mean_normalized_distance": _finite_mean(values),
        }
        for pair, values in pair_values.items()
    ]
    rows.sort(key=lambda row: float(row["mean_normalized_distance"]))
    if top_k is not None:
        rows = rows[:top_k]
    for rank, row in enumerate(rows, start=1):
        row["pair_rank"] = rank
    return rows


def _nearest_pair_rows(
    points_by_head: list[torch.Tensor],
    run_id: str,
    experiment: str,
    stage: str,
    iteration: int | None,
) -> list[dict[str, Any]]:
    rows = []
    for row in _average_pair_rows(points_by_head):
        rows.append(
            {
                "run_id": run_id,
                "experiment": experiment,
                "stage": stage,
                "iteration": iteration,
                **row,
            }
        )
    return rows


def _histogram_rows(
    points_by_head: list[torch.Tensor],
    run_id: str,
    experiment: str,
    stage: str,
    iteration: int | None,
    *,
    mode: str,
    bins: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for head_idx, points in enumerate(points_by_head, start=1):
        working = _l2_normalize(points) if mode == "normalized" else points
        distances, _ = _pairwise_distances(working)
        if distances.numel() == 0:
            continue
        values = distances.detach().float().cpu()
        min_value = 0.0
        max_value = 2.0 if mode == "normalized" else max(float(values.max().item()), 1e-12)
        counts = torch.histc(values, bins=bins, min=min_value, max=max_value)
        width = (max_value - min_value) / bins
        for bin_idx, count in enumerate(counts.tolist()):
            rows.append(
                {
                    "run_id": run_id,
                    "experiment": experiment,
                    "stage": stage,
                    "iteration": iteration,
                    "head": f"head{head_idx}",
                    "mode": mode,
                    "bin_index": bin_idx,
                    "bin_start": min_value + width * bin_idx,
                    "bin_end": min_value + width * (bin_idx + 1),
                    "count": int(count),
                }
            )
    return rows


def _append_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


_STAT_VALUE_FIELDS = [
    "raw_min",
    "raw_mean",
    "raw_median",
    "raw_std",
    "normalized_min",
    "normalized_mean",
    "normalized_median",
    "normalized_std",
    "raw_nearest_min",
    "raw_nearest_mean",
    "raw_nearest_median",
    "raw_nearest_std",
    "normalized_nearest_min",
    "normalized_nearest_mean",
    "normalized_nearest_median",
    "normalized_nearest_std",
]

_STATS_FIELDS = ["run_id", "experiment", "stage", "iteration", "head", *_STAT_VALUE_FIELDS]
_NEAREST_FIELDS = [
    "run_id",
    "experiment",
    "stage",
    "iteration",
    "head",
    "mode",
    "class_index",
    "nearest_class",
    "nearest_distance",
]
_PAIR_FIELDS = [
    "run_id",
    "experiment",
    "stage",
    "iteration",
    "pair_rank",
    "class_i",
    "class_j",
    "mean_normalized_distance",
]
_HIST_FIELDS = [
    "run_id",
    "experiment",
    "stage",
    "iteration",
    "head",
    "mode",
    "bin_index",
    "bin_start",
    "bin_end",
    "count",
]
