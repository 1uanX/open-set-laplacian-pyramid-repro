from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from copy import deepcopy

from torch import nn
from torch.utils.data import DataLoader, Dataset

from repro_dataset import EXPERIMENTS, PAPER_TARGETS, PulseGraphConverter, ReproConfig, build_all_datasets, load_sequence
from repro_model import ModelConfig, PaperPRRPADModel, PRRPADModel
from bcrps_utils import (
    BCRPSSettings,
    collect_reciprocal_point_tensors,
    compute_bcrps_loss,
    nearest_normalized_pairs,
    write_reciprocal_point_diagnostics,
)
from round4_lora_utils import (
    Round4Settings,
    apply_round4_scaffolding,
    merge_lora_adapters,
    measure_inference_latency_ms,
    parameter_report,
)

EXPERIMENT_SEEDS = {
    "exp1": 20260417,
    "exp2": 20260414,
    "exp3": 20260418,
    "exp4": 20260417,
    "exp5": 20260418,
    "exp6": 20260414,
}
OPEN_SET_EXPERIMENT_KEYS = {f"exp{experiment.experiment_id}" for experiment in EXPERIMENTS if experiment.open_set}
MODEL_ARCHITECTURES = {"mlp", "paper_prrpad"}


class PulseIndexDataset(Dataset):
    def __init__(self, dataset_root: Path, index_df: pd.DataFrame, converter: PulseGraphConverter) -> None:
        self.dataset_root = dataset_root
        self.index_df = index_df.reset_index(drop=True)
        self.converter = converter
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index_df)

    def _resolve_sequence_path(self, rel_path: str) -> Path:
        absolute = Path(rel_path)
        if absolute.is_absolute() and absolute.exists():
            return absolute
        direct = self.dataset_root / rel_path
        if direct.exists():
            return direct
        mixed = self.dataset_root / "mixed" / rel_path
        if mixed.exists():
            return mixed
        specific = self.dataset_root / "specific" / rel_path
        if specific.exists():
            return specific
        raise FileNotFoundError(f"Unable to resolve sequence path '{rel_path}' from '{self.dataset_root}'.")

    def _load_sequence(self, rel_path: str) -> np.ndarray:
        if rel_path not in self._cache:
            sequence = load_sequence(self._resolve_sequence_path(rel_path))
            self._cache[rel_path] = self.converter.vectorize_all(sequence)
        return self._cache[rel_path]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.index_df.iloc[index]
        sequence_vectors = self._load_sequence(str(row["sequence_path"]))
        vector = sequence_vectors[int(row["pulse_index"])]
        return (
            torch.from_numpy(vector).float(),
            torch.tensor(int(row["mapped_label"]), dtype=torch.long),
        )


def _filtered_index(dataset_root: Path, split: str, *, unknown_only: bool | None = None) -> pd.DataFrame:
    index_path = _index_path(dataset_root, split)
    frame = pd.read_csv(index_path)
    if unknown_only is True:
        return frame.loc[frame["is_unknown"] == 1].copy()
    if unknown_only is False:
        return frame.loc[frame["mapped_label"] >= 0].copy()
    return frame


def _index_path(dataset_root: Path, split: str) -> Path:
    candidates = [
        dataset_root / split / "pulse_index.csv",
        dataset_root / "mixed" / split / "pulse_index.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to find pulse_index.csv for split '{split}' under '{dataset_root}'.")


def build_loaders(dataset_root: Path, batch_size: int, converter: PulseGraphConverter) -> dict[str, DataLoader]:
    train_df = _filtered_index(dataset_root, "train", unknown_only=False)
    val_df = _filtered_index(dataset_root, "val", unknown_only=False)
    test_df = _filtered_index(dataset_root, "test", unknown_only=False)
    if (dataset_root / "out" / "test" / "pulse_index.csv").exists():
        out_df = _filtered_index(dataset_root, "out/test", unknown_only=True)
    else:
        out_df = pd.concat(
            [
                _filtered_index(dataset_root, "train", unknown_only=True),
                _filtered_index(dataset_root, "val", unknown_only=True),
                _filtered_index(dataset_root, "test", unknown_only=True),
            ],
            ignore_index=True,
        )
    loaders = {
        "train": DataLoader(PulseIndexDataset(dataset_root, train_df, converter), batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(PulseIndexDataset(dataset_root, val_df, converter), batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(PulseIndexDataset(dataset_root, test_df, converter), batch_size=batch_size, shuffle=False, num_workers=0),
    }
    if len(out_df):
        loaders["out"] = DataLoader(PulseIndexDataset(dataset_root, out_df, converter), batch_size=batch_size, shuffle=False, num_workers=0)
    return loaders


def build_converter(dataset_root: Path, *, input_mode: str = "vector") -> PulseGraphConverter:
    config_path = dataset_root / "config.json"
    if config_path.exists():
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        feature_ranges = config_payload["experiment"].get("global_ranges", {})
    else:
        feature_ranges = _round5_feature_ranges(dataset_root)
    return PulseGraphConverter(lenwindow=1, output_size=65, feature_ranges=feature_ranges, input_mode=input_mode)


def build_model(*, model_arch: str, num_classes: int, input_dim: int) -> PRRPADModel | PaperPRRPADModel:
    if model_arch not in MODEL_ARCHITECTURES:
        raise ValueError(f"Unsupported model architecture: {model_arch}")
    if model_arch == "paper_prrpad":
        config = ModelConfig(
            num_classes=num_classes,
            input_dim=input_dim,
            alpha1=1.0 / 3.0,
            alpha2=1.0 / 3.0,
            alpha3=1.0 / 3.0,
        )
        return PaperPRRPADModel(config)
    return PRRPADModel(ModelConfig(num_classes=num_classes, input_dim=input_dim))


def _round5_feature_ranges(dataset_root: Path) -> dict[str, tuple[float, float]]:
    experiment_key = dataset_root.name
    for parent in [dataset_root, *dataset_root.parents]:
        policy_path = parent / "normalization_policy_round5.json"
        if not policy_path.exists():
            continue
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
        stats = payload.get("experiments", {}).get(experiment_key)
        if not stats:
            continue
        return {column: (float(value["min"]), float(value["max"])) for column, value in stats.items()}
    return {}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _closed_set_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, float]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred, strict=True):
        confusion[int(true), int(pred)] += 1
    true_positive = np.diag(confusion).astype(float)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    iou = true_positive / np.maximum(confusion.sum(axis=1) + confusion.sum(axis=0) - true_positive, 1)
    return {
        "accuracy": float((y_true == y_pred).mean() * 100.0),
        "mean_recall": float(recall.mean() * 100.0),
        "mean_precision": float(precision.mean() * 100.0),
        "mean_iou": float(iou.mean() * 100.0),
    }


def _open_set_oscr(known_scores: np.ndarray, known_correct: np.ndarray, unknown_scores: np.ndarray) -> float:
    thresholds = np.unique(np.concatenate([known_scores, unknown_scores]))
    points = []
    for threshold in np.concatenate(([thresholds.max() + 1.0], thresholds[::-1], [thresholds.min() - 1.0])):
        ccr = float(np.mean((known_scores >= threshold) & known_correct))
        fpr = float(np.mean(unknown_scores >= threshold))
        points.append((fpr, ccr))
    points = sorted(points, key=lambda item: item[0])
    x = np.asarray([point[0] for point in points], dtype=float)
    y = np.asarray([point[1] for point in points], dtype=float)
    area = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
    return float(area * 100.0)


def evaluate_closed_set(model: PRRPADModel, loader: DataLoader, device: torch.device, num_classes: int) -> dict[str, float]:
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            probs = model.forward_unknown(features)["probs"]
            preds = probs.argmax(dim=1)
            all_true.append(labels.numpy())
            all_pred.append(preds.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return _closed_set_metrics(y_true, y_pred, num_classes)


def evaluate_open_set(model: PRRPADModel, known_loader: DataLoader, out_loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    known_scores: list[np.ndarray] = []
    known_correct: list[np.ndarray] = []
    acc_true: list[np.ndarray] = []
    acc_pred: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in known_loader:
            features = features.to(device)
            output = model.forward_unknown(features)
            probs = output["probs"]
            preds = probs.argmax(dim=1)
            scores = output["open_score"]
            known_scores.append(scores.cpu().numpy())
            known_correct.append((preds.cpu() == labels).numpy().astype(np.float32))
            acc_true.append(labels.numpy())
            acc_pred.append(preds.cpu().numpy())
    unknown_scores: list[np.ndarray] = []
    with torch.no_grad():
        for features, _labels in out_loader:
            features = features.to(device)
            scores = model.forward_unknown(features)["open_score"]
            unknown_scores.append(scores.cpu().numpy())
    y_true = np.concatenate(acc_true)
    y_pred = np.concatenate(acc_pred)
    known_scores_np = np.concatenate(known_scores)
    known_correct_np = np.concatenate(known_correct).astype(bool)
    unknown_scores_np = np.concatenate(unknown_scores) if unknown_scores else np.zeros(1, dtype=float)
    return {
        "ACC_known": float((y_true == y_pred).mean() * 100.0),
        "OSCR": _open_set_oscr(known_scores_np, known_correct_np, unknown_scores_np),
    }

def _next_batch(iterator: iter, loader: DataLoader) -> tuple[iter, tuple[torch.Tensor, torch.Tensor]]:
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return iterator, batch


def _trainable_parameter_count(model: PRRPADModel) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _require_trainable_parameters(parameters: list[nn.Parameter], label: str) -> list[nn.Parameter]:
    if not parameters:
        raise ValueError(f"No trainable parameters available for {label}.")
    return parameters


def _git_commit(workspace: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _peak_gpu_memory(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _timestamp_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _traceback_tail(exc: BaseException, max_lines: int = 12) -> str:
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return "".join(lines[-max_lines:]).strip()


def _write_failure_run_record(
    output_dir: Path,
    *,
    workspace: Path,
    run_id: str,
    stage: str,
    seed: int | None,
    config_path: Path | None,
    settings: BCRPSSettings | None,
    exc: BaseException,
    reports: list[dict[str, Any]] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    error_message = str(exc).replace("\n", " ")[:500]
    payload: dict[str, Any] = {
        "run_id": run_id,
        "status": "failed",
        "completion_status": "failed",
        "timestamp": _timestamp_utc(),
        "stage": stage,
        "seed": seed,
        "config_path": str(config_path.resolve()) if config_path else None,
        "config": settings.to_dict() if settings else None,
        "git_commit": _git_commit(workspace),
        "output_dir": str(output_dir),
        "log_path": str(output_dir),
        "error_type": type(exc).__name__,
        "short_error_message": error_message,
        "traceback_tail": _traceback_tail(exc),
        "partial_metrics": reports or [],
        "abnormal_events": [
            {
                "stage": stage,
                "event": "run_failed",
                "error_type": type(exc).__name__,
                "message": error_message,
            }
        ],
    }
    if settings is not None and settings.protocol_lock is not None:
        payload["protocol_lock"] = settings.protocol_lock.to_dict()
        settings.validate_failure_record_payload(payload)
    path = output_dir / "run_manifest.json"
    _write_json(path, payload)
    return path


def _bcrps_recent_summary(records: list[dict[str, float | bool]], window: int) -> dict[str, float | bool]:
    if not records:
        return {}
    recent = records[-window:]
    return {
        "bcrps_loss": float(np.mean([float(item["bcrps_loss"]) for item in recent])),
        "bcrps_unweighted_loss": float(np.mean([float(item["bcrps_unweighted_loss"]) for item in recent])),
        "bcrps_effective_lambda": float(recent[-1]["bcrps_effective_lambda"]),
        "bcrps_active": bool(recent[-1]["bcrps_active"]),
        "bcrps_active_fraction": float(np.mean([1.0 if item["bcrps_active"] else 0.0 for item in recent])),
        "bcrps_in_warmup": bool(recent[-1]["bcrps_in_warmup"]),
    }


def _resolve_optional_path(workspace: Path, raw_path: str | None, run_id: str) -> Path | None:
    if raw_path is None:
        return None
    expanded = raw_path.replace("{run_id}", run_id)
    path = Path(expanded)
    if not path.is_absolute():
        path = workspace / path
    return path


def _resolve_diagnostics_dir(workspace: Path, output_dir: Path, settings: BCRPSSettings, run_id: str) -> Path:
    configured = _resolve_optional_path(workspace, settings.diagnostic_output_dir, run_id)
    return configured if configured is not None else output_dir / "diagnostics"


def _expand_runtime_value(raw_value: str, *, run_id: str, seed_label_value: str, timestamp: str) -> str:
    return (
        raw_value.replace("{run_id}", run_id)
        .replace("{seed}", seed_label_value)
        .replace("{timestamp}", timestamp)
    )


def _resolve_runtime_path(
    workspace: Path,
    raw_path: str,
    *,
    run_id: str,
    seed_label_value: str,
    timestamp: str,
) -> Path:
    expanded = _expand_runtime_value(raw_path, run_id=run_id, seed_label_value=seed_label_value, timestamp=timestamp)
    path = Path(expanded)
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


def _same_resolved_path(left: Path, right: Path) -> bool:
    return str(left.resolve()).casefold() == str(right.resolve()).casefold()


def _resolve_round4_optional_path(
    workspace: Path,
    raw_path: str | None,
    *,
    run_id: str,
    seed_label_value: str,
    timestamp: str,
) -> Path | None:
    if raw_path is None:
        return None
    return _resolve_runtime_path(
        workspace,
        raw_path,
        run_id=run_id,
        seed_label_value=seed_label_value,
        timestamp=timestamp,
    )


def _load_round4_checkpoint_if_configured(
    model: PRRPADModel,
    workspace: Path,
    settings: Round4Settings,
    *,
    run_id: str,
    seed_label_value: str,
    timestamp: str,
    device: torch.device,
) -> str | None:
    checkpoint_path = _resolve_round4_optional_path(
        workspace,
        settings.checkpoint_path,
        run_id=run_id,
        seed_label_value=seed_label_value,
        timestamp=timestamp,
    )
    if checkpoint_path is None:
        return None
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Round4 checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise ValueError("Round4 checkpoint must contain a model state dict.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Round4 checkpoint has unexpected model keys: {sorted(unexpected)}")
    if missing:
        raise ValueError(f"Round4 checkpoint is missing model keys: {sorted(missing)}")
    return str(checkpoint_path)


def _validate_protocol_lock_runtime(
    workspace: Path,
    settings: BCRPSSettings,
    base_settings: BCRPSSettings,
    *,
    run_id: str,
    output_dir: Path,
    selected_experiments: list[str],
    seed_label_value: str,
    timestamp: str,
) -> dict[str, Any] | None:
    if not settings.strict_protocol_lock:
        return None
    protocol_lock = settings.protocol_lock
    if protocol_lock is None:
        raise ValueError("protocol_lock is required when strict_protocol_lock is true.")

    protocol_run_id = _expand_runtime_value(
        protocol_lock.run_id,
        run_id=run_id,
        seed_label_value=seed_label_value,
        timestamp=timestamp,
    )
    if protocol_run_id != run_id:
        raise ValueError("protocol_lock.run_id must match the resolved run_id.")
    if base_settings.run_id:
        config_run_id = _expand_runtime_value(
            base_settings.run_id,
            run_id=run_id,
            seed_label_value=seed_label_value,
            timestamp=timestamp,
        )
        if config_run_id != run_id:
            raise ValueError("config run_id must match protocol_lock.run_id in strict mode.")

    protocol_output_dir = _resolve_runtime_path(
        workspace,
        protocol_lock.output_dir,
        run_id=run_id,
        seed_label_value=seed_label_value,
        timestamp=timestamp,
    )
    if not _same_resolved_path(output_dir, protocol_output_dir):
        raise ValueError("protocol_lock.output_dir must match the resolved output_dir.")

    if not settings.logging_diagnostics:
        raise ValueError("logging_diagnostics must be true when strict_protocol_lock is true.")

    source = protocol_lock.frozen_similar_class_pair_source
    if settings.ablation_mode != "baseline":
        if source is None:
            raise ValueError("protocol_lock.frozen_similar_class_pair_source is required for non-baseline strict runs.")
        protocol_source = _resolve_runtime_path(
            workspace,
            source,
            run_id=run_id,
            seed_label_value=seed_label_value,
            timestamp=timestamp,
        )
        if not protocol_source.exists():
            raise FileNotFoundError(f"Frozen pair manifest not found: {protocol_source}")
        if settings.frozen_pair_manifest is None:
            raise ValueError("frozen_pair_manifest must map to protocol_lock.frozen_similar_class_pair_source.")
        settings_source = _resolve_runtime_path(
            workspace,
            settings.frozen_pair_manifest,
            run_id=run_id,
            seed_label_value=seed_label_value,
            timestamp=timestamp,
        )
        if not _same_resolved_path(settings_source, protocol_source):
            raise ValueError("frozen_pair_manifest must match protocol_lock.frozen_similar_class_pair_source.")
        if base_settings.frozen_pair_manifest is not None:
            base_source = _resolve_runtime_path(
                workspace,
                base_settings.frozen_pair_manifest,
                run_id=run_id,
                seed_label_value=seed_label_value,
                timestamp=timestamp,
            )
            if not _same_resolved_path(base_source, protocol_source):
                raise ValueError("config frozen_pair_manifest must match protocol_lock.frozen_similar_class_pair_source.")

    selected_open_set = sorted(item for item in selected_experiments if item in OPEN_SET_EXPERIMENT_KEYS)
    payload = protocol_lock.to_dict()
    payload["validated_runtime"] = {
        "run_id": run_id,
        "output_dir": str(output_dir.resolve()),
        "selected_open_set_experiments": selected_open_set,
    }
    return payload


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([_csv_safe_row(row) for row in rows])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _csv_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in row.items():
        converted = _json_safe(value)
        safe[key] = "" if converted is None else converted
    return safe


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable_payload_hash(payload: Any) -> str:
    raw = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(raw)


def _pair_class_id(item: dict[str, Any], primary_key: str, fallback_key: str, experiment_key: str) -> tuple[int, str | None]:
    if primary_key in item:
        return int(item[primary_key]), None
    if fallback_key not in item:
        raise ValueError(f"Frozen pair is missing {primary_key} and {fallback_key} for {experiment_key}: {item}")
    value = item[fallback_key]
    if isinstance(value, str):
        prefix = f"{experiment_key}:"
        if value.startswith(prefix):
            return int(value[len(prefix) :]), f"{primary_key}_from_{fallback_key}"
        if ":" in value:
            raise ValueError(f"Frozen pair {fallback_key} must match {experiment_key}:class_id, got {value!r}")
    return int(value), f"{primary_key}_from_{fallback_key}"


def _normalize_frozen_pairs(payload: dict[str, Any], experiment_key: str) -> list[dict[str, Any]]:
    if "experiments" in payload:
        experiment_payload = payload["experiments"].get(experiment_key, {})
        pair_rows = experiment_payload.get("pairs", [])
    else:
        pair_rows = payload.get("pairs", [])
    pairs = []
    for rank, item in enumerate(pair_rows, start=1):
        notes = []
        class_i, class_i_note = _pair_class_id(item, "class_i", "class_a", experiment_key)
        class_j, class_j_note = _pair_class_id(item, "class_j", "class_b", experiment_key)
        if class_i_note:
            notes.append(class_i_note)
        if class_j_note:
            notes.append(class_j_note)
        if "pair_rank" in item:
            pair_rank = int(item["pair_rank"])
        else:
            pair_rank = rank
            notes.append("pair_rank_generated_from_manifest_order")
        if "mean_normalized_distance" in item and item["mean_normalized_distance"] not in (None, ""):
            mean_normalized_distance = float(item["mean_normalized_distance"])
        else:
            mean_normalized_distance = None
            notes.append("mean_normalized_distance_missing")
        pair = {
            "pair_rank": pair_rank,
            "class_i": class_i,
            "class_j": class_j,
            "mean_normalized_distance": mean_normalized_distance,
        }
        if notes:
            pair["notes"] = "; ".join(notes)
        pairs.append(pair)
    return pairs


def _validate_frozen_pair_source(
    workspace: Path,
    settings: BCRPSSettings,
    *,
    run_id: str,
    experiment_keys: list[str],
) -> dict[str, Any] | None:
    if not settings.requires_frozen_pair_manifest():
        return None
    manifest_path = _resolve_optional_path(workspace, settings.frozen_pair_manifest, run_id)
    if manifest_path is None:
        raise ValueError("Frozen pair manifest is required for this BCRPS run.")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Frozen pair manifest not found: {manifest_path}")
    raw = manifest_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    experiments: dict[str, Any] = {}
    for experiment_key in [key for key in experiment_keys if key in OPEN_SET_EXPERIMENT_KEYS]:
        pairs = _normalize_frozen_pairs(payload, experiment_key)
        if not pairs:
            raise ValueError(f"Frozen pair manifest has no pairs for {experiment_key}: {manifest_path}")
        experiments[experiment_key] = {
            "pair_count": len(pairs),
            "pair_hash_sha256": _stable_payload_hash(pairs),
        }
    return {
        "pair_source": "loaded_frozen_pair_manifest",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_bytes(raw),
        "experiments": experiments,
    }


def _frozen_pairs_for_run(
    model: PRRPADModel,
    workspace: Path,
    settings: BCRPSSettings,
    *,
    run_id: str,
    experiment_key: str,
    num_classes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    top_k = min(3, num_classes * (num_classes - 1) // 2)
    manifest_path = _resolve_optional_path(workspace, settings.frozen_pair_manifest, run_id)
    if manifest_path is not None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Frozen pair manifest not found: {manifest_path}")
        raw = manifest_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        pairs = _normalize_frozen_pairs(payload, experiment_key)
        if not pairs:
            raise ValueError(f"Frozen pair manifest has no pairs for {experiment_key}: {manifest_path}")
        source = {
            "pair_source": "loaded_frozen_pair_manifest",
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256_bytes(raw),
            "pair_count": len(pairs),
            "pair_hash_sha256": _stable_payload_hash(pairs),
        }
    else:
        if settings.requires_frozen_pair_manifest():
            raise ValueError("Frozen pair manifest is required; refusing to recompute non-baseline diagnostic pairs.")
        pairs = nearest_normalized_pairs(model, top_k=top_k)
        source = {
            "pair_source": "baseline_validation_selected_checkpoint",
            "source_manifest": None,
            "pair_count": len(pairs),
            "pair_hash_sha256": _stable_payload_hash(pairs),
        }
    return pairs, source


def _known_class_id_to_matrix_index(
    settings: BCRPSSettings,
    experiment_key: str,
    num_classes: int,
) -> tuple[dict[int, int], set[int]]:
    if settings.protocol_lock is None:
        local_ids = set(range(num_classes))
        return {class_id: class_id for class_id in local_ids}, local_ids
    split = settings.protocol_lock.payload.get("known_unknown_split", {}).get(experiment_key)
    if not isinstance(split, dict):
        raise ValueError(f"protocol_lock.known_unknown_split is missing {experiment_key}.")
    known_ids = [int(item) for item in split.get("known", [])]
    unknown_ids = {int(item) for item in split.get("unknown", [])}
    if len(known_ids) != num_classes:
        raise ValueError(
            f"{experiment_key} known class count {len(known_ids)} does not match confusion matrix size {num_classes}."
        )
    return {class_id: index for index, class_id in enumerate(known_ids)}, set(known_ids) | unknown_ids


def _pair_matrix_index(
    pair: dict[str, Any],
    key: str,
    class_to_index: dict[int, int],
    valid_class_ids: set[int],
    *,
    experiment_key: str,
    num_classes: int,
) -> tuple[int | None, str | None]:
    class_id = int(pair[key])
    if class_id in class_to_index:
        matrix_index = int(class_to_index[class_id])
        if not 0 <= matrix_index < num_classes:
            raise ValueError(
                f"Frozen pair {key}={class_id} maps to matrix index {matrix_index}, "
                f"outside [0, {num_classes - 1}] for {experiment_key}."
            )
        return matrix_index, None
    if class_id in valid_class_ids:
        return None, f"{key}_not_in_known_confusion_matrix"
    raise ValueError(
        f"Frozen pair {key}={class_id} is not in the locked class ids for {experiment_key}; "
        f"known matrix classes are {sorted(class_to_index)}, valid locked class ids are {sorted(valid_class_ids)}."
    )


def _pair_notes(*parts: str | None) -> str:
    return "; ".join(part for part in parts if part)


def _predict_known(model: PRRPADModel, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            probs = model.forward_unknown(features)["probs"]
            preds = probs.argmax(dim=1)
            y_true.append(labels.numpy())
            y_pred.append(preds.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def _write_similar_class_confusion(
    model: PRRPADModel,
    loaders: dict[str, DataLoader],
    device: torch.device,
    diagnostics_dir: Path,
    workspace: Path,
    settings: BCRPSSettings,
    *,
    run_id: str,
    experiment_key: str,
    num_classes: int,
) -> dict[str, str]:
    pairs, source = _frozen_pairs_for_run(
        model,
        workspace,
        settings,
        run_id=run_id,
        experiment_key=experiment_key,
        num_classes=num_classes,
    )
    manifest = {
        "run_id": run_id,
        "experiment": experiment_key,
        "top_k": len(pairs),
        **source,
        "pairs": pairs,
    }
    manifest_path = diagnostics_dir / "frozen_pair_manifest.json"
    _write_json(manifest_path, manifest)

    rows: list[dict[str, Any]] = []
    class_to_index, valid_class_ids = _known_class_id_to_matrix_index(settings, experiment_key, num_classes)
    confusion_payload: dict[str, Any] = {
        "run_id": run_id,
        "experiment": experiment_key,
        "class_id_to_matrix_index": class_to_index,
        "valid_locked_class_ids": sorted(valid_class_ids),
        "splits": {},
    }
    for split_name in ("val", "test"):
        y_true, y_pred = _predict_known(model, loaders[split_name], device)
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for true, pred in zip(y_true, y_pred, strict=True):
            confusion[int(true), int(pred)] += 1
        class_counts = confusion.sum(axis=1)
        split_rates = []
        split_pair_payload = []
        for pair in pairs:
            class_i_id = int(pair["class_i"])
            class_j_id = int(pair["class_j"])
            class_i_index, class_i_note = _pair_matrix_index(
                pair,
                "class_i",
                class_to_index,
                valid_class_ids,
                experiment_key=experiment_key,
                num_classes=num_classes,
            )
            class_j_index, class_j_note = _pair_matrix_index(
                pair,
                "class_j",
                class_to_index,
                valid_class_ids,
                experiment_key=experiment_key,
                num_classes=num_classes,
            )
            notes = _pair_notes(pair.get("notes"), class_i_note, class_j_note)
            if class_i_index is None or class_j_index is None:
                count_true_i = int(class_counts[class_i_index]) if class_i_index is not None else None
                count_true_j = int(class_counts[class_j_index]) if class_j_index is not None else None
                cross_count = None
                denominator = None
                rate = None
                notes = _pair_notes(notes, "pair_not_fully_represented_in_known_confusion_matrix")
            else:
                cross_count = int(confusion[class_i_index, class_j_index] + confusion[class_j_index, class_i_index])
                count_true_i = int(class_counts[class_i_index])
                count_true_j = int(class_counts[class_j_index])
                denominator = int(count_true_i + count_true_j)
                rate = float(cross_count / denominator) if denominator else float("nan")
                split_rates.append(rate)
            pair_row = {
                "run_id": run_id,
                "experiment": experiment_key,
                "split": split_name,
                "pair_rank": int(pair["pair_rank"]),
                "class_i": class_i_id,
                "class_j": class_j_id,
                "class_i_matrix_index": class_i_index,
                "class_j_matrix_index": class_j_index,
                "mean_normalized_distance": pair.get("mean_normalized_distance"),
                "count_true_i": count_true_i,
                "count_true_j": count_true_j,
                "cross_confusion_count": cross_count,
                "denominator": denominator,
                "pair_confusion_rate": rate,
                "notes": notes,
            }
            split_pair_payload.append(pair_row)
            rows.append(
                pair_row
            )
        finite_rates = [rate for rate in split_rates if np.isfinite(rate)]
        rows.append(
            {
                "run_id": run_id,
                "experiment": experiment_key,
                "split": split_name,
                "pair_rank": "aggregate",
                "class_i": "",
                "class_j": "",
                "class_i_matrix_index": "",
                "class_j_matrix_index": "",
                "mean_normalized_distance": "",
                "count_true_i": "",
                "count_true_j": "",
                "cross_confusion_count": "",
                "denominator": "",
                "pair_confusion_rate": float(np.mean(finite_rates)) if finite_rates else float("nan"),
                "notes": "aggregate_over_finite_pair_rates",
            }
        )
        confusion_payload["splits"][split_name] = {
            "confusion_matrix": confusion.tolist(),
            "class_counts": class_counts.astype(int).tolist(),
            "diagonal_counts": np.diag(confusion).astype(int).tolist(),
            "pair_results": split_pair_payload,
        }
    confusion_path = diagnostics_dir / "similar_class_confusion.csv"
    _append_rows(confusion_path, rows)
    matrix_path = diagnostics_dir / f"{experiment_key}_known_confusion_matrix.json"
    _write_json(matrix_path, confusion_payload)
    return {"similar_class_confusion": str(confusion_path), "frozen_pair_manifest": str(manifest_path), "known_confusion_matrix": str(matrix_path)}


def train_closed_set(
    model: PRRPADModel,
    loaders: dict[str, DataLoader],
    device: torch.device,
    *,
    iterations: int,
    lr: float,
    bcrps_settings: BCRPSSettings,
    diagnostics_dir: Path | None,
    run_id: str,
    experiment_key: str,
) -> tuple[list[dict], dict[str, torch.Tensor]]:
    optim = torch.optim.SGD(_require_trainable_parameters(_trainable_parameters(model), "closed-set training"), lr=lr, momentum=0.9)
    history = []
    model.to(device)
    train_iter = iter(loaders["train"])
    losses = []
    bcrps_records: list[dict[str, float | bool]] = []
    eval_every = max(1, iterations // 10)
    best_score = float("-inf")
    best_state = deepcopy(model.state_dict())
    for iteration in range(1, iterations + 1):
        model.train()
        train_iter, (features, labels) = _next_batch(train_iter, loaders["train"])
        features = features.to(device)
        labels = labels.to(device)
        optim.zero_grad(set_to_none=True)
        out = model.forward_known(features, labels)
        total_loss = out["loss"]
        if bcrps_settings.enable_bcrps or bcrps_settings.logging_diagnostics:
            bcrps_result = compute_bcrps_loss(
                collect_reciprocal_point_tensors(model),
                bcrps_settings,
                iteration=iteration,
                total_iterations=iterations,
            )
            total_loss = total_loss + bcrps_result.loss
            bcrps_records.append(bcrps_result.log_dict())
        total_loss.backward()
        optim.step()
        losses.append(float(total_loss.item()))
        if iteration % eval_every == 0 or iteration == iterations:
            metrics = evaluate_closed_set(model, loaders["val"], device, model.config.num_classes)
            score = metrics["accuracy"] + metrics["mean_iou"]
            if score >= best_score:
                best_score = score
                best_state = deepcopy(model.state_dict())
            if diagnostics_dir is not None and bcrps_settings.logging_diagnostics:
                write_reciprocal_point_diagnostics(
                    model,
                    diagnostics_dir,
                    run_id=run_id,
                    experiment=experiment_key,
                    stage="validation_checkpoint",
                    iteration=iteration,
                )
            history.append(
                {
                    "iteration": iteration,
                    "train_loss": float(np.mean(losses[-eval_every:])),
                    **_bcrps_recent_summary(bcrps_records, eval_every),
                    **metrics,
                }
            )
    return history, best_state


def train_open_set(
    model: PRRPADModel,
    loaders: dict[str, DataLoader],
    device: torch.device,
    *,
    iterations: int,
    lr: float,
    gen_lr: float,
    bcrps_settings: BCRPSSettings,
    diagnostics_dir: Path | None,
    run_id: str,
    experiment_key: str,
    strict_validation_selection: bool = False,
) -> tuple[list[dict], dict[str, torch.Tensor]]:
    classifier_params = (
        _trainable_parameters(model.backbone)
        + _trainable_parameters(model.scale1)
        + _trainable_parameters(model.scale2)
        + _trainable_parameters(model.scale3)
        + _trainable_parameters(model.head1)
        + _trainable_parameters(model.head2)
        + _trainable_parameters(model.head3)
    )
    opt_c = torch.optim.SGD(_require_trainable_parameters(classifier_params, "open-set classifier training"), lr=lr, momentum=0.9)
    generator_params = _trainable_parameters(model.generator)
    discriminator_params = _trainable_parameters(model.discriminator)
    opt_g = torch.optim.Adam(generator_params, lr=gen_lr) if generator_params else None
    opt_d = torch.optim.Adam(discriminator_params, lr=gen_lr) if discriminator_params else None
    update_generator_discriminator = opt_g is not None and opt_d is not None
    bce = nn.BCEWithLogitsLoss()
    history = []
    model.to(device)
    train_iter = iter(loaders["train"])
    classifier_losses = []
    bcrps_records: list[dict[str, float | bool]] = []
    eval_every = max(1, iterations // 10)
    best_score = float("-inf")
    best_state = deepcopy(model.state_dict())
    for iteration in range(1, iterations + 1):
        model.train()
        train_iter, (features, labels) = _next_batch(train_iter, loaders["train"])
        features = features.to(device)
        labels = labels.to(device)
        batch_size = features.size(0)

        if update_generator_discriminator:
            opt_d.zero_grad(set_to_none=True)
            z = torch.randn(batch_size, model.config.latent_dim, device=device)
            fake = model.generator(z)
            real_logits = model.discriminator(features)
            fake_logits = model.discriminator(fake.detach())
            d_loss = bce(real_logits, torch.ones_like(real_logits)) + bce(fake_logits, torch.zeros_like(fake_logits))
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            z = torch.randn(batch_size, model.config.latent_dim, device=device)
            fake = model.generator(z)
            fake_logits = model.discriminator(fake)
            fake_probs = model.forward_unknown(fake)["probs"]
            g_loss = bce(fake_logits, torch.ones_like(fake_logits)) - model.config.beta_entropy * model.entropy(fake_probs)
            g_loss.backward()
            opt_g.step()

        opt_c.zero_grad(set_to_none=True)
        known_out = model.forward_known(features, labels)
        with torch.no_grad():
            fake = model.generator(torch.randn(batch_size, model.config.latent_dim, device=device))
        fake_probs = model.forward_unknown(fake)["probs"]
        c_loss = known_out["loss"] - model.config.beta_entropy * model.entropy(fake_probs)
        if bcrps_settings.enable_bcrps or bcrps_settings.logging_diagnostics:
            bcrps_result = compute_bcrps_loss(
                collect_reciprocal_point_tensors(model),
                bcrps_settings,
                iteration=iteration,
                total_iterations=iterations,
            )
            c_loss = c_loss + bcrps_result.loss
            bcrps_records.append(bcrps_result.log_dict())
        c_loss.backward()
        opt_c.step()
        classifier_losses.append(float(c_loss.item()))

        if iteration % eval_every == 0 or iteration == iterations:
            if strict_validation_selection:
                known_val_metrics = evaluate_closed_set(model, loaders["val"], device, model.config.num_classes)
                metrics = {
                    "ACC_known": known_val_metrics["accuracy"],
                    "OSCR": float("nan"),
                    "validation_selection": "known_val_only_no_unknown_out",
                }
                score = metrics["ACC_known"]
            else:
                metrics = evaluate_open_set(model, loaders["val"], loaders["out"], device)
                score = metrics["ACC_known"] + metrics["OSCR"]
            if score >= best_score:
                best_score = score
                best_state = deepcopy(model.state_dict())
            if diagnostics_dir is not None and bcrps_settings.logging_diagnostics:
                write_reciprocal_point_diagnostics(
                    model,
                    diagnostics_dir,
                    run_id=run_id,
                    experiment=experiment_key,
                    stage="validation_checkpoint",
                    iteration=iteration,
                )
            history.append(
                {
                    "iteration": iteration,
                    "train_loss": float(np.mean(classifier_losses[-eval_every:])),
                    **_bcrps_recent_summary(bcrps_records, eval_every),
                    **metrics,
                }
            )
    return history, best_state


def evaluate_specific_conditions(
    model: PRRPADModel,
    dataset_root: Path,
    converter: PulseGraphConverter,
    device: torch.device,
    *,
    batch_size: int,
    open_set: bool,
    max_sequences_per_condition: int | None = None,
) -> pd.DataFrame:
    rows = []
    specific_root = dataset_root / "specific"
    if not specific_root.exists():
        return pd.DataFrame()
    for split_dir in sorted(path for path in specific_root.iterdir() if path.is_dir()):
        manifest_path = split_dir / "pulse_index.csv"
        if not manifest_path.exists():
            continue
        pulse_index = pd.read_csv(manifest_path)
        if max_sequences_per_condition is not None:
            keep_sequences = pulse_index["sequence_name"].drop_duplicates().tolist()[:max_sequences_per_condition]
            pulse_index = pulse_index.loc[pulse_index["sequence_name"].isin(keep_sequences)].copy()
        known_df = pulse_index.loc[pulse_index["mapped_label"] >= 0].copy()
        loader = DataLoader(PulseIndexDataset(specific_root, known_df.assign(sequence_path=known_df["sequence_path"]), converter), batch_size=batch_size, shuffle=False, num_workers=0)
        if open_set:
            unknown_df = pulse_index.loc[pulse_index["is_unknown"] == 1].copy()
            if len(unknown_df):
                out_loader = DataLoader(PulseIndexDataset(specific_root, unknown_df.assign(sequence_path=unknown_df["sequence_path"]), converter), batch_size=batch_size, shuffle=False, num_workers=0)
                metrics = evaluate_open_set(model, loader, out_loader, device)
            else:
                metrics = {"ACC_known": np.nan, "OSCR": np.nan}
        else:
            num_classes = int(known_df["mapped_label"].max() + 1)
            metrics = evaluate_closed_set(model, loader, device, num_classes)
        rows.append({"condition": split_dir.name, **metrics})
    return pd.DataFrame(rows)


def plot_history(output_dir: Path, experiment_key: str, history: list[dict], open_set: bool) -> None:
    out_dir = output_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history_df["iteration"], history_df["train_loss"], color="#2d5f73")
    axes[0].set_title(f"{experiment_key} Train Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    if open_set:
        axes[1].plot(history_df["iteration"], history_df["ACC_known"], label="ACC_known", color="#2e8b57")
        axes[1].plot(history_df["iteration"], history_df["OSCR"], label="OSCR", color="#d65f5f")
    else:
        axes[1].plot(history_df["iteration"], history_df["accuracy"], label="ACC", color="#2e8b57")
        axes[1].plot(history_df["iteration"], history_df["mean_iou"], label="MIOU", color="#d65f5f")
    axes[1].set_title(f"{experiment_key} Validation")
    axes[1].set_xlabel("Iteration")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{experiment_key}_history.png", dpi=160)
    plt.close(fig)


def _num_classes(dataset_root: Path) -> int:
    train_index = _index_path(dataset_root, "train")
    frame = pd.read_csv(train_index, usecols=["mapped_label"])
    known_labels = frame.loc[frame["mapped_label"] >= 0, "mapped_label"]
    if len(known_labels):
        return int(known_labels.max() + 1)
    config_payload = json.loads((dataset_root / "config.json").read_text(encoding="utf-8"))
    known_emitters = [emitter for emitter in config_payload["experiment"]["emitters"] if emitter["known"]]
    return len(known_emitters)


def run_experiment(
    workspace: Path,
    experiment_key: str,
    *,
    iterations: int,
    batch_size: int,
    lr: float,
    gen_lr: float,
    evaluate_specific: bool,
    specific_limit: int | None,
    bcrps_settings: BCRPSSettings,
    round4_settings: Round4Settings,
    output_dir: Path,
    run_id: str,
    seed_override: int | None,
    seed_label_value: str,
    timestamp: str,
    dataset_family_root: Path,
    input_mode: str = "vector",
    strict_validation_selection: bool = False,
    model_arch: str = "mlp",
) -> dict:
    seed = seed_override if seed_override is not None else round4_settings.seed
    if seed is None:
        seed = bcrps_settings.seed
    if seed is None:
        seed = EXPERIMENT_SEEDS.get(experiment_key, 20260417)
    set_seed(seed)
    dataset_root = dataset_family_root / experiment_key
    converter = build_converter(dataset_root, input_mode=input_mode)
    loaders = build_loaders(dataset_root, batch_size, converter)
    sample_dim = int(loaders["train"].dataset[0][0].numel())
    model = build_model(model_arch=model_arch, num_classes=_num_classes(dataset_root), input_dim=sample_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    round4_checkpoint_path = _load_round4_checkpoint_if_configured(
        model,
        workspace,
        round4_settings,
        run_id=run_id,
        seed_label_value=seed_label_value,
        timestamp=timestamp,
        device=device,
    )
    merge_requested_for_inference = bool(round4_settings.uses_lora() and round4_settings.merge_adapters_for_inference)
    scaffolding_settings = (
        round4_settings.with_overrides(merge_adapters_for_inference=False)
        if merge_requested_for_inference
        else round4_settings
    )
    round4_payload = apply_round4_scaffolding(model, scaffolding_settings)
    round4_payload["requested_settings"] = round4_settings.to_dict()
    round4_payload["merge_requested_for_inference"] = merge_requested_for_inference
    merged_modules_for_inference: list[str] = []
    is_open_set = any(experiment.open_set for experiment in EXPERIMENTS if f"exp{experiment.experiment_id}" == experiment_key)
    diagnostics_dir = _resolve_diagnostics_dir(workspace, output_dir, bcrps_settings, run_id) if bcrps_settings.logging_diagnostics else None
    parameter_payload = parameter_report(model)
    parameter_count = int(parameter_payload["trainable_parameters"])
    model.to(device)

    train_started = time.perf_counter()
    skip_training = bool(round4_settings.enabled and round4_settings.method_variant == "frozen_no_adapter")
    if skip_training:
        history = []
        training_time_seconds = 0.0
        if merge_requested_for_inference:
            merged_modules_for_inference = merge_lora_adapters(model)
        eval_started = time.perf_counter()
        if is_open_set:
            final_metrics = evaluate_open_set(model, loaders["test"], loaders["out"], device)
        else:
            final_metrics = evaluate_closed_set(model, loaders["test"], device, model.config.num_classes)
        evaluation_time_seconds = time.perf_counter() - eval_started
    elif is_open_set:
        history, best_state = train_open_set(
            model,
            loaders,
            device,
            iterations=iterations,
            lr=lr,
            gen_lr=gen_lr,
            bcrps_settings=bcrps_settings,
            diagnostics_dir=diagnostics_dir,
            run_id=run_id,
            experiment_key=experiment_key,
            strict_validation_selection=strict_validation_selection,
        )
        model.load_state_dict(best_state)
        training_time_seconds = time.perf_counter() - train_started
        if merge_requested_for_inference:
            merged_modules_for_inference = merge_lora_adapters(model)
        eval_started = time.perf_counter()
        final_metrics = evaluate_open_set(model, loaders["test"], loaders["out"], device)
        evaluation_time_seconds = time.perf_counter() - eval_started
    else:
        history, best_state = train_closed_set(
            model,
            loaders,
            device,
            iterations=iterations,
            lr=lr,
            bcrps_settings=bcrps_settings,
            diagnostics_dir=diagnostics_dir,
            run_id=run_id,
            experiment_key=experiment_key,
        )
        model.load_state_dict(best_state)
        training_time_seconds = time.perf_counter() - train_started
        if merge_requested_for_inference:
            merged_modules_for_inference = merge_lora_adapters(model)
        eval_started = time.perf_counter()
        final_metrics = evaluate_closed_set(model, loaders["test"], device, model.config.num_classes)
        evaluation_time_seconds = time.perf_counter() - eval_started
    specific_df = pd.DataFrame()
    if evaluate_specific:
        specific_df = evaluate_specific_conditions(
            model,
            dataset_root,
            converter,
            device,
            batch_size=batch_size,
            open_set=is_open_set,
            max_sequences_per_condition=specific_limit,
        )
    if history:
        plot_history(output_dir, experiment_key, history, is_open_set)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / f"{experiment_key}_history.csv", index=False)
    if len(specific_df):
        specific_df.to_csv(output_dir / f"{experiment_key}_specific_metrics.csv", index=False)
    checkpoint_path = output_dir / f"{experiment_key}_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "model_config": asdict(model.config)}, checkpoint_path)
    inference_latency_ms: float | None = None
    if round4_settings.resource_logging:
        latency_features, _latency_labels = next(iter(loaders["test"]))
        latency_features = latency_features[: min(16, latency_features.size(0))]
        inference_latency_ms = measure_inference_latency_ms(
            model,
            latency_features,
            device,
            warmup_steps=round4_settings.latency_warmup_steps,
            repeats=round4_settings.latency_repeats,
        )

    diagnostic_outputs: dict[str, str] = {}
    if diagnostics_dir is not None:
        diagnostic_outputs.update(
            write_reciprocal_point_diagnostics(
                model,
                diagnostics_dir,
                run_id=run_id,
                experiment=experiment_key,
                stage="final_selected",
                iteration=iterations,
            )
        )
        if is_open_set:
            diagnostic_outputs.update(
                _write_similar_class_confusion(
                    model,
                    loaders,
                    device,
                    diagnostics_dir,
                    workspace,
                    bcrps_settings,
                    run_id=run_id,
                    experiment_key=experiment_key,
                    num_classes=model.config.num_classes,
                )
            )

    paper_target = PAPER_TARGETS.get(experiment_key, {})
    return {
        "experiment": experiment_key,
        "open_set": is_open_set,
        "iterations": iterations,
        "seed": seed,
        "device": str(device),
        "dataset_path": str(dataset_root),
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(output_dir / f"{experiment_key}_history.csv"),
        "specific_metrics_path": str(output_dir / f"{experiment_key}_specific_metrics.csv") if len(specific_df) else None,
        "diagnostic_outputs": diagnostic_outputs,
        "final_metrics": final_metrics,
        "paper_target": paper_target,
        "round4": round4_payload,
        "round4_merged_modules_for_inference": merged_modules_for_inference,
        "round4_checkpoint_initialization_path": round4_checkpoint_path,
        "strict_repro": {
            "input_mode": input_mode,
            "model_arch": model_arch,
            "open_set_model_selection": (
                "known_validation_only_no_unknown_out"
                if strict_validation_selection
                else "legacy_validation_uses_unknown_out"
            ),
        },
        "training_skipped": skip_training,
        "status": "completed",
        "completion_status": "completed",
        "abnormal_events": [],
        "wall_clock_training_time_seconds": training_time_seconds,
        "wall_clock_evaluation_time_seconds": evaluation_time_seconds,
        "peak_gpu_memory_bytes": _peak_gpu_memory(device),
        "inference_latency_ms": inference_latency_ms,
        "trainable_parameter_count": parameter_count,
        "total_parameter_count": int(parameter_payload["total_parameters"]),
        "adapter_parameter_count": int(parameter_payload["adapter_parameters"]),
    }


def main() -> int:
    stage = "parse_arguments"
    workspace = Path(".").resolve()
    config_path: Path | None = None
    round4_config_path: Path | None = None
    base_settings: BCRPSSettings | None = None
    settings: BCRPSSettings | None = None
    round4_settings = Round4Settings()
    output_dir = workspace / "output"
    run_id = "unresolved"
    seed_part: int | None = None
    reports: list[dict[str, Any]] = []
    validated_protocol_lock: dict[str, Any] | None = None
    startup_dry_run = False
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--workspace", default=".")
        parser.add_argument("--dataset-root", default=None)
        parser.add_argument("--config", default=None)
        parser.add_argument("--round4-config", default=None)
        parser.add_argument("--experiments", default="exp1,exp4")
        parser.add_argument("--iterations", type=int, default=100)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.01)
        parser.add_argument("--gen-lr", type=float, default=0.0002)
        parser.add_argument("--force-datasets", action="store_true")
        parser.add_argument("--skip-specific", action="store_true")
        parser.add_argument("--specific-limit", type=int, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--run-id", default=None)
        parser.add_argument("--output-dir", default=None)
        parser.add_argument("--enable-bcrps", dest="enable_bcrps", action="store_true", default=None)
        parser.add_argument("--disable-bcrps", dest="enable_bcrps", action="store_false")
        parser.add_argument("--bcrps-lambda", dest="lambda_weight", type=float, default=None)
        parser.add_argument("--bcrps-margin", dest="margin", type=float, default=None)
        parser.add_argument("--use-normalized-distance", dest="use_normalized_distance", action="store_true", default=None)
        parser.add_argument("--raw-distance", dest="use_normalized_distance", action="store_false")
        parser.add_argument("--use-warmup", dest="use_warmup", action="store_true", default=None)
        parser.add_argument("--no-warmup", dest="use_warmup", action="store_false")
        parser.add_argument("--warmup-fraction", type=float, default=None)
        parser.add_argument("--ablation-mode", default=None)
        parser.add_argument("--logging-diagnostics", dest="logging_diagnostics", action="store_true", default=None)
        parser.add_argument("--no-logging-diagnostics", dest="logging_diagnostics", action="store_false")
        parser.add_argument("--diagnostic-output-dir", default=None)
        parser.add_argument("--frozen-pair-manifest", default=None)
        parser.add_argument("--startup-dry-run", action="store_true")
        parser.add_argument("--input-mode", choices=["vector", "pdg_image"], default="vector")
        parser.add_argument("--strict-open-set-selection", action="store_true")
        parser.add_argument("--model-arch", choices=sorted(MODEL_ARCHITECTURES), default="mlp")
        args = parser.parse_args()
        startup_dry_run = bool(args.startup_dry_run)

        stage = "resolve_config"
        workspace = Path(args.workspace).resolve()
        dataset_family_root = Path(args.dataset_root) if args.dataset_root else workspace / "artifacts" / "datasets" / "pr_rpad_v1"
        if not dataset_family_root.is_absolute():
            dataset_family_root = (workspace / dataset_family_root).resolve()
        else:
            dataset_family_root = dataset_family_root.resolve()
        output_dir = workspace / "output"
        config_path = Path(args.config) if args.config else None
        if config_path is not None and not config_path.is_absolute():
            config_path = workspace / config_path
        round4_config_path = Path(args.round4_config) if args.round4_config else None
        if round4_config_path is not None and not round4_config_path.is_absolute():
            round4_config_path = workspace / round4_config_path
        base_settings = BCRPSSettings.from_json(config_path)
        round4_settings = Round4Settings.from_json(round4_config_path)
        settings = base_settings.with_overrides(
            enable_bcrps=args.enable_bcrps,
            lambda_weight=args.lambda_weight,
            margin=args.margin,
            use_normalized_distance=args.use_normalized_distance,
            use_warmup=args.use_warmup,
            warmup_fraction=args.warmup_fraction,
            seed=args.seed,
            ablation_mode=args.ablation_mode,
            logging_diagnostics=args.logging_diagnostics,
            diagnostic_output_dir=args.diagnostic_output_dir,
            run_id=args.run_id,
            frozen_pair_manifest=args.frozen_pair_manifest,
        )
        round4_settings = round4_settings.with_overrides(
            seed=args.seed,
            run_id=args.run_id,
            output_dir=args.output_dir,
        )
        if (
            settings.strict_protocol_lock
            and settings.protocol_lock is not None
            and settings.frozen_pair_manifest is None
            and settings.protocol_lock.frozen_similar_class_pair_source is not None
        ):
            settings = settings.with_overrides(
                frozen_pair_manifest=settings.protocol_lock.frozen_similar_class_pair_source
            )
        contract_mode = bool(
            config_path
            or round4_config_path
            or round4_settings.enabled
            or settings.enable_bcrps
            or settings.logging_diagnostics
            or settings.strict_protocol_lock
            or args.run_id
            or args.output_dir
        )
        run_id = (
            settings.run_id
            or (settings.protocol_lock.run_id if settings.protocol_lock is not None else "")
            or round4_settings.run_id
            or ""
        )
        timestamp = time.strftime("%Y%m%d%H%M")
        seed_part = settings.seed if settings.seed is not None else round4_settings.seed
        if seed_part is None:
            seed_part = args.seed
        seed_label_value = str(seed_part) if seed_part is not None else "legacy"
        if run_id:
            run_id = _expand_runtime_value(run_id, run_id=run_id, seed_label_value=seed_label_value, timestamp=timestamp)
            settings = settings.with_overrides(run_id=run_id)
            round4_settings = round4_settings.with_overrides(run_id=run_id)
        elif contract_mode:
            seed_label = f"seed{seed_part}" if seed_part is not None else "seedlegacy"
            run_family = "round4_contract_v1" if round4_settings.enabled else "bcrps_v1"
            run_mode = round4_settings.method_variant if round4_settings.enabled else settings.ablation_mode
            run_id = f"{run_family}_{run_mode}_{seed_label}_{timestamp}"
            settings = settings.with_overrides(run_id=run_id)
            round4_settings = round4_settings.with_overrides(run_id=run_id)
        if not run_id:
            run_id = "legacy"
        if args.output_dir is not None:
            output_dir = _resolve_runtime_path(
                workspace,
                args.output_dir,
                run_id=run_id,
                seed_label_value=seed_label_value,
                timestamp=timestamp,
            )
        elif settings.strict_protocol_lock and settings.protocol_lock is not None:
            output_dir = _resolve_runtime_path(
                workspace,
                settings.protocol_lock.output_dir,
                run_id=run_id,
                seed_label_value=seed_label_value,
                timestamp=timestamp,
            )
        elif round4_settings.output_dir is not None:
            output_dir = _resolve_runtime_path(
                workspace,
                round4_settings.output_dir,
                run_id=run_id,
                seed_label_value=seed_label_value,
                timestamp=timestamp,
            )
        elif contract_mode:
            output_family = "round4_contract_v1" if round4_settings.enabled else "bcrps_contract_v1"
            output_dir = workspace / "output" / output_family / run_id
        else:
            output_dir = workspace / "output"

        selected = [item.strip() for item in args.experiments.split(",") if item.strip()]
        round4_settings.validate_selected_experiments(selected)
        stage = "validate_protocol_lock"
        validated_protocol_lock = _validate_protocol_lock_runtime(
            workspace,
            settings,
            base_settings,
            run_id=run_id,
            output_dir=output_dir,
            selected_experiments=selected,
            seed_label_value=seed_label_value,
            timestamp=timestamp,
        )
        stage = "validate_frozen_pair_source"
        frozen_pair_source = _validate_frozen_pair_source(
            workspace,
            settings,
            run_id=run_id,
            experiment_keys=selected,
        )

        if startup_dry_run:
            summary = {
                "workspace": str(workspace),
                "dataset_root": str(dataset_family_root),
                "run_id": run_id,
                "config_path": str(config_path.resolve()) if config_path else None,
                "round4_config_path": str(round4_config_path.resolve()) if round4_config_path else None,
                "output_dir": str(output_dir),
                "status": "startup_dry_run_ok",
                "dry_run": True,
                "dataset_build_started": False,
                "training_started": False,
                "evaluation_started": False,
                "metrics_written": False,
                "input_mode": args.input_mode,
                "strict_open_set_selection": bool(args.strict_open_set_selection),
                "model_arch": args.model_arch,
                "round4": round4_settings.to_dict(),
                "selected_experiments": selected,
                "protocol_lock": validated_protocol_lock,
                "frozen_pair_source": frozen_pair_source,
            }
            print(json.dumps(summary, indent=2))
            return 0

        stage = "build_datasets"
        iterations = args.epochs if args.epochs is not None else args.iterations
        if args.dataset_root:
            dataset_paths = {
                experiment_key: dataset_family_root / experiment_key
                for experiment_key in selected
                if (dataset_family_root / experiment_key).exists()
            }
        else:
            dataset_paths = build_all_datasets(workspace, ReproConfig(), force=args.force_datasets)
        for experiment_key in selected:
            if experiment_key not in dataset_paths:
                raise ValueError(f"Unknown experiment key: {experiment_key}")
            stage = f"run_experiment:{experiment_key}"
            reports.append(
                run_experiment(
                    workspace,
                    experiment_key,
                    iterations=iterations,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    gen_lr=args.gen_lr,
                    evaluate_specific=not args.skip_specific,
                    specific_limit=args.specific_limit,
                    bcrps_settings=settings,
                    round4_settings=round4_settings,
                    output_dir=output_dir,
                    run_id=run_id,
                    seed_override=args.seed,
                    seed_label_value=seed_label_value,
                    timestamp=timestamp,
                    dataset_family_root=dataset_family_root,
                    input_mode=args.input_mode,
                    strict_validation_selection=bool(args.strict_open_set_selection),
                    model_arch=args.model_arch,
                )
            )

        stage = "write_summary"
        summary = {
            "workspace": str(workspace),
            "dataset_root": str(dataset_family_root),
            "run_id": run_id,
            "config_path": str(config_path.resolve()) if config_path else None,
            "round4_config_path": str(round4_config_path.resolve()) if round4_config_path else None,
            "config": settings.to_dict(),
            "round4": round4_settings.to_dict(),
            "input_mode": args.input_mode,
            "strict_open_set_selection": bool(args.strict_open_set_selection),
            "model_arch": args.model_arch,
            "output_dir": str(output_dir),
            "git_commit": _git_commit(workspace),
            "log_path": str(output_dir),
            "status": "completed",
            "completion_status": "completed",
            "abnormal_events": [],
            "protocol_lock": validated_protocol_lock,
            "frozen_pair_source": frozen_pair_source,
            "reports": reports,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "reproduction_report.json"
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if contract_mode:
            _write_json(output_dir / "run_manifest.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        if startup_dry_run:
            raise
        _write_failure_run_record(
            output_dir,
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            seed=seed_part,
            config_path=config_path,
            settings=settings,
            exc=exc,
            reports=reports,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
