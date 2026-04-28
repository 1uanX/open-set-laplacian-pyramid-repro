from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from momentfm import MOMENTPipeline


FEATURE_COLUMNS = ["dtoa", "rf", "pw", "pa", "doa"]
MODEL_REPO = "AutonLab/MOMENT-1-large"
LORA_VARIANTS = {
    "moment_lora_rank8": {
        "rank": 8,
        "alpha": 16,
        "target_modules": ["q", "v"],
        "dropout": 0.05,
        "use_rslora": False,
        "use_dora": False,
    },
    "moment_lora_qv_rank4": {
        "rank": 4,
        "alpha": 8,
        "target_modules": ["q", "v"],
        "dropout": 0.05,
        "use_rslora": False,
        "use_dora": False,
    },
    "moment_lora_qv_rank16": {
        "rank": 16,
        "alpha": 32,
        "target_modules": ["q", "v"],
        "dropout": 0.05,
        "use_rslora": False,
        "use_dora": False,
    },
    "moment_lora_qkvo_rank8": {
        "rank": 8,
        "alpha": 16,
        "target_modules": ["q", "k", "v", "o"],
        "dropout": 0.05,
        "use_rslora": False,
        "use_dora": False,
    },
    "moment_lora_qkvo_rank8_rslora": {
        "rank": 8,
        "alpha": 16,
        "target_modules": ["q", "k", "v", "o"],
        "dropout": 0.05,
        "use_rslora": True,
        "use_dora": False,
    },
    "moment_lora_qv_rank8_rslora": {
        "rank": 8,
        "alpha": 16,
        "target_modules": ["q", "v"],
        "dropout": 0.05,
        "use_rslora": True,
        "use_dora": False,
    },
    "moment_lora_qv_rank8_dora": {
        "rank": 8,
        "alpha": 16,
        "target_modules": ["q", "v"],
        "dropout": 0.05,
        "use_rslora": False,
        "use_dora": True,
    },
}
MODEL_CHOICES = {"moment_head", *LORA_VARIANTS.keys()}


@dataclass
class RunConfig:
    dataset_root: str
    experiments: list[str]
    models: list[str]
    output_dir: str
    cache_dir: str
    seed: int
    window_length: int
    patch_length: int
    patch_stride: int
    iterations: int
    eval_every: int
    batch_size: int
    samples_per_class: int
    train_limit: int | None
    eval_limit: int | None
    prototype_limit: int
    lr_head: float
    lr_lora: float
    weight_decay: float
    known_accept_rate: float
    device: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_feature_ranges(dataset_root: Path, experiment_key: str) -> dict[str, tuple[float, float]]:
    exp_root = dataset_root / experiment_key
    config_path = exp_root / "config.json"
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        ranges = payload.get("experiment", {}).get("global_ranges", {})
        return {key: (float(value[0]), float(value[1])) for key, value in ranges.items()}
    for parent in [exp_root, *exp_root.parents]:
        policy_path = parent / "normalization_policy_round5.json"
        if not policy_path.exists():
            continue
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
        stats = payload.get("experiments", {}).get(experiment_key)
        if stats:
            return {
                column: (float(value["min"]), float(value["max"]))
                for column, value in stats.items()
            }
    return {}


def index_path(exp_root: Path, split: str) -> Path:
    candidates = [
        exp_root / split / "pulse_index.csv",
        exp_root / "mixed" / split / "pulse_index.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find pulse_index.csv for split={split} under {exp_root}")


def load_index(exp_root: Path, split: str, *, known_only: bool = False, unknown_only: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(index_path(exp_root, split))
    if known_only:
        frame = frame.loc[frame["mapped_label"] >= 0].copy()
    if unknown_only:
        frame = frame.loc[frame["is_unknown"] == 1].copy()
    return frame.reset_index(drop=True)


def sample_per_class(frame: pd.DataFrame, samples_per_class: int, seed: int) -> pd.DataFrame:
    if samples_per_class <= 0:
        return frame.reset_index(drop=True)
    parts = []
    for label, group in frame.groupby("mapped_label", sort=True):
        take = min(samples_per_class, len(group))
        parts.append(group.sample(n=take, random_state=seed + int(label)))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def limit_frame(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(frame) <= limit:
        return frame.reset_index(drop=True)
    return frame.sample(n=limit, random_state=seed).reset_index(drop=True)


class PDWWindowDataset(Dataset):
    def __init__(
        self,
        exp_root: Path,
        index_df: pd.DataFrame,
        *,
        window_length: int,
        feature_ranges: dict[str, tuple[float, float]],
    ) -> None:
        self.exp_root = exp_root
        self.index_df = index_df.reset_index(drop=True)
        self.window_length = int(window_length)
        self.feature_ranges = feature_ranges
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.index_df)

    def _resolve_sequence_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute() and path.exists():
            return path
        direct = self.exp_root / value
        if direct.exists():
            return direct
        mixed = self.exp_root / "mixed" / value
        if mixed.exists():
            return mixed
        raise FileNotFoundError(f"Cannot resolve sequence path {value}")

    def _normalize(self, sequence: pd.DataFrame) -> np.ndarray:
        work = sequence.loc[:, FEATURE_COLUMNS].copy()
        for column in FEATURE_COLUMNS:
            values = work[column].to_numpy(dtype=np.float32)
            if column in self.feature_ranges:
                lo, hi = self.feature_ranges[column]
            else:
                lo = float(np.min(values))
                hi = float(np.max(values))
            if math.isclose(lo, hi):
                work[column] = 0.0
            else:
                work[column] = (values - lo) / (hi - lo)
            work[column] = work[column].clip(0.0, 1.0)
        return work.to_numpy(dtype=np.float32)

    def _load_sequence(self, sequence_path: str) -> tuple[np.ndarray, np.ndarray]:
        if sequence_path not in self._cache:
            resolved = self._resolve_sequence_path(sequence_path)
            frame = pd.read_csv(resolved)
            features = self._normalize(frame)
            labels = frame["mapped_label"].to_numpy(dtype=np.int64)
            self._cache[sequence_path] = (features, labels)
        return self._cache[sequence_path]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.index_df.iloc[idx]
        features, labels = self._load_sequence(str(row["sequence_path"]))
        center = int(row["pulse_index"])
        start = center - (self.window_length // 2)
        indices = np.arange(start, start + self.window_length)
        indices = np.clip(indices, 0, len(features) - 1)
        window = features[indices]
        label = int(row["mapped_label"])
        if label < 0:
            label = int(labels[center])
        # MOMENT expects [channels, sequence_length].
        return torch.from_numpy(window.T.copy()), torch.tensor(label, dtype=torch.long)


def build_loader(
    exp_root: Path,
    index_df: pd.DataFrame,
    *,
    window_length: int,
    feature_ranges: dict[str, tuple[float, float]],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = PDWWindowDataset(
        exp_root,
        index_df,
        window_length=window_length,
        feature_ranges=feature_ranges,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def count_trainable(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


def build_model(
    *,
    model_name: str,
    num_classes: int,
    cache_dir: Path,
    window_length: int,
    patch_length: int,
    patch_stride: int,
    local_files_only: bool,
) -> torch.nn.Module:
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"Unsupported model {model_name}")
    freeze_encoder = model_name == "moment_head"
    model = MOMENTPipeline.from_pretrained(
        MODEL_REPO,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        model_kwargs={
            "task_name": "classification",
            "n_channels": len(FEATURE_COLUMNS),
            "num_class": num_classes,
            "seq_len": window_length,
            "patch_len": patch_length,
            "patch_stride_len": patch_stride,
            "freeze_encoder": freeze_encoder,
            "freeze_embedder": True,
            "freeze_head": False,
        },
    )
    model.init()
    if hasattr(model.encoder, "gradient_checkpointing_disable"):
        model.encoder.gradient_checkpointing_disable()

    for parameter in model.parameters():
        parameter.requires_grad = False

    if model_name in LORA_VARIANTS:
        variant = LORA_VARIANTS[model_name]
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=int(variant["rank"]),
            lora_alpha=int(variant["alpha"]),
            lora_dropout=float(variant["dropout"]),
            target_modules=list(variant["target_modules"]),
            bias="none",
            use_rslora=bool(variant["use_rslora"]),
            use_dora=bool(variant["use_dora"]),
        )
        model.encoder = get_peft_model(model.encoder, lora_config)
        try:
            model.encoder.base_model.model.gradient_checkpointing_disable()
        except AttributeError:
            pass

    for parameter in model.head.parameters():
        parameter.requires_grad = True
    return model


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state_dict(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        parameters[name].data.copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))


def model_forward(model: torch.nn.Module, batch_x: torch.Tensor, *, use_amp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(x_enc=batch_x)
            logits = outputs.logits
    else:
        outputs = model(x_enc=batch_x)
        logits = outputs.logits
    embeddings = outputs.embeddings
    if embeddings.ndim == 3:
        embeddings = embeddings.mean(dim=1)
    return logits, embeddings


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_items: int | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    labels_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    conf_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    emb_all: list[np.ndarray] = []
    seen = 0
    for batch_x, batch_y in loader:
        if max_items is not None and seen >= max_items:
            break
        if max_items is not None and seen + len(batch_y) > max_items:
            keep = max_items - seen
            batch_x = batch_x[:keep]
            batch_y = batch_y[:keep]
        batch_x = batch_x.to(device, non_blocking=True)
        logits, embeddings = model_forward(model, batch_x, use_amp=False)
        probs = torch.softmax(logits.float(), dim=1)
        conf, preds = probs.max(dim=1)
        labels_all.append(batch_y.numpy())
        preds_all.append(preds.cpu().numpy())
        conf_all.append(conf.cpu().numpy())
        logits_all.append(logits.float().cpu().numpy())
        probs_all.append(probs.cpu().numpy())
        emb_all.append(embeddings.float().cpu().numpy())
        seen += len(batch_y)
    return {
        "labels": np.concatenate(labels_all) if labels_all else np.asarray([], dtype=np.int64),
        "preds": np.concatenate(preds_all) if preds_all else np.asarray([], dtype=np.int64),
        "confidence": np.concatenate(conf_all) if conf_all else np.asarray([], dtype=np.float32),
        "logits": np.concatenate(logits_all) if logits_all else np.asarray([], dtype=np.float32),
        "probs": np.concatenate(probs_all) if probs_all else np.asarray([], dtype=np.float32),
        "embeddings": np.concatenate(emb_all) if emb_all else np.asarray([], dtype=np.float32),
    }


def fit_distance_models(embeddings: np.ndarray, labels: np.ndarray, num_classes: int) -> dict[str, np.ndarray]:
    means = []
    variances = []
    global_var = np.var(embeddings, axis=0) + 1e-4
    for label in range(num_classes):
        class_embeddings = embeddings[labels == label]
        if len(class_embeddings) == 0:
            means.append(np.zeros((embeddings.shape[1],), dtype=np.float32))
            variances.append(global_var.astype(np.float32))
            continue
        means.append(class_embeddings.mean(axis=0).astype(np.float32))
        if len(class_embeddings) > 1:
            variances.append((class_embeddings.var(axis=0) + 1e-4).astype(np.float32))
        else:
            variances.append(global_var.astype(np.float32))
    return {"means": np.stack(means), "variances": np.stack(variances)}


def distance_scores(embeddings: np.ndarray, distance_model: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    means = distance_model["means"]
    variances = distance_model["variances"]
    diff = embeddings[:, None, :] - means[None, :, :]
    centroid_distance = np.sum(diff * diff, axis=2)
    mahalanobis_distance = np.sum((diff * diff) / variances[None, :, :], axis=2)
    return {
        "centroid": -np.min(centroid_distance, axis=1),
        "mahalanobis": -np.min(mahalanobis_distance, axis=1),
    }


def oscr(known_scores: np.ndarray, known_correct: np.ndarray, unknown_scores: np.ndarray) -> float:
    if len(known_scores) == 0 or len(unknown_scores) == 0:
        return float("nan")
    thresholds = np.unique(np.concatenate([known_scores, unknown_scores]))
    points = []
    for threshold in np.concatenate(([thresholds.max() + 1.0], thresholds[::-1], [thresholds.min() - 1.0])):
        ccr = float(np.mean((known_scores >= threshold) & known_correct))
        fpr = float(np.mean(unknown_scores >= threshold))
        points.append((fpr, ccr))
    points = sorted(points, key=lambda item: item[0])
    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    return float(np.trapz(y, x) * 100.0)


def auroc_known_vs_unknown(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    if len(known_scores) == 0 or len(unknown_scores) == 0:
        return float("nan")
    y_true = np.concatenate([np.ones_like(known_scores), np.zeros_like(unknown_scores)])
    y_score = np.concatenate([known_scores, unknown_scores])
    return float(roc_auc_score(y_true, y_score) * 100.0)


def fpr95(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    if len(known_scores) == 0 or len(unknown_scores) == 0:
        return float("nan")
    threshold = float(np.quantile(known_scores, 0.05))
    return float(np.mean(unknown_scores >= threshold) * 100.0)


def logsumexp(values: np.ndarray, axis: int = 1) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(np.sum(np.exp(values - maximum), axis=axis))


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(float(temperature), 1e-6)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 1.0
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = F.cross_entropy(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).clamp(0.05, 20.0).detach().item())


def calibration_metrics(probs: np.ndarray, labels: np.ndarray, *, prefix: str, bins: int = 15) -> dict[str, float]:
    if len(labels) == 0:
        return {
            f"{prefix}_NLL": float("nan"),
            f"{prefix}_Brier": float("nan"),
            f"{prefix}_ECE": float("nan"),
        }
    labels = labels.astype(np.int64)
    clipped = np.clip(probs, 1e-8, 1.0)
    nll = -np.log(clipped[np.arange(len(labels)), labels]).mean()
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    confidence = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == labels).astype(np.float32)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        if high == 1.0:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask) * abs(np.mean(confidence[mask]) - np.mean(correctness[mask])))
    return {
        f"{prefix}_NLL": float(nll),
        f"{prefix}_Brier": float(brier),
        f"{prefix}_ECE": float(ece * 100.0),
    }


def standardize_from_val(val_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if len(val_scores) == 0:
        return scores
    mean = float(np.mean(val_scores))
    std = float(np.std(val_scores))
    if std < 1e-8:
        std = 1.0
    return (scores - mean) / std


def score_metrics(
    *,
    score_name: str,
    val_known: dict[str, np.ndarray],
    test_known: dict[str, np.ndarray],
    out_unknown: dict[str, np.ndarray],
    train_distance_model: dict[str, np.ndarray] | None,
    known_accept_rate: float,
) -> dict[str, float]:
    if score_name == "confidence":
        val_scores = val_known["confidence"]
        known_scores = test_known["confidence"]
        unknown_scores = out_unknown["confidence"]
    elif score_name == "energy":
        val_scores = logsumexp(val_known["logits"])
        known_scores = logsumexp(test_known["logits"])
        unknown_scores = logsumexp(out_unknown["logits"])
    else:
        assert train_distance_model is not None
        val_scores = distance_scores(val_known["embeddings"], train_distance_model)[score_name]
        known_scores = distance_scores(test_known["embeddings"], train_distance_model)[score_name]
        unknown_scores = distance_scores(out_unknown["embeddings"], train_distance_model)[score_name]
    known_correct = test_known["labels"] == test_known["preds"]
    quantile = max(0.0, min(1.0, 1.0 - known_accept_rate))
    threshold = float(np.quantile(val_scores, quantile)) if len(val_scores) else float("nan")
    return {
        f"{score_name}_OSCR": oscr(known_scores, known_correct, unknown_scores),
        f"{score_name}_AUROC": auroc_known_vs_unknown(known_scores, unknown_scores),
        f"{score_name}_FPR95": fpr95(known_scores, unknown_scores),
        f"{score_name}_unknown_accept_at_val95": float(np.mean(unknown_scores >= threshold) * 100.0)
        if len(unknown_scores)
        else float("nan"),
        f"{score_name}_threshold_val95": threshold,
    }


def direct_score_metrics(
    *,
    score_name: str,
    val_scores: np.ndarray,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_correct: np.ndarray,
    known_accept_rate: float,
) -> dict[str, float]:
    quantile = max(0.0, min(1.0, 1.0 - known_accept_rate))
    threshold = float(np.quantile(val_scores, quantile)) if len(val_scores) else float("nan")
    return {
        f"{score_name}_OSCR": oscr(known_scores, known_correct, unknown_scores),
        f"{score_name}_AUROC": auroc_known_vs_unknown(known_scores, unknown_scores),
        f"{score_name}_FPR95": fpr95(known_scores, unknown_scores),
        f"{score_name}_unknown_accept_at_val95": float(np.mean(unknown_scores >= threshold) * 100.0)
        if len(unknown_scores)
        else float("nan"),
        f"{score_name}_threshold_val95": threshold,
    }


def evaluate_all(
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    *,
    num_classes: int,
    prototype_limit: int,
    known_accept_rate: float,
) -> dict[str, float]:
    train_proto = collect_predictions(model, loaders["prototype"], device, max_items=prototype_limit)
    val_known = collect_predictions(model, loaders["val"], device)
    test_known = collect_predictions(model, loaders["test"], device)
    out_unknown = collect_predictions(model, loaders["out"], device)
    distance_model = fit_distance_models(train_proto["embeddings"], train_proto["labels"], num_classes)
    metrics: dict[str, float] = {
        "val_ACC_known": float(np.mean(val_known["labels"] == val_known["preds"]) * 100.0),
        "test_ACC_known": float(np.mean(test_known["labels"] == test_known["preds"]) * 100.0),
        "num_val_known": float(len(val_known["labels"])),
        "num_test_known": float(len(test_known["labels"])),
        "num_out_unknown": float(len(out_unknown["labels"])),
    }
    metrics.update(calibration_metrics(test_known["probs"], test_known["labels"], prefix="test"))
    for score_name in ["confidence", "energy", "centroid", "mahalanobis"]:
        metrics.update(
            score_metrics(
                score_name=score_name,
                val_known=val_known,
                test_known=test_known,
                out_unknown=out_unknown,
                train_distance_model=distance_model,
                known_accept_rate=known_accept_rate,
            )
        )
    known_correct = test_known["labels"] == test_known["preds"]
    temperature = fit_temperature(val_known["logits"], val_known["labels"])
    val_temp_probs = softmax_np(val_known["logits"], temperature)
    known_temp_probs = softmax_np(test_known["logits"], temperature)
    unknown_temp_probs = softmax_np(out_unknown["logits"], temperature)
    metrics["temperature"] = temperature
    metrics.update(calibration_metrics(known_temp_probs, test_known["labels"], prefix="temperature_test"))
    metrics.update(
        direct_score_metrics(
            score_name="temperature_confidence",
            val_scores=np.max(val_temp_probs, axis=1),
            known_scores=np.max(known_temp_probs, axis=1),
            unknown_scores=np.max(unknown_temp_probs, axis=1),
            known_correct=known_correct,
            known_accept_rate=known_accept_rate,
        )
    )

    val_distance = distance_scores(val_known["embeddings"], distance_model)
    known_distance = distance_scores(test_known["embeddings"], distance_model)
    unknown_distance = distance_scores(out_unknown["embeddings"], distance_model)
    val_conf_z = standardize_from_val(val_known["confidence"], val_known["confidence"])
    known_conf_z = standardize_from_val(val_known["confidence"], test_known["confidence"])
    unknown_conf_z = standardize_from_val(val_known["confidence"], out_unknown["confidence"])
    val_maha_z = standardize_from_val(val_distance["mahalanobis"], val_distance["mahalanobis"])
    known_maha_z = standardize_from_val(val_distance["mahalanobis"], known_distance["mahalanobis"])
    unknown_maha_z = standardize_from_val(val_distance["mahalanobis"], unknown_distance["mahalanobis"])
    metrics.update(
        direct_score_metrics(
            score_name="confidence_mahalanobis_fusion",
            val_scores=0.5 * val_conf_z + 0.5 * val_maha_z,
            known_scores=0.5 * known_conf_z + 0.5 * known_maha_z,
            unknown_scores=0.5 * unknown_conf_z + 0.5 * unknown_maha_z,
            known_correct=known_correct,
            known_accept_rate=known_accept_rate,
        )
    )
    return metrics


def train_one(
    *,
    model_name: str,
    exp_key: str,
    num_classes: int,
    loaders: dict[str, DataLoader],
    output_dir: Path,
    config: RunConfig,
    local_files_only: bool,
) -> dict:
    device = torch.device(config.device)
    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        cache_dir=Path(config.cache_dir),
        window_length=config.window_length,
        patch_length=config.patch_length,
        patch_stride=config.patch_stride,
        local_files_only=local_files_only,
    ).to(device)
    trainable, total = count_trainable(model)
    lr = config.lr_lora if model_name in LORA_VARIANTS else config.lr_head
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict] = []
    best_state = trainable_state_dict(model)
    best_val_acc = -1.0
    train_iter = iter(loaders["train"])
    started = time.perf_counter()

    for step in range(1, config.iterations + 1):
        try:
            batch_x, batch_y = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"])
            batch_x, batch_y = next(train_iter)
        model.train()
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        use_amp = device.type == "cuda"
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(x_enc=batch_x)
                loss = F.cross_entropy(outputs.logits.float(), batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x_enc=batch_x)
            loss = F.cross_entropy(outputs.logits, batch_y)
            loss.backward()
            optimizer.step()

        should_eval = step == 1 or step == config.iterations or step % config.eval_every == 0
        row = {
            "step": step,
            "loss": float(loss.detach().cpu().item()),
            "val_ACC_known": float("nan"),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        if should_eval:
            val = collect_predictions(model, loaders["val"], device)
            val_acc = float(np.mean(val["labels"] == val["preds"]) * 100.0)
            row["val_ACC_known"] = val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = trainable_state_dict(model)
        history.append(row)
        print(
            json.dumps(
                {
                    "event": "train_step",
                    "experiment": exp_key,
                    "model": model_name,
                    "step": step,
                    "loss": row["loss"],
                    "val_ACC_known": row["val_ACC_known"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    load_trainable_state_dict(model, best_state)
    final_metrics = evaluate_all(
        model,
        loaders,
        device,
        num_classes=num_classes,
        prototype_limit=config.prototype_limit,
        known_accept_rate=config.known_accept_rate,
    )
    model_dir = output_dir / exp_key / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model_name,
            "experiment": exp_key,
            "num_classes": num_classes,
            "trainable_state_dict": best_state,
            "config": asdict(config),
        },
        model_dir / "checkpoint_trainable.pt",
    )
    write_history(model_dir / "history.csv", history)
    metrics_payload = {
        "experiment": exp_key,
        "model": model_name,
        "lora_variant": LORA_VARIANTS.get(model_name),
        "best_val_ACC_known": best_val_acc,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": (trainable / total) * 100.0,
        "training_seconds": time.perf_counter() - started,
        **final_metrics,
    }
    write_json(model_dir / "metrics.json", metrics_payload)
    return metrics_payload


def prepare_loaders(
    dataset_root: Path,
    exp_key: str,
    config: RunConfig,
) -> tuple[dict[str, DataLoader], int, dict[str, int]]:
    exp_root = dataset_root / exp_key
    feature_ranges = read_feature_ranges(dataset_root, exp_key)
    train_df = load_index(exp_root, "train", known_only=True)
    val_df = load_index(exp_root, "val", known_only=True)
    test_df = load_index(exp_root, "test", known_only=True)
    out_df = load_index(exp_root, "out/test", unknown_only=True)
    train_df = sample_per_class(train_df, config.samples_per_class, config.seed)
    train_df = limit_frame(train_df, config.train_limit, config.seed)
    val_df = limit_frame(val_df, config.eval_limit, config.seed + 1)
    test_df = limit_frame(test_df, config.eval_limit, config.seed + 2)
    out_df = limit_frame(out_df, config.eval_limit, config.seed + 3)
    num_classes = int(train_df["mapped_label"].max()) + 1
    loaders = {
        "train": build_loader(
            exp_root,
            train_df,
            window_length=config.window_length,
            feature_ranges=feature_ranges,
            batch_size=config.batch_size,
            shuffle=True,
        ),
        "prototype": build_loader(
            exp_root,
            train_df,
            window_length=config.window_length,
            feature_ranges=feature_ranges,
            batch_size=config.batch_size,
            shuffle=False,
        ),
        "val": build_loader(
            exp_root,
            val_df,
            window_length=config.window_length,
            feature_ranges=feature_ranges,
            batch_size=config.batch_size,
            shuffle=False,
        ),
        "test": build_loader(
            exp_root,
            test_df,
            window_length=config.window_length,
            feature_ranges=feature_ranges,
            batch_size=config.batch_size,
            shuffle=False,
        ),
        "out": build_loader(
            exp_root,
            out_df,
            window_length=config.window_length,
            feature_ranges=feature_ranges,
            batch_size=config.batch_size,
            shuffle=False,
        ),
    }
    sizes = {name: len(loader.dataset) for name, loader in loaders.items()}
    return loaders, num_classes, sizes


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_config_file(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--experiments", default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument("--patch-length", type=int, default=None)
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--prototype-limit", type=int, default=None)
    parser.add_argument("--lr-head", type=float, default=None)
    parser.add_argument("--lr-lora", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--known-accept-rate", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--startup-dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    file_config = load_config_file(Path(args.config) if args.config else None)
    base_dir = Path(__file__).resolve().parents[1]
    defaults = {
        "dataset_root": r"D:\keyan\ai_research_workflow_base\13_paper_aligned_repro\datasets\pr_rpad_paper_aligned_v6",
        "experiments": ["exp4", "exp5", "exp6"],
        "models": ["moment_head", "moment_lora_rank8"],
        "output_dir": str(base_dir / "results" / "rejection_safe_lora_v0"),
        "cache_dir": str(base_dir / "hf_cache"),
        "seed": 20260414,
        "window_length": 32,
        "patch_length": 8,
        "patch_stride": 8,
        "iterations": 60,
        "eval_every": 10,
        "batch_size": 32,
        "samples_per_class": 1024,
        "train_limit": None,
        "eval_limit": None,
        "prototype_limit": 4096,
        "lr_head": 1e-3,
        "lr_lora": 5e-4,
        "weight_decay": 1e-4,
        "known_accept_rate": 0.95,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    merged = {**defaults, **file_config}
    cli_overrides = {
        "dataset_root": args.dataset_root,
        "experiments": parse_list(args.experiments) if args.experiments else None,
        "models": parse_list(args.models) if args.models else None,
        "output_dir": args.output_dir,
        "cache_dir": args.cache_dir,
        "seed": args.seed,
        "window_length": args.window_length,
        "patch_length": args.patch_length,
        "patch_stride": args.patch_stride,
        "iterations": args.iterations,
        "eval_every": args.eval_every,
        "batch_size": args.batch_size,
        "samples_per_class": args.samples_per_class,
        "train_limit": args.train_limit,
        "eval_limit": args.eval_limit,
        "prototype_limit": args.prototype_limit,
        "lr_head": args.lr_head,
        "lr_lora": args.lr_lora,
        "weight_decay": args.weight_decay,
        "known_accept_rate": args.known_accept_rate,
        "device": args.device,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            merged[key] = value
    if args.run_id:
        merged["output_dir"] = str(Path(merged["output_dir"]) / args.run_id)
    config = RunConfig(**merged)
    invalid = sorted(set(config.models) - MODEL_CHOICES)
    if invalid:
        raise ValueError(f"Unsupported models: {invalid}")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    set_seed(config.seed)
    dataset_root = Path(config.dataset_root)
    output_dir = Path(config.output_dir)
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": asdict(config),
        "model_repo": MODEL_REPO,
        "local_files_only": bool(args.local_files_only),
        "startup_dry_run": bool(args.startup_dry_run),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(output_dir / "run_manifest.json", manifest)

    if args.startup_dry_run:
        checks = {
            "dataset_root_exists": dataset_root.exists(),
            "experiments": {},
            "cuda_available": torch.cuda.is_available(),
        }
        for exp_key in config.experiments:
            exp_root = dataset_root / exp_key
            checks["experiments"][exp_key] = {
                "exp_root_exists": exp_root.exists(),
                "train_index_exists": (exp_root / "mixed" / "train" / "pulse_index.csv").exists(),
                "val_index_exists": (exp_root / "mixed" / "val" / "pulse_index.csv").exists(),
                "test_index_exists": (exp_root / "mixed" / "test" / "pulse_index.csv").exists(),
                "out_index_exists": (exp_root / "out" / "test" / "pulse_index.csv").exists(),
            }
        write_json(output_dir / "startup_dry_run.json", checks)
        print(json.dumps(checks, indent=2, ensure_ascii=False))
        return 0

    summary_rows: list[dict] = []
    for exp_key in config.experiments:
        loaders, num_classes, sizes = prepare_loaders(dataset_root, exp_key, config)
        write_json(output_dir / exp_key / "dataset_sizes.json", {"num_classes": num_classes, **sizes})
        print(json.dumps({"event": "dataset_ready", "experiment": exp_key, "num_classes": num_classes, "sizes": sizes}, ensure_ascii=False), flush=True)
        for model_name in config.models:
            result = train_one(
                model_name=model_name,
                exp_key=exp_key,
                num_classes=num_classes,
                loaders=loaders,
                output_dir=output_dir,
                config=config,
                local_files_only=bool(args.local_files_only),
            )
            summary_rows.append(result)
            write_json(output_dir / "latest_summary.json", {"rows": summary_rows})

    summary_path = output_dir / "summary_metrics.csv"
    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = sorted({key for row in summary_rows for key in row.keys()})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    write_json(output_dir / "run_manifest.json", {**manifest, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "summary_metrics": str(summary_path)})
    print(json.dumps({"event": "finished", "summary_metrics": str(summary_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
