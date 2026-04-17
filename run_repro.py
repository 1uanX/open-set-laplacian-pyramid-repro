from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from copy import deepcopy

from torch import nn
from torch.utils.data import DataLoader, Dataset

from repro_dataset import EXPERIMENTS, PAPER_TARGETS, PulseGraphConverter, ReproConfig, build_all_datasets, load_sequence
from repro_model import ModelConfig, PRRPADModel

EXPERIMENT_SEEDS = {
    "exp1": 20260417,
    "exp2": 20260414,
    "exp3": 20260418,
    "exp4": 20260417,
    "exp5": 20260418,
    "exp6": 20260414,
}


class PulseIndexDataset(Dataset):
    def __init__(self, dataset_root: Path, index_df: pd.DataFrame, converter: PulseGraphConverter) -> None:
        self.dataset_root = dataset_root
        self.index_df = index_df.reset_index(drop=True)
        self.converter = converter
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index_df)

    def _resolve_sequence_path(self, rel_path: str) -> Path:
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
    index_path = dataset_root / "mixed" / split / "pulse_index.csv"
    frame = pd.read_csv(index_path)
    if unknown_only is True:
        return frame.loc[frame["is_unknown"] == 1].copy()
    if unknown_only is False:
        return frame.loc[frame["mapped_label"] >= 0].copy()
    return frame


def build_loaders(dataset_root: Path, batch_size: int, converter: PulseGraphConverter) -> dict[str, DataLoader]:
    train_df = _filtered_index(dataset_root, "train", unknown_only=False)
    val_df = _filtered_index(dataset_root, "val", unknown_only=False)
    test_df = _filtered_index(dataset_root, "test", unknown_only=False)
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


def build_converter(dataset_root: Path) -> PulseGraphConverter:
    config_payload = json.loads((dataset_root / "config.json").read_text(encoding="utf-8"))
    feature_ranges = config_payload["experiment"].get("global_ranges", {})
    return PulseGraphConverter(lenwindow=1, output_size=65, feature_ranges=feature_ranges)


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


def train_closed_set(model: PRRPADModel, loaders: dict[str, DataLoader], device: torch.device, *, iterations: int, lr: float) -> tuple[list[dict], dict[str, torch.Tensor]]:
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    history = []
    model.to(device)
    train_iter = iter(loaders["train"])
    losses = []
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
        out["loss"].backward()
        optim.step()
        losses.append(float(out["loss"].item()))
        if iteration % eval_every == 0 or iteration == iterations:
            metrics = evaluate_closed_set(model, loaders["val"], device, model.config.num_classes)
            score = metrics["accuracy"] + metrics["mean_iou"]
            if score >= best_score:
                best_score = score
                best_state = deepcopy(model.state_dict())
            history.append({"iteration": iteration, "train_loss": float(np.mean(losses[-eval_every:])), **metrics})
    return history, best_state


def train_open_set(model: PRRPADModel, loaders: dict[str, DataLoader], device: torch.device, *, iterations: int, lr: float, gen_lr: float) -> tuple[list[dict], dict[str, torch.Tensor]]:
    classifier_params = list(model.backbone.parameters()) + list(model.scale1.parameters()) + list(model.scale2.parameters()) + list(model.scale3.parameters()) + list(model.head1.parameters()) + list(model.head2.parameters()) + list(model.head3.parameters())
    opt_c = torch.optim.SGD(classifier_params, lr=lr, momentum=0.9)
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=gen_lr)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=gen_lr)
    bce = nn.BCEWithLogitsLoss()
    history = []
    model.to(device)
    train_iter = iter(loaders["train"])
    classifier_losses = []
    eval_every = max(1, iterations // 10)
    best_score = float("-inf")
    best_state = deepcopy(model.state_dict())
    for iteration in range(1, iterations + 1):
        model.train()
        train_iter, (features, labels) = _next_batch(train_iter, loaders["train"])
        features = features.to(device)
        labels = labels.to(device)
        batch_size = features.size(0)

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
        fake = model.generator(torch.randn(batch_size, model.config.latent_dim, device=device)).detach()
        fake_probs = model.forward_unknown(fake)["probs"]
        c_loss = known_out["loss"] - model.config.beta_entropy * model.entropy(fake_probs)
        c_loss.backward()
        opt_c.step()
        classifier_losses.append(float(c_loss.item()))

        if iteration % eval_every == 0 or iteration == iterations:
            metrics = evaluate_open_set(model, loaders["val"], loaders["out"], device)
            score = metrics["ACC_known"] + metrics["OSCR"]
            if score >= best_score:
                best_score = score
                best_state = deepcopy(model.state_dict())
            history.append({"iteration": iteration, "train_loss": float(np.mean(classifier_losses[-eval_every:])), **metrics})
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


def plot_history(workspace: Path, experiment_key: str, history: list[dict], open_set: bool) -> None:
    out_dir = workspace / "output" / "figures"
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
) -> dict:
    set_seed(EXPERIMENT_SEEDS.get(experiment_key, 20260417))
    dataset_root = workspace / "artifacts" / "datasets" / "pr_rpad_v1" / experiment_key
    converter = build_converter(dataset_root)
    loaders = build_loaders(dataset_root, batch_size, converter)
    sample_dim = int(loaders["train"].dataset[0][0].numel())
    model = PRRPADModel(ModelConfig(num_classes=_num_classes(dataset_root), input_dim=sample_dim))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_open_set = any(experiment.open_set for experiment in EXPERIMENTS if f"exp{experiment.experiment_id}" == experiment_key)

    if is_open_set:
        history, best_state = train_open_set(model, loaders, device, iterations=iterations, lr=lr, gen_lr=gen_lr)
        model.load_state_dict(best_state)
        final_metrics = evaluate_open_set(model, loaders["test"], loaders["out"], device)
    else:
        history, best_state = train_closed_set(model, loaders, device, iterations=iterations, lr=lr)
        model.load_state_dict(best_state)
        final_metrics = evaluate_closed_set(model, loaders["test"], device, model.config.num_classes)
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
    plot_history(workspace, experiment_key, history, is_open_set)

    output_dir = workspace / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / f"{experiment_key}_history.csv", index=False)
    if len(specific_df):
        specific_df.to_csv(output_dir / f"{experiment_key}_specific_metrics.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "model_config": asdict(model.config)}, output_dir / f"{experiment_key}_checkpoint.pt")

    paper_target = PAPER_TARGETS.get(experiment_key, {})
    return {
        "experiment": experiment_key,
        "open_set": is_open_set,
        "iterations": iterations,
        "device": str(device),
        "final_metrics": final_metrics,
        "paper_target": paper_target,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--experiments", default="exp1,exp4")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gen-lr", type=float, default=0.0002)
    parser.add_argument("--force-datasets", action="store_true")
    parser.add_argument("--skip-specific", action="store_true")
    parser.add_argument("--specific-limit", type=int, default=None)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    iterations = args.epochs if args.epochs is not None else args.iterations
    dataset_paths = build_all_datasets(workspace, ReproConfig(), force=args.force_datasets)
    selected = [item.strip() for item in args.experiments.split(",") if item.strip()]
    reports = []
    for experiment_key in selected:
        if experiment_key not in dataset_paths:
            raise ValueError(f"Unknown experiment key: {experiment_key}")
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
            )
        )

    summary = {
        "workspace": str(workspace),
        "dataset_root": str(workspace / "artifacts" / "datasets" / "pr_rpad_v1"),
        "reports": reports,
    }
    output_path = workspace / "output" / "reproduction_report.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
