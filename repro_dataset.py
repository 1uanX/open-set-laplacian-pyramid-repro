from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class ParameterSpec:
    low: float
    high: float
    mode: str


@dataclass(frozen=True)
class EmitterMode:
    rf: ParameterSpec
    pw: ParameterSpec
    pri: ParameterSpec
    pa: ParameterSpec
    doa_low: float
    doa_high: float


@dataclass(frozen=True)
class EmitterConfig:
    tag: int
    known: bool
    modes: tuple[EmitterMode, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: int
    name: str
    open_set: bool
    emitters: tuple[EmitterConfig, ...]


@dataclass(frozen=True)
class ReproConfig:
    mixed_sequences: int = 200
    sequence_length: int = 400
    specific_sequences: int = 100
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    lenwindow: int = 1
    pdg_size: int = 65
    pulse_loss_range: tuple[float, float] = (0.0, 0.5)
    noise_ratio_range: tuple[float, float] = (0.0, 0.5)
    specific_noise_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    seed: int = 20260417


def _mode(rf: tuple[float, float, str], pw: tuple[float, float, str], pri: tuple[float, float, str], pa: tuple[float, float, str], doa: tuple[float, float]) -> EmitterMode:
    return EmitterMode(
        rf=ParameterSpec(*rf),
        pw=ParameterSpec(*pw),
        pri=ParameterSpec(*pri),
        pa=ParameterSpec(*pa),
        doa_low=doa[0],
        doa_high=doa[1],
    )


EXPERIMENTS: tuple[ExperimentConfig, ...] = (
    ExperimentConfig(
        experiment_id=1,
        name="exp1_closed_single_function",
        open_set=False,
        emitters=(
            EmitterConfig(1, True, (_mode((9650, 9950, "F"), (7.2, 8.6, "F"), (61, 92, "S"), (-45, -38, "M"), (79, 84)),)),
            EmitterConfig(2, True, (_mode((9820, 10200, "A"), (1.2, 3.8, "S"), (28, 32, "DS"), (-42, -35, "P1"), (76, 81)),)),
            EmitterConfig(3, True, (_mode((1205, 1450, "S"), (4.2, 5.7, "A"), (59.1, 59.1, "F"), (-54, -49, "P2"), (80, 81)),)),
            EmitterConfig(4, True, (_mode((8200, 12300, "F"), (1.4, 6.2, "F"), (18.7, 48.7, "S"), (-51, -43, "P1"), (82, 88)),)),
        ),
    ),
    ExperimentConfig(
        experiment_id=2,
        name="exp2_closed_multifunction",
        open_set=False,
        emitters=(
            EmitterConfig(1, True, (_mode((9650, 9950, "F"), (7.2, 8.6, "F"), (61, 92, "S"), (-45, -38, "M"), (79, 84)),)),
            EmitterConfig(2, True, (_mode((9820, 10200, "A"), (1.2, 3.8, "S"), (28, 32, "DS"), (-42, -35, "P1"), (76, 81)),)),
            EmitterConfig(3, True, (_mode((9650, 9950, "F"), (7.2, 8.6, "F"), (61, 92, "S"), (-40, -33, "M"), (75, 80)),)),
            EmitterConfig(4, True, (_mode((1205, 1450, "S"), (4.2, 5.7, "A"), (59.1, 59.1, "F"), (-54, -49, "P2"), (80, 81)),)),
            EmitterConfig(
                5,
                True,
                (
                    _mode((8200, 10100, "F"), (1.4, 4.5, "F"), (18.7, 28.7, "S"), (-51, -43, "P1"), (82, 88)),
                    _mode((10200, 12300, "S"), (3.2, 6.2, "A"), (29, 68.7, "DS"), (-44, -37, "P1"), (82, 88)),
                ),
            ),
        ),
    ),
    ExperimentConfig(
        experiment_id=3,
        name="exp3_closed_jittered",
        open_set=False,
        emitters=(
            EmitterConfig(1, True, (_mode((9650, 9950, "F"), (7.2, 8.6, "F"), (61, 92, "S"), (-45, -38, "M"), (79, 84)),)),
            EmitterConfig(
                2,
                True,
                (
                    _mode((9820, 10200, "A"), (1.2, 3.8, "S"), (28, 32, "DS"), (-42, -35, "P1"), (76, 81)),
                    _mode((10800, 11200, "F"), (2.2, 6.8, "A"), (18, 46, "J"), (-45, -38, "P1"), (76, 81)),
                ),
            ),
            EmitterConfig(3, True, (_mode((9650, 9950, "F"), (7.2, 8.6, "F"), (61, 92, "S"), (-40, -33, "M"), (75, 80)),)),
            EmitterConfig(4, True, (_mode((1205, 1450, "S"), (4.2, 5.7, "A"), (59.1, 59.1, "F"), (-54, -49, "P2"), (80, 81)),)),
            EmitterConfig(
                5,
                True,
                (
                    _mode((8200, 9100, "F"), (1.4, 4.5, "F"), (18.7, 28.7, "S"), (-51, -43, "P1"), (82, 88)),
                    _mode((9120, 10200, "S"), (3.2, 5.2, "A"), (29, 68.7, "DS"), (-44, -37, "P1"), (82, 88)),
                    _mode((11200, 12300, "A"), (4.1, 6.4, "S"), (21.4, 34.7, "J"), (-48, -45, "P1"), (82, 88)),
                ),
            ),
            EmitterConfig(6, True, (_mode((5400, 5870, "S"), (1.1, 11.2, "F"), (990, 1007, "F"), (-41, -34, "P2"), (65, 69)),)),
            EmitterConfig(7, True, (_mode((3100, 3490, "A"), (22.4, 29.1, "F"), (857, 1250, "J"), (-39, -32, "P1"), (85, 86)),)),
        ),
    ),
    ExperimentConfig(
        experiment_id=4,
        name="exp4_open_single_function",
        open_set=True,
        emitters=(
            EmitterConfig(1, True, (_mode((9650, 9950, "F"), (7.2, 8.6, "F"), (61, 92, "S"), (-45, -38, "M"), (79, 84)),)),
            EmitterConfig(2, False, (_mode((9820, 10300, "A"), (1.2, 3.8, "S"), (28, 32, "DS"), (-42, -35, "P1"), (79, 81)),)),
            EmitterConfig(3, True, (_mode((1205, 1450, "S"), (4.2, 5.7, "A"), (59.1, 59.1, "F"), (-54, -49, "P2"), (80, 81)),)),
            EmitterConfig(4, True, (_mode((8200, 12300, "F"), (1.4, 6.2, "F"), (18.7, 48.7, "S"), (-51, -43, "P1"), (82, 88)),)),
        ),
    ),
    ExperimentConfig(
        experiment_id=5,
        name="exp5_open_multifunction",
        open_set=True,
        emitters=(
            EmitterConfig(1, True, (_mode((9650, 9950, "F"), (7.2, 8.6, "F"), (61, 92, "S"), (-45, -38, "M"), (79, 84)),)),
            EmitterConfig(2, False, (_mode((9820, 10200, "A"), (1.2, 3.8, "S"), (28, 32, "DS"), (-42, -35, "P1"), (76, 81)),)),
            EmitterConfig(3, False, (_mode((9650, 9950, "F"), (7.2, 8.6, "F"), (61, 92, "S"), (-40, -33, "M"), (75, 80)),)),
            EmitterConfig(4, True, (_mode((1205, 1450, "S"), (4.2, 5.7, "A"), (59.1, 59.1, "F"), (-54, -49, "P2"), (80, 81)),)),
            EmitterConfig(
                5,
                True,
                (
                    _mode((8200, 10100, "F"), (1.4, 4.5, "F"), (18.7, 28.7, "S"), (-51, -43, "P1"), (82, 88)),
                    _mode((10200, 12300, "S"), (3.2, 6.2, "A"), (29, 68.7, "DS"), (-44, -37, "P1"), (82, 88)),
                ),
            ),
        ),
    ),
    ExperimentConfig(
        experiment_id=6,
        name="exp6_open_jittered",
        open_set=True,
        emitters=(
            EmitterConfig(1, True, (_mode((9650, 9950, "F"), (7.2, 8.6, "F"), (61, 92, "S"), (-45, -38, "M"), (79, 84)),)),
            EmitterConfig(
                2,
                False,
                (
                    _mode((9820, 10200, "A"), (1.2, 3.8, "S"), (28, 32, "DS"), (-42, -35, "P1"), (76, 81)),
                    _mode((10800, 11200, "F"), (2.2, 6.8, "A"), (18, 46, "J"), (-45, -38, "P1"), (76, 81)),
                ),
            ),
            EmitterConfig(3, False, (_mode((9650, 9950, "F"), (7.2, 8.6, "F"), (61, 92, "S"), (-40, -33, "M"), (75, 80)),)),
            EmitterConfig(4, True, (_mode((1205, 1450, "S"), (4.2, 5.7, "A"), (59.1, 59.1, "F"), (-54, -49, "P2"), (80, 81)),)),
            EmitterConfig(
                5,
                True,
                (
                    _mode((8200, 9100, "F"), (1.4, 4.5, "F"), (18.7, 28.7, "S"), (-51, -43, "P1"), (82, 88)),
                    _mode((9120, 10200, "S"), (3.2, 5.2, "A"), (29, 68.7, "DS"), (-44, -37, "P1"), (82, 88)),
                    _mode((11200, 12300, "A"), (4.1, 6.4, "S"), (21.4, 34.7, "J"), (-48, -45, "P1"), (82, 88)),
                ),
            ),
            EmitterConfig(6, False, (_mode((5400, 5870, "S"), (1.1, 11.2, "F"), (990, 1007, "F"), (-41, -34, "P2"), (65, 69)),)),
            EmitterConfig(7, True, (_mode((3100, 3490, "A"), (22.4, 29.1, "F"), (857, 1250, "J"), (-39, -32, "P1"), (85, 86)),)),
        ),
    ),
)


PAPER_TARGETS = {
    "exp1": {"ACC_closed_set": 100.000, "MR": 100.000, "MP": 100.000, "MIOU": 100.000},
    "exp2": {"ACC_closed_set": 99.382, "MR": 99.020, "MP": 99.043, "MIOU": 98.098},
    "exp3": {"ACC_closed_set": 98.184, "MR": 97.763, "MP": 98.272, "MIOU": 96.226},
    "exp4": {"ACC_known": 99.983, "OSCR": 99.966},
    "exp5": {"ACC_known": 100.000, "OSCR": 82.391},
    "exp6": {"ACC_known": 99.613, "OSCR": 72.140},
}


def _sample_mode_schedule(emitter: EmitterConfig, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(emitter.modes) == 1:
        return np.zeros(count, dtype=int)
    min_dwell = max(20, count // (len(emitter.modes) * 4))
    schedule = []
    mode_index = int(rng.integers(0, len(emitter.modes)))
    while len(schedule) < count:
        dwell = int(rng.integers(min_dwell, min_dwell * 2 + 1))
        schedule.extend([mode_index] * dwell)
        mode_index = (mode_index + 1) % len(emitter.modes)
    return np.asarray(schedule[:count], dtype=int)


def _smooth_doa(low: float, high: float, count: int, rng: np.random.Generator) -> np.ndarray:
    base = np.linspace(low, high, count, dtype=float)
    if count <= 3:
        return base
    wobble = np.sin(np.linspace(0, 2 * np.pi, count, dtype=float) + float(rng.uniform(0.0, np.pi))) * (high - low) * 0.12
    return np.clip(base + wobble, low, high)


def _scan_pa(low: float, high: float, mode: str, count: int, rng: np.random.Generator) -> np.ndarray:
    phase = float(rng.uniform(0.0, 2 * np.pi))
    grid = np.linspace(0.0, 1.0, count, dtype=float)
    if mode == "M":
        values = 0.5 * (1.0 + np.sin(2 * np.pi * grid + phase))
    elif mode == "P1":
        values = 2.0 * np.abs(((2 * grid + phase / (2 * np.pi)) % 1.0) - 0.5)
    else:
        values = 0.5 * (1.0 + np.sin(2 * np.pi * grid + phase) * np.cos(4 * np.pi * grid + phase / 2.0))
    return low + values * (high - low)


def _sample_parameter(spec: ParameterSpec, count: int, rng: np.random.Generator) -> np.ndarray:
    low, high, mode = spec.low, spec.high, spec.mode
    if mode == "F":
        return np.full(count, float(rng.uniform(low, high)), dtype=float)
    if mode == "A":
        return rng.uniform(low, high, size=count).astype(float)
    if mode == "S":
        level_count = 2 if abs(high - low) <= 20 else 3
        levels = np.linspace(low, high, num=level_count, dtype=float)
        jitter = rng.normal(0.0, max((high - low) * 0.03, 1e-3), size=levels.shape)
        values = np.clip(levels + jitter, low, high)
        return np.resize(values, count).astype(float)
    if mode == "DS":
        level_count = 2 if abs(high - low) <= 20 else 3
        levels = np.linspace(low, high, num=level_count, dtype=float)
        dwell = max(8, count // (level_count * 3))
        chunks: list[float] = []
        for level in levels:
            chunks.extend([float(level)] * dwell)
        tiled = np.resize(np.asarray(chunks, dtype=float), count)
        return tiled
    if mode == "J":
        center = 0.5 * (low + high)
        sigma = max((high - low) / 6.0, 1e-3)
        values = rng.normal(center, sigma, size=count)
        return np.clip(values, low, high).astype(float)
    raise ValueError(f"Unsupported parameter mode: {mode}")


def _emit_emitter_pulses(emitter: EmitterConfig, count: int, rng: np.random.Generator) -> pd.DataFrame:
    mode_schedule = _sample_mode_schedule(emitter, count, rng)
    frames: list[pd.DataFrame] = []
    start = 0
    while start < count:
        mode_index = int(mode_schedule[start])
        end = start
        while end < count and int(mode_schedule[end]) == mode_index:
            end += 1
        mode = emitter.modes[mode_index]
        segment_count = end - start
        pri = _sample_parameter(mode.pri, segment_count, rng)
        toa = np.cumsum(pri, dtype=float) + float(rng.uniform(0.0, max(mode.pri.high, 1.0)))
        rf = _sample_parameter(mode.rf, segment_count, rng)
        pw = _sample_parameter(mode.pw, segment_count, rng)
        pa = _scan_pa(mode.pa.low, mode.pa.high, mode.pa.mode, segment_count, rng)
        doa = _smooth_doa(mode.doa_low, mode.doa_high, segment_count, rng)
        frames.append(
            pd.DataFrame(
                {
                    "tag": emitter.tag,
                    "known": emitter.known,
                    "mode_index": mode_index,
                    "toa": toa + (frames[-1]["toa"].iloc[-1] if frames else 0.0),
                    "rf": rf,
                    "pw": pw,
                    "pri": pri,
                    "pa": pa,
                    "doa": doa,
                }
            )
        )
        start = end
    return pd.concat(frames, ignore_index=True)


def _global_ranges(experiment: ExperimentConfig) -> dict[str, tuple[float, float]]:
    rf_vals = []
    pw_vals = []
    pri_vals = []
    pa_vals = []
    doa_vals = []
    for emitter in experiment.emitters:
        for mode in emitter.modes:
            rf_vals.extend([mode.rf.low, mode.rf.high])
            pw_vals.extend([mode.pw.low, mode.pw.high])
            pri_vals.extend([mode.pri.low, mode.pri.high])
            pa_vals.extend([mode.pa.low, mode.pa.high])
            doa_vals.extend([mode.doa_low, mode.doa_high])
    return {
        "rf": (min(rf_vals), max(rf_vals)),
        "pw": (min(pw_vals), max(pw_vals)),
        "pri": (min(pri_vals), max(pri_vals)),
        "pa": (min(pa_vals), max(pa_vals)),
        "doa": (min(doa_vals), max(doa_vals)),
    }


def _noise_frame(experiment: ExperimentConfig, count: int, rng: np.random.Generator) -> pd.DataFrame:
    ranges = _global_ranges(experiment)
    toa = np.sort(rng.uniform(0.0, max(ranges["pri"][1] * count / 3.0, 1.0), size=count))
    return pd.DataFrame(
        {
            "tag": 0,
            "known": True,
            "mode_index": -1,
            "toa": toa,
            "rf": rng.uniform(*ranges["rf"], size=count),
            "pw": rng.uniform(*ranges["pw"], size=count),
            "pri": np.nan,
            "pa": rng.uniform(*ranges["pa"], size=count),
            "doa": rng.uniform(*ranges["doa"], size=count),
        }
    )


def _build_sequence(experiment: ExperimentConfig, config: ReproConfig, pulse_loss_rate: float, noise_ratio: float, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    active_emitters = list(experiment.emitters)
    radar_count = len(active_emitters)
    observed_per_emitter = max(24, int(config.sequence_length // max(radar_count + noise_ratio, 1.0)))
    noise_count = max(0, config.sequence_length - radar_count * observed_per_emitter)
    emitter_frames = []
    for emitter in active_emitters:
        pre_loss_count = int(np.ceil(observed_per_emitter / max(1.0 - pulse_loss_rate, 0.1)))
        frame = _emit_emitter_pulses(emitter, pre_loss_count, rng)
        keep = rng.choice(len(frame), size=observed_per_emitter, replace=False)
        frame = frame.iloc[np.sort(keep)].reset_index(drop=True)
        emitter_frames.append(frame)
    noise_frame = _noise_frame(experiment, noise_count, rng)
    full = pd.concat([*emitter_frames, noise_frame], ignore_index=True)
    full = full.sort_values("toa", kind="mergesort").reset_index(drop=True)
    if len(full) > config.sequence_length:
        full = full.iloc[: config.sequence_length].copy()
    elif len(full) < config.sequence_length:
        extra = _noise_frame(experiment, config.sequence_length - len(full), rng)
        full = pd.concat([full, extra], ignore_index=True).sort_values("toa", kind="mergesort").reset_index(drop=True)
    full["pulse_index"] = np.arange(len(full), dtype=int)
    full["dtoa"] = np.concatenate([[0.0], np.diff(full["toa"].to_numpy(dtype=float))])
    known_tags = [emitter.tag for emitter in experiment.emitters if emitter.known]
    unknown_tags = [emitter.tag for emitter in experiment.emitters if not emitter.known]
    class_tags = [*known_tags]
    class_map = {tag: idx for idx, tag in enumerate(class_tags)}
    full["mapped_label"] = full["tag"].map(class_map).fillna(-1).astype(int)
    full["is_unknown"] = full["tag"].isin(unknown_tags).astype(int)
    full["is_noise"] = (full["tag"] == 0).astype(int)
    metadata = {
        "pulse_loss_rate": pulse_loss_rate,
        "noise_ratio": noise_ratio,
        "known_tags": known_tags,
        "unknown_tags": unknown_tags,
        "class_tags": class_tags,
        "sequence_length": int(len(full)),
    }
    return full, metadata


def _save_sequences(root: Path, split_name: str, sequences: list[pd.DataFrame], metadatas: list[dict]) -> pd.DataFrame:
    split_dir = root / split_name / "sequences"
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (sequence, metadata) in enumerate(zip(sequences, metadatas, strict=True)):
        sequence_name = f"{split_name}_{idx:03d}"
        sequence_path = split_dir / f"{sequence_name}.csv"
        meta_path = split_dir / f"{sequence_name}.json"
        sequence.to_csv(sequence_path, index=False)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        rows.append(
            {
                "sequence_name": sequence_name,
                "path": str(sequence_path.relative_to(root)),
                "metadata_path": str(meta_path.relative_to(root)),
                "pulse_loss_rate": metadata["pulse_loss_rate"],
                "noise_ratio": metadata["noise_ratio"],
                "known_tags": ",".join(map(str, metadata["known_tags"])),
                "class_tags": ",".join(map(str, metadata["class_tags"])),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(root / split_name / "manifest.csv", index=False)
    return manifest


def _write_index(root: Path, split_name: str, manifest: pd.DataFrame) -> None:
    rows = []
    for row in manifest.itertuples(index=False):
        seq = pd.read_csv(root / row.path)
        for pulse in seq.itertuples(index=False):
            rows.append(
                {
                    "sequence_name": row.sequence_name,
                    "sequence_path": row.path,
                    "pulse_index": pulse.pulse_index,
                    "tag": pulse.tag,
                    "mapped_label": pulse.mapped_label,
                    "is_unknown": pulse.is_unknown,
                    "is_noise": pulse.is_noise,
                }
            )
    pd.DataFrame(rows).to_csv(root / split_name / "pulse_index.csv", index=False)


def _dataset_is_current(root: Path) -> bool:
    config_path = root / "config.json"
    train_index = root / "mixed" / "train" / "pulse_index.csv"
    if not config_path.exists() or not train_index.exists():
        return False
    try:
        frame = pd.read_csv(train_index, nrows=1)
    except Exception:
        return False
    return "is_noise" in frame.columns


def build_experiment_dataset(workspace: Path, experiment: ExperimentConfig, config: ReproConfig, *, force: bool = False) -> Path:
    rng = np.random.default_rng(config.seed + experiment.experiment_id * 101)
    root = workspace / "artifacts" / "datasets" / "pr_rpad_v1" / f"exp{experiment.experiment_id}"
    if not force and _dataset_is_current(root):
        return root
    root.mkdir(parents=True, exist_ok=True)

    hybrid_sequences = []
    hybrid_meta = []
    for _ in range(config.mixed_sequences):
        pulse_loss_rate = float(rng.uniform(*config.pulse_loss_range))
        noise_ratio = float(rng.uniform(*config.noise_ratio_range))
        sequence, metadata = _build_sequence(experiment, config, pulse_loss_rate, noise_ratio, rng)
        hybrid_sequences.append(sequence)
        hybrid_meta.append(metadata)

    train_end = int(round(config.mixed_sequences * config.train_ratio))
    val_end = train_end + int(round(config.mixed_sequences * config.val_ratio))
    split_map = {
        "train": (hybrid_sequences[:train_end], hybrid_meta[:train_end]),
        "val": (hybrid_sequences[train_end:val_end], hybrid_meta[train_end:val_end]),
        "test": (hybrid_sequences[val_end:], hybrid_meta[val_end:]),
    }
    split_manifests = {}
    for split_name, (sequences, metadatas) in split_map.items():
        manifest = _save_sequences(root / "mixed", split_name, sequences, metadatas)
        _write_index(root / "mixed", split_name, manifest)
        split_manifests[split_name] = manifest

    specific_root = root / "specific"
    for noise_ratio in config.specific_noise_grid:
        sequences = []
        metadatas = []
        for _ in range(config.specific_sequences):
            sequence, metadata = _build_sequence(experiment, config, 0.0, float(noise_ratio), rng)
            sequences.append(sequence)
            metadatas.append(metadata)
        split_name = f"loss0_noise{int(round(noise_ratio * 10)):02d}"
        manifest = _save_sequences(specific_root, split_name, sequences, metadatas)
        _write_index(specific_root, split_name, manifest)
    for level in config.specific_noise_grid:
        sequences = []
        metadatas = []
        for _ in range(config.specific_sequences):
            sequence, metadata = _build_sequence(experiment, config, float(level), float(level), rng)
            sequences.append(sequence)
            metadatas.append(metadata)
        split_name = f"loss{int(round(level * 10)):02d}_noise{int(round(level * 10)):02d}"
        manifest = _save_sequences(specific_root, split_name, sequences, metadatas)
        _write_index(specific_root, split_name, manifest)

    config_payload = {
        "repro_config": asdict(config),
        "experiment": {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "open_set": experiment.open_set,
            "global_ranges": _global_ranges(experiment),
            "emitters": [
                {
                    "tag": emitter.tag,
                    "known": emitter.known,
                    "modes": [asdict(mode) for mode in emitter.modes],
                }
                for emitter in experiment.emitters
            ],
        },
        "paper_targets": PAPER_TARGETS.get(f"exp{experiment.experiment_id}", {}),
    }
    (root / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    return root


def build_all_datasets(workspace: Path, config: ReproConfig, *, force: bool = False) -> dict[str, Path]:
    return {f"exp{experiment.experiment_id}": build_experiment_dataset(workspace, experiment, config, force=force) for experiment in EXPERIMENTS}


def load_sequence(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def iter_mixed_split(dataset_root: Path, split: str) -> Iterable[Path]:
    manifest = pd.read_csv(dataset_root / "mixed" / split / "manifest.csv")
    for rel in manifest["path"].tolist():
        yield dataset_root / "mixed" / rel if not str(rel).startswith("mixed") else dataset_root / rel


def otsu_threshold(values: np.ndarray) -> float:
    hist, bin_edges = np.histogram(values.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_edges[:-1]) / np.maximum(weight1, 1e-6)
    mean2 = (np.cumsum((hist * bin_edges[:-1])[::-1]) / np.maximum(weight2[::-1], 1e-6))[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    index = int(np.argmax(variance12)) if len(variance12) else 0
    return float(bin_edges[index])


class PulseGraphConverter:
    def __init__(self, *, lenwindow: int = 1, output_size: int = 65, feature_ranges: dict[str, tuple[float, float]] | None = None) -> None:
        self.lenwindow = lenwindow
        self.output_size = output_size
        self.feature_ranges = feature_ranges or {}

    def _normalize(self, sequence: pd.DataFrame) -> pd.DataFrame:
        work = sequence.loc[:, ["dtoa", "rf", "pw", "pa", "doa"]].copy()
        for column in work.columns:
            values = work[column].to_numpy(dtype=float)
            if column in self.feature_ranges:
                lo, hi = self.feature_ranges[column]
            else:
                lo = float(values.min())
                hi = float(values.max())
            if np.isclose(lo, hi):
                work[column] = 0.0
            else:
                work[column] = (values - lo) / (hi - lo)
            work[column] = work[column].clip(0.0, 1.0)
        return work

    def _window_rows(self, normalized: pd.DataFrame, center_index: int) -> np.ndarray:
        radius = (self.lenwindow - 1) // 2
        rows = []
        for offset in range(-radius, radius + 1):
            idx = min(max(center_index + offset, 0), len(normalized) - 1)
            rows.append(normalized.iloc[idx].to_numpy(dtype=float))
        return np.vstack(rows)

    def _enhance(self, matrix: np.ndarray) -> np.ndarray:
        image = Image.fromarray(np.uint8(np.clip(matrix, 0.0, 1.0) * 255.0), mode="L")
        upscaled = image.resize((self.output_size, self.output_size), resample=Image.Resampling.BICUBIC)
        enhanced = upscaled.filter(ImageFilter.UnsharpMask(radius=1, percent=180, threshold=2))
        arr = np.asarray(enhanced, dtype=np.float32) / 255.0
        threshold = otsu_threshold(arr)
        mask = (arr >= threshold).astype(np.float32)
        arr = np.clip(0.85 * arr + 0.15 * mask, 0.0, 1.0)
        return arr

    def _pairwise_matrix(self, vector: np.ndarray) -> np.ndarray:
        xmax = float(vector.max())
        xmin = float(vector.min())
        denom = max(xmax - xmin, 1e-6)
        return np.abs(vector[:, None] - vector[None, :]) / denom

    def vectorize(self, sequence: pd.DataFrame, center_index: int) -> np.ndarray:
        normalized = self._normalize(sequence)
        block = self._window_rows(normalized, center_index)
        vector = block.reshape(-1).astype(np.float32)
        matrix = self._pairwise_matrix(vector).astype(np.float32).reshape(-1)
        return np.concatenate([vector, matrix], axis=0).astype(np.float32)

    def vectorize_all(self, sequence: pd.DataFrame) -> np.ndarray:
        normalized = self._normalize(sequence)
        vectors = []
        for center_index in range(len(normalized)):
            block = self._window_rows(normalized, center_index)
            vector = block.reshape(-1).astype(np.float32)
            matrix = self._pairwise_matrix(vector).astype(np.float32).reshape(-1)
            vectors.append(np.concatenate([vector, matrix], axis=0))
        return np.asarray(vectors, dtype=np.float32)

    def transform(self, sequence: pd.DataFrame, center_index: int) -> np.ndarray:
        normalized = self._normalize(sequence)
        block = self._window_rows(normalized, center_index)
        vector = block.reshape(-1)
        matrix = self._pairwise_matrix(vector)
        return self._enhance(matrix.astype(np.float32))
