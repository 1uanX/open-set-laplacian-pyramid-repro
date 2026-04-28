from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_index(exp_root: Path, split: str) -> pd.DataFrame:
    candidates = [
        exp_root / split / "pulse_index.csv",
        exp_root / "mixed" / split / "pulse_index.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError(f"Cannot find pulse_index.csv for {exp_root} split={split}")


def _write_index(root: Path, split: str, frame: pd.DataFrame) -> None:
    out_dir = root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "pulse_index.csv", index=False)


def _fewshot_train(frame: pd.DataFrame, samples_per_class: int, seed: int) -> pd.DataFrame:
    known = frame.loc[frame["mapped_label"] >= 0].copy()
    rows = []
    for label, group in known.groupby("mapped_label", sort=True):
        if len(group) < samples_per_class:
            raise ValueError(f"Class {label} has only {len(group)} rows; requested {samples_per_class}.")
        rows.append(group.sample(n=samples_per_class, random_state=seed + int(label)))
    return pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--experiments", default="exp4,exp5,exp6")
    parser.add_argument("--samples-per-class", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260428)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "samples_per_class": args.samples_per_class,
        "seed": args.seed,
        "experiments": {},
        "protocol": {
            "train": "fixed known-only few-shot pulse_index sampled per mapped_label",
            "val": "full known validation pulse_index copied from source",
            "test": "full known test pulse_index copied from source",
            "out": "full true unknown out/test pulse_index copied from source",
            "sequence_files": "not copied; sequence_path remains absolute source path",
        },
    }

    for exp_key in experiments:
        exp_src = source_root / exp_key
        exp_dst = output_root / exp_key
        train = _read_index(exp_src, "train")
        val = _read_index(exp_src, "val")
        test = _read_index(exp_src, "test")
        out = _read_index(exp_src, "out/test")

        train_few = _fewshot_train(train, args.samples_per_class, args.seed)
        val_known = val.loc[val["mapped_label"] >= 0].reset_index(drop=True)
        test_known = test.loc[test["mapped_label"] >= 0].reset_index(drop=True)
        out_unknown = out.loc[out["is_unknown"] == 1].reset_index(drop=True)

        _write_index(exp_dst, "mixed/train", train_few)
        _write_index(exp_dst, "mixed/val", val_known)
        _write_index(exp_dst, "mixed/test", test_known)
        _write_index(exp_dst, "out/test", out_unknown)

        label_counts = train_few.groupby("mapped_label").size().astype(int).to_dict()
        manifest["experiments"][exp_key] = {
            "num_classes": int(train_few["mapped_label"].max()) + 1,
            "train_rows": int(len(train_few)),
            "val_rows": int(len(val_known)),
            "test_rows": int(len(test_known)),
            "out_unknown_rows": int(len(out_unknown)),
            "train_label_counts": {str(k): int(v) for k, v in label_counts.items()},
        }

    (output_root / "fewshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
