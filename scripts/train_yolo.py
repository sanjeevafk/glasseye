#!/usr/bin/env python3
"""Fine-tune an Ultralytics YOLO model and record immutable model metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from app.paths import data_root, models_root
from app.synthetic import DATASET_SEED

DEFAULT_MODEL_VERSION = "glasseye-yolo-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def selected_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def float_or_none(value: Any) -> float | None:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def metrics_report(
    metrics: Any, names: dict[int, str]
) -> tuple[dict[str, float | None], dict[str, dict[str, float | None]]]:
    values = getattr(metrics, "results_dict", {})
    summary = {
        "mAP50": float_or_none(values.get("metrics/mAP50(B)")),
        "mAP50_95": float_or_none(values.get("metrics/mAP50-95(B)")),
        "precision": float_or_none(values.get("metrics/precision(B)")),
        "recall": float_or_none(values.get("metrics/recall(B)")),
    }
    box = getattr(metrics, "box", None)
    ap50 = list(getattr(box, "ap50", [])) if box is not None else []
    ap = list(getattr(box, "ap", [])) if box is not None else []
    per_class = {
        name: {
            "mAP50": float_or_none(ap50[index]) if index < len(ap50) else None,
            "mAP50_95": float_or_none(ap[index]) if index < len(ap) else None,
        }
        for index, name in sorted(names.items())
    }
    return summary, per_class


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--base-model", type=Path, default=models_root() / "base" / "yolov8n.pt"
    )
    parser.add_argument("--if-missing", action="store_true")
    parser.add_argument("--data-yaml", type=Path, default=data_root() / "data.yaml")
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    args = parser.parse_args()

    destination = models_root() / args.model_version
    checkpoint = destination / "best.pt"
    manifest = destination / "model_manifest.json"
    if args.if_missing and checkpoint.exists() and manifest.exists():
        print(json.dumps(json.loads(manifest.read_text(encoding="utf-8")), indent=2))
        return 0

    data_yaml = args.data_yaml.resolve()
    hash_path = data_yaml.parent / "dataset_hash.txt"
    manifest_path = data_yaml.parent / "manifest.json"
    if not data_yaml.exists():
        raise SystemExit(
            f"Dataset configuration is absent at {data_yaml}. Prepare the dataset before training."
        )
    dataset_hash = (
        hash_path.read_text(encoding="utf-8").strip()
        if hash_path.exists()
        else sha256(manifest_path if manifest_path.exists() else data_yaml)
    )

    import torch
    import ultralytics
    from ultralytics import YOLO

    destination.mkdir(parents=True, exist_ok=True)
    run_parent = destination / "training_runs"
    device = selected_device(args.device)
    args.base_model.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.base_model))
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "workers": 0,
        "seed": DATASET_SEED,
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": 0.002,
        "lrf": 0.01,
        "mosaic": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.01,
        "hsv_s": 0.2,
        "hsv_v": 0.15,
        "patience": 0,
        "pretrained": True,
        "save": True,
        "plots": True,
        "project": str(run_parent),
        "name": "seeded_v1",
        "exist_ok": True,
        "verbose": True,
    }
    model.train(**train_kwargs)
    trained_checkpoint = run_parent / "seeded_v1" / "weights" / "best.pt"
    if not trained_checkpoint.exists():
        raise RuntimeError(f"Ultralytics did not produce {trained_checkpoint}.")
    shutil.copy2(trained_checkpoint, checkpoint)

    loaded = YOLO(str(checkpoint))
    validation = loaded.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        plots=True,
        project=str(run_parent),
        name="test_metrics",
        exist_ok=True,
        verbose=False,
    )
    names = {int(key): str(value) for key, value in loaded.names.items()}
    metrics, per_class = metrics_report(validation, names)
    result = {
        "model_version": args.model_version,
        "base_model": "yolov8n.pt",
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "python_version": platform.python_version(),
        "dataset_hash": dataset_hash,
        "dataset_yaml": str(data_yaml),
        "class_map": names,
        "image_size": args.imgsz,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "seed": DATASET_SEED,
        "augmentations": {
            "mosaic": 0.0,
            "fliplr": 0.5,
            "hsv_h": 0.01,
            "hsv_s": 0.2,
            "hsv_v": 0.15,
        },
        "metrics": metrics,
        "per_class_metrics": per_class,
        "best_checkpoint": str(checkpoint.resolve()),
        "best_checkpoint_sha256": sha256(checkpoint),
        "training_artifacts": str((run_parent / "seeded_v1").resolve()),
        "validation_artifacts": str((run_parent / "test_metrics").resolve()),
    }
    (destination / "metrics_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    manifest_data = {
        key: result[key]
        for key in (
            "model_version",
            "base_model",
            "ultralytics_version",
            "dataset_hash",
            "class_map",
            "best_checkpoint",
            "best_checkpoint_sha256",
        )
    }
    manifest.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    # This load is deliberately after copy, proving the project checkpoint itself can be opened.
    YOLO(str(checkpoint))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
