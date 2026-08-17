"""Standalone training script for Google Colab GPU."""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from ultralytics import YOLO

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
DATASET_REPO = "sanjeevafk/glasseye-bfdd-cubit-640"
MODEL_REPO = "sanjeevafk/glasseye-yolo-bfdd-cubit-v1"


def main():
    print("==================================================")
    print("🚀 GLASSEYE YOLOv8 640px CLOUD GPU TRAINING")
    print("==================================================")

    # 1. Device check
    device = "0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Detected: {gpu_name}")
    else:
        print("⚠️ Warning: Running on CPU (recommend switching to T4 GPU in Colab Runtime)")

    # 2. Download dataset from Hugging Face
    dataset_dir = Path("./glasseye_dataset_640")
    if not dataset_dir.exists():
        print(f"\n📥 Downloading 640px dataset from Hugging Face ({DATASET_REPO})...")
        zip_file = hf_hub_download(
            repo_id=DATASET_REPO,
            filename="glasseye_dataset_640.zip",
            repo_type="dataset",
            token=HF_TOKEN,
        )
        print(f"📦 Extracting dataset archive...")
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(dataset_dir)
        print(f"✅ Dataset ready at {dataset_dir}")
    else:
        print(f"✅ Dataset already present at {dataset_dir}")

    # Ensure data.yaml has absolute path
    yaml_path = dataset_dir / "data.yaml"
    yaml_content = f"""path: {dataset_dir.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: defect
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")

    # 3. Train YOLOv8n at 640px
    print("\n🏋️ Starting YOLOv8n fine-tuning at 640px native resolution...")
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(yaml_path.resolve()),
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        seed=20260815,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=15,
        save=True,
        plots=True,
        verbose=True,
    )

    # 4. Evaluate checkpoint
    best_pt = Path(model.trainer.best)
    print(f"\n✅ Training complete! Best checkpoint saved at: {best_pt}")

    print("\n📊 Running final validation on held-out split...")
    val_results = model.val(data=str(yaml_path.resolve()), imgsz=640, device=device)
    map50 = round(float(val_results.box.map50), 6)
    map50_95 = round(float(val_results.box.map), 6)
    print(f"  - Validation mAP@50:    {map50:.4f}")
    print(f"  - Validation mAP@50-95: {map50_95:.4f}")

    # 5. Push to Hugging Face Model Hub
    print(f"\n🚀 Uploading updated best.pt to Hugging Face ({MODEL_REPO})...")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)

    # Upload best.pt
    api.upload_file(
        path_or_fileobj=str(best_pt),
        path_in_repo="best.pt",
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message=f"Update YOLOv8n 640px checkpoint (val mAP@50: {map50:.4f})",
    )

    # Upload metrics report
    metrics_data = {
        "model_version": "glasseye-yolo-bfdd-cubit-640-v2",
        "imgsz": 640,
        "epochs": 50,
        "validation_map50": map50,
        "validation_map50_95": map50_95,
    }
    metrics_path = Path("metrics_report.json")
    metrics_path.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(metrics_path),
        path_in_repo="metrics_report.json",
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="Add 640px validation metrics report",
    )

    print("\n==================================================")
    print("🎉 ALL DONE! Model updated on Hugging Face:")
    print(f"👉 https://huggingface.co/{MODEL_REPO}")
    print("==================================================")


if __name__ == "__main__":
    main()
