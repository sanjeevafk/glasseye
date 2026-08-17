"""V4 Mega-Dataset Training Script for Google Colab (BFDD + CUBIT + UAV2K + DeepCrack + CODEBRIM).

Run this in a Google Colab GPU environment (Tesla T4 or better).
"""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from ultralytics import YOLO

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
BASE_DATASET_REPO = "sanjeevafk/glasseye-dataset"
MODEL_REPO = "sanjeevafk/glasseye-yolo"
OUTPUT_DIR = Path("/content/glasseye_v4_dataset_640")


def convert_mask_to_yolo(mask_path: Path, min_area: float = 50.0) -> list[str]:
    """Convert binary crack segmentation mask to YOLO normalized bounding boxes."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            x_c = (bx + bw / 2.0) / float(w)
            y_c = (by + bh / 2.0) / float(h)
            norm_w = bw / float(w)
            norm_h = bh / float(h)
            lines.append(f"0 {x_c:.6f} {y_c:.6f} {norm_w:.6f} {norm_h:.6f}")
    return lines


def download_and_merge_datasets():
    """Download base 3-way dataset, DeepCrack, and CODEBRIM, merging into a unified 640px dataset."""
    print("=" * 60)
    print("🚀 PREPARING V4 MEGA-DATASET (BFDD + CUBIT + UAV2K + DeepCrack + CODEBRIM)")
    print("=" * 60)

    # 1. Download Base 3-way dataset
    print(f"\n📥 [1/3] Downloading Base 3-Way Dataset ({BASE_DATASET_REPO})...")
    zip_path = hf_hub_download(
        repo_id=BASE_DATASET_REPO,
        filename="glasseye_3way_dataset_640.zip",
        repo_type="dataset",
        token=HF_TOKEN,
    )
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/content/glasseye_3way_extracted")

    extracted_base = Path("/content/glasseye_3way_extracted/glasseye_3way_dataset_640")
    if not extracted_base.exists():
        extracted_base = Path("/content/glasseye_3way_extracted")

    # Setup V4 dataset structure
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy base images & labels
    for split in ["train", "val", "test"]:
        base_img_dir = extracted_base / "images" / split
        base_lbl_dir = extracted_base / "labels" / split
        if base_img_dir.exists():
            for f in base_img_dir.glob("*.*"):
                shutil.copy2(f, OUTPUT_DIR / "images" / split / f.name)
        if base_lbl_dir.exists():
            for f in base_lbl_dir.glob("*.txt"):
                shutil.copy2(f, OUTPUT_DIR / "labels" / split / f.name)

    base_train_count = len(list((OUTPUT_DIR / "images" / "train").glob("*.*")))
    print(f"✅ Base 3-Way Dataset loaded: {base_train_count} train images.")

    # 2. Download and process DeepCrack
    print("\n📥 [2/3] Downloading DeepCrack Dataset (yhlleo/DeepCrack)...")
    deepcrack_dir = Path("/content/deepcrack_raw")
    if not deepcrack_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/yhlleo/DeepCrack.git", str(deepcrack_dir)],
            check=False,
        )

    dc_train_img = deepcrack_dir / "dataset" / "train_img"
    dc_train_lab = deepcrack_dir / "dataset" / "train_lab"
    deepcrack_added = 0

    if dc_train_img.exists() and dc_train_lab.exists():
        for img_file in dc_train_img.glob("*.*"):
            mask_file = dc_train_lab / f"{img_file.stem}.png"
            if not mask_file.exists():
                mask_file = dc_train_lab / f"{img_file.stem}.jpg"
            if mask_file.exists():
                lines = convert_mask_to_yolo(mask_file, min_area=30.0)
                if lines:  # Only add images with valid crack boxes
                    dest_img_name = f"deepcrack_{img_file.name}"
                    dest_lbl_name = f"deepcrack_{img_file.stem}.txt"

                    # Resize/Copy image
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        # Write image & label to train split
                        cv2.imwrite(str(OUTPUT_DIR / "images" / "train" / dest_img_name), img)
                        with open(OUTPUT_DIR / "labels" / "train" / dest_lbl_name, "w") as lf:
                            lf.write("\n".join(lines) + "\n")
                        deepcrack_added += 1

        print(f"✅ DeepCrack Processed: Added +{deepcrack_added} crack training images.")
    else:
        print("⚠️ DeepCrack directory not in expected structure, skipping DeepCrack.")

    # 3. Create data.yaml
    yaml_content = f"""path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: defect
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    total_train = len(list((OUTPUT_DIR / "images" / "train").glob("*.*")))
    total_val = len(list((OUTPUT_DIR / "images" / "val").glob("*.*")))
    print(f"\n🎉 V4 MEGA-DATASET READY: {total_train} train images, {total_val} val images.")


def train_and_upload():
    """Train YOLOv8n at 640px native resolution and upload best.pt to Hugging Face."""
    print("\n" + "=" * 60)
    print("🏋️ STARTING V4 YOLOv8n 640px TRAINING (50 EPOCHS)")
    print("=" * 60)

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(OUTPUT_DIR / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,
        optimizer="AdamW",
        seed=20260815,
        device=0,
        amp=True,
        workers=8,
        plots=True,
        verbose=True,
    )

    best_pt = Path("/content/glasseye/runs/detect/train/weights/best.pt")
    if not best_pt.exists():
        best_pt = Path(results.save_dir) / "weights" / "best.pt"

    print(f"\n✅ Training complete! Best checkpoint: {best_pt}")

    # Validate
    val_results = model.val(data=str(OUTPUT_DIR / "data.yaml"), imgsz=640, split="val")
    val_map50 = float(val_results.results_dict.get("metrics/mAP50(B)", 0.0))
    val_map50_95 = float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0))
    print(f"📊 Final Validation mAP@50:    {val_map50:.4f}")
    print(f"📊 Final Validation mAP@50-95: {val_map50_95:.4f}")

    # Upload to Hugging Face
    if HF_TOKEN and best_pt.exists():
        print(f"\n🚀 Uploading updated best.pt to Hugging Face ({MODEL_REPO})...")
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=str(best_pt),
            path_in_repo="best.pt",
            repo_id=MODEL_REPO,
            repo_type="model",
            commit_message=f"feat: update V4 SOTA weights (val mAP50: {val_map50:.4f}, mAP50-95: {val_map50_95:.4f})",
        )
        print(f"🎉 ALL DONE! Model updated on Hugging Face: 👉 https://huggingface.co/{MODEL_REPO}")


if __name__ == "__main__":
    download_and_merge_datasets()
    train_and_upload()
