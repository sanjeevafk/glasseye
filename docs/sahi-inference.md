# Sliced Aided Hyper Inference (SAHI) & Benchmark Evaluation

Date: 2026-08-17  
Module: [`backend/app/detector.py`](file:///home/sanjeev/Downloads/glasseye/backend/app/detector.py)  
CLI Benchmarks: [`scripts/benchmark_real_data.py`](file:///home/sanjeev/Downloads/glasseye/scripts/benchmark_real_data.py), [`scripts/benchmark_uav2k_data.py`](file:///home/sanjeev/Downloads/glasseye/scripts/benchmark_uav2k_data.py)

---

## 1. Overview & Problem Statement

Standard full-frame resizing downscales high-resolution drone captures to 320×320 pixels. Thin structural hairline cracks (often 2–4 pixels wide in sensor space) are decimated into sub-pixel blur, leading to zero-recall on distant drone surveys.

**Sliced Aided Hyper Inference (SAHI)** resolves this without model retraining by slicing high-resolution images into overlapping patches, running native-resolution YOLO inference per patch, translating local box coordinates back to full image space, and merging cross-tile predictions with Non-Maximum Suppression (NMS).

---

## 2. Architecture & Pipeline

```
High-Resolution Image (>640px)
 ├── 1. Global Full-Frame Pass (Macro stains & large spalling)
 └── 2. Sliding Window Tiling (480×480 slices, 25% overlap)
         ├── Patch 0,0: YOLO predict (320px) ──> Offset (x0, y0)
         ├── Patch 0,1: YOLO predict (320px) ──> Offset (x1, y0)
         └── Patch N,M: YOLO predict (320px) ──> Offset (xN, yM)
 └── 3. Cross-Tile NMS Merging (IoU threshold = 0.45)
 └── 4. Top-20 Detections Normalized [ymin, xmin, ymax, xmax]
```

### Memory & Concurrency Optimization
- **Sequential Tile Evaluation:** Tiles are processed in a tight single-thread loop with immediate array disposal, keeping memory overhead under **<10 MB RAM** (fully compliant with Render Free Tier 512 MB ceiling).
- **Inference Mode:** Wrapped in `torch.inference_mode()` with `torch.set_num_threads(1)`.

---

## 3. Measured Benchmark Results

Evaluated on the production checkpoint: `models/glasseye-yolo-bfdd-cubit-v1/best.pt`.

![UAV2K Real-Defect Detection Improvement](uav2k-benchmark-comparison.png)

### Evaluation Set 1: Untouched UAV2K Real Drone Holdout Benchmark (200-Image Split)
*Distant high-altitude drone surveys of full building facades (527 ground-truth defect boxes, building-disjoint from training).*

| Model & Mode | Predictions | True Positives | Precision | Recall | AP@50 | False Alarms |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **320px Baseline (Standard YOLO)** | 3 | 0 | 0.0% | 0.0% | 0.0000 | 3 |
| **320px + SAHI** | 649 | 46 | 7.09% | 8.73% | 0.0362 | 603 |
| **640px + SAHI (Before UAV2K)** | 536 | 60 | 11.19% | 11.39% | 0.0357 | 476 |
| **3-Way Unified V1 + SAHI** | 523 | 187 | 35.76% | 35.48% | 0.2316 | 336 |
| **3-Way Unified V2 + SAHI (Latest)** | **1,764** | **`322`** | **18.25%** | **`61.10%`** | **`0.2774`** | **1,442** |

*Takeaway: SAHI inference with the fine-tuned 640px model captured **61.10% of all real defects** on untouched drone surveys, reaching a benchmark record of **0.2774 AP@50**.*

### Evaluation Set 2: 3-Way Combined Validation Set (289 Held-Out Images)
*Combined multi-domain validation set evaluated during model convergence.*

| Metric | BFDD-only Baseline | BFDD + CUBIT (640px) | 3-Way Unified (640px Latest) |
| :--- | :--- | :--- | :--- |
| **Validation Precision** | 22.1% | 36.2% | **`47.8%`** |
| **Validation Recall** | 9.9% | 35.9% | **`34.3%`** |
| **Validation mAP@50** | 0.2130 | 0.3094 | **`0.3483`** |
| **Validation mAP@50-95** | 0.0937 | 0.1648 | **`0.1750`** |

---

## 4. CLI Reproduction Commands

Run benchmarks with and without SAHI to verify metrics:

```bash
# BFDD Test Set Benchmark
python scripts/benchmark_real_data.py --checkpoint models/glasseye-yolo-bfdd-cubit-v1/best.pt --bfdd-split test --min-component-area 512 --sahi

# Untouched UAV2K Drone Benchmark
python scripts/benchmark_uav2k_data.py --checkpoint models/glasseye-yolo-bfdd-cubit-v1/best.pt --sahi
```
