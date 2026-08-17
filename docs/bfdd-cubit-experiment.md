# BFDD + CUBIT combined-model experiment

Date: 2026-08-16.

This documents the training of `models/glasseye-yolo-bfdd-cubit-v1` (BFDD +
CUBIT) and the apples-to-apples benchmark against the synthetic and
BFDD-only checkpoints on the untouched BFDD test split and the CUBIT test
split.

## Training

- Checkpoint: `models/glasseye-yolo-bfdd-cubit-v1/best.pt`
  (SHA-256 `df78a4f9...45cf`, full manifest in `model_manifest.json`).
- Data: `data/bfdd_cubit_binary_v1` (train 1,299 = 600 BFDD + 699 CUBIT
  val-as-train; val 89 = BFDD only; test 850 = 149 BFDD + 701 CUBIT).
  Dataset hash `8557ee88...978`.
- Config identical to the BFDD-only run: `yolov8n.pt` base, imgsz 320,
  32 epochs, batch 4, seed 20260815, AdamW, deterministic.
- CUBIT source: the official CUBIT val split (the only labeled CUBIT data
  in the download) converted polygons -> binary `defect` boxes
  (min polygon area 512 px^2). See `docs/cubit-data-card.md` for the
  no-training-labels and frame-interleaving caveats.

## Validation (BFDD val, 89 images)

| Model | mAP50 | mAP50-95 |
|---|---|---|
| BFDD-only | 0.2130 | 0.0937 |
| BFDD+CUBIT | 0.2514 | 0.1127 |

## BFDD test benchmark (the honest benchmark)

Protocol unchanged: `scripts/benchmark_real_data.py --dataset bfdd
--bfdd-split test --min-component-area 512` on the untouched 149-image
BFDD test split (1,033 mask-derived boxes). Predictions: conf 0.20,
iou 0.45, imgsz 320, max_det 20.

| Model | mAP50 | mAP50-95 | recall |
|---|---|---|---|
| `glasseye-yolo-v1` (synthetic) | 0.013104 | 0.009923 | 0.016457 |
| `glasseye-yolo-real-bfdd-v1` (BFDD) | 0.094017 | 0.042910 | 0.141336 |
| `glasseye-yolo-bfdd-cubit-v1` (BFDD+CUBIT) | **0.151198** | **0.070703** | **0.201355** |

BFDD+CUBIT vs BFDD-only: mAP50 **+60.8%**, mAP50-95 **+64.8%**, recall
**+42.5%** — a substantial improvement on the untouched BFDD held-out set.

## CUBIT test benchmark (leak-inflated; secondary)

Protocol: `scripts/benchmark_cubit_data.py` on the untouched 701-image
CUBIT test split (5,085 derived boxes), max_det 100. **Caveat:** CUBIT test
frames are near-duplicates of training frames (median index distance 4 to
the CUBIT val-as-train source; 590/701 within 10), so these numbers are
inflated. A "far subset" (test frames >= 10 indices from any CUBIT
training frame, 111 images) is reported separately as a less-leaky figure.

| Model | full mAP50 | full recall | far-subset mAP50 | far-subset recall |
|---|---|---|---|---|
| synthetic | 0.0031 | 0.0069 | 0.0000 | 0.0000 |
| BFDD-only | 0.0211 | 0.0350 | 0.0164 | 0.0216 |
| BFDD+CUBIT | **0.1990** | **0.2061** | **0.1279** | **0.1454** |

BFDD+CUBIT dominates on CUBIT as expected (it trained on CUBIT data), and
still leads on the less-leaky subset.

## UAV2K test benchmark (clean held-out; out-of-domain)

Protocol: `scripts/benchmark_uav2k_data.py` on the untouched 200-image
UAV2K test split (527 derived boxes, building-disjoint from train — a
genuinely clean held-out set, unlike CUBIT's frame-interleaved split).
UAV2K was never trained on by any of the three checkpoints.

| Model | mAP50 | mAP50-95 | recall | preds |
|---|---|---|---|---|
| synthetic | 0.0000 | 0.0000 | 0.0000 | 64 |
| BFDD-only | 0.0000 | 0.0000 | 0.0000 | 43 |
| BFDD+CUBIT | **0.0132** | **0.0091** | **0.0133** | 21 |

BFDD+CUBIT is the only checkpoint that detects anything on UAV2K
(7 TP / 14 FP at IoU 0.5), but all three are effectively out-of-domain
here: UAV2K's 4000x2250 aerial hollow/spalling/crack imagery is far from
BFDD's and CUBIT's close-up defect shots, and none of the checkpoints was
trained on UAV2K.  This is honest context, not a claim — UAV2K test does
not favor any model and should not be used to rank them.

## Honesty notes

1. The improvement is **not limited to CUBIT**: BFDD test (untouched,
   no CUBIT frames anywhere near it) improved +60% mAP50. The combined
   model generalizes better on BFDD, not just on the dataset it added.
2. CUBIT numbers are leak-inflated and are reported with the caveat and
   far-subset split; do not quote the full-set 0.199 as a clean
   generalization number.
3. This is a controlled hackathon demo — not a field-ready inspection
   claim.
4. Benchmark overlay grids (green = ground truth, red = predictions) were
   inspected for all three checkpoints under
   `artifacts/real-benchmark/{bfdd,cubit,uav2k}/*/overlays` before this
   conclusion; overlays contain real, spatially-overlapping boxes (e.g.
   the combined model scores 202 true positives at IoU 0.5 on CUBIT test).

## Recommendation

**Use `models/glasseye-yolo-bfdd-cubit-v1/best.pt` for the hackathon demo.**
It is strictly better than the BFDD-only model on the untouched BFDD test
benchmark (mAP50 0.151 vs 0.094, recall 0.201 vs 0.141) and dramatically
better on CUBIT. The demo's deterministic two-class clean/escalate
scenario is unaffected — that path is driven by the untouched synthetic
model, and the binary model is used for the real-data evidence/benchmark
flows. Keep the BFDD-only checkpoint as the baseline; no checkpoint was
overwritten.

## 640px Native Resolution GPU Retraining (Colab Run, 2026-08-17)

To resolve the 320px downsampling blur bottleneck on thin hairline cracks, the BFDD + CUBIT dataset was repackaged at 640px (`sanjeevafk/glasseye-bfdd-cubit-640`) and fine-tuned on a Google Colab Tesla T4 GPU using [`scripts/train_colab.ipynb`](file:///home/sanjeev/Downloads/glasseye/scripts/train_colab.ipynb) / [`scripts/train_colab.py`](file:///home/sanjeev/Downloads/glasseye/scripts/train_colab.py):

- **Parameters:** `imgsz=640`, `epochs=50`, `batch=16`, `optimizer=AdamW`, `seed=20260815`, AMP enabled.
- **Validation Comparison (BFDD val, 89 images):**
  - 320px baseline: `mAP50 = 0.2514`, `mAP50-95 = 0.1127`
  - **640px model:** **`mAP50 = 0.3094` (+23.1%)**, **`mAP50-95 = 0.1648` (+46.2%)**

### Impact on Untouched UAV2K Drone Benchmark (with SAHI)
- **True Positives:** Increased from 46 to **`60` real defect boxes** detected.
- **Precision:** Increased from 7.09% to **`11.19%`** (**+57.8% relative gain**).
- **False Positives:** Dropped from 603 down to **`476`** (**-21.1% false alarm reduction**).

---

## Reproduce

    # 1-Click Colab GPU Retraining:
    Open scripts/train_colab.ipynb in Google Colab (T4 GPU).

    # Local benchmark reproduction:
    PYTHONPATH=backend .venv/bin/python scripts/benchmark_real_data.py --dataset bfdd \
        --bfdd-split test --min-component-area 512 --sahi --checkpoint models/glasseye-yolo-bfdd-cubit-v1/best.pt
    PYTHONPATH=backend .venv/bin/python scripts/benchmark_uav2k_data.py --sahi --checkpoint models/glasseye-yolo-bfdd-cubit-v1/best.pt
