# GlassEye

## Hackathon Project Specification

**Tagline:** An AI-powered facade inspection and remediation simulator that detects visible building defects, decides what should happen next, and proves whether the action worked.

## 1. What GlassEye is

GlassEye inspects a building facade from images or drone-style video. A fine-tuned YOLO model finds visible defects. The system keeps evidence, decides whether an issue can be cleaned or must be escalated, simulates the action, then inspects the same area again to verify the result.

The demo is deliberately a **simulated inspection system**, not a claim of safe real-world autonomous drone or repair operation.

```text
Image or video
      ↓
YOLO defect detection
      ↓
Tracking + evidence
      ↓
Facade/panel location
      ↓
Policy decision
      ↓
Simulated clean OR escalate
      ↓
Reinspection + verification
      ↓
Dashboard and replay
```

## 2. Problem

Building-facade inspection is slow, expensive, and difficult to repeat consistently. Teams need a simple way to show:

- where a visible issue was found;
- what evidence supports it;
- whether it is likely cleanable or needs human review;
- what action was taken; and
- whether the issue changed after that action.

GlassEye makes that inspection-to-verification loop visible and auditable.

## 3. Hackathon goal

Build one reliable, repeatable end-to-end demo:

1. A synthetic or recorded facade video is inspected.
2. GlassEye detects defects in the footage.
3. The system tracks the same defect across frames and stores evidence.
4. The defect is placed on a known facade panel.
5. Policy selects `CLEAN` or `ESCALATE`.
6. Cleaning is simulated for a cleanable issue.
7. The location is reinspected.
8. The dashboard shows `RESOLVED`, `UNRESOLVED`, or `ESCALATED`.

The main demo story should take about 30 seconds and should be deterministic: the same seeded scenario produces the same result every time.

## 4. Scope

### In scope

- Fine-tuned Ultralytics YOLO model.
- Still-image and recorded-video inference.
- Detection tracking and evidence capture.
- Known facade/panel coordinates.
- Optional VLM review for ambiguous or important detections.
- Rules-based policy engine.
- Cleaning and reinspection simulator.
- Event log and replay.
- Simple dashboard showing facade state and mission timeline.

### Out of scope for the hackathon

- Flying a real drone.
- Controlling a real sprayer.
- SLAM, photogrammetry, or arbitrary-building reconstruction.
- Thermal-camera integration.
- Weather-aware routing.
- Autonomous structural repair.
- A real safety certification claim.

## 5. User-facing outcome

For each issue, the dashboard should show:

- defect type and confidence;
- image evidence;
- facade/panel location;
- policy result: `CLEAN`, `ESCALATE`, or `REVIEW`;
- action taken in the simulator;
- reinspection result; and
- an event timeline explaining why the system made each decision.

## 6. Model strategy

### Recommended MVP classes

Use two classes first:

- `cleanable_surface_issue`: dirt or stain that can be simulated as cleaned.
- `structural_issue`: crack, spalling, or sealant damage that must be escalated.

This is more reliable for a hackathon than four separate classes with too few examples.

If enough reviewed data exists for every class, expand later to:

- `dirt`
- `water_stain`
- `crack`
- `sealant_damage`

### Bounding boxes versus segmentation

A bounding box is a rectangle around a defect. It is enough to say, "there is a crack here."

Segmentation draws the exact outline of a defect. Use it only for `cleanable_surface_issue` when the product needs the exact dirty area for simulated cleaning or area estimates. It needs more careful annotation work.

The API must allow `mask` to be absent, so the rest of the product works with normal detection before segmentation is ready.

### Dataset sources

Start with these public sources, after checking their terms before redistributing any data:

- **BFD-UAV2K**: 2,000 full-frame UAV facade images, already annotated in YOLO TXT and COCO JSON. Use as the first detection baseline. Its release page says licence information will be added, so use it for prototyping/evaluation unless the authors give clear reuse permission: https://huggingface.co/datasets/RealUAV-SD/UAV2K
- **CUBIT-Det and CUBIT-Seg**: UAV/DSLR defect data for cracks, spalling, and moisture, including pixel-level crack/spalling labels: https://github.com/CUHK-USR-Group/Defect-Dataset
- **BFDD**: RGB/IR facade data with pixel-level labels for cracks, peeling, hollow areas, stains, and erosion. Use RGB only in this MVP. It is CC BY 4.0: https://data.mendeley.com/datasets/9ych7czvyg/1
- **BD3**: useful extra visual variety for stains, cracks, peeling, and spalling, but it is a classification dataset and is not ready-made YOLO detection data: https://github.com/Praveenkottari/BD3-Dataset

### Target data

Add 100-300 manually reviewed project-specific images or video frames. These should look like the exact facade, lighting, camera distance, and demo footage used by GlassEye. Keep a small group of these images aside as the final test set; never train on them.

Do not split consecutive frames from one video across training and test sets. Split by capture session or source video so results are honest.

### Dataset contract

```text
data/glasseye_v1/
├── data.yaml
├── images/train/
├── images/val/
├── images/test/
├── labels/train/
├── labels/val/
├── labels/test/
├── manifest.csv
├── dataset_hash.txt
└── README.md
```

`manifest.csv` records source, image ID, split, classes, annotation status, and checksum. `data.yaml` is the sole source of truth for class IDs and class names.

### Annotation rules

Write these rules before any annotation work starts:

- What counts as one defect instance?
- What is the minimum visible defect size?
- How should partly hidden defects be handled?
- Can two defect classes overlap?
- Which images are normal/negative images?
- When is a mask required rather than a box?

## 7. Training and inference pipeline

```text
Raw media
   ↓
Dataset audit and annotation checks
   ↓
Train/validation/test split
   ↓
Ultralytics training
   ↓
Validation + visual error review
   ↓
Confidence and IoU threshold tuning
   ↓
Immutable best checkpoint
   ↓
Image/video inference adapter
   ↓
Tracking and evidence
```

The model agent must record:

- model base and Ultralytics version;
- dataset hash and class map;
- image size, epochs, batch size, seed, and augmentations;
- mAP50, mAP50-95, precision, recall, and per-class metrics;
- confusion matrix/error examples;
- example predictions;
- inference speed; and
- exact `best.pt` checkpoint location.

Training completion alone is not a pass. The model must load, predict on images and video, return valid JSON, and produce a clear metrics report.

## 8. Detector contract

```json
{
  "frame_id": "frame-00123",
  "timestamp": 12.3,
  "image_id": "img-00123",
  "model_version": "glasseye-yolo-v1",
  "detections": [
    {
      "class_name": "cleanable_surface_issue",
      "class_id": 0,
      "confidence": 0.91,
      "bbox_xyxy": [120, 80, 260, 190],
      "mask": null,
      "track_id": null
    }
  ]
}
```

`mask` is optional. Downstream code must not assume segmentation exists.

## 9. Intelligence and policy

YOLO is the primary detector. The VLM is optional and advisory.

```text
YOLO detection
    ↓
High confidence + simple case ──→ policy
    ↓
Ambiguous or high-impact case ─→ VLM review ─→ policy
```

The VLM receives a cropped image and structured metadata. It may return `confirm`, `reject`, or `escalate`; it never directly controls an actuator.

Initial policy rules:

| Input | Decision |
|---|---|
| High-confidence `cleanable_surface_issue` with stable track | `CLEAN` |
| `structural_issue` | `ESCALATE` |
| Low confidence, unstable track, or conflicting VLM result | `REVIEW` |
| No defect after reinspection | `RESOLVED` |
| Defect still visible after cleaning | `UNRESOLVED` and `ESCALATE` |

Keep the policy rules in configuration, not hidden in UI code or prompts.

## 10. System design

Use a modular monolith for the hackathon: one backend process with clear modules. This is simpler to test and easier for agents to integrate than many network services.

```text
Frontend dashboard
        ↓ API
GlassEye backend
├── detector adapter
├── tracker
├── evidence/artifact store
├── facade localization
├── VLM adapter
├── policy engine
├── mission state machine
├── cleaning simulator
├── verification service
└── event log + replay
```

### State flow

```text
IDLE → INSPECTING → DETECTED → EVIDENCE_READY → DECIDED
                                               ├→ CLEANING → REINSPECTING → RESOLVED
                                               └→ ESCALATED
```

Every transition must create an event with a timestamp, reason code, evidence references, and payload.

## 11. Reuse from Vanrakshak

Use `/home/sanjeev/Downloads/vanrakshak` as the source of implementation patterns.

Most useful files:

- `backend/app/perception.py`: detector/tracker seam, confidence filter, crop selection, evidence artifacts, and event emission.
- `backend/app/schemas.py`: Pydantic request/response structure.
- `backend/app/state_machines.py`: clear testable transitions.
- `backend/app/policies.py`, `events.py`, `vlm.py`, `actuator.py`, and `replay.py`: patterns for policy, event logging, advisory VLM calls, simulated action, and replay.
- `backend/tests/`: testing patterns.
- `docs/YOLO_FINETUNING_IMPLEMENTATION.md`: an agent-task template only. Its Roboflow dataset slugs are placeholders and must not be used as real links.

Do not copy forest-specific classes, threat scoring, mission names, or hardware claims. Rename and adapt them around facade issues, inspection evidence, remediation, and verification.

## 12. Agent plan and gates

| Phase | Owner | Depends on | Completion gate |
|---|---|---|---|
| 0. Foundation | Architect | None | Fresh checkout installs and runs smoke tests; schemas are frozen. |
| 1. Dataset | Data/CV agent | Phase 0 | Manifest, split report, class map, label validator, and visual samples exist. |
| 2. YOLO model | CV agent | Phase 1 | `best.pt`, metrics report, example predictions, and valid image/video inference exist. |
| 3. Perception | Perception agent | Phase 2 | Tracks and evidence records are produced from recorded video. |
| 4. Localization | Geometry agent | Phase 3 | A tracked defect maps to a known facade panel. |
| 5. Policy/VLM | Intelligence agent | Phases 3-4 | Deterministic clean/escalate/review decisions pass tests. |
| 6. Simulator | Simulation agent | Phase 5 | Cleaning and reinspection change facade state correctly. |
| 7. Dashboard | Frontend agent | Phases 3-6 | Dashboard displays issues, outcomes, and event timeline. |
| 8. Release | Integration/QA agent | All prior phases | One command runs a seeded E2E demo and replay. |

Each agent must provide:

- role and exact owned files;
- input/output contracts;
- commands to run;
- automated acceptance tests;
- artifacts produced; and
- short handoff report, including known limits.

No agent should rewrite another agent's interface without approval from the architect/integration owner.

## 13. Acceptance tests

### Dataset and YOLO

```text
[ ] Dataset validator rejects invalid labels.
[ ] Class IDs match data.yaml exactly.
[ ] Split report shows no source-video leakage.
[ ] Training configuration and dataset hash are saved.
[ ] best.pt loads on a clean environment.
[ ] Validation emits mAP50, mAP50-95, precision, recall, and per-class metrics.
[ ] Image inference returns valid detector-contract JSON.
[ ] Video inference includes frame IDs and timestamps.
[ ] Masks are valid or explicitly null.
[ ] Inference benchmark is recorded.
```

### End-to-end demo

```text
[ ] A seeded facade scenario starts.
[ ] Video frames are inspected.
[ ] Defects are detected and tracked.
[ ] Evidence is saved.
[ ] Each defect maps to a facade/panel location.
[ ] Policy creates a visible CLEAN, REVIEW, or ESCALATE decision.
[ ] CLEAN changes the simulated facade state.
[ ] Reinspection produces RESOLVED or UNRESOLVED.
[ ] ESCALATE never triggers cleaning.
[ ] Dashboard shows final status and timeline.
[ ] Event log replay reproduces the same result.
```

## 14. Suggested repository layout

```text
glasseye/
├── backend/
│   ├── app/
│   │   ├── perception.py
│   │   ├── schemas.py
│   │   ├── events.py
│   │   ├── localization.py
│   │   ├── policies.py
│   │   ├── state_machines.py
│   │   ├── simulator.py
│   │   ├── verification.py
│   │   └── main.py
│   └── tests/
├── frontend/
├── data/                 # ignored raw data; manifests may be tracked
├── models/               # ignored weights; checkpoint manifest tracked
├── scripts/
│   ├── validate_dataset.py
│   ├── train_yolo.py
│   ├── run_inference.py
│   └── run_demo.py
├── docs/
│   ├── architecture.md
│   └── data-card.md
└── README.md
```

## 15. Demo script

1. Start with a clean facade dashboard.
2. Start inspection footage.
3. Show a `cleanable_surface_issue` detection and evidence crop.
4. Show GlassEye choosing `CLEAN`.
5. Show simulated cleaning and reinspection.
6. Show the issue becoming `RESOLVED`.
7. Show a `structural_issue` detection.
8. Show `ESCALATE`, with no cleaning attempt.
9. Open the timeline/replay to prove each decision was logged.

## 16. Definition of done

GlassEye is done for the hackathon when a fresh checkout can run one documented command that executes a fixed inspection scenario through detection, tracking, evidence, localization, policy, simulated action or escalation, reinspection, verification, dashboard update, and event replay.

The goal is not a perfect production inspection platform. The goal is a credible, explainable, end-to-end proof that the closed loop works.
