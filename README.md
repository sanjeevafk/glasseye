# GlassEye

**AI-Powered Façade Inspection & Remediation Simulator**

GlassEye is an end-to-end building façade inspection platform. It detects visible defects from imagery or drone video with a fine-tuned YOLO model, localizes them to façade panels, recommends clean-or-escalate actions, and simulates remediation — then reinspects the same area to verify the outcome.

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-glasseye--yolo-blue)](https://huggingface.co/sanjeevafk/glasseye-yolo)

---

## Features

- **Aerial Drone Video Scanner** — Upload drone survey recordings (MP4/WebM/MOV) or pick a 1-click flight preset. Runs native-resolution YOLO inference with configurable frame sampling, synchronized playback with a live bounding-box HUD, a flight timeline scrubber, and a 4×3 cumulative damage heatmap.
- **Interactive Façade Scanner** — Upload high-resolution façade photos or pick a 1-click test preset for instant defect boxes, 4×3 panel grid coordinates, and a 0–100 Façade Integrity Index.
- **Automated Dispatch Policy** — Recommends next steps (`SIMULATED CLEAN APPROVAL`, `MANDATORY STRUCTURAL ESCALATION`, `MAINTENANCE SCHEDULE`).
- **Advisory VLM Second Opinions** — Routes high-impact defect crops to a Vision-Language Model for independent verification.
- **Closed-Loop Drone Simulation** — Replays a full flight with video inference, IOU tracking, simulated cleaning, and post-remediation verification.
- **Three.js 3D Panel Map** — Visualizes real-time status (`resolved`, `escalated`, `active`) across building geometry, with automatic 2D fallback for headless environments.

---

## Quickstart

```bash
make setup
```

Start the backend (FastAPI) in one terminal:

```bash
make backend
```

Start the frontend (Vite / React) in a second terminal:

```bash
make frontend
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

### Tests

```bash
# Backend: ruff lint + pytest
.venv/bin/ruff check .
PYTHONPATH=backend .venv/bin/pytest backend/tests -q

# Frontend: Playwright E2E browser tests
npm --prefix frontend run test:e2e
```

---

## Trained Model

The active production model is trained on the Building Façade Defect Dataset (BFDD), CUBIT concrete defects, and high-altitude UAV2K aerial survey footage.

- **Hugging Face Hub**: [`sanjeevafk/glasseye-yolo`](https://huggingface.co/sanjeevafk/glasseye-yolo)
- **Local Checkpoint**: `models/glasseye-yolo-bfdd-cubit-v1/best.pt`

```python
from ultralytics import YOLO

model = YOLO("models/glasseye-yolo-bfdd-cubit-v1/best.pt")
results = model.predict("backend/app/samples/spalling_damage_sample.jpg", conf=0.15)
results[0].show()
```

---

## Deployment

GlassEye ships as a single self-contained Docker container serving the compiled React frontend and FastAPI backend on one port:

```bash
docker build -t glasseye-demo .
docker run -p 8000:8000 glasseye-demo
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). A `render.yaml` blueprint is included for 1-click cloud deployment.

---

## Documentation

- [`docs/engineering-story.md`](docs/engineering-story.md) — From 0% synthetic baseline to 61.1% drone defect recall
- [`docs/architecture.md`](docs/architecture.md) — Core system design and state machines
- [`docs/sahi-inference.md`](docs/sahi-inference.md) — SAHI inference and high-res drone benchmark metrics
- [`docs/data-card.md`](docs/data-card.md) — Dataset sources, licensing, and annotation schemas
- [`docs/bfdd-cubit-experiment.md`](docs/bfdd-cubit-experiment.md) — Benchmark results across BFDD, CUBIT, and UAV2K
- [`docs/cubit-data-card.md`](docs/cubit-data-card.md) — CUBIT concrete defect dataset card
- [`docs/uav2k-data-card.md`](docs/uav2k-data-card.md) — UAV2K high-resolution drone façade dataset card