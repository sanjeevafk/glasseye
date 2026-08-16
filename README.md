# GlassEye

GlassEye is a deterministic, simulated facade-inspection loop: a fine-tuned
YOLO model finds visible facade defects in recorded video, tracks and locates
them, records evidence, applies a rules-based simulated response, reinspects
the same panel, and shows the outcome in a Three.js dashboard.

It makes no claim of real drone autonomy, physical spraying, repair safety, or
field-ready defect performance.

## Start here

    make setup
    make e2e-demo

The second command is the release gate: it prepares and validates the
dataset, trains or reuses the checkpoint, runs real image and video inference,
executes and replays the deterministic mission, then verifies the FastAPI +
Vite + Three.js dashboard in Chromium.

For a local presentation after the checkpoint exists:

    make demo
    make backend

In another terminal:

    make frontend

Open http://127.0.0.1:5173 and use RUN DETERMINISTIC DEMO.

## Advisory VLM review (optional)

An advisory vision-language review runs on selected evidence crops before the
policy decision: structural (high-impact) detections are always reviewed, and
ambiguous cleanable detections are reviewed before any decision. The VLM never
controls an actuator; it can only move a cleanable case to REVIEW or to human
escalation, and any VLM failure routes to REVIEW, never CLEAN.

The deterministic fixture mode is the default and keeps the E2E demo
repeatable:

    DEMO_VLM_MODE=fixture make demo

To use a real OpenAI-compatible VLM once an API key exists:

    DEMO_VLM_MODE=http GLASSEYE_VLM_API_KEY=... make demo

See backend/app/vlm.py for provider configuration.

## External real-data benchmark

After extracting the user-provided BFDD archive to `data/external/bfdd`, create
the grouped training/validation/test split:

    make prepare-bfdd

Benchmark the original synthetic model and the BFDD-tuned checkpoint on the
same held-out 149-image test split:

    PYTHONPATH=backend .venv/bin/python scripts/benchmark_real_data.py --dataset bfdd --bfdd-split test --min-component-area 512 --checkpoint models/glasseye-yolo-v1/best.pt
    PYTHONPATH=backend .venv/bin/python scripts/benchmark_real_data.py --dataset bfdd --bfdd-split test --min-component-area 512 --checkpoint models/glasseye-yolo-real-bfdd-v1/best.pt

The script converts all non-background BFDD mask components to binary defect
boxes and writes an isolated report/overlays directory for each checkpoint
under `artifacts/real-benchmark/bfdd/`. This is a reproducible domain-shift
check, not an official BFDD segmentation score.

## Production deployment

The app can run as a single self-contained container: the image builds the
frontend, and at boot regenerates the deterministic demo (dataset, YOLO
checkpoint, scenario video, mission events), then serves the API and the
built SPA on one port.

    docker build -t glasseye-demo .
    docker run -p 8000:8000 glasseye-demo

Open http://127.0.0.1:8000. A `render.yaml` blueprint deploys the same image
to Render (free tier):

    render blueprint launch

## Layout

- backend/app: detector adapter, tracking, evidence, localization, policy,
  state machine, simulator, verification, events, replay, and API
- frontend: Vite/React dashboard with a native Three.js facade panel map
- scripts: dataset audit/validation, training, inference, and mission runner
- docs: architecture, dataset licence/annotation audit, and demo runbook
- Dockerfile / render.yaml / .dockerignore: single-container production deploy

See docs/data-card.md before using any downloaded source data and
docs/demo-runbook.md for the deterministic proof path. For continuation work,
see docs/handoff-prompt.md.
