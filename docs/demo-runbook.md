# Deterministic GlassEye demo runbook

## Fresh checkout

1. Install Python 3.12, Node 22 or later, and FFmpeg-compatible video support.
2. Run make setup. The first run installs Python and browser dependencies.
3. Run make e2e-demo.

The e2e-demo target performs all of the following in order:

1. creates the deterministic labelled facade dataset;
2. validates labels, class map, checksums, and split isolation;
3. fine-tunes a YOLOv8n checkpoint and writes its metrics report;
4. runs real image/video inference on the generated MP4 scenario;
5. runs the closed-loop mission and replays events;
6. starts FastAPI and Vite; and
7. drives the Three.js dashboard in Chromium, asserting RESOLVED, ESCALATED,
   evidence images, timeline events, and canvas visibility.

The first run downloads the Ultralytics base model. Subsequent runs reuse the
immutable project checkpoint at models/glasseye-yolo-v1/best.pt unless it is
removed deliberately.

## Live presentation

In two terminals after a successful make demo:

    make backend

    make frontend

Open http://127.0.0.1:5173 and select RUN DETERMINISTIC DEMO.

Tell the story in this order:

1. The dashboard displays a known panel grid and clearly marks the scenario as
   simulated.
2. The preinspection video produces a tracked cleanable B2 issue and a
   structural C3 issue, each with a YOLO evidence crop.
3. B2 has a stable high-confidence track, so configuration selects CLEAN.
4. The cleaning simulator records an action; no physical device is controlled.
5. Actual YOLO reinspection no longer sees B2, so verification marks it
   RESOLVED.
6. C3 is structurally classified, so policy selects ESCALATE and explicitly
   never triggers cleaning.
7. Use RESET, STEP, and PLAY to show that the timeline is a replay of the
   same JSONL event log. The replay digest is displayed with the result.

## Produced artefacts

- artifacts/dataset-validation.json
- artifacts/inference-contract.json
- artifacts/demo/glasseye-seed-20260815/events.jsonl
- artifacts/demo/glasseye-seed-20260815/result.json
- models/glasseye-yolo-v1/best.pt
- models/glasseye-yolo-v1/metrics_report.json
- frontend/test-results/glasseye-dashboard.png
