# GlassEye — submission package

One-page orientation for judges, reviewers, and anyone picking up the repo.

## What it is

GlassEye is a **deterministic façade-inspection software pipeline**. A
fine-tuned YOLO model finds visible facade defects in recorded video; the
system tracks them across frames, localizes them to a panel map, crops
evidence, runs a rules-based policy with an optional advisory vision-language
review, simulates a response (cleaning command for cleanable surface issues,
maintenance dispatch for structural issues), reinspects the same panel, and
projects the whole mission onto a Three.js dashboard — driven entirely by an
append-only event log that can be replayed.

It is explicitly **simulation-only**: no physical drone or actuator is
controlled, and no field-ready inspection performance is claimed. That honesty
is part of the design: every synthetic artefact is labelled as synthetic.

## The deterministic demo (the star)

`make e2e-demo` (or the deployed container) runs the full loop:

1. A seeded synthetic facade scene with known ground truth (deliberately
   marked simulated).
2. Real YOLO inference on the pre-inspection video produces two tracked
   issues: a cleanable surface issue and a structural issue.
3. The cleanable issue passes the confidence/stability gate, an advisory VLM
   review, and policy — emitting a **software-only cleaning command** modal
   (simulated complete), then real YOLO reinspection confirms it is gone:
   `RESOLVED`.
4. The structural issue is never cleaned: policy routes it to
   `ESCALATED` with a **structural maintenance dispatch** modal.
5. The Three.js panel map, evidence cards, VLM verdicts, and timeline are all
   projections of the same JSONL event log (28 events), with replay.

## What is real vs simulated

| Piece | Status |
|---|---|
| YOLO inference on recorded video | real (yolov8n, fine-tuned) |
| Tracking, localization, evidence crops | real pipeline code |
| Policy decisions + VLM advisory review | real pipeline code (VLM optional; fixture default for determinism) |
| Cleaning / actuator behaviour | simulated — software-only command signal |
| Facade scene, scenario video, ground truth | synthetic, labelled as such |
| Real-data benchmark | honest numbers below, on untouched held-out data |

## Honest real-data numbers

Trained on the same config (yolov8n, imgsz 320, 32 epochs, seed 20260815),
benchmarked on the **untouched 149-image BFDD test split** (1,033
mask-derived defect boxes), identical protocol:

| Model | mAP50 | mAP50-95 | recall |
|---|---|---|---|
| synthetic baseline | 0.013 | 0.010 | 0.016 |
| BFDD-tuned | 0.094 | 0.043 | 0.141 |
| **BFDD + CUBIT** | **0.151** | **0.071** | **0.201** |

Adding CUBIT data improved the held-out BFDD benchmark by **+61% mAP50 /
+65% mAP50-95 / +42% recall**. CUBIT test and UAV2K test were also
benchmarked (reports under `artifacts/real-benchmark/`); CUBIT numbers are
leak-inflated (frame-interleaved split) and UAV2K is out-of-domain for all
three models — those are documented context, not ranking evidence. Full
detail: `docs/bfdd-cubit-experiment.md`.

This is a curated hackathon demo, not a field-ready inspection claim.

## Reproduce locally

    make setup
    make e2e-demo          # release gate: dataset -> train -> demo -> browser test

Live presentation after a checkpoint exists:

    make demo
    make backend           # terminal 1
    make frontend          # terminal 2  -> http://127.0.0.1:5173

## Run as a deployed single container

    docker build -t glasseye-demo .
    docker run -p 8000:8000 glasseye-demo
    # open http://127.0.0.1:8000

The image builds the frontend and regenerates the deterministic demo at boot,
then serves the SPA and API on one port (no GPU needed). `render.yaml`
deploys the same image to Render free tier.

## Submission assets

- Demo video: `frontend/tests/demo/recordings/glasseye-demo.webm` (40s,
  cursor + subtitles; regenerate with `node
  frontend/tests/demo/glasseye-demo.cjs` against a running server)
- Final dashboard screenshot:
  `frontend/tests/demo/recordings/glasseye-dashboard-final.png`
- Live walkthrough runbook: `docs/demo-runbook.md`
- Architecture: `docs/architecture.md`
- Data licence/audit: `docs/data-card.md`, `docs/cubit-data-card.md`,
  `docs/uav2k-data-card.md`
- Real-data experiment: `docs/bfdd-cubit-experiment.md`
