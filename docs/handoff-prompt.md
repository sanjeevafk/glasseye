# GlassEye handoff prompt

You are continuing the GlassEye project. Read `README.md`,
`docs/architecture.md`, `docs/data-card.md`, and `docs/demo-runbook.md` first.

GlassEye is a deterministic façade-inspection software pipeline: YOLO runs on
recorded video; detections become tracked issues, evidence crops, façade panel
locations, policy decisions, append-only events, replay, and a Three.js
dashboard. It is simulation-only: no physical drone or actuator is controlled.

## Current demo flow

1. YOLO detects a seeded cleanable surface issue and a seeded structural issue.
2. The cleanable issue is approved by policy, emits a software-only cleaning
   command modal with `SIMULATED_COMPLETE`, is reinspected, and becomes
   `RESOLVED`.
3. The structural issue is never cleaned; it becomes `ESCALATED` for human
   structural review, with an advisory VLM review of its evidence crop.
4. The Three.js panel map, evidence cards, command signal, and timeline are
   projections of the same event log.

Run the complete proof with `make setup` followed by `make e2e-demo`.

## Live deployment

The app is deployed and verified at **https://glasseye-td75.onrender.com**
(Render, free tier, single Docker container). Auto-deploy is on: every push to
`main` rebuilds the image and ships it (~4-6 min; the demo model is baked in,
so there is no training at build or boot).

- **Boot is instant**: the deterministic demo (dataset, checkpoint, scenario
  video, mission events, VLM review) is baked into the image at build time.
- **RUN DEMO is instant**: `POST /api/demo/run` serves the baked result
  (~0.7 s) instead of re-running inference — the demo is fully seeded, so
  every run is identical anyway. The pipeline only actually runs when no baked
  result exists (local dev before `make demo`).
- **Free tier caveat**: the instance spins down after ~15 min idle; first load
  after idle takes ~30-60 s to wake. Warm it ~5 min before presenting.
- **Deploy plumbing**: `Dockerfile` (CPU-only torch; frontend dist COPY is
  last so frontend-only deploys reuse the cached demo layer), `render.yaml`,
  `.dockerignore`. The 6 MB synthetic checkpoint is committed
  (`models/glasseye-yolo-v1/`) so `train_yolo.py --if-missing` skips training
  at build; the `training_runs/` subdir stays gitignored.

## Advisory VLM review (live on the deployed site)

`backend/app/vlm.py` is provider-neutral:

- `FixtureVlmProvider` (default, `DEMO_VLM_MODE=fixture`): deterministic
  rule-based reviewer, no network.
- `HttpVlmProvider` (`DEMO_VLM_MODE=http`): OpenAI-compatible
  chat-completions provider gated behind `GLASSEYE_VLM_API_KEY`, with
  `GLASSEYE_VLM_BASE_URL`, `GLASSEYE_VLM_MODEL`, `GLASSEYE_VLM_TIMEOUT_SECONDS`.
- **The deployed demo uses a real NVIDIA vision model**:
  `DEMO_VLM_MODE=http`, base URL `https://integrate.api.nvidia.com/v1`,
  model `meta/llama-3.2-11b-vision-instruct`. The key lives as a Render
  service env var, which Render injects as a Docker build arg so the VLM
  review is baked into the demo at build time. (The 90B model was too slow;
  11B cold ~22 s, warm ~2 s.)
- Only the selected YOLO evidence crop plus structured metadata is submitted;
  never raw frames. Strict verdicts: `confirm`, `reject`, `escalate`.
- Routing: structural (high-impact) detections are always reviewed; cleanable
  detections only when the deterministic policy would send them to REVIEW.
  The VLM never controls an actuator. Structural issues always ESCALATE
  regardless of verdict, so a live VLM cannot break the demo outcome.
- On VLM failure/timeout/malformed output/unavailable credentials the
  decision routes to `REVIEW` (never `CLEAN`); the demo degrades to
  `VLM_UNAVAILABLE` events instead of failing.
- `VLM_REVIEW_REQUESTED` / `VLM_REVIEW_RESULT` events are appended to the log;
  the verdict/rationale/provider/latency render in the issue card and replay.

**Security note (hackathon tradeoff)**: the NVIDIA key is in the Render env
config and, via the build arg, touches the image's build layer. Acceptable for
the demo; rotate the key after the event if it matters.

To run the demo locally with the real provider:

    DEMO_VLM_MODE=http GLASSEYE_VLM_API_KEY=... make backend

To run with the deterministic fixture (no key):

    make backend

## Frontend hardening (done)

- **WebGL fallback**: `FacadeScene` catches `WebGLRenderer` creation failure
  (blocked GPU — common on VMs/remote desktops) and renders a static panel
  map instead of crashing. This fixed a real blank-screen bug on the live
  site.
- **ErrorBoundary** wraps the scene as a safety net — no single component can
  blank the whole dashboard.
- **Theme**: the default AI-blue palette was replaced with the project's own
  aesthetic from the architecture diagram — dark ink-slate (`#2d3142`),
  paper (`#f5f5f5`), burnt orange (`#eb6c36`) — warm industrial console look,
  verified 0% blue pixels across dashboard and modals.

## Model and data status

- Synthetic checkpoint: `models/glasseye-yolo-v1/best.pt` (committed, 6 MB).
- BFDD-tuned checkpoint: `models/glasseye-yolo-real-bfdd-v1/best.pt`.
- **Recommended demo model**: `models/glasseye-yolo-bfdd-cubit-v1/best.pt`
  (BFDD + CUBIT combined).
- On the held-out 149-image BFDD test split (untouched, mask-to-box protocol
  `--min-component-area 512`):

  | Model | mAP50 | mAP50-95 | recall |
  |---|---|---|---|
  | `glasseye-yolo-v1` (synthetic) | 0.013104 | 0.009923 | — |
  | `glasseye-yolo-real-bfdd-v1` (BFDD) | 0.094017 | 0.042910 | 0.141336 |
  | `glasseye-yolo-bfdd-cubit-v1` (BFDD+CUBIT) | **0.151198** | **0.070703** | **0.201355** |

  BFDD+CUBIT vs BFDD-only: mAP50 **+60.8%**, mAP50-95 **+64.8%**, recall
  **+42.5%**. Full breakdown, CUBIT benchmark (leak-inflated — frame
  interleaving), UAV2K out-of-domain benchmark (all models near zero; combined
  is least-bad), and the honest recommendation are in
  `docs/bfdd-cubit-experiment.md`.
- This supports a curated hackathon demo, not a field-ready building-inspection
  claim. UAV2K/CUBIT/BFDD raw datasets, generated datasets, model weights,
  archives, videos, and reports are local and gitignored.

## Verification at handoff

The following passed at handoff:

    .venv/bin/ruff check .
    PYTHONPATH=backend .venv/bin/pytest backend/tests -q        # 26 passed
    npm --prefix frontend run build
    npm --prefix frontend run test:e2e

The E2E browser test verifies trained-YOLO inference, dashboard rendering,
command-modal acknowledgement, final `RESOLVED`/`ESCALATED` statuses,
evidence, timeline, and replay. The live site was additionally verified in a
real browser (no-WebGL fallback + normal 3D scene, both zero console errors).

## Submission assets (hackathon package)

- Live URL: https://glasseye-td75.onrender.com
- Demo video: `frontend/tests/demo/recordings/glasseye-demo.webm` (~40 s,
  cursor + subtitles) — recorded before the re-theme; fine as-is.
- Screenshots: `frontend/tests/demo/recordings/` (`glasseye-themed.png`,
  `modal-cleaning-themed.png`, `modal-maintenance-themed.png`,
  `glasseye-dashboard-final.png`).
- `docs/submission-package.md` (judge-facing orientation + honest metrics).
- `docs/live-walkthrough-notes.md` (5-7 min pitch + Q&A prep).

## Reproduce commands

BFDD test comparison (after extracting BFDD under
`data/external/bfdd/Dataset_1x`):

    make prepare-bfdd
    PYTHONPATH=backend .venv/bin/python scripts/benchmark_real_data.py --dataset bfdd --bfdd-split test --min-component-area 512 --checkpoint models/glasseye-yolo-v1/best.pt
    PYTHONPATH=backend .venv/bin/python scripts/benchmark_real_data.py --dataset bfdd --bfdd-split test --min-component-area 512 --checkpoint models/glasseye-yolo-real-bfdd-v1/best.pt
    PYTHONPATH=backend .venv/bin/python scripts/benchmark_real_data.py --dataset bfdd --bfdd-split test --min-component-area 512 --checkpoint models/glasseye-yolo-bfdd-cubit-v1/best.pt

Combined-dataset prep is reproducible via
`scripts/prepare_bfdd_cubit_dataset.py` and
`scripts/prepare_bfdd_cubit_uav2k_dataset.py` (three-source dataset prepared
and verified at `data/bfdd_cubit_uav2k_binary_v1` — train 2,899 / val 289 /
test 1,050; not yet trained). UAV2K benchmark:
`scripts/benchmark_uav2k_data.py`.

## Follow-up priorities

1. **Optional**: train the three-source model (BFDD + CUBIT + UAV2K) with the
   same config as before (~2 h on the MX550, ~20 min on a Colab T4 once the
   11 GB dataset is uploaded/resized) and re-run the apples-to-apples BFDD
   benchmark. The recommendation stands either way — the current
   `glasseye-yolo-bfdd-cubit-v1` is the demo model.
2. Re-record the demo video with the new theme + live VLM verdict if the
   current one feels dated.
3. Rotate the NVIDIA VLM API key after the hackathon.
4. Keep the BFDD test split untouched forever — it is the comparison
   benchmark; never train or tune on it.
