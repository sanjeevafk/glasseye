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
   structural review.
4. The Three.js panel map, evidence cards, command signal, and timeline are
   projections of the same event log.

Run the complete proof with `make setup` followed by `make e2e-demo`.

## Verification at handoff

The following passed before handoff:

    .venv/bin/ruff check .
    PYTHONPATH=backend .venv/bin/pytest backend/tests -q
    npm --prefix frontend run build
    npm --prefix frontend run test:e2e

The E2E browser test verifies trained-YOLO inference, dashboard rendering,
command-modal acknowledgement, final `RESOLVED`/`ESCALATED` statuses, evidence,
timeline, and replay.

## Model and data status

- Synthetic checkpoint: `models/glasseye-yolo-v1/best.pt`.
- BFDD-tuned checkpoint: `models/glasseye-yolo-real-bfdd-v1/best.pt`.
- On the held-out 149-image BFDD test split, the synthetic checkpoint reached
  mAP50 `0.013104` / mAP50-95 `0.009923`; the BFDD-tuned checkpoint reached
  mAP50 `0.094017` / mAP50-95 `0.042910`, with recall `0.141336`.
- This supports a curated hackathon demo, not a field-ready building-inspection
  claim.
- Generated datasets, model weights, archives, videos, and reports are local
  and intentionally gitignored.

To reproduce the BFDD test comparison after extracting BFDD under
`data/external/bfdd/Dataset_1x`:

    make prepare-bfdd
    PYTHONPATH=backend .venv/bin/python scripts/benchmark_real_data.py --dataset bfdd --bfdd-split test --min-component-area 512 --checkpoint models/glasseye-yolo-v1/best.pt
    PYTHONPATH=backend .venv/bin/python scripts/benchmark_real_data.py --dataset bfdd --bfdd-split test --min-component-area 512 --checkpoint models/glasseye-yolo-real-bfdd-v1/best.pt

## Advisory VLM review (implemented)

The advisory VLM path is implemented in `backend/app/vlm.py` behind a
provider-neutral adapter:

- `FixtureVlmProvider` (default, `DEMO_VLM_MODE=fixture`): deterministic
  rule-based reviewer for tests and presentation runs.
- `HttpVlmProvider` (`DEMO_VLM_MODE=http`): OpenAI-compatible
  chat-completions provider gated behind `GLASSEYE_VLM_API_KEY` (with
  `GLASSEYE_VLM_BASE_URL`, `GLASSEYE_VLM_MODEL`, `GLASSEYE_VLM_TIMEOUT_SECONDS`).
- Only the selected YOLO evidence crop plus structured metadata is submitted;
  never raw frames.
- Strict verdicts: `confirm`, `reject`, or `escalate` with a concise rationale.
- Routing: structural (high-impact) detections are always reviewed; cleanable
  detections only when the deterministic policy would send them to REVIEW.
  The VLM never controls an actuator — it can only move a cleanable case away
  from CLEAN (reject → REVIEW) or toward human escalation (escalate →
  ESCALATE); a confirm verdict still requires the confidence/stability gate.
- On VLM failure, timeout, malformed output, or unavailable credentials the
  decision routes to `REVIEW` (never `CLEAN`); a structural issue still
  escalates, and a cleanable issue that cannot be cleared causes the mission
  to abort rather than clean.
- `VLM_REVIEW_REQUESTED` / `VLM_REVIEW_RESULT` events are appended to the log
  and the verdict/rationale is shown in the issue card and replay timeline.

To run the demo with the real provider once an API key is set:

    DEMO_VLM_MODE=http GLASSEYE_VLM_API_KEY=... make backend

## Follow-up priorities

1. Add a matching maintenance-dispatch signal/modal for structural escalation.
2. ~~Resolve UAV2K extraction and validate its annotations~~ — done:
   extraction is under `data/external/uav2k/`, 90 damaged label files were
   regenerated from the intact COCO JSONs and validated
   (`scripts/recover_uav2k_labels.py`, `artifacts/uav2k-recovery-report.json`,
   `docs/uav2k-data-card.md`). Still blocked on license selection
   (`LICENSE_SELECTION_REQUIRED.txt`) — local use only.
3. Improve BFDD mask-to-box conversion and train a multi-source model while
   preserving the BFDD test split untouched. A three-source dataset
   (BFDD + CUBIT + UAV2K) is prepared and verified at
   `data/bfdd_cubit_uav2k_binary_v1` (`scripts/prepare_bfdd_cubit_uav2k_dataset.py`);
   training it is the natural next experiment.
4. Re-run the CUBIT audit/preparation and the combined-model benchmark
   (`scripts/audit_cubit_dataset.py`, `scripts/prepare_bfdd_cubit_dataset.py`,
   `scripts/benchmark_real_data.py --dataset bfdd`, `scripts/benchmark_cubit_data.py`)
   and record the final recommendation in `docs/bfdd-cubit-experiment.md`.
