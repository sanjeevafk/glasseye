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

## Required next feature: advisory VLM review

The original specification requires an optional advisory VLM path for ambiguous
or high-impact detections. It has **not** been implemented yet. Current policy
is deterministic-only: high-confidence cleanable issues clean, structural
issues escalate, and other cases review.

Implement this next:

1. Add `backend/app/vlm.py` behind a provider-neutral adapter.
2. Submit only a selected YOLO evidence crop plus structured metadata; do not
   send every video frame.
3. Validate a strict structured verdict: `confirm`, `reject`, or `escalate`,
   with a concise rationale.
4. Route ambiguous/high-impact detections through VLM before policy. A VLM must
   never directly control an actuator.
5. On VLM failure, timeout, malformed output, or unavailable credentials,
   route to `REVIEW`, never `CLEAN`.
6. Emit VLM request/result events and show the verdict/rationale in the issue
   card and replay timeline.
7. Add a deterministic fixture adapter for tests and presentation runs, exposed
   with `DEMO_VLM_MODE=fixture`. Keep the real provider mode behind an API key.

Expected implementation effort is 4–6 hours including adapter, policy/event/UI
integration, tests, and fallback handling. For a hackathon, VLM usage should be
low-cost because it should review only 2–10 selected crops per demo. The main
tradeoffs are cloud/API-key dependency, review latency, and nondeterminism;
the fixture mode keeps the E2E demo repeatable.

## Follow-up priorities

1. Implement the advisory VLM path above.
2. Add a matching maintenance-dispatch signal/modal for structural escalation.
3. Resolve UAV2K extraction and validate its annotations.
4. Improve BFDD mask-to-box conversion and train a multi-source model while
   preserving the BFDD test split untouched.
