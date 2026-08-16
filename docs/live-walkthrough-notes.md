# GlassEye — live walkthrough notes

Timing target: **5–7 minutes** for the demo, then Q&A. The flow below follows
the deterministic demo; every claim maps to something visible on screen.

## Setup before you present

1. Ensure the demo has run: `make e2e-demo` (or use the deployed container).
2. Start the servers (`make backend` + `make frontend`, or one container).
3. Pre-open the dashboard and let the canvas render so there is no white
   flash during the pitch.
4. If presenting the deployed URL, load it once before the pitch so the boot
   demo is already generated.

## The pitch (0:00–1:00) — what it is

> "GlassEye is a deterministic façade-inspection pipeline: real YOLO
> detection on recorded video, tracking, evidence, policy with an optional
> vision-language review, simulated remediation, and full replay — all
> projected onto a Three.js panel map."

Key line: *"It's simulation-only — we never control a physical drone or
actuator. That honesty is part of the design."*

## Demo (1:00–5:00)

### 1. Dashboard orientation (~30s)
- Point at the facade panel grid; note the scene is **labelled synthetic**.
- Point at the timeline: it's the same event log that drives everything.

### 2. Run the mission (~1:30, mostly wait)
- Click RUN DETERMINISTIC DEMO.
- While YOLO runs, say what's happening under the hood: "real inference on a
  seeded video — two issues are tracked: a cleanable surface issue and a
  structural issue, each with an evidence crop."

### 3. Cleaning command modal (~45s)
- When the modal appears: "policy approved the cleanable issue after the
  confidence/stability gate and an advisory VLM review. This is a
  software-only command signal — simulated complete."
- Acknowledge.

### 4. Structural maintenance modal (~45s)
- "The structural issue is never cleaned — it's escalated for human
  structural review with a maintenance dispatch."
- Acknowledge.

### 5. Final state (~1:30)
- Cleanable issue card: `RESOLVED` — "real YOLO reinspection confirmed the
  defect is gone."
- Structural issue card: `ESCALATED` — VLM verdict visible on the card.
- Timeline: 28 events including VLM REVIEW RESULT.
- Use STEP/PLAY to show replay: "this is a projection of the same JSONL event
  log — replayable, hash-verified."

## Honest-numbers segment (if asked / time allows)

- Same config, untouched 149-image BFDD test split:
  - synthetic baseline: mAP50 0.013
  - BFDD-tuned: mAP50 0.094
  - **BFDD+CUBIT: mAP50 0.151** (+61%), recall 0.201 (+42%)
- One-line framing: *"Adding a second real dataset improved the held-out
  benchmark substantially; details are in the repo, and this is a curated
  hackathon demo, not a field-ready claim."*

## Likely Q&A

- **Is the VLM real?** Yes, provider-neutral; deterministic fixture by
  default, OpenAI-compatible HTTP provider available. It never controls an
  actuator — worst case it routes to human review.
- **Why not real drone control?** Simulation-only by design; the pipeline
  (detect → track → evidence → decide → verify → replay) is the demo.
- **Can it run without a GPU?** Yes — the deterministic demo runs on CPU;
  the container needs no GPU.
- **What are the numbers?** Point at the table above; stress the untouched
  held-out split and the honesty caveats (CUBIT leakage, UAV2K out-of-domain).

## Pitfalls

- Don't claim real-world readiness. Say "hackathon demo, honest about it."
- Don't quote CUBIT 0.199 as a clean number — it's leak-inflated.
- If the modal doesn't appear within ~2 min, say "the pipeline is running
  real inference" and wait — do not click twice.
