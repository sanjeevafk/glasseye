# GlassEye architecture

GlassEye is a local modular monolith. The dashboard never invents mission state:
it renders the result and append-only event log returned by the backend.

## Components

| Component | Responsibility | Contract |
|---|---|---|
| detector.py | Load the fine-tuned Ultralytics YOLO checkpoint and produce per-frame detections with SAHI support | DetectorFrame with nullable mask |
| video_inspector.py | Slices drone flight footage at configurable FPS, computes 4×3 panel damage heatmap, timeline scrubber, and summary | VideoInspectionResult |
| image_inspector.py | Evaluates static high-res facade photos, calculates 0-100 Integrity Index, and triggers advisory VLM second opinions | ImageInspectionResult |
| perception.py and tracker.py | Read recorded video, apply deterministic IoU tracking, and save evidence crops | VideoInference and EvidenceRecord |
| localization.py | Place detection centroids on a fixed 4 × 3 facade grid | FacadeLocation |
| policies.py | Decide CLEAN, ESCALATE, or REVIEW from configuration | PolicyDecision |
| state_machines.py | Enforce issue lifecycle transitions | IssueStatus transition |
| simulator.py | Record a simulated cleaning operation only | SimulatedCleaningResult |
| verification.py | Compare actual YOLO reinspection output against the original issue location | RESOLVED or UNRESOLVED |
| events.py and replay.py | Store JSONL events and independently replay the final state | ReplayProjection |
| frontend | Render panel state, evidence, drone video player, damage heatmaps, final outcomes, and 3D event replay | FastAPI JSON API |

The architecture adapts the useful seams observed in Vanrakshak: Pydantic
contracts, an event-first replay model, a side-effect-free policy engine, and
testable state transitions. No forest, threat, hardware, or mission-specific
code was copied.

## Fixed deterministic scenario

The synthetic scenario seed is 20260815. It creates two recorded MP4 files:

1. preinspection.mp4 contains a brown cleanable surface issue on panel B2 and a
   branching structural issue on panel C3;
2. the policy allows a simulated cleaning action only for B2;
3. reinspection.mp4 removes the B2 issue but retains C3;
4. actual YOLO inference runs on both videos;
5. verification marks B2 RESOLVED and C3 ESCALATED;
6. replay hashes the append-only events and must reproduce both states.

If YOLO does not detect both preinspection defects or the persistent structural
defect, the runner fails. It does not substitute canned detections.

## Evented lifecycle

Each issue uses the following constrained lifecycle:

IDLE → INSPECTING → DETECTED → EVIDENCE_READY → DECIDED

For a cleanable issue:

DECIDED → CLEANING → REINSPECTING → RESOLVED or UNRESOLVED → ESCALATED

For a structural issue:

DECIDED → ESCALATED

Every transition is appended to events.jsonl with a deterministic sequence,
timestamp, reason code, issue/track IDs, payload, and evidence references.
