#!/usr/bin/env python3
"""Execute the full deterministic GlassEye closed-loop demo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.demo import DemoRunner
from app.replay import replay_log
from app.schemas import IssueStatus


def main() -> int:
    result = DemoRunner().run()
    statuses = {issue.class_name.value: issue.status.value for issue in result.issues}
    if statuses.get("cleanable_surface_issue") != IssueStatus.RESOLVED.value:
        raise RuntimeError(f"Expected resolved cleanable issue; got {statuses}.")
    if statuses.get("structural_issue") != IssueStatus.ESCALATED.value:
        raise RuntimeError(f"Expected escalated structural issue; got {statuses}.")
    projection = replay_log(ROOT / result.event_log_ref)
    if projection.replay_digest != result.replay_digest:
        raise RuntimeError(
            "Event log replay digest does not match the live mission result."
        )
    report = {
        "mission_id": result.mission_id,
        "model_version": result.model_version,
        "issues": statuses,
        "event_count": len(result.events),
        "replay_digest": result.replay_digest,
        "event_log": result.event_log_ref,
        "inference_benchmark": result.inference_benchmark,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
