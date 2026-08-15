"""Deterministic event-log replay and canonical result hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .events import EventLog
from .schemas import IssueStatus, MissionEvent, ReplayProjection


def canonical_event_digest(events: list[MissionEvent]) -> str:
    canonical = [
        {
            "event_id": event.event_id,
            "sequence": event.sequence,
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "issue_id": event.issue_id,
            "track_id": event.track_id,
            "reason_code": event.reason_code,
            "evidence_refs": event.evidence_refs,
            "payload": event.payload,
        }
        for event in events
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def replay_events(events: list[MissionEvent]) -> ReplayProjection:
    mission_id = events[0].mission_id if events else "unknown"
    statuses: dict[str, IssueStatus] = {}
    for event in events:
        if event.event_type == "ISSUE_STATUS_CHANGED" and event.issue_id:
            statuses[event.issue_id] = IssueStatus(event.payload["next_state"])
    return ReplayProjection(
        mission_id=mission_id,
        event_count=len(events),
        issue_statuses=statuses,
        replay_digest=canonical_event_digest(events),
    )


def replay_log(path: Path) -> ReplayProjection:
    return replay_events(EventLog.load(path))
