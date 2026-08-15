"""Append-only deterministic mission events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import MissionEvent


@dataclass
class DeterministicClock:
    timestamp: float = 0.0
    step_seconds: float = 1.25

    def tick(self) -> float:
        current = self.timestamp
        self.timestamp = round(self.timestamp + self.step_seconds, 6)
        return current


class EventLog:
    """Append JSONL events with deterministic identifiers and an in-memory projection."""

    def __init__(self, mission_id: str, path: Path, clock: DeterministicClock | None = None) -> None:
        self.mission_id = mission_id
        self.path = path
        self.clock = clock or DeterministicClock()
        self.events: list[MissionEvent] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def emit(
        self,
        event_type: str,
        *,
        source: str,
        reason_code: str,
        issue_id: str | None = None,
        track_id: int | None = None,
        evidence_refs: list[str] | None = None,
        payload: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> MissionEvent:
        sequence = len(self.events) + 1
        event = MissionEvent(
            event_id=f"evt-{self.mission_id}-{sequence:04d}",
            mission_id=self.mission_id,
            sequence=sequence,
            timestamp=self.clock.tick() if timestamp is None else round(timestamp, 6),
            event_type=event_type,
            source=source,
            issue_id=issue_id,
            track_id=track_id,
            reason_code=reason_code,
            evidence_refs=evidence_refs or [],
            payload=payload or {},
        )
        self.events.append(event)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n")
        return event

    @classmethod
    def load(cls, path: Path) -> list[MissionEvent]:
        if not path.exists():
            return []
        return [
            MissionEvent.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
