"""A clearly simulated cleaning adapter; it never controls physical hardware."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulatedCleaningResult:
    action_id: str
    status: str
    description: str


class CleaningSimulator:
    def run(self, issue_id: str, panel_id: str) -> SimulatedCleaningResult:
        return SimulatedCleaningResult(
            action_id=f"sim-clean-{issue_id}",
            status="SIMULATED_COMPLETE",
            description=f"Simulated surface cleaning on facade panel {panel_id}; no physical actuator was used.",
        )
