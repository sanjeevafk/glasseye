"""Explicit and testable issue lifecycle transitions."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import IssueStatus


class InvalidTransition(ValueError):
    pass


_ALLOWED: dict[IssueStatus, set[IssueStatus]] = {
    IssueStatus.IDLE: {IssueStatus.INSPECTING},
    IssueStatus.INSPECTING: {IssueStatus.DETECTED},
    IssueStatus.DETECTED: {IssueStatus.EVIDENCE_READY},
    IssueStatus.EVIDENCE_READY: {IssueStatus.DECIDED},
    IssueStatus.DECIDED: {IssueStatus.CLEANING, IssueStatus.ESCALATED, IssueStatus.REVIEW},
    IssueStatus.CLEANING: {IssueStatus.REINSPECTING},
    IssueStatus.REINSPECTING: {IssueStatus.RESOLVED, IssueStatus.UNRESOLVED},
    IssueStatus.RESOLVED: set(),
    IssueStatus.UNRESOLVED: {IssueStatus.ESCALATED},
    IssueStatus.ESCALATED: set(),
    IssueStatus.REVIEW: set(),
}


@dataclass
class IssueStateMachine:
    state: IssueStatus = IssueStatus.IDLE

    def transition(self, next_state: IssueStatus) -> tuple[IssueStatus, IssueStatus]:
        if next_state not in _ALLOWED[self.state]:
            raise InvalidTransition(f"{self.state} -> {next_state} is not allowed")
        previous = self.state
        self.state = next_state
        return previous, next_state
