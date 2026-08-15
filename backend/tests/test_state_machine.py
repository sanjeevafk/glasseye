import pytest

from app.schemas import IssueStatus
from app.state_machines import InvalidTransition, IssueStateMachine


def test_lifecycle_enforces_closed_loop_transition_order():
    machine = IssueStateMachine()
    for state in (
        IssueStatus.INSPECTING,
        IssueStatus.DETECTED,
        IssueStatus.EVIDENCE_READY,
        IssueStatus.DECIDED,
        IssueStatus.CLEANING,
        IssueStatus.REINSPECTING,
        IssueStatus.RESOLVED,
    ):
        machine.transition(state)

    assert machine.state == IssueStatus.RESOLVED
    with pytest.raises(InvalidTransition):
        machine.transition(IssueStatus.CLEANING)
