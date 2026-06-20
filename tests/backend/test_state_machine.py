import pytest

from backend.db.state_machine import (
    ANALYZING,
    CLOSED,
    DLQ_INVESTIGATION,
    MORE_INFO_REQUIRED,
    PROCESSING_FAILED,
    QUEUED,
    READY_FOR_REVIEW,
    REFERRED,
    SUBMITTED,
    UNDER_REVIEW,
    VALIDATING,
    VALIDATION_FAILED,
    InvalidTransitionError,
    is_valid_transition,
    transition,
)


class _FakeCase:
    def __init__(self, status: str) -> None:
        self.status = status
        self.updated_at = None


def test_happy_path_transitions_are_valid():
    assert is_valid_transition(SUBMITTED, VALIDATING)
    assert is_valid_transition(VALIDATING, "privacy_processing")
    assert is_valid_transition(QUEUED, ANALYZING)
    assert is_valid_transition(ANALYZING, READY_FOR_REVIEW)
    assert is_valid_transition(READY_FOR_REVIEW, UNDER_REVIEW)
    assert is_valid_transition(UNDER_REVIEW, REFERRED)
    assert is_valid_transition(UNDER_REVIEW, CLOSED)


def test_more_info_required_reenters_validating():
    assert is_valid_transition(UNDER_REVIEW, MORE_INFO_REQUIRED)
    assert is_valid_transition(MORE_INFO_REQUIRED, VALIDATING)


def test_failure_and_recovery_paths():
    assert is_valid_transition(VALIDATING, VALIDATION_FAILED)
    assert is_valid_transition(VALIDATION_FAILED, VALIDATING)
    assert is_valid_transition(VALIDATION_FAILED, CLOSED)

    assert is_valid_transition(ANALYZING, PROCESSING_FAILED)
    assert is_valid_transition(PROCESSING_FAILED, QUEUED)
    assert is_valid_transition(PROCESSING_FAILED, DLQ_INVESTIGATION)
    assert is_valid_transition(DLQ_INVESTIGATION, QUEUED)
    assert is_valid_transition(DLQ_INVESTIGATION, CLOSED)


def test_illegal_transition_rejected():
    assert not is_valid_transition(SUBMITTED, REFERRED)
    assert not is_valid_transition(REFERRED, UNDER_REVIEW)
    assert not is_valid_transition(CLOSED, VALIDATING)


def test_transition_applies_on_success():
    case = _FakeCase(status=SUBMITTED)
    transition(case, VALIDATING)
    assert case.status == VALIDATING
    assert case.updated_at is not None


def test_transition_raises_and_does_not_mutate_on_illegal_edge():
    case = _FakeCase(status=SUBMITTED)
    with pytest.raises(InvalidTransitionError):
        transition(case, REFERRED)
    assert case.status == SUBMITTED


def test_terminal_states_have_no_outgoing_transitions():
    assert not is_valid_transition(REFERRED, REFERRED)
    assert not is_valid_transition(CLOSED, CLOSED)