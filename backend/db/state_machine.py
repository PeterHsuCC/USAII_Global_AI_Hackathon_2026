"""Case status state machine (v6 Section 15.1).

Encodes the same transition graph the doc specifies as a Postgres
`BEFORE UPDATE` trigger, enforced here at the application layer instead
(SQLite has no equivalent trigger ergonomics) -- see plan decision #3.
"""

from datetime import datetime, timezone

SUBMITTED = "submitted"
VALIDATING = "validating"
PRIVACY_PROCESSING = "privacy_processing"
QUEUED = "queued"
ANALYZING = "analyzing"
READY_FOR_REVIEW = "ready_for_review"
UNDER_REVIEW = "under_review"
REFERRED = "referred"
CLOSED = "closed"
MORE_INFO_REQUIRED = "more_information_required"
VALIDATION_FAILED = "validation_failed"
PROCESSING_FAILED = "processing_failed"
DLQ_INVESTIGATION = "dlq_investigation"

# Doc text: "Failure states (from any processing stage)... Validation Failed
# .../ Processing Failed ...". The doc doesn't enumerate which forward state
# maps to which failure type, so this maps Validating -> Validation Failed
# (the validation stage itself) and the three downstream processing stages
# -> Processing Failed, which is the most direct reading of the two failure
# names against the forward graph in the same section.
TRANSITIONS: dict[str, set[str]] = {
    SUBMITTED: {VALIDATING},
    VALIDATING: {PRIVACY_PROCESSING, VALIDATION_FAILED},
    PRIVACY_PROCESSING: {QUEUED, PROCESSING_FAILED},
    QUEUED: {ANALYZING, PROCESSING_FAILED},
    ANALYZING: {READY_FOR_REVIEW, PROCESSING_FAILED},
    READY_FOR_REVIEW: {UNDER_REVIEW},
    UNDER_REVIEW: {REFERRED, CLOSED, MORE_INFO_REQUIRED},
    MORE_INFO_REQUIRED: {VALIDATING},
    VALIDATION_FAILED: {VALIDATING, CLOSED},
    PROCESSING_FAILED: {QUEUED, DLQ_INVESTIGATION},
    DLQ_INVESTIGATION: {QUEUED, CLOSED},
    REFERRED: set(),
    CLOSED: set(),
}

ALL_STATUSES: tuple[str, ...] = tuple(TRANSITIONS.keys())


class InvalidTransitionError(Exception):
    def __init__(self, current: str, new: str) -> None:
        super().__init__(f"Illegal case status transition: {current!r} -> {new!r}")
        self.current = current
        self.new = new


def is_valid_transition(current: str, new: str) -> bool:
    return new in TRANSITIONS.get(current, set())


def transition(case, new_status: str) -> None:
    """Validates and applies a case status transition in place.

    Raises InvalidTransitionError without mutating `case` on an illegal edge,
    so the caller's surrounding DB transaction can roll back cleanly -- the
    same rollback-on-violation behavior the doc's trigger gives Postgres.
    """
    if not is_valid_transition(case.status, new_status):
        raise InvalidTransitionError(case.status, new_status)
    case.status = new_status
    case.updated_at = datetime.now(timezone.utc)