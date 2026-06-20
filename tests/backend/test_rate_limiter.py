import uuid

import pytest

from backend.rate_limit import limiter


@pytest.fixture(autouse=True)
def _reset_limiter():
    limiter.reset_all()
    yield
    limiter.reset_all()


def test_job_submission_allows_up_to_per_analyst_limit():
    analyst_id, org_id = uuid.uuid4(), uuid.uuid4()
    for _ in range(5):
        limiter.check_job_submission(analyst_id, org_id)

    with pytest.raises(limiter.RateLimitExceeded):
        limiter.check_job_submission(analyst_id, org_id)


def test_job_submission_per_org_limit_shared_across_analysts():
    org_id = uuid.uuid4()
    # 4 analysts x 5 jobs = 20, hitting the org cap before any individual
    # analyst hits their own 5/min limit.
    for _ in range(4):
        analyst_id = uuid.uuid4()
        for _ in range(5):
            limiter.check_job_submission(analyst_id, org_id)

    with pytest.raises(limiter.RateLimitExceeded):
        limiter.check_job_submission(uuid.uuid4(), org_id)


def test_concurrent_job_slot_enforces_limit_and_release_frees_it():
    analyst_id = uuid.uuid4()
    for _ in range(3):
        limiter.acquire_concurrent_job_slot(analyst_id)

    with pytest.raises(limiter.RateLimitExceeded):
        limiter.acquire_concurrent_job_slot(analyst_id)

    limiter.release_concurrent_job_slot(analyst_id)
    limiter.acquire_concurrent_job_slot(analyst_id)  # slot freed, should succeed


def test_read_limit_independent_per_analyst():
    analyst_a, analyst_b = uuid.uuid4(), uuid.uuid4()
    for _ in range(120):
        limiter.check_read(analyst_a)

    with pytest.raises(limiter.RateLimitExceeded):
        limiter.check_read(analyst_a)

    limiter.check_read(analyst_b)  # independent counter, not affected


def test_bulk_export_limit_per_org():
    org_id = uuid.uuid4()
    limiter.check_bulk_export(org_id)
    limiter.check_bulk_export(org_id)

    with pytest.raises(limiter.RateLimitExceeded):
        limiter.check_bulk_export(org_id)