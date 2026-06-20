"""In-memory rate limiting (v6 Section 6, illustrative limits in 6.1).

Counter state lives in plain in-memory structures behind a small
interface -- standing in for Redis (plan decision #5). Swapping in a real
Redis-backed limiter later only touches this module.
"""

import threading
import time
from collections import defaultdict, deque
from uuid import UUID

from backend.config import settings


class RateLimitExceeded(Exception):
    def __init__(self, retry_after_seconds: float) -> None:
        super().__init__(f"Rate limit exceeded, retry after {retry_after_seconds:.1f}s")
        self.retry_after_seconds = retry_after_seconds


class _SlidingWindowCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def hit(self, key: str, limit: int, window_seconds: float) -> None:
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            cutoff = now - window_seconds
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = window_seconds - (now - bucket[0])
                raise RateLimitExceeded(max(retry_after, 0.0))
            bucket.append(now)


class _ConcurrencyGauge:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = defaultdict(int)

    def acquire(self, key: str, limit: int) -> None:
        with self._lock:
            if self._counts[key] >= limit:
                raise RateLimitExceeded(retry_after_seconds=5.0)
            self._counts[key] += 1

    def release(self, key: str) -> None:
        with self._lock:
            if self._counts[key] > 0:
                self._counts[key] -= 1


_job_submit_counter = _SlidingWindowCounter()
_read_counter = _SlidingWindowCounter()
_bulk_export_counter = _SlidingWindowCounter()
_concurrent_jobs = _ConcurrencyGauge()


def check_job_submission(analyst_id: UUID, organization_id: UUID) -> None:
    _job_submit_counter.hit(f"analyst:{analyst_id}", settings.rate_limit_job_submit_per_analyst_per_min, 60.0)
    _job_submit_counter.hit(f"org:{organization_id}", settings.rate_limit_job_submit_per_org_per_min, 60.0)


def acquire_concurrent_job_slot(analyst_id: UUID) -> None:
    """Held from submission until the job reaches a terminal state
    (succeeded or DLQ) -- released in backend.jobs.queue.process_job_once,
    not on a scheduled retry, since the job is still in flight then."""
    _concurrent_jobs.acquire(f"analyst:{analyst_id}", settings.rate_limit_concurrent_jobs_per_analyst)


def release_concurrent_job_slot(analyst_id: UUID) -> None:
    _concurrent_jobs.release(f"analyst:{analyst_id}")


def check_read(analyst_id: UUID) -> None:
    _read_counter.hit(f"analyst:{analyst_id}", settings.rate_limit_reads_per_analyst_per_min, 60.0)


def check_bulk_export(organization_id: UUID) -> None:
    _bulk_export_counter.hit(f"org:{organization_id}", settings.rate_limit_bulk_export_per_org_per_hour, 3600.0)


def reset_all() -> None:
    """Test-only hook -- clears all in-memory counters."""
    global _job_submit_counter, _read_counter, _bulk_export_counter, _concurrent_jobs
    _job_submit_counter = _SlidingWindowCounter()
    _read_counter = _SlidingWindowCounter()
    _bulk_export_counter = _SlidingWindowCounter()
    _concurrent_jobs = _ConcurrencyGauge()