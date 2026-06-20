"""In-memory counters standing in for the Section 12 monitoring stack
(Prometheus/Grafana etc. in production) -- exposed via GET /admin/metrics
for this hackathon build."""

import threading
from collections import defaultdict

_lock = threading.Lock()
_counters: dict[str, int] = defaultdict(int)


def increment(name: str, by: int = 1) -> None:
    with _lock:
        _counters[name] += by


def snapshot() -> dict[str, int]:
    with _lock:
        return dict(_counters)


def reset_all() -> None:
    """Test-only hook."""
    with _lock:
        _counters.clear()