import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from train_grooming import CachedLLMSafetyExtractor, save_artifact  # noqa: E402


def test_save_does_not_corrupt_existing_cache_if_the_write_fails(tmp_path):
    cache_path = tmp_path / "llm_signal_cache.json"
    original = {"existing:0-1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    cache_path.write_text(json.dumps(original), encoding="utf-8")

    extractor = CachedLLMSafetyExtractor(cache_path)
    extractor._cache["new:0-1"] = [0.0] * 6
    extractor._dirty = True

    with patch("json.dump", side_effect=RuntimeError("simulated crash mid-write")):
        try:
            extractor.save()
        except RuntimeError:
            pass

    # The real cache file must be untouched -- still the old, fully valid
    # content -- not truncated or left as invalid JSON by the failed write.
    on_disk = json.loads(cache_path.read_text(encoding="utf-8"))
    assert on_disk == original


def test_save_then_reload_round_trips_through_the_real_path(tmp_path):
    cache_path = tmp_path / "llm_signal_cache.json"

    extractor = CachedLLMSafetyExtractor(cache_path)
    extractor._cache["a:0-1"] = [0.1] * 6
    extractor._dirty = True
    extractor.save()

    assert cache_path.exists()
    assert not cache_path.with_suffix(cache_path.suffix + ".tmp").exists()

    reloaded = CachedLLMSafetyExtractor(cache_path)
    assert reloaded._cache == {"a:0-1": [0.1] * 6}


def test_save_artifact_does_not_corrupt_latest_copy_if_the_write_fails(tmp_path):
    latest_path = tmp_path / "grooming_head_C.json"
    latest_path.write_text(json.dumps({"epoch": 1}), encoding="utf-8")

    with patch("json.dump", side_effect=RuntimeError("simulated crash mid-write")):
        try:
            save_artifact({"epoch": 2}, tmp_path, "grooming_head_C", "20260101_0000", as_json=True)
        except RuntimeError:
            pass

    # The overwritable "latest" copy must still hold the old, valid content --
    # not be truncated/corrupted by the failed write attempt.
    assert json.loads(latest_path.read_text(encoding="utf-8")) == {"epoch": 1}


def test_save_artifact_writes_latest_and_history_copies(tmp_path):
    save_artifact({"epoch": 3}, tmp_path, "grooming_head_C", "20260101_0000", as_json=True)

    latest_path = tmp_path / "grooming_head_C.json"
    history_path = tmp_path / "grooming_head_C" / "grooming_head_C_20260101_0000.json"

    assert json.loads(latest_path.read_text(encoding="utf-8")) == {"epoch": 3}
    assert json.loads(history_path.read_text(encoding="utf-8")) == {"epoch": 3}
    assert not latest_path.with_suffix(latest_path.suffix + ".tmp").exists()
