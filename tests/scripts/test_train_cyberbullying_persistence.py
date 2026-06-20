import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from train_cyberbullying import save_artifact  # noqa: E402


def test_save_artifact_does_not_corrupt_latest_copy_if_the_write_fails(tmp_path):
    latest_path = tmp_path / "cyberbullying_head.json"
    latest_path.write_text(json.dumps({"epoch": 1}), encoding="utf-8")

    with patch("json.dump", side_effect=RuntimeError("simulated crash mid-write")):
        try:
            save_artifact({"epoch": 2}, tmp_path, "cyberbullying_head", "20260101_0000", as_json=True)
        except RuntimeError:
            pass

    # The overwritable "latest" copy must still hold the old, valid content --
    # not be truncated/corrupted by the failed write attempt.
    assert json.loads(latest_path.read_text(encoding="utf-8")) == {"epoch": 1}


def test_save_artifact_writes_latest_and_history_copies(tmp_path):
    save_artifact({"epoch": 3}, tmp_path, "cyberbullying_head", "20260101_0000", as_json=True)

    latest_path = tmp_path / "cyberbullying_head.json"
    history_path = tmp_path / "cyberbullying_head" / "cyberbullying_head_20260101_0000.json"

    assert json.loads(latest_path.read_text(encoding="utf-8")) == {"epoch": 3}
    assert json.loads(history_path.read_text(encoding="utf-8")) == {"epoch": 3}
    assert not latest_path.with_suffix(latest_path.suffix + ".tmp").exists()
