"""backend/config.py must load .env before Settings() reads os.environ --
otherwise ANTHROPIC_API_KEY (and anything else only set in .env, not the
real shell environment) is invisible to the backend process.
LLMSafetySignalExtractor/EmotionalDependencyExtractor construct
`anthropic.Anthropic()` with no explicit api_key, which reads os.environ
directly, so a case submitted in real mode failed with "Could not resolve
authentication method" even though .env had the key -- caught via a live
real-mode test, not assumed.
"""

import importlib
from unittest.mock import patch

import backend.config
from backend.paths import PROJECT_ROOT


def test_config_module_loads_dotenv_from_project_root():
    # Patch dotenv.load_dotenv itself, not backend.config.load_dotenv: a
    # reload re-executes `from dotenv import load_dotenv`, which would
    # immediately rebind the latter back to the real function and silently
    # defeat the mock.
    with patch("dotenv.load_dotenv") as mock_load_dotenv:
        importlib.reload(backend.config)

    mock_load_dotenv.assert_called_once_with(PROJECT_ROOT / ".env")

    importlib.reload(backend.config)  # restore the real (un-mocked) module for later tests
