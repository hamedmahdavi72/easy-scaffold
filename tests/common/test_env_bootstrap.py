"""Tests for `.env` bootstrap (platform env precedence)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from easy_scaffold.common import env_bootstrap


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    (tmp_path / ".env").write_text("FROM_FILE=1\nALSO=fromfile\n", encoding="utf-8")
    return tmp_path


def test_load_local_dotenv_does_not_override_existing_env(tmp_root: Path) -> None:
    os.environ["FROM_FILE"] = "from_platform"
    try:
        env_bootstrap.load_local_dotenv(tmp_root)
        assert os.environ["FROM_FILE"] == "from_platform"
        assert os.environ.get("ALSO") == "fromfile"
    finally:
        os.environ.pop("FROM_FILE", None)
        os.environ.pop("ALSO", None)


def test_load_local_dotenv_skipped_when_flag_off(tmp_root: Path) -> None:
    with patch("easy_scaffold.common.env_bootstrap.load_dotenv") as mock_ld:
        with patch.dict(os.environ, {"EASY_SCAFFOLD_LOAD_DOTENV": "0"}):
            env_bootstrap.load_local_dotenv(tmp_root)
    mock_ld.assert_not_called()


def test_load_local_dotenv_passes_override_false(tmp_root: Path) -> None:
    with patch("easy_scaffold.common.env_bootstrap.load_dotenv") as mock_ld:
        env_bootstrap.load_local_dotenv(tmp_root)
    mock_ld.assert_called_once()
    _, kwargs = mock_ld.call_args
    assert kwargs.get("override") is False
