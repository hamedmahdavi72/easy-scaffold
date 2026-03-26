"""Local `.env` loading for CLI entrypoints (pattern A: production uses platform-injected env)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# If set to 0/false/no/off, skip reading `.env` (e.g. AWS where secrets are injected as env vars).
_ENV_FLAG = "EASY_SCAFFOLD_LOAD_DOTENV"


def _should_load_dotenv_file() -> bool:
    raw = os.environ.get(_ENV_FLAG)
    if raw is None or raw.strip() == "":
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def load_local_dotenv(project_root: Path, *, dotenv_name: str = ".env") -> None:
    """
    Load ``project_root / dotenv_name`` into ``os.environ`` with ``override=False``.

    Variables already set in the process environment (e.g. ECS/Lambda secrets mapped
    to env by your platform) are never overwritten by the file.
    """
    if not _should_load_dotenv_file():
        return
    path = project_root / dotenv_name
    load_dotenv(path, override=False)
