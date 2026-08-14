"""Load machine-specific project settings from the ignored .env file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parent


def load_dotenv(env_path: Optional[Path] = None) -> None:
    """Load simple KEY=VALUE entries without requiring python-dotenv."""
    path = env_path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return

    with path.open(encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv()


def get_setting(env_key: str, default: str) -> str:
    """Return a non-empty environment setting or its shared default."""
    return os.environ.get(env_key, "").strip() or default


def get_path(env_key: str, default: Union[str, Path]) -> Path:
    """Resolve a configurable path, treating relative values as project-relative."""
    raw_value = os.environ.get(env_key, "").strip()
    path = Path(raw_value).expanduser() if raw_value else Path(default).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()
