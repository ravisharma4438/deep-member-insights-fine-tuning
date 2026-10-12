"""Configuration and secrets, read from the environment or a gitignored .env.

Every key is listed in .env.example. Code calls `require()` for keys a command
can't run without, so a missing key fails with a message naming it.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

# .env in the working directory wins, then the repo root (Studio may run from either).
# Real environment variables always take precedence over both.
load_dotenv(Path.cwd() / ".env")
load_dotenv(REPO_ROOT / ".env")


def get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def require(name: str) -> str:
    value = get(name)
    if value is None:
        raise SystemExit(f"{name} is not set. Add it to .env (see .env.example).")
    return value


def s3_uri(*parts: str) -> str:
    """s3://$DMI_S3_BUCKET/$DMI_S3_PREFIX/<parts...>"""
    bucket = require("DMI_S3_BUCKET")
    prefix = get("DMI_S3_PREFIX", "dmi").strip("/")
    return "s3://" + "/".join([bucket, prefix, *(p.strip("/") for p in parts)])
