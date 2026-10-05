"""Read-only access to the platform database."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def database_url() -> str:
    # .env in the working directory, then the repo root (Studio runs from either)
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit(
            "DATABASE_URL is not set. Add it to .env (see .env.example); "
            "prefer a read-only Neon role."
        )
    return url


def connect():
    """Connection whose transactions are READ ONLY.

    Uses BEGIN READ ONLY per transaction rather than a session setting, which
    Neon's pgbouncer pooler would drop between transactions.
    """
    import psycopg

    conn = psycopg.connect(database_url(), connect_timeout=15)
    conn.read_only = True
    return conn
