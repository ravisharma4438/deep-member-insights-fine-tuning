"""Read-only access to the platform database."""

from __future__ import annotations

from dmi import settings


def connect():
    """Connection whose transactions are READ ONLY.

    Uses BEGIN READ ONLY per transaction rather than a session setting, which
    Neon's pgbouncer pooler would drop between transactions.
    """
    import psycopg

    conn = psycopg.connect(settings.require("DATABASE_URL"), connect_timeout=15)
    conn.read_only = True
    return conn
