"""model_registry.py — SQLite-backed versioned artifact metadata.

Tracks every trained model artifact by timestamp and evaluation metrics
so older versions can be audited and rolled back without touching Postgres.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import structlog

log = structlog.get_logger()

REGISTRY_PATH: Final[Path] = Path("model_registry.db")

_DDL = """
    CREATE TABLE IF NOT EXISTS model_versions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        version     TEXT    NOT NULL,
        model_path  TEXT    NOT NULL,
        accuracy    REAL,
        created_at  TEXT    NOT NULL
    )
"""


def _get_conn(registry_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(registry_path)
    conn.execute(_DDL)
    conn.commit()
    return conn


def register_model(
    model_path: Path = Path("models/model.joblib"),
    accuracy: float | None = None,
    registry_path: Path = REGISTRY_PATH,
) -> str:
    """Insert a new model version record and return the version string.

    Args:
        model_path:    Path to the saved joblib artifact.
        accuracy:      Optional accuracy metric from evaluate.py.
        registry_path: SQLite database file path.

    Returns:
        ISO-8601 version string used as the version identifier.
    """
    now = datetime.now(timezone.utc)
    version = now.strftime("%Y%m%dT%H%M%SZ")

    conn = _get_conn(registry_path)
    try:
        conn.execute(
            "INSERT INTO model_versions (version, model_path, accuracy, created_at)"
            " VALUES (?, ?, ?, ?)",
            (version, str(model_path), accuracy, now.isoformat()),
        )
        conn.commit()
    finally:
        conn.close()

    log.info("model.registered", version=version, path=str(model_path), accuracy=accuracy)
    return version


def list_versions(registry_path: Path = REGISTRY_PATH) -> list[dict]:
    """Return all model versions ordered newest-first."""
    conn = _get_conn(registry_path)
    try:
        rows = conn.execute(
            "SELECT version, model_path, accuracy, created_at"
            " FROM model_versions ORDER BY id DESC"
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "version": r[0],
            "model_path": r[1],
            "accuracy": r[2],
            "created_at": r[3],
        }
        for r in rows
    ]


def latest_version(registry_path: Path = REGISTRY_PATH) -> dict | None:
    """Return the most recently registered model version, or None if empty."""
    versions = list_versions(registry_path)
    return versions[0] if versions else None
