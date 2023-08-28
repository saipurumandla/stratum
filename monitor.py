"""monitor.py — aggregates rolling prediction window statistics from the DB.

Fetches recent predictions, deserialises input_json, and assembles a
DataFrame suitable for passing to drift.check_drift.
"""
import json
import logging
from dataclasses import dataclass
from typing import Final

import pandas as pd
from sqlalchemy import text

import db

logger = logging.getLogger(__name__)

WINDOW_HOURS: Final[int] = 24
MIN_WINDOW_ROWS: Final[int] = 30  # skip drift check if window is too sparse


@dataclass(frozen=True)
class WindowStats:
    total_predictions: int
    positive_rate: float
    avg_prob_high: float


def fetch_recent_predictions(window_hours: int = WINDOW_HOURS) -> pd.DataFrame:
    """Return a DataFrame of input features from the last N hours of predictions.

    Columns match the model's feature schema so the result can be passed
    directly to drift.check_drift as the `current` argument.
    """
    with db.get_conn() as conn:
        rows = conn.execute(
            text(
                """
                SELECT input_json, prediction, prob_high
                FROM predictions
                WHERE created_at > NOW() - make_interval(hours => :hours)
                ORDER BY created_at DESC
                """
            ),
            {"hours": window_hours},
        ).fetchall()

    if not rows:
        logger.info("no predictions in window hours=%d", window_hours)
        return pd.DataFrame()

    records = []
    for row in rows:
        record = json.loads(row[0])
        record["_prediction"] = row[1]
        record["_prob_high"] = row[2]
        records.append(record)

    df = pd.DataFrame(records)
    logger.info("fetched predictions window_hours=%d rows=%d", window_hours, len(df))
    return df


def compute_window_stats(window_hours: int = WINDOW_HOURS) -> WindowStats:
    """Compute aggregate statistics for the rolling prediction window."""
    df = fetch_recent_predictions(window_hours)

    if df.empty:
        return WindowStats(
            total_predictions=0,
            positive_rate=0.0,
            avg_prob_high=0.0,
        )

    total = len(df)
    positive_rate = float(df["_prediction"].mean())
    avg_prob_high = float(df["_prob_high"].mean())

    logger.info(
        "window stats hours=%d total=%d positive_rate=%.3f avg_prob_high=%.3f",
        window_hours,
        total,
        positive_rate,
        avg_prob_high,
    )
    return WindowStats(
        total_predictions=total,
        positive_rate=positive_rate,
        avg_prob_high=avg_prob_high,
    )


def is_window_sufficient(window_hours: int = WINDOW_HOURS) -> bool:
    """Return True if the prediction window has enough rows for reliable drift detection."""
    df = fetch_recent_predictions(window_hours)
    sufficient = len(df) >= MIN_WINDOW_ROWS
    if not sufficient:
        logger.warning(
            "window too sparse for drift check rows=%d min=%d",
            len(df),
            MIN_WINDOW_ROWS,
        )
    return sufficient
