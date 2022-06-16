"""Round-trip tests for prediction storage and retrieval.

Requires a running Postgres instance. Set DATABASE_URL to run:

    DATABASE_URL=postgresql://user:pass@localhost/test_stratum pytest tests/test_db.py
"""
import json
import os

import pytest
from sqlalchemy import text

import db

_DATABASE_URL = os.getenv("DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not _DATABASE_URL,
    reason="DATABASE_URL not set — skipping DB integration tests",
)


@pytest.fixture(scope="module")
def engine():
    """Initialise engine and ensure predictions table exists for the test run."""
    eng = db.init_engine(_DATABASE_URL)
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id          SERIAL PRIMARY KEY,
                    input_json  TEXT        NOT NULL,
                    prediction  INT         NOT NULL,
                    prob_low    FLOAT       NOT NULL,
                    prob_high   FLOAT       NOT NULL,
                    model_ver   TEXT        NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        )
    yield eng
    # Clean up rows written during this test run
    with eng.begin() as conn:
        conn.execute(text("DELETE FROM predictions"))


def test_store_prediction_happy_path(engine: None) -> None:
    """Storing a valid prediction inserts a row that can be read back."""
    input_data = {
        "age": 35,
        "education": "Bachelors",
        "occupation": "Tech-support",
        "hours_per_week": 40,
        "capital_gain": 0,
        "capital_loss": 0,
    }
    db.store_prediction(
        input_data=input_data,
        prediction=1,
        probability=[0.3, 0.7],
    )

    with db.get_conn() as conn:
        row = conn.execute(
            text(
                "SELECT input_json, prediction, prob_low, prob_high, model_ver"
                " FROM predictions ORDER BY id DESC LIMIT 1"
            )
        ).fetchone()

    assert row is not None
    stored = json.loads(row[0])
    assert stored["age"] == 35
    assert stored["education"] == "Bachelors"
    assert row[1] == 1
    assert abs(row[2] - 0.3) < 1e-6
    assert abs(row[3] - 0.7) < 1e-6
    assert row[4] == db.MODEL_VERSION


def test_store_prediction_zero_class(engine: None) -> None:
    """Prediction class 0 (income <= 50K) round-trips correctly."""
    input_data = {
        "age": 22,
        "education": "HS-grad",
        "occupation": "Handlers-cleaners",
        "hours_per_week": 20,
        "capital_gain": 0,
        "capital_loss": 0,
    }
    db.store_prediction(
        input_data=input_data,
        prediction=0,
        probability=[0.85, 0.15],
    )

    with db.get_conn() as conn:
        row = conn.execute(
            text(
                "SELECT prediction, prob_low, prob_high"
                " FROM predictions ORDER BY id DESC LIMIT 1"
            )
        ).fetchone()

    assert row[0] == 0
    assert abs(row[1] - 0.85) < 1e-6
    assert abs(row[2] - 0.15) < 1e-6


def test_input_json_stored_verbatim(engine: None) -> None:
    """input_json must preserve all field names and values exactly — it is immutable."""
    input_data = {
        "age": 50,
        "education": "Masters",
        "occupation": "Exec-managerial",
        "hours_per_week": 55,
        "capital_gain": 5000,
        "capital_loss": 0,
    }
    db.store_prediction(
        input_data=input_data,
        prediction=1,
        probability=[0.1, 0.9],
    )

    with db.get_conn() as conn:
        row = conn.execute(
            text("SELECT input_json FROM predictions ORDER BY id DESC LIMIT 1")
        ).fetchone()

    assert json.loads(row[0]) == input_data


def test_get_engine_raises_before_init() -> None:
    """get_engine raises RuntimeError when called before init_engine."""
    original = db._engine
    db._engine = None
    try:
        with pytest.raises(RuntimeError, match="not initialised"):
            db.get_engine()
    finally:
        db._engine = original
