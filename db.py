import json
import logging
from contextlib import contextmanager
from typing import Any, Final, Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

logger = logging.getLogger(__name__)

MODEL_VERSION: Final = "1.0.0"

_engine: Engine | None = None


def init_engine(database_url: str) -> Engine:
    """Create and cache the SQLAlchemy engine with connection pooling."""
    global _engine
    _engine = create_engine(
        database_url,
        pool_size=5,
        max_overflow=10,
    )
    logger.info("db engine initialised")
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("DB engine not initialised — call init_engine first")
    return _engine


@contextmanager
def get_conn() -> Generator[Connection, None, None]:
    """Yield a read-only connection, auto-closed on exit."""
    with get_engine().connect() as conn:
        yield conn


@contextmanager
def get_transaction() -> Generator[Connection, None, None]:
    """Yield a connection inside a transaction; commits on exit, rolls back on error."""
    with get_engine().begin() as conn:
        yield conn


def store_prediction(
    input_data: dict[str, Any],
    prediction: int,
    probability: list[float],
) -> None:
    """Insert one prediction row into the predictions table.

    Uses %s-style placeholders via SQLAlchemy text() with named params
    to prevent SQL injection — never string-format user data into SQL.
    """
    with get_transaction() as conn:
        conn.execute(
            text(
                """
                INSERT INTO predictions
                    (input_json, prediction, prob_low, prob_high, model_ver)
                VALUES
                    (:input_json, :prediction, :prob_low, :prob_high, :model_ver)
                """
            ),
            {
                "input_json": json.dumps(input_data),
                "prediction": prediction,
                "prob_low": probability[0],
                "prob_high": probability[1],
                "model_ver": MODEL_VERSION,
            },
        )
    logger.info("prediction stored prediction=%d", prediction)
