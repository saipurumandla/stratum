"""Run database migrations to bring the schema up to date.

Uses Alembic programmatic API so there is no separate alembic/ directory
needed for a single-table service at this scale.
"""
import logging
import os
from pathlib import Path

from alembic import command
from alembic.config import Config

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def get_alembic_config(database_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


def run_migrations(database_url: str | None = None) -> None:
    """Apply all pending Alembic migrations."""
    url = database_url or os.environ["DATABASE_URL"]
    cfg = get_alembic_config(url)
    logger.info("running migrations url=%s", url)
    command.upgrade(cfg, "head")
    logger.info("migrations complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_migrations()
