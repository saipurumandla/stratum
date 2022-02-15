"""create predictions table

Revision ID: 0001
Revises:
Create Date: 2022-02-14 12:00:00.000000
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
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


def downgrade() -> None:
    op.drop_table("predictions")
