-- Predictions table — stores every inference request and its result.
-- Created by migrate.py on first run.

CREATE TABLE IF NOT EXISTS predictions (
    id          SERIAL PRIMARY KEY,
    input_json  TEXT        NOT NULL,   -- raw JSON of PredictRequest, never modified
    prediction  INT         NOT NULL,   -- 0 or 1
    prob_low    FLOAT       NOT NULL,   -- P(income <= 50K)
    prob_high   FLOAT       NOT NULL,   -- P(income  > 50K)
    model_ver   TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
