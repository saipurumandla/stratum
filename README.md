# Stratum

![CI](https://github.com/saipurumandla/stratum/actions/workflows/train-and-deploy.yml/badge.svg)
![GitLab CI](https://gitlab.com/saipurumandla/stratum/badges/main/pipeline.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)

A Python MLOps service that trains a logistic regression on the UCI Adult Income
dataset and exposes predictions via a FastAPI REST API with Postgres storage,
drift detection, and automated nightly retraining.

---

## Architecture

```
                ┌─────────────────────────────────────────┐
                │               Stratum API                │
                │                                          │
  HTTP POST ───►│  /predict  ──► model_loader  ──► sklearn │
  HTTP GET  ───►│  /health                                  │
  HTTP GET  ───►│  /metrics  ──► prometheus_client          │
                └────────────┬────────────────┬────────────┘
                             │                │
                     writes  │        reads   │
                             ▼                ▼
                      ┌──────────┐     ┌────────────┐
                      │ Postgres │     │ baseline   │
                      │predictions    │ .json      │
                      └──────────┘     └────────────┘
                             │
                     feeds   │
                             ▼
                      ┌──────────────────────┐
                      │  monitor.py          │
                      │  drift.py (Evidently)│
                      │  alerts.py (httpx)   │
                      └──────────────────────┘

          ┌──────────────────────────────────────────┐
          │  retrain.py (APScheduler — 02:00 UTC)    │
          │    train.py ──► model_registry.py        │
          │    (SQLite artifact versioning)           │
          └──────────────────────────────────────────┘
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Train the initial model
python train.py

# Save training distribution baseline for drift detection
python baseline.py

# Start the API
uvicorn app:app --reload

# Start nightly retraining scheduler (separate process)
python retrain.py
```

---

## API

| Method | Path       | Description                    |
|--------|-----------|--------------------------------|
| POST   | /predict  | Run inference on input features |
| GET    | /health   | Liveness check                 |
| GET    | /metrics  | Prometheus metrics endpoint    |

**Request body:**
```json
{
  "age": 35,
  "education": "Bachelors",
  "occupation": "Tech-support",
  "hours_per_week": 40,
  "capital_gain": 0,
  "capital_loss": 0
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": [0.32, 0.68]
}
```

---

## Environment Variables

| Variable           | Default                  | Description                       |
|--------------------|--------------------------|-----------------------------------|
| `MODEL_PATH`       | `models/model.joblib`    | Path to the trained joblib artifact|
| `DATABASE_URL`     | *(empty)*                | Postgres connection string        |
| `LOG_LEVEL`        | `INFO`                   | Logging level                     |
| `ALERT_WEBHOOK_URL`| *(empty)*                | Webhook URL for drift alerts       |

---

## CI/CD

Two pipelines are provided and kept in sync:

- **GitHub Actions** — `.github/workflows/train-and-deploy.yml`
- **GitLab CI** — `.gitlab-ci.yml`

Both run the full test matrix on Python 3.11 and 3.12, then retrain the
model artifact on pushes to `main`.

---

## CHANGELOG

### v2.0.0 — Jun 2025
- APScheduler nightly retraining (`retrain.py`)
- SQLite model registry with version history (`model_registry.py`)
- GitHub Actions + GitLab CI pipelines (Python 3.11 / 3.12 matrix)
- Fix: drift false positives on sparse categorical buckets (`drift.py`)
- Custom Prometheus CollectorRegistry (`metrics.py`)
- Pydantic v2 migration (`model_dump`, `ConfigDict`, `pydantic-settings`)
- structlog structured logging in new modules

### v1.0.0 — Aug 2023
- Evidently drift detection + webhook alerts
- Prometheus `/metrics` endpoint
- FastAPI lifespan migration (replaces `@app.on_event`)
- Postgres predictions storage with SQLAlchemy Core
- Structured JSON logging via `python-json-logger`
