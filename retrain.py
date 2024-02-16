"""retrain.py — APScheduler-driven nightly model retraining.

Loads the latest data, retrains the pipeline, and registers the new
artifact in the model registry. Runs as a long-lived process alongside
the API server — start it with `python retrain.py`.
"""
from pathlib import Path

import structlog
from apscheduler.schedulers.blocking import BlockingScheduler

from model_registry import register_model
from train import train

log = structlog.get_logger()

DATA_PATH: Path = Path("data/adult.csv")
MODEL_PATH: Path = Path("models/model.joblib")
RETRAIN_HOUR: int = 2  # 02:00 UTC nightly


def retrain_job() -> None:
    """Load latest data, retrain the pipeline, and register the artifact."""
    log.info("retrain_job.started")
    try:
        pipeline = train(data_path=DATA_PATH, model_path=MODEL_PATH)
        version = register_model(model_path=MODEL_PATH)
        log.info("retrain_job.completed", version=version)
    except Exception as exc:
        log.error("retrain_job.failed", error=str(exc))


if __name__ == "__main__":
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )
    scheduler = BlockingScheduler()
    scheduler.add_job(retrain_job, "cron", hour=RETRAIN_HOUR, minute=0)
    log.info("retrain_scheduler.started", schedule=f"daily at {RETRAIN_HOUR:02d}:00 UTC")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("retrain_scheduler.stopped")
