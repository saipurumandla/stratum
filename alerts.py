"""alerts.py — sends a webhook alert when feature drift is detected.

Uses httpx to POST a JSON payload to ALERT_WEBHOOK_URL. No-ops if the
env var is not set so the service can run without alerting configured.
"""
import logging
import os
from typing import Final

import httpx

from drift import DriftResult

logger = logging.getLogger(__name__)

ALERT_WEBHOOK_URL: Final[str] = os.getenv("ALERT_WEBHOOK_URL", "")
ALERT_TIMEOUT_SECONDS: Final[float] = 5.0


def send_drift_alert(result: DriftResult) -> None:
    """POST a drift-detected payload to the configured webhook URL.

    Args:
        result: The DriftResult produced by drift.check_drift.

    Silently no-ops if ALERT_WEBHOOK_URL is not configured.
    Logs a warning on HTTP failure but does not propagate the exception
    so a broken alerting backend cannot take down the prediction service.
    """
    if not ALERT_WEBHOOK_URL:
        logger.debug("ALERT_WEBHOOK_URL not set — skipping drift alert")
        return

    payload = {
        "alert": "drift_detected",
        "drifted_features": result.drifted_features,
        "feature_scores": result.feature_scores,
    }

    try:
        response = httpx.post(
            ALERT_WEBHOOK_URL,
            json=payload,
            timeout=ALERT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        logger.info("drift alert sent status=%d", response.status_code)
    except httpx.HTTPStatusError as exc:
        logger.error(
            "drift alert returned error status=%d url=%s",
            exc.response.status_code,
            ALERT_WEBHOOK_URL,
        )
    except httpx.HTTPError as exc:
        logger.error("drift alert request failed error=%s", exc)
