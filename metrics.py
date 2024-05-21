"""metrics.py — Prometheus metrics with a custom CollectorRegistry.

Isolates application metrics from the default global registry so they
can be tested independently and exported without polluting process-level
metrics from other libraries.
"""
from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest

REGISTRY = CollectorRegistry()

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total number of predictions served",
    registry=REGISTRY,
)
DRIFT_DETECTED = Gauge(
    "drift_detected",
    "1 if drift was detected in the last monitoring run, 0 otherwise",
    registry=REGISTRY,
)
RETRAIN_TOTAL = Counter(
    "retrain_total",
    "Total number of model retraining runs completed",
    registry=REGISTRY,
)


def get_metrics_output() -> bytes:
    """Return the latest metrics in Prometheus text exposition format."""
    return generate_latest(REGISTRY)
