"""drift.py — compares live feature distributions against the training baseline.

Uses Evidently DataDriftPreset to produce a per-feature drift report.
"""
import logging
from dataclasses import dataclass, field
from typing import Final

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD: Final[float] = 0.05  # p-value cutoff used by Evidently internally


class DriftCheckError(Exception):
    """Raised when the Evidently drift check cannot complete."""


@dataclass(frozen=True)
class DriftResult:
    drifted: bool
    drifted_features: list[str]
    feature_scores: dict[str, float]


def check_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> DriftResult:
    """Run Evidently DataDriftPreset and return a structured result.

    Args:
        reference: Training-time data (the reference distribution).
        current:   Recent live predictions assembled as a DataFrame.

    Returns:
        DriftResult with per-feature drift scores and an overall drift flag.

    Raises:
        DriftCheckError: If Evidently cannot produce a report.
    """
    report = Report(metrics=[DataDriftPreset()])
    try:
        report.run(reference_data=reference, current_data=current)
    except Exception as exc:
        raise DriftCheckError("Evidently report failed") from exc

    result_dict = report.as_dict()
    # metrics[0] is DatasetDriftMetric — the dataset-level summary produced
    # by DataDriftPreset; it contains per-column drift information.
    dataset_metric = result_dict["metrics"][0]["result"]

    feature_scores: dict[str, float] = {}
    drifted_features: list[str] = []

    for feat, info in dataset_metric.get("drift_by_columns", {}).items():
        score = float(info.get("drift_score", 1.0))
        feature_scores[feat] = score
        if info.get("drift_detected", False):
            drifted_features.append(feat)

    overall_drifted = len(drifted_features) > 0
    if overall_drifted:
        logger.warning(
            "drift detected features=%s scores=%s",
            drifted_features,
            {f: round(feature_scores[f], 4) for f in drifted_features},
        )
    else:
        logger.info("no drift detected feature_scores=%s", feature_scores)

    return DriftResult(
        drifted=overall_drifted,
        drifted_features=drifted_features,
        feature_scores=feature_scores,
    )
