"""drift.py — compares live feature distributions against the training baseline.

Uses Evidently DataDriftPreset to produce a per-feature drift report.

Fix (2024-04-28): sparse categorical buckets were triggering false-positive
drift alerts. Categories with fewer than MIN_CATEGORY_COUNT observations in
the current window are now collapsed to an '_other' bucket before the report
runs, preventing chi-square tests from firing on single-observation cells.
"""
import logging
from dataclasses import dataclass
from typing import Final

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD: Final[float] = 0.05
MIN_CATEGORY_COUNT: Final[int] = 5  # cells below this are collapsed to '_other'


class DriftCheckError(Exception):
    """Raised when the Evidently drift check cannot complete."""


@dataclass(frozen=True)
class DriftResult:
    drifted: bool
    drifted_features: list[str]
    feature_scores: dict[str, float]


def _collapse_sparse_categories(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    min_count: int = MIN_CATEGORY_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse rare categories to '_other' in both reference and current.

    Categories whose count in `current` falls below `min_count` are
    remapped to '_other' in both DataFrames so Evidently's chi-square
    test does not treat each sparse cell as a meaningful distribution shift.
    """
    reference = reference.copy()
    current = current.copy()

    for col in current.select_dtypes(include="object").columns:
        counts = current[col].value_counts()
        sparse_cats = counts[counts < min_count].index
        if len(sparse_cats) == 0:
            continue
        reference[col] = reference[col].where(
            ~reference[col].isin(sparse_cats), other="_other"
        )
        current[col] = current[col].where(
            ~current[col].isin(sparse_cats), other="_other"
        )
        logger.debug(
            "collapsed sparse categories col=%s count=%d",
            col,
            len(sparse_cats),
        )

    return reference, current


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
    reference, current = _collapse_sparse_categories(reference, current)

    report = Report(metrics=[DataDriftPreset()])
    try:
        report.run(reference_data=reference, current_data=current)
    except Exception as exc:
        raise DriftCheckError("Evidently report failed") from exc

    result_dict = report.as_dict()
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
