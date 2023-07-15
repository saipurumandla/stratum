"""test_monitor.py — drift detection validated on synthetic shifted datasets.

These tests exercise drift.check_drift directly with known distributions
so they are deterministic and require no external services.
"""
import numpy as np
import pandas as pd
import pytest

from drift import DriftCheckError, DriftResult, check_drift

# Feature columns matching the model's input schema
NUMERIC_COLS = ["age", "hours-per-week", "capital-gain", "capital-loss"]
CATEGORICAL_COLS = ["education", "occupation"]
ALL_COLS = NUMERIC_COLS + CATEGORICAL_COLS

_RNG = np.random.default_rng(42)

EDUCATION_VALS = ["Bachelors", "HS-grad", "Masters", "Some-college", "Assoc-voc"]
OCCUPATION_VALS = ["Tech-support", "Craft-repair", "Exec-managerial", "Sales", "Other-service"]


def _make_reference(n: int = 500) -> pd.DataFrame:
    """Generate a stable reference distribution matching training data statistics."""
    return pd.DataFrame(
        {
            "age": _RNG.integers(25, 60, size=n).astype(float),
            "hours-per-week": _RNG.integers(35, 50, size=n).astype(float),
            "capital-gain": _RNG.integers(0, 1000, size=n).astype(float),
            "capital-loss": _RNG.integers(0, 200, size=n).astype(float),
            "education": _RNG.choice(EDUCATION_VALS, size=n),
            "occupation": _RNG.choice(OCCUPATION_VALS, size=n),
        }
    )


def test_no_drift_on_identical_distribution() -> None:
    """check_drift returns drifted=False when current matches reference."""
    reference = _make_reference(500)
    current = _make_reference(200)
    result = check_drift(reference, current)

    assert isinstance(result, DriftResult)
    assert result.drifted is False
    assert result.drifted_features == []


def test_drift_detected_on_shifted_numeric() -> None:
    """check_drift returns drifted=True when a numeric feature shifts significantly."""
    reference = _make_reference(500)

    # Shift age distribution far outside training range
    shifted = reference.copy()
    shifted["age"] = _RNG.integers(70, 90, size=len(shifted)).astype(float)

    result = check_drift(reference, shifted)

    assert result.drifted is True
    assert "age" in result.drifted_features


def test_drift_detected_on_shifted_categorical() -> None:
    """check_drift returns drifted=True when categorical distribution changes sharply."""
    reference = _make_reference(500)

    # Force occupation to a single unseen-heavy category
    shifted = reference.copy()
    shifted["occupation"] = "Exec-managerial"

    result = check_drift(reference, shifted)

    assert result.drifted is True
    assert "occupation" in result.drifted_features


def test_feature_scores_present_for_all_columns() -> None:
    """feature_scores dict is populated for every column in the dataset."""
    reference = _make_reference(300)
    current = _make_reference(100)
    result = check_drift(reference, current)

    for col in ALL_COLS:
        assert col in result.feature_scores, f"missing score for {col}"


def test_drift_result_is_immutable() -> None:
    """DriftResult is a frozen dataclass — mutation must raise FrozenInstanceError."""
    reference = _make_reference(200)
    current = _make_reference(100)
    result = check_drift(reference, current)

    with pytest.raises(Exception):
        result.drifted = True  # type: ignore[misc]
