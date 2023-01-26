"""baseline.py — saves training-time feature distribution statistics as JSON.

Run once after training to capture the reference distribution that
drift.py will compare live predictions against.
"""
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from data_loader import load_data
from preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

BASELINE_PATH: Final[Path] = Path("baseline.json")
DATA_PATH: Final[Path] = Path("data/adult.csv")


@dataclass(frozen=True)
class NumericStats:
    mean: float
    std: float
    min: float
    p25: float
    p50: float
    p75: float
    max: float


def _numeric_stats(series: pd.Series) -> NumericStats:
    desc = series.describe(percentiles=[0.25, 0.5, 0.75])
    return NumericStats(
        mean=float(desc["mean"]),
        std=float(desc["std"]),
        min=float(desc["min"]),
        p25=float(desc["25%"]),
        p50=float(desc["50%"]),
        p75=float(desc["75%"]),
        max=float(desc["max"]),
    )


def compute_baseline(
    data_path: Path = DATA_PATH,
    output_path: Path = BASELINE_PATH,
) -> dict:
    """Compute and persist training-data feature statistics.

    Numeric features: mean, std, percentiles.
    Categorical features: normalised value frequencies.
    """
    train_df, _ = load_data(data_path)

    stats: dict = {"numeric": {}, "categorical": {}}

    for col in NUMERIC_FEATURES:
        stats["numeric"][col] = asdict(_numeric_stats(train_df[col]))

    for col in CATEGORICAL_FEATURES:
        freq = train_df[col].value_counts(normalize=True).to_dict()
        stats["categorical"][col] = freq

    output_path.write_text(json.dumps(stats, indent=2))
    logger.info(
        "baseline saved path=%s numeric=%d categorical=%d",
        output_path,
        len(NUMERIC_FEATURES),
        len(CATEGORICAL_FEATURES),
    )
    return stats


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    """Load persisted baseline statistics from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Baseline not found at {path} — run baseline.py first")
    return json.loads(path.read_text())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    compute_baseline()
