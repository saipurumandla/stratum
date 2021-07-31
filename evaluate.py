import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from data_loader import load_data
from preprocess import get_X_y

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/model.joblib")
DATA_PATH = Path("data/adult.csv")


@dataclass
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float


def evaluate(
    model_path: Path = MODEL_PATH,
    data_path: Path = DATA_PATH,
) -> EvalResult:
    """Load the saved model and compute metrics on the held-out test set."""
    pipeline: Pipeline = joblib.load(model_path)
    _, test_df = load_data(data_path)
    X_test, y_test = get_X_y(test_df)

    y_pred = pipeline.predict(X_test)

    result = EvalResult(
        accuracy=accuracy_score(y_test, y_pred),
        # pos_label=1 targets the >50K class throughout
        precision=precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        recall=recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        f1=f1_score(y_test, y_pred, pos_label=1, zero_division=0),
    )
    logger.info(
        "eval accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        result.accuracy,
        result.precision,
        result.recall,
        result.f1,
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = evaluate()
    print(f"Accuracy:  {result.accuracy:.4f}")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall:    {result.recall:.4f}")
    print(f"F1:        {result.f1:.4f}")
