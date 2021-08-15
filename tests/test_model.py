import shutil
from pathlib import Path

import pytest

from data_loader import load_data
from evaluate import EvalResult, evaluate
from train import train


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a model into a temp directory and return the artifact path."""
    model_dir = tmp_path_factory.mktemp("models")
    model_path = model_dir / "model.joblib"
    train(data_path=Path("data/adult.csv"), model_path=model_path)
    return model_path


def test_artifact_loads(trained_model: Path) -> None:
    """The saved artifact must load without error."""
    import joblib

    pipeline = joblib.load(trained_model)
    assert pipeline is not None


def test_accuracy_above_threshold(trained_model: Path) -> None:
    """Accuracy on the held-out test set must be at least 0.80."""
    result: EvalResult = evaluate(
        model_path=trained_model,
        data_path=Path("data/adult.csv"),
    )
    assert result.accuracy >= 0.80, (
        f"Accuracy {result.accuracy:.4f} is below required 0.80"
    )


def test_precision_above_threshold(trained_model: Path) -> None:
    """Precision on >50K class must be at least 0.70."""
    result: EvalResult = evaluate(
        model_path=trained_model,
        data_path=Path("data/adult.csv"),
    )
    assert result.precision >= 0.70, (
        f"Precision {result.precision:.4f} is below required 0.70"
    )


def test_recall_above_threshold(trained_model: Path) -> None:
    """Recall on >50K class must be at least 0.55."""
    result: EvalResult = evaluate(
        model_path=trained_model,
        data_path=Path("data/adult.csv"),
    )
    assert result.recall >= 0.55, (
        f"Recall {result.recall:.4f} is below required 0.55"
    )
