from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app
from train import train

KNOWN_INPUT = {
    "age": 35,
    "education": "Bachelors",
    "occupation": "Prof-specialty",
    "hours_per_week": 40,
    "capital_gain": 0,
    "capital_loss": 0,
}


@pytest.fixture(scope="module", autouse=True)
def trained_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a model and inject it into app state before tests run."""
    model_path = tmp_path_factory.mktemp("models") / "model.joblib"
    train(data_path=Path("data/adult.csv"), model_path=model_path)
    from model_loader import load_model

    app.state.model = load_model(model_path)
    return model_path


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_response(client: TestClient) -> None:
    response = client.post("/predict", json=KNOWN_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in (0, 1)
    assert "probability" in data
    assert len(data["probability"]) == 2


def test_predict_probability_sums_to_one(client: TestClient) -> None:
    response = client.post("/predict", json=KNOWN_INPUT)
    assert response.status_code == 200
    probs = response.json()["probability"]
    assert abs(sum(probs) - 1.0) < 1e-6


def test_predict_invalid_age_rejected(client: TestClient) -> None:
    bad_input = {**KNOWN_INPUT, "age": -1}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_predict_invalid_hours_rejected(client: TestClient) -> None:
    bad_input = {**KNOWN_INPUT, "hours_per_week": 200}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422
