import logging
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request

from model_loader import load_model
from schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/model.joblib")

app = FastAPI(title="Stratum", version="0.1.0")


@app.on_event("startup")
def startup_event() -> None:
    """Load the model artifact into app state at startup."""
    try:
        app.state.model = load_model(MODEL_PATH)
        logger.info("model loaded path=%s", MODEL_PATH)
    except Exception as exc:
        logger.error("failed to load model: %s", exc)
        app.state.model = None


def get_model(request: Request):
    """FastAPI dependency that returns the loaded model or raises 503."""
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return request.app.state.model


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, model=Depends(get_model)) -> PredictResponse:
    """Run inference on the input features and return a prediction."""
    if body.age <= 0:
        raise HTTPException(status_code=422, detail="age must be positive")
    if not (1 <= body.hours_per_week <= 99):
        raise HTTPException(
            status_code=422, detail="hours_per_week must be between 1 and 99"
        )

    import pandas as pd

    row = pd.DataFrame(
        [
            {
                "age": body.age,
                "education": body.education,
                "occupation": body.occupation,
                "hours-per-week": body.hours_per_week,
                "capital-gain": body.capital_gain,
                "capital-loss": body.capital_loss,
            }
        ]
    )
    prediction = int(model.predict(row)[0])
    probability = model.predict_proba(row)[0].tolist()
    return PredictResponse(prediction=prediction, probability=probability)
