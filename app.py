import logging
import logging.handlers
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request

from model_loader import load_model
from schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/model.joblib")
LOG_PATH = Path("predictions.log")

# Rotating file handler for flat prediction log — 5 MB per file, 3 backups
_pred_logger = logging.getLogger("predictions")
_pred_logger.setLevel(logging.INFO)
_pred_logger.propagate = False
_handler = logging.handlers.RotatingFileHandler(
    LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3
)
_handler.setFormatter(
    logging.Formatter("%(asctime)s %(message)s")
)
_pred_logger.addHandler(_handler)

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

    _pred_logger.info(
        "prediction=%d probability=%s age=%d education=%s occupation=%s "
        "hours_per_week=%d capital_gain=%d capital_loss=%d",
        prediction,
        probability,
        body.age,
        body.education,
        body.occupation,
        body.hours_per_week,
        body.capital_gain,
        body.capital_loss,
    )
    return PredictResponse(prediction=prediction, probability=probability)
