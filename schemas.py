from pydantic import BaseModel, ConfigDict


class PredictRequest(BaseModel):
    model_config = ConfigDict(strict=False)

    age: int
    education: str
    occupation: str
    hours_per_week: int
    capital_gain: int
    capital_loss: int


class PredictResponse(BaseModel):
    prediction: int
    probability: list[float]
