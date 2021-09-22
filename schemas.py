from pydantic import BaseModel


class PredictRequest(BaseModel):
    age: int
    education: str
    occupation: str
    hours_per_week: int
    capital_gain: int
    capital_loss: int


class PredictResponse(BaseModel):
    prediction: int
    probability: list[float]
