import logging
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data_loader import load_data
from preprocess import build_preprocessor, get_X_y

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/model.joblib")
DATA_PATH = Path("data/adult.csv")


def build_pipeline() -> Pipeline:
    """Combine preprocessor and logistic regression into a single sklearn Pipeline."""
    preprocessor = build_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    return pipeline


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> Pipeline:
    """Train and save the model pipeline. Returns the fitted pipeline."""
    train_df, _ = load_data(data_path)
    X_train, y_train = get_X_y(train_df)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    logger.info("training complete classes=%s", pipeline.classes_.tolist())

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("model saved path=%s", model_path)
    return pipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
