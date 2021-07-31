import logging
from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import FEATURES, TARGET

logger = logging.getLogger(__name__)

NUMERIC_FEATURES: Final[list[str]] = [
    "age",
    "hours-per-week",
    "capital-gain",
    "capital-loss",
]
CATEGORICAL_FEATURES: Final[list[str]] = [
    "education",
    "occupation",
]


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that scales numeric and encodes categorical features.

    OneHotEncoder uses handle_unknown='ignore' so unseen categories at
    inference time produce a zero vector rather than raising an error.
    """
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))]
    )
    # sparse_threshold=0 forces dense output regardless of individual transformer output
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )
    return preprocessor


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into feature matrix X and binary target y.

    Target encoding: '>50K' → 1, '<=50K' → 0.
    """
    X = df[FEATURES]
    y = (df[TARGET] == ">50K").astype(int)
    return X, y


if __name__ == "__main__":
    import logging
    from pathlib import Path

    from data_loader import load_data

    logging.basicConfig(level=logging.INFO)
    train_df, test_df = load_data(Path("data/adult.csv"))
    preprocessor = build_preprocessor()
    X_train, y_train = get_X_y(train_df)
    X_transformed = preprocessor.fit_transform(X_train)
    logger.info("preprocessed shape: %s", X_transformed.shape)
