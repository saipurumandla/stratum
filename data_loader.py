import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FEATURES = [
    "age",
    "education",
    "occupation",
    "hours-per-week",
    "capital-gain",
    "capital-loss",
]
TARGET = "income"


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read adult income CSV and return (train_df, test_df).

    Strips leading whitespace from string columns — the UCI file has
    a space before each value when read with the default separator.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Strip whitespace from object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET])

    # Keep only the features we use plus the target
    keep = FEATURES + [TARGET]
    df = df[keep].dropna()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(
        "loaded dataset path=%s train=%d test=%d",
        path,
        len(train_df),
        len(test_df),
    )
    return train_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_path = Path("data/adult.csv")
    train, test = load_data(data_path)
    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    print(train.head(3))
