import logging
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def load_model(path: Path) -> Pipeline:
    """Load a joblib-serialised sklearn Pipeline from disk.

    Raises FileNotFoundError if the artifact is missing — caller decides
    whether to abort startup or continue with model=None.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    pipeline: Pipeline = joblib.load(path)
    logger.info("model loaded path=%s", path)
    return pipeline
