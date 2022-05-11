"""Application configuration loaded from environment variables.

Uses Pydantic BaseSettings so that any missing required variable
raises a clear ValidationError at startup rather than silently
using a wrong default.
"""
from pathlib import Path
from typing import Final

from pydantic import BaseSettings

LOG_LEVEL_DEFAULT: Final = "INFO"


class Settings(BaseSettings):
    model_path: Path = Path("models/model.joblib")
    database_url: str = ""
    log_level: str = LOG_LEVEL_DEFAULT

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
