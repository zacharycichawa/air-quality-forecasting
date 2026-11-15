"""Settings loaded from environment variables / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SQL_DIR = PROJECT_ROOT / "sql"

POLLUTANTS = ("PM2.5", "PM10", "NO2")
TARGET_POLLUTANT = "PM2.5"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class DbSettings:
    host: str
    port: int
    user: str
    password: str
    database: str

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


def db_settings(database: str | None = None, *, reader: bool = False) -> DbSettings:
    """Owner credentials by default; `reader=True` gives the assistant's read-only role."""
    return DbSettings(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=READER_ROLE if reader else os.getenv("POSTGRES_USER", "postgres"),
        password=(os.getenv("AQ_READER_PASSWORD", "") if reader
                  else os.getenv("POSTGRES_PASSWORD", "")),
        database=database or os.getenv("POSTGRES_DB", "airquality"),
    )


# LLM assistant (local model served by Ollama)
READER_ROLE = "aq_reader"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
KNOWLEDGE_DIR = PROJECT_ROOT / "docs" / "knowledge"
