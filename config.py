"""CompositeGuard AI - Configuration module.

Defines configurable paths, model settings, and hardware parameters.
All paths are relative to the project root and use pathlib for cross-platform compatibility.
"""

from pathlib import Path
from pydantic import BaseModel, Field, validator
import os

# Project root at E:\
PROJECT_ROOT = Path("E:/compositeguard-ai")

# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, SRC_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Model settings
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
DEVICE = os.getenv("DEVICE", "cpu")

# Serial settings
SERIAL_BAUD_RATE = int(os.getenv("SERIAL_BAUD_RATE", "115200"))
SERIAL_TIMEOUT = float(os.getenv("SERIAL_TIMEOUT", "1.0"))

# Risk assessment defaults
FMEA_DEFAULTS = {
    "severity_scale": "1-10",
    "occurrence_scale": "1-10",
    "detection_scale": "1-10",
    "risk_threshold_low": 10,
    "risk_threshold_high": 30,
}


class Settings(BaseModel):
    """Project settings with validation."""

    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum confidence for reliable predictions"
    )
    image_size: int = Field(
        default=224, ge=64, le=1024,
        description="Input image size for model"
    )
    serial_baud_rate: int = Field(
        default=115200, ge=9600, le=230400,
        description="Serial communication baud rate"
    )
    device: str = Field(
        default="cpu", pattern="^(cpu|cuda)$",
        description="Computation device"
    )

    @validator("device")
    def device_must_be_valid(cls, v):
        if v not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        return v


# Singleton instance
settings = Settings()


def get_db_path() -> Path:
    """Get the path to the SQLite database."""
    return LOGS_DIR / "inspections.db"


def get_logs_dir() -> Path:
    """Get the logs directory path."""
    return LOGS_DIR


def get_models_dir() -> Path:
    """Get the models directory path."""
    return MODELS_DIR


def get_data_dir() -> Path:
    """Get the data directory path."""
    return DATA_DIR