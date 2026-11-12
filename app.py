import json
import os
import sys
from pathlib import Path

# Ensure local config and modules are importable when running from the repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

try:
    from config import settings, PROJECT_ROOT, get_db_path, get_logs_dir
except Exception as exc:
    raise SystemExit(f"config import failed: {exc}") from exc

try:
    import numpy as np
    from PIL import Image
except Exception as exc:
    raise SystemExit(f"image backend deps unavailable: {exc}") from exc

try:
    import tensorflow as tf  # noqa: F401
    TF_AVAILABLE = bool(tf is not None)
except Exception as exc:
    TF_AVAILABLE = False

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DB_PATH = LOGS_DIR / "inspections.jsonl"
for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[dev] CompositeGuard AI app starting at {PROJECT_ROOT}")

from fastapi import FastAPI

app = FastAPI(title="CompositeGuard AI App")

def _has_ml():
    return TF_AVAILABLE


SPINNET_READY = False


def _spinnet_attempt_once(image_path: str) -> None:
    pass


def _current_timestamp_iso():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def main():
    print("hello from app")


if __name__ == "__main__":
    main()
