from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(os.environ.get("KRISHI_CONFIG", ROOT / "ml" / "configs" / "config.yaml"))
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def load_config(path: str | Path | None = None) -> dict:
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except Exception:  # TF not needed for data scripts
        pass


def list_images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)
