from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _f(name, default):
    return type(default)(os.environ.get(name, default))


@dataclass
class Settings:
    model_dir: Path = Path(
    os.environ.get(
        "MODEL_DIR",
        ROOT / "ml" / "models" / "mobilenetv3s-v1"
    )
)
    storage_dir: Path = Path(os.environ.get("STORAGE_DIR", ROOT / "backend" / "storage"))
    kb_path: Path = Path(os.environ.get("KB_PATH", ROOT / "backend" / "app" / "data" / "knowledge_base.json"))
    database_url: str = os.environ.get("DATABASE_URL", "")
    max_upload_mb: float = _f("MAX_UPLOAD_MB", 8.0)
    allowed_types: tuple = ("image/jpeg", "image/png", "image/jpg")
    # quality gate (calibrate with ml.src.data.quality_stats)
    min_side: int = _f("MIN_SIDE", 128)
    blur_threshold: float = _f("BLUR_THRESHOLD", 40.0)
    dark_threshold: float = _f("DARK_THRESHOLD", 40.0)
    bright_threshold: float = _f("BRIGHT_THRESHOLD", 220.0)
    # uncertainty fallback if model_meta.json has no calibrated thresholds
    default_thresholds: dict = field(default_factory=lambda: {
        "confidence": _f("CONF_THRESHOLD", 0.80), "margin": _f("MARGIN_THRESHOLD", 0.20),
        "entropy": _f("ENTROPY_THRESHOLD", 0.60)})
    cors_origins: tuple = tuple(os.environ.get("CORS_ORIGINS", "*").split(","))

    def __post_init__(self):
        self.model_dir, self.storage_dir = Path(self.model_dir), Path(self.storage_dir)
        if not self.database_url:
            self.database_url = f"sqlite:///{self.storage_dir / 'krishidrishti.db'}"
