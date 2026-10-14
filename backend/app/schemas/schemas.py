from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel

DISCLAIMER = ("This is an AI-assisted visual assessment, not a guaranteed agricultural diagnosis. "
              "Consult a local agricultural expert before taking action.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    num_classes: Optional[int] = None


class TopK(BaseModel):
    class_id: str
    label: str
    probability: float


class Severity(BaseModel):
    available: bool
    percent: Optional[float] = None
    level: Optional[str] = None
    note: Optional[str] = None
    method: str = "classical HSV colour segmentation (heuristic, unvalidated unless stated in docs)"


class PredictionResponse(BaseModel):
    prediction_id: int
    reliable: bool
    message: Optional[str] = None
    crop: Optional[str] = None
    class_id: Optional[str] = None
    disease: Optional[str] = None
    confidence: float
    uncertainty: dict
    top_k: list[TopK]
    severity: Optional[Severity] = None
    advisory: Optional[dict] = None
    quality: dict
    gradcam_url: Optional[str] = None
    severity_url: Optional[str] = None
    model_version: str
    language: str
    warnings: list[str] = []
    disclaimer: str = DISCLAIMER


class HistoryItem(BaseModel):
    id: int
    created_at: datetime
    crop: Optional[str]
    predicted_disease: Optional[str]
    class_id: Optional[str]
    confidence: float
    reliable: bool
    severity_percentage: Optional[float]
    severity_level: Optional[str]
    model_version: str
    image_url: str


class HistoryDetail(HistoryItem):
    result: dict[str, Any]
