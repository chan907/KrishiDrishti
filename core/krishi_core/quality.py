"""Basic input-quality gate. Runs BEFORE the classifier.

Thresholds are heuristics. Calibrate them on your own data with
`python -m ml.src.data.quality_stats` and set them in the config/env.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

QUALITY_MESSAGE = (
    "Image quality is insufficient. Please capture another image in good "
    "lighting with the affected leaf clearly visible."
)


@dataclass
class QualityReport:
    ok: bool
    issues: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def compute_quality_metrics(img: Image.Image) -> dict:
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    scale = 512.0 / max(h, w)
    if scale < 1:  # normalise so blur score is comparable across resolutions
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "width": int(w),
        "height": int(h),
        "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),  # low = blurry
        "brightness": float(gray.mean()),  # 0..255
    }


def assess_quality(img: Image.Image, min_side: int = 128, blur_threshold: float = 40.0,
                   dark_threshold: float = 40.0, bright_threshold: float = 220.0) -> QualityReport:
    m = compute_quality_metrics(img)
    issues = []
    if min(m["width"], m["height"]) < min_side:
        issues.append("image_too_small")
    if m["blur_score"] < blur_threshold:
        issues.append("image_too_blurry")
    if m["brightness"] < dark_threshold:
        issues.append("image_too_dark")
    if m["brightness"] > bright_threshold:
        issues.append("image_too_bright")
    return QualityReport(ok=not issues, issues=issues, metrics=m)
