"""Visible-lesion severity estimation using classical computer vision (HSV).

severity % = lesion_pixels / leaf_pixels * 100

IMPORTANT LIMITATIONS (also in docs/SEVERITY_METHOD.md):
  * No severity-labelled data was used. This is a colour-threshold HEURISTIC.
  * Yellow/brown healthy senescence and soil/brown backgrounds can be miscounted.
  * Works best for a single leaf on a plain background. Validate against
    expert masks with `ml/src/severity/validate_severity.py` before trusting it.
  * Result is an *estimate of visible area affected*, not a yield-loss figure.
OpenCV HSV ranges: H 0..179, S 0..255, V 0..255.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

DEFAULT_LEVELS = [(10.0, "Mild"), (30.0, "Moderate"), (60.0, "High"), (float("inf"), "Severe")]


def severity_level(percent: float, levels=DEFAULT_LEVELS) -> str:
    for upper, name in levels:
        if percent <= upper:
            return name
    return "Severe"


def estimate_severity(img: Image.Image, max_side: int = 512, min_leaf_fraction: float = 0.08) -> dict:
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(rgb, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # 1) candidate leaf = green..yellow..brown hues with some colour saturation
    leaf_cand = ((H >= 8) & (H <= 95) & (S > 35) & (V > 30)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    leaf_cand = cv2.morphologyEx(leaf_cand, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(leaf_cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = rgb.shape[0] * rgb.shape[1]
    if not contours:
        return _fail(rgb, "no_leaf_region_found")
    biggest = max(contours, key=cv2.contourArea)
    leaf_mask = np.zeros(leaf_cand.shape, np.uint8)
    cv2.drawContours(leaf_mask, [biggest], -1, 255, thickness=cv2.FILLED)  # fills holes (dark lesions)
    leaf_area = int((leaf_mask > 0).sum())
    if leaf_area / total < min_leaf_fraction:
        return _fail(rgb, "leaf_too_small_in_frame")

    # 2) healthy green inside the leaf; everything else in the leaf is "visibly affected"
    healthy = ((H >= 35) & (H <= 95) & (S > 40) & (V > 40)).astype(np.uint8) * 255
    lesion = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(healthy))
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    lesion_area = int((lesion > 0).sum())
    pct = round(100.0 * lesion_area / leaf_area, 2)

    overlay = rgb.copy()
    overlay[lesion > 0] = (0.5 * overlay[lesion > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    cv2.drawContours(overlay, [biggest], -1, (0, 120, 255), 2)
    return {"available": True, "percent": pct, "level": severity_level(pct),
            "leaf_fraction_of_image": round(leaf_area / total, 3), "overlay": overlay, "note": None}


def _fail(rgb, reason):
    return {"available": False, "percent": None, "level": None, "leaf_fraction_of_image": None,
            "overlay": rgb, "note": reason}


def iou_and_percent_error(pred_lesion: np.ndarray, true_lesion: np.ndarray, pred_leaf, true_leaf) -> dict:
    """Helper for validate_severity.py"""
    inter = np.logical_and(pred_lesion, true_lesion).sum()
    union = np.logical_or(pred_lesion, true_lesion).sum()
    p_pred = 100 * pred_lesion.sum() / max(pred_leaf.sum(), 1)
    p_true = 100 * true_lesion.sum() / max(true_leaf.sum(), 1)
    return {"lesion_iou": float(inter / union) if union else 1.0,
            "percent_pred": float(p_pred), "percent_true": float(p_true),
            "abs_error_pp": float(abs(p_pred - p_true))}
