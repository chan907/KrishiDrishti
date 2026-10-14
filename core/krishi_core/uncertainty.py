"""Turn a softmax vector into a reliability decision.

Softmax confidence alone is NOT a trustworthy out-of-distribution detector
(networks are often over-confident on unseen inputs). We therefore combine:
  1. top-1 confidence      2. margin between top-1 and top-2
  3. normalised entropy    (+ crop-consistency check in the backend)
These reduce, but do not eliminate, wrong confident answers.
"""
from __future__ import annotations

import numpy as np

LOW_CONFIDENCE_MESSAGE = "Unable to make a reliable prediction."

DEFAULT_THRESHOLDS = {"confidence": 0.80, "margin": 0.20, "entropy": 0.60}


def analyze_probs(probs, class_names, thresholds=None, top_k: int = 3) -> dict:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    order = np.argsort(-p)
    top1 = int(order[0])
    conf = float(p[top1])
    margin = float(p[order[0]] - p[order[1]]) if len(p) > 1 else conf
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
    reasons = []
    if conf < t["confidence"]:
        reasons.append("low_confidence")
    if margin < t["margin"]:
        reasons.append("small_margin_between_top_classes")
    if ent > t["entropy"]:
        reasons.append("high_prediction_entropy")
    return {
        "class_index": top1,
        "class_id": class_names[top1],
        "confidence": conf,
        "margin": margin,
        "normalized_entropy": ent,
        "reliable": not reasons,
        "reasons": reasons,
        "top_k": [{"class_id": class_names[int(i)], "probability": float(p[i])} for i in order[:top_k]],
    }
