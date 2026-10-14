"""Print blur/brightness distributions on the training set to CALIBRATE quality thresholds.
Run:  python -m ml.src.data.quality_stats
"""
from __future__ import annotations

import random

import numpy as np
from krishi_core.preprocess import load_pil
from krishi_core.quality import compute_quality_metrics

from ml.src.common.config import list_images, load_config, resolve


def main(n: int = 500):
    cfg = load_config()
    imgs = list_images(resolve(cfg["dataset"]["split_root"]) / "train")
    random.Random(0).shuffle(imgs)
    rows = [compute_quality_metrics(load_pil(p)) for p in imgs[:n]]
    for key in ("blur_score", "brightness"):
        v = np.array([r[key] for r in rows])
        print(key, {f"p{q}": round(float(np.percentile(v, q)), 1) for q in (1, 5, 25, 50, 75, 95, 99)})
    print("Tip: set blur threshold near the 1st percentile of GOOD images, then test with real phone photos.")


if __name__ == "__main__":
    main()
