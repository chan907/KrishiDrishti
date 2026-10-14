"""PHASE 13 - Validate the classical severity method against EXPERT MASKS.

You must supply images + masks yourself (e.g. a small hand-annotated set, 30-100 images):
  ml/data/severity_val/images/xxx.jpg
  ml/data/severity_val/leaf_masks/xxx.png      (white = leaf)
  ml/data/severity_val/lesion_masks/xxx.png    (white = diseased)
Run: python -m ml.src.severity.validate_severity
Reports lesion IoU and absolute severity error (percentage points). Without this, severity
numbers are UNVALIDATED estimates and must be described that way.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
from krishi_core.preprocess import load_pil
from krishi_core.severity import estimate_severity

from ml.src.common.config import ROOT, resolve


def main():
    base = resolve("ml/data/severity_val")
    imgs = sorted((base / "images").glob("*"))
    if not imgs:
        raise SystemExit(f"No images in {base/'images'} - see the docstring for the expected layout.")
    rows = []
    for p in imgs:
        img = load_pil(p)
        res = estimate_severity(img, max_side=max(img.size))
        if not res["available"]:
            rows.append({"file": p.name, "skipped": res["note"]}); continue
        leaf = cv2.imread(str(base / "leaf_masks" / (p.stem + ".png")), 0) > 127
        les = cv2.imread(str(base / "lesion_masks" / (p.stem + ".png")), 0) > 127
        pct_true = 100 * les.sum() / max(leaf.sum(), 1)
        rows.append({"file": p.name, "percent_pred": res["percent"], "percent_true": round(float(pct_true), 2),
                     "abs_error_pp": round(abs(res["percent"] - pct_true), 2)})
    ok = [r for r in rows if "abs_error_pp" in r]
    summary = {"n": len(rows), "evaluated": len(ok),
               "mean_abs_error_pp": round(float(np.mean([r["abs_error_pp"] for r in ok])), 2) if ok else None,
               "rows": rows}
    (ROOT / "ml" / "reports" / "severity_validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
