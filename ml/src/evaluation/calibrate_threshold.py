"""PHASE 11 - Choose uncertainty thresholds on the VALIDATION set (never test).

For each confidence threshold we report coverage (fraction answered) and accuracy of
the answered images. We pick the lowest threshold reaching target accuracy and store
it in model_meta.json['uncertainty'].  Run:
  python -m ml.src.evaluation.calibrate_threshold --run-name mobilenetv3s-v1
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from tensorflow import keras

from ml.src.common.config import ROOT, load_config, resolve
from ml.src.evaluation.evaluate import predict_all
from ml.src.training.data import load_split


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-name", required=True)
    a = ap.parse_args()
    cfg = load_config()
    run = ROOT / "ml" / "models" / a.run_name
    meta = json.loads((run / "model_meta.json").read_text())
    model = keras.models.load_model(run / "model.keras")
    ds = load_split(resolve(cfg["dataset"]["split_root"]) / "val", meta["input_size"], 32, False, cfg["seed"], meta["class_names"])
    probs, y = predict_all(model, ds)
    conf = probs.max(1); correct = probs.argmax(1) == y
    target = cfg["uncertainty"]["target_accepted_accuracy"]
    chosen, table = 0.99, []
    for t in np.arange(0.30, 0.995, 0.05):
        m = conf >= t
        cov = float(m.mean()); acc = float(correct[m].mean()) if m.any() else float("nan")
        table.append({"threshold": round(float(t), 2), "coverage": round(cov, 3), "accepted_accuracy": round(acc, 4)})
    for row in table:
        if row["accepted_accuracy"] >= target:
            chosen = row["threshold"]; break
    margin_thr = 0.2
    meta["uncertainty"] = {"confidence": chosen, "margin": margin_thr, "entropy": 0.6,
                           "calibrated_on": "validation", "target_accepted_accuracy": target, "table": table}
    (run / "model_meta.json").write_text(json.dumps(meta, indent=2))
    for r in table:
        print(r)
    print(f"chosen confidence threshold = {chosen}")


if __name__ == "__main__":
    main()
