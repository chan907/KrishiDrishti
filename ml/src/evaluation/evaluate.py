"""PHASES 8-9 - Evaluation + error analysis on a held-out split.

  python -m ml.src.evaluation.evaluate --run-name mobilenetv3s-v1            # test split
  python -m ml.src.evaluation.evaluate --run-name mobilenetv3s-v1 --data-dir ml/data/realworld --report-name realworld
(--data-dir must contain class folders using the SAME class ids. Use for field/phone photos.)
Use the TEST split only for final reporting - never to tune the model.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from tensorflow import keras

from ml.src.common.config import ROOT, load_config, resolve
from ml.src.training.data import load_split


def predict_all(model, ds):
    probs, y = [], []
    for xb, yb in ds:
        probs.append(model.predict(xb, verbose=0)); y.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(y)


def latency_ms(model, size, runs=50):
    x = np.random.RandomState(0).uniform(0, 255, (1, size, size, 3)).astype("float32")
    for _ in range(5):
        model(x, training=False)
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); model(x, training=False); ts.append((time.perf_counter() - t) * 1000)
    return {"mean_ms": round(float(np.mean(ts)), 2), "p95_ms": round(float(np.percentile(ts, 95)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-dir")
    ap.add_argument("--report-name")
    a = ap.parse_args()
    cfg = load_config()
    run = ROOT / "ml" / "models" / a.run_name
    meta = json.loads((run / "model_meta.json").read_text())
    class_names = meta["class_names"]
    model = keras.models.load_model(run / "model.keras")
    data_dir = resolve(a.data_dir) if a.data_dir else resolve(cfg["dataset"]["split_root"]) / a.split
    name = a.report_name or a.split
    size = meta["input_size"]
    ds = load_split(data_dir, size, 32, False, cfg["seed"], class_names)
    files = list(ds.file_paths)
    probs, y = predict_all(model, ds)
    pred = probs.argmax(1)
    present = sorted(set(y) | set(pred))
    labels_present = [class_names[i] for i in present]

    p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=present, average="macro", zero_division=0)
    report = classification_report(y, pred, labels=present, target_names=labels_present, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=present)
    out = run / f"eval_{name}"
    out.mkdir(exist_ok=True)

    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(0.7 * len(present) + 4, 0.7 * len(present) + 3))
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(present))); ax.set_yticks(range(len(present)))
    ax.set_xticklabels(labels_present, rotation=90, fontsize=7); ax.set_yticklabels(labels_present, fontsize=7)
    for i in range(len(present)):
        for j in range(len(present)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(f"Confusion matrix ({name}, row-normalised colours)")
    fig.colorbar(im); plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=150); plt.close()

    # ---- error analysis ----
    conf = probs.max(1)
    err = pd.DataFrame({"file": files, "actual": [class_names[i] for i in y],
                        "predicted": [class_names[i] for i in pred], "confidence": conf})
    errors = err[err.actual != err.predicted].sort_values("confidence", ascending=False)
    errors.to_csv(out / "misclassified.csv", index=False)
    (out / "worst_errors").mkdir(exist_ok=True)
    for _, row in errors.head(30).iterrows():
        src = row.file
        shutil.copy2(src, out / "worst_errors" / f"{row.actual}__as__{row.predicted}__{row.confidence:.2f}_{os.path.basename(src)}")
    pairs = errors.groupby(["actual", "predicted"]).size().sort_values(ascending=False).head(10)

    metrics = {
        "split": name, "num_images": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1),
        "per_class": {k: v for k, v in report.items() if k in labels_present},
        "top_confusions": {f"{k[0]} -> {k[1]}": int(v) for k, v in pairs.items()},
        "model_size_mb": round((run / "model.keras").stat().st_size / 1e6, 2),
        "params": int(model.count_params()), "latency_cpu_batch1": latency_ms(model, size),
        "note": "Latency is hardware dependent; compare models on the SAME machine.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "classification_report.txt").write_text(
        classification_report(y, pred, labels=present, target_names=labels_present, zero_division=0))
    if name == "test":
        (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
        meta["metrics"] = {k: metrics[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "num_images")}
        (run / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(open(out / "classification_report.txt").read())
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "model_size_mb", "latency_cpu_batch1")}, indent=2))
    print(f"errors: {len(errors)}/{len(y)} -> {out}")


if __name__ == "__main__":
    main()
