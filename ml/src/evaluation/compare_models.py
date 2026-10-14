"""PHASE 6/8 - Build a model comparison table from real metrics.json files.
Run:  python -m ml.src.evaluation.compare_models
Mobile suitability is derived from MEASURED size/latency (rule of thumb: <= 25 MB and < 100 ms CPU).
"""
from __future__ import annotations

import json

import pandas as pd

from ml.src.common.config import ROOT


def main():
    rows = []
    for m in sorted((ROOT / "ml" / "models").glob("*/metrics.json")):
        d = json.loads(m.read_text())
        meta = json.loads((m.parent / "model_meta.json").read_text())
        mobile_ok = d["model_size_mb"] <= 25 and d["latency_cpu_batch1"]["mean_ms"] < 100
        rows.append({"run": m.parent.name, "arch": meta["architecture"], "accuracy": round(d["accuracy"], 4),
                     "macro_precision": round(d["macro_precision"], 4), "macro_recall": round(d["macro_recall"], 4),
                     "macro_f1": round(d["macro_f1"], 4), "params_M": round(d["params"] / 1e6, 2),
                     "size_MB": d["model_size_mb"], "cpu_ms": d["latency_cpu_batch1"]["mean_ms"],
                     "train_min": round(meta["training_seconds"] / 60, 1), "mobile_suitable": "yes" if mobile_ok else "no"})
    if not rows:
        raise SystemExit("No metrics.json found - run evaluate.py for each trained model first.")
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    out = ROOT / "docs"
    df.to_csv(out / "model_comparison.csv", index=False)
    (out / "model_comparison.md").write_text(df.to_markdown(index=False))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
