"""PHASE 20/21 - Compare Keras vs every TFLite variant on the TEST split.
  python -m ml.src.export.validate_tflite --run-name mobilenetv3s-v1
Reports: accuracy, top-1 agreement with Keras, max prob difference, size, latency.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ml.src.common.config import ROOT, load_config, resolve
from ml.src.training.data import load_split


def run_tflite(path, xs):
    it = tf.lite.Interpreter(model_path=str(path)); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    outs, ts = [], []
    for x in xs:
        it.set_tensor(i["index"], x[None, ...].astype(np.float32))
        t = time.perf_counter(); it.invoke(); ts.append((time.perf_counter() - t) * 1000)
        outs.append(it.get_tensor(o["index"])[0])
    return np.array(outs), float(np.mean(ts))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-name", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = whole test split")
    a = ap.parse_args()
    cfg = load_config()
    run = ROOT / "ml" / "models" / a.run_name
    meta = json.loads((run / "model_meta.json").read_text())
    model = keras.models.load_model(run / "model.keras")
    ds = load_split(resolve(cfg["dataset"]["split_root"]) / "test", meta["input_size"], 32, False, cfg["seed"], meta["class_names"])
    xs, ys = [], []
    for xb, yb in ds:
        xs.append(xb.numpy()); ys.append(yb.numpy())
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    if a.limit:
        xs, ys = xs[: a.limit], ys[: a.limit]
    kp = model.predict(xs, batch_size=32, verbose=0)
    res = {"keras": {"accuracy": float((kp.argmax(1) == ys).mean()),
                     "size_mb": round((run / "model.keras").stat().st_size / 1e6, 2)}}
    for f in sorted((run / "tflite").glob("*.tflite")):
        tp, ms = run_tflite(f, xs)
        res[f.stem] = {"accuracy": float((tp.argmax(1) == ys).mean()),
                       "top1_agreement_with_keras": float((tp.argmax(1) == kp.argmax(1)).mean()),
                       "max_abs_prob_diff": float(np.abs(tp - kp).max()),
                       "size_mb": round(f.stat().st_size / 1e6, 2), "latency_ms_pc_cpu": round(ms, 2)}
    (run / "tflite" / "validation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("Note: PC latency is NOT phone latency - benchmark on a real device.")


if __name__ == "__main__":
    main()
