"""PHASE 20 - Convert the Keras model to TensorFlow Lite variants.

  python -m ml.src.export.convert_tflite --run-name mobilenetv3s-v1

Variants:  float32 | dynamic (weights int8) | float16 | int8_fallback (full-integer w/ float fallback,
needs a representative dataset = 200 training images).
Quantization = storing weights (and maybe activations) with fewer bits -> smaller/faster model,
possibly slightly less accurate. We MEASURE it (validate_tflite.py) rather than assume.
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import tensorflow as tf
from krishi_core.preprocess import load_for_model
from tensorflow import keras

from ml.src.common.config import ROOT, list_images, load_config, resolve


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-name", required=True)
    ap.add_argument("--rep-images", type=int, default=200)
    a = ap.parse_args()
    cfg = load_config()
    run = ROOT / "ml" / "models" / a.run_name
    out = run / "tflite"; out.mkdir(exist_ok=True)
    model = keras.models.load_model(run / "model.keras")
    size = cfg["model"]["input_size"]
    train_imgs = list_images(resolve(cfg["dataset"]["split_root"]) / "train")
    random.Random(cfg["seed"]).shuffle(train_imgs)
    rep = train_imgs[: a.rep_images]

    def representative():
        for p in rep:
            yield [load_for_model(p, (size, size))[None, ...].astype(np.float32)]

    def convert(name, setup):
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        setup(conv)
        data = conv.convert()
        (out / f"{name}.tflite").write_bytes(data)
        print(f"{name}: {len(data)/1e6:.2f} MB")

    convert("float32", lambda c: None)
    convert("dynamic", lambda c: setattr(c, "optimizations", [tf.lite.Optimize.DEFAULT]))

    def f16(c):
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_types = [tf.float16]
    convert("float16", f16)

    def i8(c):
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = representative
    try:
        convert("int8_fallback", i8)
    except Exception as e:  # some ops may not quantise - report, don't crash
        print("int8_fallback FAILED:", e)


if __name__ == "__main__":
    main()
