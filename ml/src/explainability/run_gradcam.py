"""PHASE 12 - Generate Grad-CAM overlays for images.
  python -m ml.src.explainability.run_gradcam --run-name mobilenetv3s-v1 --image path/to/leaf.jpg
  python -m ml.src.explainability.run_gradcam --run-name mobilenetv3s-v1 --folder ml/data/test/Tomato___Early_blight --n 8
Grad-CAM shows which regions influenced the MODEL's output. It is not proof of biological cause.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from krishi_core.gradcam import make_gradcam, overlay_heatmap
from krishi_core.preprocess import load_pil, pil_to_model_input, to_batch
from tensorflow import keras

from ml.src.common.config import ROOT, list_images, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--image"); ap.add_argument("--folder"); ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    run = ROOT / "ml" / "models" / a.run_name
    meta = json.loads((run / "model_meta.json").read_text())
    model = keras.models.load_model(run / "model.keras")
    paths = [Path(a.image)] if a.image else random.Random(0).sample(list_images(resolve(a.folder)), a.n)
    out = run / "gradcam"; out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(paths), 2, figsize=(6, 3 * len(paths)), squeeze=False)
    for ax_row, p in zip(axes, paths):
        img = load_pil(p)
        x = to_batch(pil_to_model_input(img, (meta["input_size"],) * 2))
        probs = model.predict(x, verbose=0)[0]
        cam, idx = make_gradcam(model, x)
        rgb = np.asarray(img)
        ax_row[0].imshow(rgb); ax_row[0].set_title(p.parent.name[:28], fontsize=7)
        ax_row[1].imshow(overlay_heatmap(rgb, cam))
        ax_row[1].set_title(f"{meta['class_names'][idx][:28]} {probs[idx]:.2f}", fontsize=7)
        for ax in ax_row:
            ax.axis("off")
    plt.tight_layout(); plt.savefig(out / "gradcam_samples.png", dpi=140)
    print(f"saved {out / 'gradcam_samples.png'}")


if __name__ == "__main__":
    main()
