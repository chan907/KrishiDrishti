"""Grad-CAM for the KrishiDrishti model layout.

Model layout (see ml/src/training/models.py):
  image -> backbone (nested model, output = last conv feature map)
        -> gap -> dropout -> logits -> probs(softmax)
We take gradients of the PRE-softmax logit of the target class w.r.t. the
backbone's feature map (softmax gradients saturate when confidence ~1).

Grad-CAM is a model-interpretation tool. It does not prove biological causality.
"""
from __future__ import annotations

import cv2
import numpy as np


def make_gradcam(model, x_batch: np.ndarray, class_index: int | None = None):
    import tensorflow as tf

    backbone = model.get_layer("backbone")
    head = [model.get_layer(n) for n in ("gap", "dropout", "logits")]
    x = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        feats = backbone(x, training=False)
        tape.watch(feats)
        y = feats
        for layer in head:
            y = layer(y, training=False)
        if class_index is None:
            class_index = int(tf.argmax(y[0]))
        score = y[:, class_index]
    grads = tape.gradient(score, feats)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(feats * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype(np.float32), int(class_index)


def overlay_heatmap(rgb_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    h, w = rgb_uint8.shape[:2]
    heat = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0, 1)
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_uint8, 1 - alpha, colored, alpha, 0)
