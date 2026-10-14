from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from ml.src.common.config import list_images, resolve


def augmentation_layer():
    """Realistic, mild augmentation - TRAINING ONLY. No vertical flip / hue shifts
    (they would create unrealistic leaf appearance and distort disease colour cues)."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomBrightness(0.15, value_range=(0, 255)),
        layers.RandomContrast(0.15),
    ], name="augmentation")


def load_split(split_dir: Path, size: int, batch: int, shuffle: bool, seed: int, class_names=None):
    ds = keras.utils.image_dataset_from_directory(
        split_dir, labels="inferred", label_mode="int", class_names=class_names,
        image_size=(size, size), batch_size=batch, shuffle=shuffle, seed=seed, interpolation="bilinear")
    return ds


def load_datasets(cfg: dict):
    root = resolve(cfg["dataset"]["split_root"])
    size, batch, seed = cfg["model"]["input_size"], cfg["model"]["batch_size"], cfg["seed"]
    class_names = json.loads((root / "class_names.json").read_text())
    train = load_split(root / "train", size, batch, True, seed, class_names)
    val = load_split(root / "val", size, batch, False, seed, class_names)
    aug = augmentation_layer()
    train = train.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE), class_names


def class_weights_for(split_dir: Path, class_names):
    labels = []
    for i, c in enumerate(class_names):
        labels += [i] * len(list_images(split_dir / c))
    w = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=np.array(labels))
    return {i: float(v) for i, v in enumerate(w)}
