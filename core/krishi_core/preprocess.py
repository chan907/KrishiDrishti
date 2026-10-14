"""Preprocessing shared by training scripts, FastAPI and offline evaluation.

Design decision (avoids train/serve skew):
  * The Keras model takes RAW float32 RGB pixels in the range 0..255.
  * Model-specific normalisation (MobileNetV3 / EfficientNet / ResNetV2 scaling)
    lives INSIDE the model graph, so TFLite exports it too.
  * The only step outside the model is: decode -> EXIF-rotate -> RGB -> resize.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

INPUT_SIZE = (224, 224)
Source = Union[str, Path, bytes, io.BytesIO]


def load_pil(source: Source) -> Image.Image:
    """Open an image, fix camera orientation, and convert to RGB."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    img = Image.open(source)
    img.load()  # force decode now so corrupt files fail here
    img = ImageOps.exif_transpose(img)  # phones store rotation in EXIF
    return img.convert("RGB")


def pil_to_model_input(img: Image.Image, size=INPUT_SIZE) -> np.ndarray:
    """RGB PIL image -> float32 array (H, W, 3) with values 0..255.

    Uses tf.image.resize (bilinear), the same op used by
    keras.utils.image_dataset_from_directory during training.
    """
    import tensorflow as tf  # lazy import keeps non-TF unit tests light

    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    resized = tf.image.resize(arr, size, method="bilinear")
    return np.asarray(resized, dtype=np.float32)


def to_batch(arr: np.ndarray) -> np.ndarray:
    return arr[None, ...].astype(np.float32)


def load_for_model(source: Source, size=INPUT_SIZE) -> np.ndarray:
    return pil_to_model_input(load_pil(source), size)
