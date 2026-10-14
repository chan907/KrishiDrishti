"""Test fixtures. The tiny RANDOM-WEIGHT model below exists ONLY to test API plumbing
(validation, error handling, storage). It is never used by the real application."""
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from tensorflow import keras
from tensorflow.keras import layers

from app.config.settings import Settings
from app.main import create_app

CLASSES = ["Tomato___Early_blight", "Tomato___healthy", "Potato___Late_blight"]


def build_tiny_model():
    inp = keras.Input((224, 224, 3), name="backbone_input")
    x = layers.Rescaling(1 / 255.0)(inp)
    x = layers.Conv2D(4, 3, strides=4, activation="relu")(x)
    backbone = keras.Model(inp, x, name="backbone")
    i = keras.Input((224, 224, 3), name="image")
    x = layers.GlobalAveragePooling2D(name="gap")(backbone(i))
    x = layers.Dropout(0.1, name="dropout")(x)
    x = layers.Dense(len(CLASSES), name="logits", kernel_initializer="glorot_uniform")(x)
    return keras.Model(i, layers.Softmax(name="probs")(x))


def make_settings(tmp_path, model_dir, thresholds=None):
    return Settings(model_dir=model_dir, storage_dir=tmp_path / "storage", max_upload_mb=1.0,
                    default_thresholds=thresholds or {"confidence": 0.0, "margin": 0.0, "entropy": 1.0})


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    build_tiny_model().save(d / "model.keras")
    (d / "model_meta.json").write_text(json.dumps({"model_version": "test-0", "class_names": CLASSES, "input_size": 224}))
    return d


@pytest.fixture()
def client(tmp_path, tiny_model_dir):
    with TestClient(create_app(make_settings(tmp_path, tiny_model_dir))) as c:
        yield c


@pytest.fixture()
def strict_client(tmp_path, tiny_model_dir):
    """Thresholds so strict that a random model can never be 'reliable'."""
    s = make_settings(tmp_path, tiny_model_dir, {"confidence": 0.999, "margin": 0.9, "entropy": 0.01})
    with TestClient(create_app(s)) as c:
        yield c


@pytest.fixture()
def no_model_client(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, tmp_path / "missing"))) as c:
        yield c


def leaf_jpeg(size=300, color=(40, 140, 50)) -> bytes:
    img = Image.new("RGB", (size, size), (235, 235, 235))
    ImageDraw.Draw(img).ellipse([40, 20, size - 40, size - 20], fill=color)
    a = np.asarray(img).astype(np.int16) + np.random.RandomState(1).randint(-25, 25, (size, size, 3))
    buf = io.BytesIO()
    Image.fromarray(np.clip(a, 0, 255).astype("uint8")).save(buf, "JPEG")
    return buf.getvalue()
