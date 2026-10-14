"""Model builders. Every model:  image(0..255 float) -> backbone -> gap -> dropout -> logits -> probs

Normalisation is INSIDE the graph (see krishi_core.preprocess docstring).
Layer names 'backbone','gap','dropout','logits','probs' are relied on by Grad-CAM.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

ARCHS = ["small_cnn", "mobilenetv3_small", "mobilenetv3_large", "efficientnet_b0", "resnet50v2"]


def _backbone(arch: str, size: int, pretrained: bool):
    shape = (size, size, 3)
    inp = keras.Input(shape=shape, name="backbone_input")
    weights = "imagenet" if pretrained else None
    x, base = inp, None
    if arch == "small_cnn":  # BASELINE: no pretraining
        x = layers.Rescaling(1 / 255.0)(x)
        for f in (32, 64, 128, 128):
            x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D()(x)
        feats = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    else:
        if arch == "mobilenetv3_small":
            base = keras.applications.MobileNetV3Small(input_shape=shape, include_top=False, weights=weights)
        elif arch == "mobilenetv3_large":
            base = keras.applications.MobileNetV3Large(input_shape=shape, include_top=False, weights=weights)
        elif arch == "efficientnet_b0":
            base = keras.applications.EfficientNetB0(input_shape=shape, include_top=False, weights=weights)
        elif arch == "resnet50v2":
            # ResNet50V2 expects [-1, 1]; we scale inside the model
            x = layers.Rescaling(1 / 127.5, offset=-1.0)(x)
            base = keras.applications.ResNet50V2(input_shape=shape, include_top=False, weights=weights)
        else:
            raise ValueError(f"Unknown arch {arch}. Choose from {ARCHS}")
        # MobileNetV3 and EfficientNet (Keras) include their own 0..255 preprocessing.
        feats = base(x)
    return keras.Model(inp, feats, name="backbone"), base


def build_model(arch: str, num_classes: int, input_size: int = 224, dropout: float = 0.3, pretrained: bool = True):
    backbone, base = _backbone(arch, input_size, pretrained)
    inputs = keras.Input(shape=(input_size, input_size, 3), name="image")
    feats = backbone(inputs)
    x = layers.GlobalAveragePooling2D(name="gap")(feats)
    x = layers.Dropout(dropout, name="dropout")(x)
    logits = layers.Dense(num_classes, name="logits")(x)
    probs = layers.Softmax(name="probs")(logits)
    model = keras.Model(inputs, probs, name=f"krishidrishti_{arch}")
    return model, base


def freeze_backbone(model):
    model.get_layer("backbone").trainable = False


def unfreeze_for_finetune(model, base, last_n: int):
    """Unfreeze only the last `last_n` layers of the pretrained base.
    BatchNorm layers stay frozen (standard practice - protects running statistics)."""
    backbone = model.get_layer("backbone")
    backbone.trainable = True
    if base is None:
        return
    base.trainable = True
    for l in base.layers[:-last_n]:
        l.trainable = False
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
