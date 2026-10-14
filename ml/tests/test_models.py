import numpy as np
import pytest

from ml.src.data.classes import normalise_class_folder
from ml.src.data.prepare_dataset import split_clusters
from ml.src.training.models import ARCHS, build_model

import random


def test_class_normalisation():
    assert normalise_class_folder("Corn_(maize)___Common_rust_")[2] == "Corn_maize___Common_rust"
    got = normalise_class_folder("Tomato___Spider_mites Two-spotted_spider_mite")[2]
    assert got == "Tomato___Spider_mites_Two_spotted_spider_mite"


def test_split_keeps_clusters_together_and_disjoint():
    clusters = {i: [f"{i}_{j}" for j in range(1 + i % 4)] for i in range(200)}
    assign = split_clusters(clusters, {"train": .7, "val": .15, "test": .15}, random.Random(0))
    for cid, files in clusters.items():
        assert len({assign[f] for f in files}) == 1  # near-duplicates never straddle splits
    assert set(assign.values()) == {"train", "val", "test"}


@pytest.mark.parametrize("arch", ["small_cnn", "mobilenetv3_small"])
def test_model_io_shape(arch):
    model, _ = build_model(arch, num_classes=5, input_size=224, pretrained=False)
    out = model.predict(np.random.uniform(0, 255, (2, 224, 224, 3)).astype("float32"), verbose=0)
    assert out.shape == (2, 5) and np.allclose(out.sum(1), 1, atol=1e-4)
    for name in ("backbone", "gap", "dropout", "logits", "probs"):
        model.get_layer(name)  # Grad-CAM relies on these names


def test_all_archs_registered():
    assert {"mobilenetv3_small", "efficientnet_b0", "resnet50v2"} <= set(ARCHS)
