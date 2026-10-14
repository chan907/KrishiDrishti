import numpy as np
from PIL import Image, ImageDraw

from krishi_core.quality import assess_quality
from krishi_core.severity import estimate_severity, severity_level
from krishi_core.uncertainty import analyze_probs

CLASSES = ["A___healthy", "A___blight", "A___rust"]


def synthetic_leaf(diseased_fraction=0.0, size=300):
    img = Image.new("RGB", (size, size), (235, 235, 235))
    d = ImageDraw.Draw(img)
    mx, my = int(size * 0.13), int(size * 0.07)
    d.ellipse([mx, my, size - mx, size - my], fill=(40, 140, 50))
    if diseased_fraction:
        r = int((size / 2 - mx) * (diseased_fraction ** 0.5))
        c = size // 2
        d.ellipse([c - r, c - r, c + r, c + r], fill=(120, 70, 30))
    return img


def noisy(img, seed=0):
    a = np.asarray(img).astype(np.int16)
    a = a + np.random.RandomState(seed).randint(-25, 25, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))


def test_severity_levels():
    assert severity_level(5) == "Mild" and severity_level(10) == "Mild"
    assert severity_level(11) == "Moderate" and severity_level(45) == "High" and severity_level(80) == "Severe"


def test_severity_healthy_leaf_near_zero():
    r = estimate_severity(synthetic_leaf(0.0))
    assert r["available"] and r["percent"] < 5


def test_severity_tracks_lesion_area():
    low = estimate_severity(synthetic_leaf(0.10))["percent"]
    high = estimate_severity(synthetic_leaf(0.40))["percent"]
    assert 3 < low < 25 and 25 < high < 60 and high > low


def test_severity_no_leaf():
    r = estimate_severity(Image.new("RGB", (200, 200), (128, 128, 128)))
    assert not r["available"]


def test_quality_flags_dark_and_small():
    dark = assess_quality(Image.new("RGB", (300, 300), (5, 5, 5)))
    assert "image_too_dark" in dark.issues and not dark.ok
    small = assess_quality(noisy(synthetic_leaf(0.2, 64)))
    assert "image_too_small" in small.issues


def test_quality_accepts_sharp_image():
    assert assess_quality(noisy(synthetic_leaf(0.2))).ok


def test_uncertainty_confident_vs_uncertain():
    sure = analyze_probs([0.97, 0.02, 0.01], CLASSES)
    assert sure["reliable"] and sure["class_id"] == "A___healthy"
    unsure = analyze_probs([0.40, 0.35, 0.25], CLASSES)
    assert not unsure["reliable"] and "low_confidence" in unsure["reasons"]
    assert len(unsure["top_k"]) == 3
