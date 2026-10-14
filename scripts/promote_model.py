"""Copy a trained run into ml/models/current (what the backend/Docker load).
  python scripts/promote_model.py mobilenetv3s-v1
Optionally also copies a TFLite variant for the mobile app:  --tflite float16
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--tflite", default=None)
a = ap.parse_args()
src, dst = ROOT / "ml" / "models" / a.run, ROOT / "ml" / "models" / "current"
dst.mkdir(parents=True, exist_ok=True)
for f in ("model.keras", "model_meta.json", "class_names.json"):
    shutil.copy2(src / f, dst / f)
print(f"promoted {a.run} -> {dst}")
if a.tflite:
    assets = ROOT / "mobile" / "assets" / "model"
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "tflite" / f"{a.tflite}.tflite", assets / "model.tflite")
    shutil.copy2(src / "model_meta.json", assets / "model_meta.json")
    kb = ROOT / "backend" / "app" / "data" / "knowledge_base.json"
    shutil.copy2(kb, ROOT / "mobile" / "assets" / "kb" / "knowledge_base.json")
    print(f"copied {a.tflite}.tflite + meta + KB to mobile/assets")
