"""Copy a field-image dataset into ml/data/realworld/<class_id>/ using ml/configs/realworld_mapping.yaml.
  python scripts/prepare_realworld.py --src path/to/PlantDoc-Dataset
Then:  python -m ml.src.evaluation.evaluate --run-name <run> --data-dir ml/data/realworld --report-name realworld
These images must NEVER be used for training if you want an honest real-world number.
"""
import argparse
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True)
a = ap.parse_args()
mapping = yaml.safe_load((ROOT / "ml" / "configs" / "realworld_mapping.yaml").read_text())["mapping"]
dst = ROOT / "ml" / "data" / "realworld"
n = 0
for folder in Path(a.src).rglob("*"):
    if folder.is_dir() and folder.name in mapping:
        out = dst / mapping[folder.name]
        out.mkdir(parents=True, exist_ok=True)
        for f in folder.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copy2(f, out / f"{folder.parent.name}_{f.name}"); n += 1
print(f"copied {n} images into {dst}")
if n == 0:
    print("Nothing copied: edit folder names in ml/configs/realworld_mapping.yaml to match your dataset.")
