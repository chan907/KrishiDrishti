import json
from ml.src.common.config import ROOT, load_config, resolve
from ml.src.training.data import load_datasets

cfg = load_config()

_, _, class_names = load_datasets(cfg)

print("Classes:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

size = cfg["model"]["input_size"]

print("\nInput size:", size)

run_dir = ROOT / "ml" / "models" / "baseline-v1"

class_names_file = run_dir / "class_names.json"
class_names_file.write_text(json.dumps(class_names, indent=2))

ds_info = json.loads(
    (resolve(cfg["dataset"]["split_root"]) / "dataset_version.json").read_text()
)

meta = {
    "model_version": "1.0.0",
    "run_name": "baseline-v1",
    "architecture": "small_cnn",
    "input_size": size,
    "input_range": "float32 RGB 0..255 (normalisation inside model)",
    "class_names": class_names,
    "dataset": ds_info,
    "params": 392017,
    "metrics": None,
    "uncertainty": None,
}

(run_dir / "model_meta.json").write_text(
    json.dumps(meta, indent=2)
)

print("\nMetadata recovered successfully.")
print("Saved:", run_dir / "model_meta.json")