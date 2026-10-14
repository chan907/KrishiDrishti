"""PHASES 5-7 - Baseline, transfer learning, fine-tuning.

Examples (run from repo root):
  python -m ml.src.training.train --arch small_cnn --run-name baseline-smallcnn-v1 --no-finetune
  python -m ml.src.training.train --arch mobilenetv3_small --run-name mobilenetv3s-v1
  python -m ml.src.training.train --arch efficientnet_b0   --run-name effnetb0-v1
  python -m ml.src.training.train --arch resnet50v2        --run-name resnet50v2-v1
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

from ml.src.common.config import ROOT, load_config, resolve, set_seed
from ml.src.training.data import class_weights_for, load_datasets
from ml.src.training.models import ARCHS, build_model, freeze_backbone, unfreeze_for_finetune


def callbacks(ckpt, cfg):
    t = cfg["training"]
    return [
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=t["early_stopping_patience"],
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=t["reduce_lr_patience"], min_lr=1e-7),
    ]


def plot_history(hist: dict, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in ("accuracy", "val_accuracy"):
        ax[0].plot(hist[k], label=k)
    for k in ("loss", "val_loss"):
        ax[1].plot(hist[k], label=k)
    ax[0].set_title("Accuracy"); ax[1].set_title("Loss")
    for a in ax:
        a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=ARCHS)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--epochs-head", type=int)
    ap.add_argument("--epochs-finetune", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--no-finetune", action="store_true")
    ap.add_argument("--mlflow", action="store_true", help="optional experiment tracking")
    a = ap.parse_args()

    cfg = load_config()
    if a.batch_size:
        cfg["model"]["batch_size"] = a.batch_size
    t = cfg["training"]
    eh = a.epochs_head or t["epochs_head"]
    ef = 0 if a.no_finetune or a.arch == "small_cnn" else (a.epochs_finetune or t["epochs_finetune"])
    set_seed(cfg["seed"])

    run_dir = ROOT / "ml" / "models" / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, class_names = load_datasets(cfg)
    cw = class_weights_for(resolve(cfg["dataset"]["split_root"]) / "train", class_names) if t["use_class_weights"] else None
    size = cfg["model"]["input_size"]
    model, base = build_model(a.arch, len(class_names), size, cfg["model"]["dropout"], pretrained=a.arch != "small_cnn")
    print(f"{a.arch}: {model.count_params():,} params, {len(class_names)} classes")

    if a.mlflow:
        import mlflow
        mlflow.set_experiment("krishidrishti")
        mlflow.start_run(run_name=a.run_name)
        mlflow.log_params({"arch": a.arch, **cfg["model"], **t, "epochs_head": eh, "epochs_ft": ef})

    t0 = time.time()
    history = {}

    def merge(h):
        for k, v in h.history.items():
            history.setdefault(k, []).extend(v)

    # ---- Stage 1: train only the new head (backbone frozen) ----
    if a.arch != "small_cnn":
        freeze_backbone(model)
    model.compile(keras.optimizers.Adam(t["lr_head"]), "sparse_categorical_crossentropy", metrics=["accuracy"])
    merge(model.fit(train_ds, validation_data=val_ds, epochs=eh, class_weight=cw,
                    callbacks=callbacks(run_dir / "best_stage1.keras", cfg)))

    # ---- Stage 2: fine-tune last N layers with a LOW learning rate ----
    if ef > 0:
        unfreeze_for_finetune(model, base, t["finetune_last_layers"])
        model.compile(keras.optimizers.Adam(t["lr_finetune"]), "sparse_categorical_crossentropy", metrics=["accuracy"])
        merge(model.fit(train_ds, validation_data=val_ds, epochs=ef, class_weight=cw,
                        callbacks=callbacks(run_dir / "best_stage2.keras", cfg)))

    train_seconds = time.time() - t0
    model.save(run_dir / "model.keras")  # EarlyStopping restored best-val-loss weights
    (run_dir / "history.json").write_text(
    json.dumps(history, default=lambda x: float(x))
)
    plot_history(history, run_dir / "training_curves.png")
    (run_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))
    ds_info = json.loads((resolve(cfg["dataset"]["split_root"]) / "dataset_version.json").read_text())
    meta = {
        "model_version": a.version, "run_name": a.run_name, "architecture": a.arch,
        "input_size": size, "input_range": "float32 RGB 0..255 (normalisation inside model)",
        "class_names": class_names, "dataset": ds_info, "trained_at": datetime.now().isoformat(timespec="seconds"),
        "training_seconds": round(train_seconds, 1), "params": int(model.count_params()),
        "training_config": {"seed": cfg["seed"], **cfg["model"], **t, "epochs_head": eh, "epochs_finetune": ef},
        "best_val_loss": min(history["val_loss"]), "best_val_accuracy": max(history["val_accuracy"]),
        "metrics": None, "uncertainty": None,
    }
    (run_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    if a.mlflow:
        import mlflow
        mlflow.log_metrics({"best_val_loss": meta["best_val_loss"], "best_val_accuracy": meta["best_val_accuracy"]})
        mlflow.end_run()
    print(f"Saved -> {run_dir}  (train time {train_seconds/60:.1f} min)")


if __name__ == "__main__":
    main()
