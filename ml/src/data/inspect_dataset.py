"""PHASE 2/3 - Dataset inspection + EDA.

Outputs (ml/reports/eda/): class_distribution.png, image_samples.png,
dimension_stats.png, eda_summary.json, eda_report.md
Run:  python -m ml.src.data.inspect_dataset
"""
from __future__ import annotations

import json
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from ml.src.common.config import ROOT, list_images, load_config, resolve
from ml.src.data.classes import find_class_dirs


def main():
    cfg = load_config()
    d = cfg["dataset"]
    classes = find_class_dirs(resolve(d["raw_dir"]), d["crops"], d["include_classes"])
    if not classes:
        raise SystemExit(f"No class folders found under {resolve(d['raw_dir'])}. See docs/DATASET.md")
    out = ROOT / "ml" / "reports" / "eda"
    out.mkdir(parents=True, exist_ok=True)

    counts, corrupt, sizes = {}, [], []
    for cid, folder in classes.items():
        imgs = list_images(folder)
        counts[cid] = len(imgs)
        for p in imgs:
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    sizes.append(im.size)
            except Exception:
                corrupt.append(str(p))

    total = sum(counts.values())
    mx, mn = max(counts.values()), min(counts.values())
    imbalance = round(mx / max(mn, 1), 2)
    size_counter = Counter(sizes)

    # 1) class distribution
    names = sorted(counts, key=counts.get)
    plt.figure(figsize=(10, 0.4 * len(names) + 2))
    plt.barh(names, [counts[n] for n in names], color="#2e7d32")
    plt.xlabel("images"); plt.title("Class distribution"); plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=150); plt.close()

    # 2) sample grid
    random.seed(cfg["seed"])
    cols = 4
    rows = (len(classes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (cid, folder) in zip(axes.flat, classes.items()):
        ax.imshow(Image.open(random.choice(list_images(folder))).convert("RGB"))
        ax.set_title(cid.replace("___", "\n"), fontsize=7)
    plt.tight_layout(); plt.savefig(out / "image_samples.png", dpi=130); plt.close()

    # 3) dimensions
    plt.figure(figsize=(5, 4))
    plt.scatter([s[0] for s in sizes[::10]], [s[1] for s in sizes[::10]], s=4, alpha=0.4)
    plt.xlabel("width"); plt.ylabel("height"); plt.title("Image dimensions (sampled)")
    plt.tight_layout(); plt.savefig(out / "dimension_stats.png", dpi=150); plt.close()

    summary = {
        "dataset": d["name"], "version": d["version"], "num_classes": len(counts), "total_images": total,
        "images_per_class": counts, "imbalance_ratio_max_over_min": imbalance,
        "corrupted_files": corrupt, "most_common_sizes": [[list(k), v] for k, v in size_counter.most_common(5)],
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    md = ["# EDA report (auto-generated from the real dataset)\n",
          f"- classes: **{len(counts)}**, images: **{total}**, imbalance max/min: **{imbalance}**",
          f"- corrupted files: **{len(corrupt)}**", "", "| class | images |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    md += ["", "![dist](class_distribution.png)", "![samples](image_samples.png)"]
    (out / "eda_report.md").write_text("\n".join(md))
    print(json.dumps({k: summary[k] for k in ("num_classes", "total_images", "imbalance_ratio_max_over_min")}, indent=2))
    print(f"corrupted: {len(corrupt)} | reports in {out}")


if __name__ == "__main__":
    main()
