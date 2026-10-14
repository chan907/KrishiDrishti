"""PHASE 2 - Build leakage-safe train/val/test folders.

Steps: select crops -> drop corrupt -> exact (md5) + near duplicate (dHash) clustering
-> split WHOLE CLUSTERS per class (stratified) -> copy into ml/data/{train,val,test}.
Writes: class_names.json, split_manifest.csv, dataset_version.json, duplicates_report.json
Run:  python -m ml.src.data.prepare_dataset
"""
from __future__ import annotations

import csv
import json
import random
import shutil
from collections import defaultdict
from datetime import date

from PIL import Image
from tqdm import tqdm

from ml.src.common.config import list_images, load_config, resolve, set_seed
from ml.src.data.classes import find_class_dirs
from ml.src.data.dedupe import cluster_near_duplicates, dhash, md5_file


def split_clusters(clusters: dict, ratios: dict, rng: random.Random) -> dict:
    """clusters: {cluster_id: [paths]} -> {path: split}. Greedy fill toward target sizes."""
    total = sum(len(v) for v in clusters.values())
    target = {k: ratios[k] * total for k in ("train", "val", "test")}
    got = {k: 0 for k in target}
    ids = list(clusters)
    rng.shuffle(ids)
    ids.sort(key=lambda c: -len(clusters[c]))  # big clusters first, then fill gaps
    assign = {}
    for cid in ids:
        # choose split with largest remaining deficit (relative)
        split = max(target, key=lambda k: (target[k] - got[k]) / max(target[k], 1e-9))
        got[split] += len(clusters[cid])
        for p in clusters[cid]:
            assign[p] = split
    return assign


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    d = cfg["dataset"]
    rng = random.Random(cfg["seed"])
    classes = find_class_dirs(resolve(d["raw_dir"]), d["crops"], d["include_classes"])
    if not classes:
        raise SystemExit("No classes found. Download the dataset first (docs/DATASET.md).")
    root = resolve(d["split_root"])
    for s in ("train", "val", "test"):
        if (root / s).exists():
            shutil.rmtree(root / s)

    manifest, dup_report, kept_classes = [], {}, []
    for cid, folder in classes.items():
        paths = []
        for p in list_images(folder):
            try:
                with Image.open(p) as im:
                    im.verify()
                paths.append(p)
            except Exception:
                dup_report.setdefault("corrupt", []).append(str(p))
        if len(paths) < d["min_images_per_class"]:
            print(f"SKIP {cid}: only {len(paths)} images")
            continue
        kept_classes.append(cid)
        # exact duplicates
        seen, exact = {}, 0
        uniq = []
        for p in tqdm(paths, desc=f"hash {cid}", leave=False):
            h = md5_file(p)
            if h in seen:
                exact += 1
                continue
            seen[h] = p
            uniq.append(p)
        hashes = [dhash(p) for p in uniq]
        cl = cluster_near_duplicates(hashes, d["near_duplicate_hamming"])
        clusters = defaultdict(list)
        for p, c in zip(uniq, cl):
            clusters[c].append(p)
        multi = sum(1 for v in clusters.values() if len(v) > 1)
        biggest = max(len(v) for v in clusters.values())
        dup_report[cid] = {"images": len(paths), "exact_duplicates_removed": exact,
                           "near_duplicate_clusters": multi, "clusters_total": len(clusters),
                           "largest_cluster": biggest}
        if biggest > 0.15 * len(uniq):
            # Union-find chains similar hashes together. One giant cluster cannot be spread over
            # train/val/test, which makes the split badly unbalanced. Lower the threshold.
            print(f"WARNING {cid}: largest near-duplicate cluster = {biggest}/{len(uniq)} images. "
                  f"Lower dataset.near_duplicate_hamming in config.yaml (try 2, then 0) and re-run.")
        assign = split_clusters(clusters, d["ratios"], rng)
        for p, split in assign.items():
            dest = root / split / cid
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / p.name
            if target.exists():
                target = dest / f"{p.stem}_{md5_file(p)[:6]}{p.suffix}"
            shutil.copy2(p, target)
            manifest.append((str(target.relative_to(root)), cid, split))

    kept_classes.sort()  # Keras image_dataset_from_directory sorts alphabetically too
    (root / "class_names.json").write_text(json.dumps(kept_classes, indent=2))
    with open(root / "split_manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "class", "split"]); w.writerows(manifest)
    counts = defaultdict(lambda: defaultdict(int))
    for _, cid, split in manifest:
        counts[split][cid] += 1
    info = {"dataset": d["name"], "version": d["version"], "prepared": str(date.today()),
            "seed": cfg["seed"], "ratios": d["ratios"], "num_classes": len(kept_classes),
            "split_counts": {s: sum(v.values()) for s, v in counts.items()}}
    (root / "dataset_version.json").write_text(json.dumps(info, indent=2))
    (root / "duplicates_report.json").write_text(json.dumps(dup_report, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
