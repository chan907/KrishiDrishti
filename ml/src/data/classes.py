"""Class-name handling. Class ids come from dataset folder names - never hardcoded."""
from __future__ import annotations

import re


def norm(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def normalise_class_folder(name: str) -> tuple[str, str, str]:
    """'Corn_(maize)___Common_rust_' -> ('Corn_maize', 'Common_rust', 'Corn_maize___Common_rust')"""
    if "___" in name:
        crop, disease = name.split("___", 1)
    else:
        crop, disease = name, "unknown"
    crop, disease = norm(crop), norm(disease)
    return crop, disease, f"{crop}___{disease}"


def find_class_dirs(raw_dir, crops, include=None):
    """Return {class_id: folder_path} for classes whose crop is selected.
    Searches up to 2 levels deep because Kaggle zips often nest 'color/'."""
    from pathlib import Path
    raw = Path(raw_dir)
    wanted = {norm(c) for c in crops}
    found = {}
    for d in sorted(p for p in raw.rglob("*") if p.is_dir() and "___" in p.name):
        crop, _, cid = normalise_class_folder(d.name)
        if crop in wanted and (include is None or cid in include):
            if cid not in found or len(list(d.iterdir())) > len(list(found[cid].iterdir())):
                found[cid] = d  # prefer the folder with more files (e.g. color over segmented)
    return found
