"""Exact + near-duplicate detection (dHash) and grouping.

Why: PlantVillage contains many near-identical shots (same leaf, rotated/flipped).
If copies land in both train and test, test accuracy is inflated (data leakage).
We cluster near-duplicates and keep every cluster inside ONE split.
"""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def md5_file(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path, size: int = 8) -> int:
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _hamming_to_all(h: np.uint64, arr: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(arr, h)
    return _POP[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def cluster_near_duplicates(hashes: list[int], max_hamming: int = 4) -> list[int]:
    """Union-find clustering. Returns a cluster id per image."""
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    arr = np.array(hashes, dtype=np.uint64)
    for i in range(n):
        d = _hamming_to_all(arr[i], arr[i + 1:])
        for j in np.nonzero(d <= max_hamming)[0]:
            ri, rj = find(i), find(i + 1 + int(j))
            if ri != rj:
                parent[rj] = ri
    return [find(i) for i in range(n)]
