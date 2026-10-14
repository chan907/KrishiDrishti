# Dataset strategy (PHASE 1-2)

## Candidate sources
| Dataset | Type | Use here | Main limitation |
|---|---|---|---|
| **PlantVillage** (Hughes & Salathé, 2015) | ~54k leaf images, 38 classes, mostly **lab/controlled background** | training / validation / test | Models score very high on it but often drop sharply on field photos; many near-duplicate shots of the same leaf |
| **PlantDoc** (Singh et al., 2020) | ~2.5k **field** images, 27 classes | **real-world evaluation** (and optionally fine-tuning later) | Small, noisy labels, fewer classes, some images show whole plants |
| Own phone photos | field | real-world evaluation | Needs labelling by an expert; take only with permission |

## Before using any dataset - checklist (you must do this yourself)
1. Open the dataset page, **read the licence/terms**, and copy them into this file. Do not assume - I have not verified licence terms for you.
2. Record the source URL, download date and version.
3. Record numbers of images, crops, classes (use `python -m ml.src.data.inspect_dataset`).
4. Check class imbalance, corrupt files, duplicates (scripts do this).
5. Keep real-world images **out of training** if you want an honest real-world score.

| Field | Your entry |
|---|---|
| Source URL | |
| Licence | |
| Download date | |
| Images / classes (from `eda_summary.json`) | |

## Scope
Default crops: Tomato, Potato, Corn (maize). Classes are **discovered from the dataset folders**. For PlantVillage that gives (if all present) Tomato 10, Potato 3, Corn 4 classes. If you prefer the smaller list from the brief (e.g. only 5 tomato classes), set `dataset.include_classes` in `ml/configs/config.yaml`. The knowledge base covers those 17 PlantVillage classes; a class not in the knowledge base returns "no curated advisory" rather than invented text.

## Why leakage-safe splitting matters
PlantVillage contains many shots of the *same leaf* (rotations/flips). A random image-level split puts siblings in both train and test and inflates accuracy. We cluster near-duplicates (64-bit dHash, Hamming distance <= `near_duplicate_hamming`) and keep each cluster in a single split.

## Known weakness to be aware of
Single-linkage clustering can chain many similar images into one huge cluster. `prepare_dataset.py` prints a warning when a cluster exceeds 15% of a class; lower the Hamming threshold when that happens.
