# Real-world evaluation report (fill with YOUR results)

Status: **NOT YET PERFORMED.** Do not claim field robustness until this page contains real numbers.

## Procedure
1. Obtain field-condition images (PlantDoc and/or your own phone photos). Record source + licence.
2. Map folders with `ml/configs/realworld_mapping.yaml`; run `scripts/prepare_realworld.py`.
3. `python -m ml.src.evaluation.evaluate --run-name <run> --data-dir ml/data/realworld --report-name realworld`
4. Copy numbers from `ml/models/<run>/eval_realworld/metrics.json` below.

## Results
| Metric | PlantVillage test | Real-world set |
|---|---|---|
| Images | | |
| Accuracy | | |
| Macro F1 | | |
| % flagged "unreliable" by the uncertainty gate | | |

## Failure analysis (from `misclassified.csv` / `worst_errors/`)
- Lighting / shadows:
- Background clutter:
- Multiple leaves / whole-plant photos:
- Visually similar diseases:
- Other:

## Conclusions and next steps
(e.g. add field images to training with a *separate* held-out field test set; stronger augmentation; leaf detection before classification.)
