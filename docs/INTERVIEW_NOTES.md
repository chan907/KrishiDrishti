# Interview cheat-sheet (fill in your own measured numbers)

| Question | Where in the project | Your numbers / notes |
|---|---|---|
| Problem & users | README, brief | |
| Dataset choice & limits | docs/DATASET.md, `eda_report.md` | |
| Cleaning + leakage-safe split | `ml/src/data/prepare_dataset.py`, `dedupe.py` | dup clusters found: |
| Why transfer learning | `training/models.py`, `train.py` | baseline vs pretrained F1: |
| Why this architecture | `docs/model_comparison.md` | size/latency/F1 trade-off |
| Preprocessing & no train/serve skew | `core/krishi_core/preprocess.py` | |
| Overfitting control | augmentation, dropout, early stopping, LR schedule, `training_curves.png` | |
| Accuracy vs precision/recall/F1, macro | `evaluate.py`, `classification_report.txt` | |
| Confusion matrix reading | `eval_test/confusion_matrix.png` | top confusions: |
| Lab vs field gap | `docs/REALWORLD_EVALUATION.md` | drop: |
| Uncertainty / unreliable output | `uncertainty.py`, `calibrate_threshold.py` | threshold, coverage: |
| Grad-CAM | `gradcam.py` (gradients of logit wrt last conv maps) | |
| Severity | `severity.py`, `docs/SEVERITY_METHOD.md` | MAE vs masks: |
| FastAPI serving | `backend/app/main.py`, `model_service.py` (load once) | |
| TFLite offline | `export/*`, `mobile/lib/services/tflite_service.dart` | size, agreement: |
| Testing | `core/tests`, `ml/tests`, `backend/tests`, `mobile/test` | |
| Docker & CI | `backend/Dockerfile`, `.github/workflows/ci.yml` | |
| Limitations (be candid) | `docs/MODEL_CARD.md` | |
