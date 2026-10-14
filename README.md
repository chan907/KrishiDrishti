# KrishiDrishti - AI-Powered Crop Disease Detection & Advisory

An end-to-end ML engineering project: dataset preparation -> leakage-safe splitting -> transfer learning -> evaluation -> uncertainty handling -> Grad-CAM -> severity estimation -> FastAPI serving -> Flutter app with offline TensorFlow Lite -> Docker + CI.

> **Important:** this repository contains the *code and pipeline*. **No model is shipped and no accuracy numbers are claimed here.** Every metric in your documentation must come from running the scripts on your data (`docs/MODEL_CARD.md` is generated from real outputs). The output is an AI-assisted visual assessment, **not** a guaranteed agricultural diagnosis.

## Start here
**`docs/SETUP_GUIDE.md`** - complete step-by-step instructions for VS Code (Windows/macOS/Linux).

## Features
- Image quality gate (size, blur, brightness) before any prediction
- Transfer learning with MobileNetV3 / EfficientNetB0 / ResNet50V2 + small-CNN baseline, fair comparison table
- Reliability decision from confidence + margin + entropy + crop-consistency; thresholds calibrated on the validation set
- Grad-CAM explanations (model interpretation, not causality)
- Severity = lesion area / leaf area (classical HSV segmentation, documented limits)
- Curated multilingual (English / Hindi / Marathi) advisory knowledge base - no generative text, no pesticide doses
- FastAPI backend, prediction history (SQLite -> PostgreSQL ready), EXIF-stripped storage
- Flutter app: camera/gallery, online + automatic offline fallback via TFLite, history, language switch, optional text-to-speech
- Tests (core, ML, backend, Flutter), Docker, GitHub Actions

## Repository layout
```
core/       shared krishi_core package (preprocess, quality, uncertainty, severity, gradcam)
ml/         configs, data scripts, training, evaluation, explainability, severity validation, TFLite export
backend/    FastAPI app, tests, Dockerfile
mobile/     Flutter app (lib/, assets/, test/)
docs/       setup guide, dataset, severity method, architecture, real-world report, model card
scripts/    promote_model, prepare_realworld, generate_model_card
```
Architecture diagrams: `docs/ARCHITECTURE.md`.

## Quick commands
```bash
pip install -r requirements.txt
pytest core/tests ml/tests -q && (cd backend && pytest -q)
python -m ml.src.data.inspect_dataset && python -m ml.src.data.prepare_dataset
python -m ml.src.training.train --arch mobilenetv3_small --run-name mobilenetv3s-v1
python -m ml.src.evaluation.evaluate --run-name mobilenetv3s-v1
python scripts/promote_model.py mobilenetv3s-v1 && (cd backend && uvicorn app.main:app --reload)
```

## Dataset, results, screenshots
Fill these from your own runs: EDA (`ml/reports/eda`), comparison (`docs/model_comparison.md`), confusion matrix and Grad-CAM images (`ml/models/<run>/`), real-world report (`docs/REALWORLD_EVALUATION.md`), app screenshots.

## Known limitations
Lab-image bias of PlantVillage; limited crops/classes; softmax-based uncertainty is imperfect; severity is a heuristic; Hindi/Marathi translations need expert review (`knowledge_base.json -> review_status`); Flutter code needs a first-run compile check; offline mode has no severity/Grad-CAM yet.

## Future work
Field-image fine-tuning with separate field test set, leaf detection stage, U-Net lesion segmentation, calibrated OOD detection (e.g. energy score / Mahalanobis), reviewed translations, PostgreSQL + auth, model monitoring.
