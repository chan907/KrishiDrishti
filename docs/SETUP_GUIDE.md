# KrishiDrishti - Step-by-step setup in VS Code

Follow the steps in order. Every command is run from the **repo root** (the folder that contains `ml/`, `backend/`, `mobile/`) unless stated.
Commands are shown for **Windows PowerShell** and **macOS/Linux** where they differ.

---
## STEP 0 - Install the tools (once)
| Tool | Why | Notes |
|---|---|---|
| **Python 3.10 or 3.11** | ML + backend | TensorFlow 2.15 does **not** support 3.12/3.13. Check: `python --version` |
| **VS Code** + extensions: *Python, Pylance, Dart, Flutter, Docker, YAML* | editor | |
| **Git** | version control | |
| **Flutter SDK** (+ Android Studio for the emulator/SDK) | mobile app | `flutter doctor` must be green for Android |
| **Docker Desktop** | container (optional until step 14) | |
| **Kaggle account + API token** | dataset download | kaggle.com > Settings > Create New Token |

GPU is optional. Training on CPU works but is slow (PlantVillage subset: expect hours). Easiest speed-up on Windows: run in **WSL2**, or use Google Colab for the training step only and copy `ml/models/<run>/` back.

---
## STEP 1 - Open the project and create the environment
```powershell
# Windows PowerShell
cd KrishiDrishti
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```
```bash
# macOS / Linux
cd KrishiDrishti
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
In VS Code: `Ctrl+Shift+P` > **Python: Select Interpreter** > choose `.venv`.

`requirements.txt` also installs `core/` (the shared `krishi_core` package) in editable mode. This package holds preprocessing, quality checks, uncertainty, severity and Grad-CAM, and is used by **both** training and the backend, so they can never drift apart (no train/serve skew).

### Verify the install
```bash
pytest core/tests ml/tests -q
cd backend && pytest -q && cd ..
```
Expected: all tests pass (`ml/tests` builds models with random weights, so nothing is downloaded). The backend tests use a tiny *random-weight test model* purely to test API behaviour (validation, errors, storage); it is never used by the real app.

---
## STEP 2 - Choose and download the dataset (PHASE 1-2)
Read `docs/DATASET.md` first. **Verify the licence on the dataset page yourself** before using or redistributing it, and record it in your README.

```bash
pip install kaggle
# put kaggle.json in ~/.kaggle/ (Windows: C:\Users\<you>\.kaggle\kaggle.json)
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p ml/data/raw --unzip
```
Then make sure the class folders (e.g. `Tomato___Early_blight`) are somewhere under `ml/data/raw/plantvillage/`. The Kaggle zip has `color/`, `grayscale/`, `segmented/` sub-folders; the scripts search recursively and **prefer the folder with more files per class, so delete `grayscale/` and `segmented/`** to be safe:
```
ml/data/raw/plantvillage/color/Tomato___Early_blight/*.JPG ...
```
Edit `ml/configs/config.yaml` if you want other crops (`dataset.crops`). Class ids are **discovered from folder names**, never hard-coded.

---
## STEP 3 - Exploratory data analysis (PHASE 3)
```bash
python -m ml.src.data.inspect_dataset
```
Creates `ml/reports/eda/` with `class_distribution.png`, `image_samples.png`, `dimension_stats.png`, `eda_summary.json`, `eda_report.md`. Open them, and write down what you observe (imbalance ratio, corrupt files, image sizes) in your README. **Only numbers produced by this script may appear in your documentation.**

---
## STEP 4 - Leakage-safe train/validation/test split
```bash
python -m ml.src.data.prepare_dataset
```
What it does: drops corrupt files -> removes exact duplicates (MD5) -> groups near-duplicates (dHash) -> splits **whole groups** 70/15/15 per class -> copies to `ml/data/train|val|test` -> writes `class_names.json`, `split_manifest.csv`, `duplicates_report.json`.

> **If you see `WARNING ... largest near-duplicate cluster`** the grouping chained too many images together and the split will be unbalanced. Lower `dataset.near_duplicate_hamming` in `config.yaml` (4 -> 2 -> 0) and run again. Always check `split_counts` in the printed summary is close to 70/15/15.

Optional: `python -m ml.src.data.quality_stats` prints blur/brightness percentiles to calibrate the quality gate thresholds (`BLUR_THRESHOLD`, etc. - environment variables for the backend).

---
## STEP 5 - Baseline model (PHASE 5)
**CNN** = a network of learned image filters. The baseline (small CNN from scratch) tells you how much transfer learning really adds.
```bash
python -m ml.src.training.train --arch small_cnn --run-name baseline-smallcnn-v1 --no-finetune --epochs-head 15
python -m ml.src.evaluation.evaluate --run-name baseline-smallcnn-v1
```
Outputs in `ml/models/baseline-smallcnn-v1/`: `model.keras`, `training_curves.png`, `eval_test/confusion_matrix.png`, `classification_report.txt`, `misclassified.csv`, `metrics.json`.

---
## STEP 6 - Transfer learning + fine-tuning (PHASE 6-7)
**Transfer learning** = start from a network pretrained on ImageNet and train a new classification head. **Fine-tuning** = then unfreeze the last layers and train with a *very low* learning rate (1e-5) so pretrained features adapt without being destroyed. BatchNorm layers stay frozen.
```bash
python -m ml.src.training.train --arch mobilenetv3_small --run-name mobilenetv3s-v1
python -m ml.src.training.train --arch efficientnet_b0   --run-name effnetb0-v1
python -m ml.src.training.train --arch resnet50v2        --run-name resnet50v2-v1
```
Callbacks: `EarlyStopping` (stop when val loss stops improving; restores best weights), `ModelCheckpoint`, `ReduceLROnPlateau`. Optional tracking: `pip install mlflow`, add `--mlflow`, view with `mlflow ui`.
(ResNet50**V2** is used because its input scaling can live inside the model; document that choice.)

Overfitting check: open `training_curves.png` - a growing gap between train and validation loss means overfitting.

---
## STEP 7 - Evaluate and compare (PHASE 8-9, 14)
```bash
python -m ml.src.evaluation.evaluate --run-name mobilenetv3s-v1
python -m ml.src.evaluation.evaluate --run-name effnetb0-v1
python -m ml.src.evaluation.evaluate --run-name resnet50v2-v1
python -m ml.src.evaluation.compare_models        # writes docs/model_comparison.md/.csv
```
- **Precision** = of images predicted as disease X, how many were really X. **Recall** = of real X images, how many we found. **F1** = harmonic mean. **Macro** = average over classes, so rare classes count equally.
- The **confusion matrix** shows which classes are mixed up (rows = actual, columns = predicted).
- Error analysis: open `eval_test/misclassified.csv` and `eval_test/worst_errors/` (most confident mistakes). Write down patterns: lighting, similar diseases, background.

Pick the final model by the **measured trade-off** (macro F1 vs size vs latency), not accuracy alone. Run the evaluation on the test split **once per model for reporting**; do not tune on it.

---
## STEP 8 - Uncertainty, Grad-CAM (PHASE 11-12)
```bash
python -m ml.src.evaluation.calibrate_threshold --run-name mobilenetv3s-v1     # uses VALIDATION set
python -m ml.src.explainability.run_gradcam --run-name mobilenetv3s-v1 --folder ml/data/test/Tomato___Early_blight --n 6
```
Calibration prints coverage vs accepted-accuracy per threshold and stores the chosen thresholds in `model_meta.json`. **Grad-CAM** weights the last convolutional feature maps by the gradient of the class score, so bright areas are where the model "looked". It is a model-interpretation tool, not proof of biological cause.

---
## STEP 9 - Real-world evaluation (PHASE 47)
Download a **field-condition** dataset (e.g. PlantDoc) and/or collect your own phone photos (with permission). Edit `ml/configs/realworld_mapping.yaml` to match its folder names, then:
```bash
python scripts/prepare_realworld.py --src path/to/PlantDoc-Dataset
python -m ml.src.evaluation.evaluate --run-name mobilenetv3s-v1 --data-dir ml/data/realworld --report-name realworld
```
Expect a **large drop** compared with PlantVillage test accuracy - that is normal and is the honest finding. Document it in `docs/REALWORLD_EVALUATION.md`, including failure cases. Never train on these images if you want the number to mean anything.

---
## STEP 10 - Severity estimation (PHASE 13)
Severity is a **separate problem** from classification. This project uses classical HSV colour segmentation (`core/krishi_core/severity.py`); read `docs/SEVERITY_METHOD.md`. To *validate* it you must provide a few dozen expert-annotated masks (layout in `ml/src/severity/validate_severity.py`), then:
```bash
python -m ml.src.severity.validate_severity
```
Until you do, call the output an "estimated visible affected area (unvalidated heuristic)".

---
## STEP 11 - Run the backend (PHASE 15-16)
```bash
python scripts/promote_model.py mobilenetv3s-v1          # copies the chosen model to ml/models/current
python scripts/generate_model_card.py mobilenetv3s-v1    # docs/MODEL_CARD.md from real metrics
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000/docs** (interactive Swagger UI). Try `GET /api/health` (shows `model_loaded: true`), then `POST /api/predict` with an image, e.g.:
```bash
curl -X POST http://localhost:8000/api/predict -F "image=@leaf.jpg" -F "crop=Tomato" -F "language=hi"
```
In VS Code you can also press **F5** and choose *FastAPI backend (reload)*.
The model is loaded **once** at startup (`ModelService.load_model`). If it is missing the API still starts; `/api/predict` returns **503**.

Backend tests: `cd backend && pytest -q`.

---
## STEP 12 - TensorFlow Lite (PHASE 20-21)
**Quantization** stores weights (and maybe activations) with fewer bits: smaller and often faster, possibly slightly less accurate - so we measure.
```bash
python -m ml.src.export.convert_tflite  --run-name mobilenetv3s-v1
python -m ml.src.export.validate_tflite --run-name mobilenetv3s-v1
```
`validate_tflite` runs the **same test images** through Keras and each TFLite variant and reports accuracy, top-1 agreement, max probability difference, size, latency. Choose a variant from those numbers (float16 is a common safe choice), then:
```bash
python scripts/promote_model.py mobilenetv3s-v1 --tflite float16
```
This also copies the TFLite model, `model_meta.json` and the knowledge base into `mobile/assets/`.
PC latency is **not** phone latency - benchmark on a real device.

---
## STEP 13 - Flutter app (PHASE 17-19)
`mobile/` contains the Dart code (`lib/`), `pubspec.yaml`, assets and tests, but not the generated Android/iOS folders. Generate them safely:
```bash
flutter create --org com.krishidrishti --project-name krishidrishti --platforms=android,ios mobile_tmp
# copy platform folders into mobile/ then delete the temp project
```
```powershell
Copy-Item -Recurse mobile_tmp\android mobile\android ; Copy-Item -Recurse mobile_tmp\ios mobile\ios ; Remove-Item -Recurse -Force mobile_tmp
```
```bash
cp -r mobile_tmp/android mobile/android && cp -r mobile_tmp/ios mobile/ios && rm -rf mobile_tmp
```
Then edit **`mobile/android/app/src/main/AndroidManifest.xml`** (inside `<manifest>`, before `<application>`):
```xml
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.CAMERA"/>
```
and add `android:usesCleartextTraffic="true"` to the `<application ...>` tag (needed only because local development uses `http://`; use HTTPS in production).
In `mobile/android/app/build.gradle(.kts)` set **minSdk to 26**.
For iOS add `NSCameraUsageDescription` and `NSPhotoLibraryUsageDescription` to `ios/Runner/Info.plist`.

```bash
cd mobile
flutter pub get
flutter test                  # model/JSON round-trip tests
flutter run                   # emulator or USB-connected phone
```
- **Android emulator**: server address `http://10.0.2.2:8000` (default; the host PC as seen from the emulator).
- **Real phone**: run uvicorn with `--host 0.0.0.0`, allow port 8000 in the PC firewall, put the phone on the same Wi-Fi, and in the app *Settings > Server address* enter `http://<your-PC-LAN-IP>:8000`.
- **Offline**: Settings > *Offline mode* (needs `model.tflite` bundled - step 12). Without internet the app also automatically falls back to the on-device model.
- **Languages**: Settings > Language (English / Hindi / Marathi). UI strings are in `mobile/assets/i18n/*.json`; advisory text comes from the curated knowledge base.

> I could not run Flutter in my build environment, so the Dart code is **unexecuted**. Expect to fix small compile errors / package-version differences on first run (`flutter analyze` helps). The Python side was executed and tested.

Offline limitations in v1: no Grad-CAM and no severity on-device (shown honestly in the UI). The Dart image resize differs very slightly from `tf.image.resize`; compare a handful of images on-device against the server output before claiming equal accuracy.

---
## STEP 14 - Docker (PHASE 23)
```bash
docker build -f backend/Dockerfile -t krishidrishti-api .       # build context = repo root
docker compose up --build                                       # mounts ml/models/current read-only at /models
curl http://localhost:8000/api/health
```
The image contains only inference dependencies (no pandas/sklearn/matplotlib/mlflow) and **no model** - the model is mounted, so you can swap versions without rebuilding. Run `scripts/promote_model.py` first so `ml/models/current` exists. Prediction images/history live in the `krishi_data` volume. For PostgreSQL set `DATABASE_URL=postgresql+psycopg2://...` and add `psycopg2-binary` to requirements.

---
## STEP 15 - Git, GitHub, CI (PHASE 24)
```bash
git init && git add . && git commit -m "KrishiDrishti initial commit"
git branch -M main
git remote add origin https://github.com/<you>/KrishiDrishti.git
git push -u origin main
```
`.gitignore` already excludes datasets, models, reports and storage. GitHub Actions (`.github/workflows/ci.yml`) lints, runs core/ML/backend tests and builds the Docker image on every push/PR. Training is **not** run in CI.
Publish trained models via GitHub Releases or cloud storage, and note the model version in the README.

---
## STEP 16 - Finish the documentation (PHASE 26)
1. `python scripts/generate_model_card.py <run>` -> `docs/MODEL_CARD.md`
2. Copy the comparison table, confusion matrix, EDA charts, Grad-CAM samples and screenshots into the README *from your own run outputs*.
3. Fill `docs/REALWORLD_EVALUATION.md` with the real-world results and failure analysis.
4. Complete `docs/INTERVIEW_NOTES.md` with your own numbers.

## Troubleshooting
| Problem | Fix |
|---|---|
| `No matching distribution for tensorflow==2.15.1` | You are on Python 3.12+. Use 3.10/3.11 |
| `No class folders found` | Check `dataset.raw_dir` and that folders contain `___` in their names |
| Split counts far from 70/15/15 | Lower `near_duplicate_hamming` (Step 4) |
| Out-of-memory during training | `--batch-size 16` |
| Backend `model_loaded: false` | Run `scripts/promote_model.py`; check `MODEL_DIR`; read the server log |
| Phone cannot reach server | Same Wi-Fi, `--host 0.0.0.0`, firewall, correct LAN IP |
| `422 Image quality is insufficient` for good photos | Thresholds too strict: calibrate with `quality_stats` and set `BLUR_THRESHOLD` etc. |
