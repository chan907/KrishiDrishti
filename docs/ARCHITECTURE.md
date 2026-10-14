# Architecture

```mermaid
flowchart TD
  C[Flutter / Web client] -->|multipart image| API[FastAPI]
  API --> V[Image validation + quality gate]
  V -->|poor quality| R1[422: image quality insufficient]
  V --> P[Preprocess - krishi_core]
  P --> M[Keras model - loaded once]
  M --> U[Uncertainty: confidence, margin, entropy, crop check]
  U -->|unreliable| R2["Unable to make a reliable prediction"]
  U -->|reliable| S[Severity - HSV segmentation]
  U --> G[Grad-CAM]
  U --> A[Advisory - curated knowledge base]
  S --> OUT[Result JSON]
  G --> OUT
  A --> OUT
  OUT --> H[(SQLite / PostgreSQL history)]
```

```mermaid
flowchart LR
  CAM[Camera] --> PRE[Resize to 224, 0..255 float] --> TFL[TFLite model] --> UNC[Same uncertainty rules] --> KB[Local knowledge base JSON] --> UI[Result screen]
```

## Key design decisions
| Decision | Reason |
|---|---|
| Normalisation inside the model graph | the same graph is exported to TFLite; removes a major train/serve skew source |
| `core/krishi_core` shared package | preprocessing, quality, uncertainty, severity, Grad-CAM exist exactly once |
| Grad-CAM on pre-softmax logits | softmax gradients vanish at ~100% confidence |
| Advisory from a curated JSON, never a generative model | no invented pesticide names/doses |
| Model mounted into Docker, not baked in | small image, swappable model versions |
| Image re-encoded on upload | strips EXIF/GPS metadata |
| 503 if model missing, server still boots | graceful failure; mobile falls back to offline model |

## Backend layout
`backend/app/{api,config,database,schemas,services,data}` - routes only handle HTTP; services hold logic (`ModelService`, `ImageService`, `SeverityService`, `ExplainService`, `AdvisoryService`, `HistoryService`).

## API summary
`GET /api/health` - `GET /api/crops` - `GET /api/crops/{crop}/diseases` - `POST /api/predict` (image, crop?, language, explain) - `GET /api/predictions` - `GET /api/predictions/{id}` - `DELETE /api/predictions/{id}` - `GET /api/diseases/{class_id}` - `GET /api/advisory/{class_id}`.
(`/api/predict/advanced` from the brief is merged into `/api/predict`, which already returns severity + Grad-CAM; pass `explain=false` to skip the heatmap.)
