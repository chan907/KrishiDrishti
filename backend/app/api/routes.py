from __future__ import annotations

import json
import logging
import uuid

import cv2
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from krishi_core.quality import QUALITY_MESSAGE
from krishi_core.uncertainty import LOW_CONFIDENCE_MESSAGE

from app.schemas.schemas import HealthResponse, HistoryDetail, HistoryItem, PredictionResponse
from app.services.image_service import InvalidImage
from app.services.model_service import ModelUnavailable

log = logging.getLogger("krishidrishti.api")
router = APIRouter(prefix="/api")


def _svc(request: Request):
    return request.app.state.svc


def _save_png(media_dir, arr_rgb) -> str:
    name = f"{uuid.uuid4().hex}.png"
    cv2.imwrite(str(media_dir / name), cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))
    return name


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    m = _svc(request)["model"]
    return HealthResponse(status="ok", model_loaded=m.loaded,
                          model_version=m.get_model_version() if m.loaded else None,
                          num_classes=len(m.class_names) if m.loaded else None)


@router.get("/crops")
def crops(request: Request):
    svc = _svc(request)
    supported = {c.split("___")[0] for c in svc["model"].class_names} if svc["model"].loaded else set()
    return [c for c in svc["advisory"].crops() if not supported or c["id"] in supported]


@router.get("/crops/{crop}/diseases")
def crop_diseases(crop: str, request: Request):
    svc = _svc(request)
    trained = set(svc["model"].class_names)
    items = [d for d in svc["advisory"].diseases_for_crop(crop) if not trained or d["class_id"] in trained]
    if not items:
        raise HTTPException(404, f"Crop '{crop}' is not supported.")
    return items


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, image: UploadFile = File(...), crop: str | None = Form(None),
                  language: str = Form("en"), explain: bool = Form(True)):
    svc = _svc(request)
    model, imgs, adv = svc["model"], svc["image"], svc["advisory"]
    settings = svc["settings"]
    if not model.loaded:
        raise HTTPException(503, "Model is not available on the server. Please try again later or use offline mode.")

    # crop validation
    supported_crops = {c.split("___")[0] for c in model.class_names}
    selected = None
    if crop:
        match = [c for c in supported_crops if c.lower() == crop.lower()]
        if not match:
            raise HTTPException(400, f"Unsupported crop '{crop}'. Supported: {sorted(supported_crops)}")
        selected = match[0]

    data = await image.read()
    try:
        img = imgs.validate_and_load(data, image.content_type)
    except InvalidImage as e:
        code = 413 if "too large" in str(e) else 400
        raise HTTPException(code, str(e))

    q = imgs.quality(img)
    if not q.ok:
        raise HTTPException(422, detail={"message": QUALITY_MESSAGE, "issues": q.issues, "metrics": q.metrics})

    try:
        batch = imgs.to_batch(img, model.input_size)
        probs = model.predict(batch)
    except ModelUnavailable:
        raise HTTPException(503, "Model is not available.")
    except Exception:
        log.exception("inference failed")
        raise HTTPException(500, "Prediction failed unexpectedly. The image was not analysed.")

    res = model.analyze(probs)
    warnings = []
    pred_crop = res["class_id"].split("___")[0]
    if selected and pred_crop != selected:
        res["reliable"] = False
        res["reasons"].append("predicted_crop_differs_from_selected_crop")
        warnings.append("The predicted crop differs from the crop you selected.")
    for r in res["reasons"]:
        if r != "predicted_crop_differs_from_selected_crop":
            warnings.append(r.replace("_", " "))

    lang = language if language in ("en", "hi", "mr") else "en"
    top_k = [{"class_id": t["class_id"],
              "label": f"{adv.crop_name(t['class_id'].split('___')[0], lang)} - {adv.disease_name(t['class_id'], lang)}",
              "probability": t["probability"]} for t in res["top_k"]]

    media = settings.storage_dir / "media"
    image_ref = f"{uuid.uuid4().hex}.jpg"
    (media / image_ref).write_bytes(imgs.clean_jpeg_bytes(img))

    out = {"reliable": res["reliable"], "message": None, "crop": None, "class_id": None, "disease": None,
           "confidence": res["confidence"],
           "uncertainty": {k: res[k] for k in ("margin", "normalized_entropy", "reasons")} | {"thresholds": model.thresholds},
           "top_k": top_k, "severity": None, "advisory": None, "quality": q.metrics,
           "gradcam_url": None, "severity_url": None, "model_version": model.get_model_version(),
           "language": lang, "warnings": warnings}
    gradcam_ref = sev_ref = None
    sev_pct = sev_level = None

    if res["reliable"]:
        class_id = res["class_id"]
        is_healthy = class_id.lower().endswith("___healthy")
        out.update(class_id=class_id, crop=adv.crop_name(pred_crop, lang), disease=adv.disease_name(class_id, lang))
        sev = svc["severity"].estimate(img, is_healthy)
        if sev["overlay"] is not None and sev["available"]:
            sev_ref = _save_png(media, sev["overlay"])
            out["severity_url"] = f"/media/{sev_ref}"
        sev_pct, sev_level = sev["percent"], sev["level"]
        out["severity"] = {k: sev[k] for k in ("available", "percent", "level", "note")}
        out["advisory"] = adv.get_advisory(class_id, sev_level, lang)
        if explain:
            try:
                ov = svc["explain"].gradcam_overlay(model.model, batch, img, res["class_index"])
                gradcam_ref = _save_png(media, ov)
                out["gradcam_url"] = f"/media/{gradcam_ref}"
            except Exception:
                log.exception("gradcam failed")
                warnings.append("explanation heatmap unavailable")
    else:
        out["message"] = LOW_CONFIDENCE_MESSAGE + " Prediction uncertain - please capture another image or consult an agricultural expert."

    row = svc["history"].save(
        image_reference=image_ref, gradcam_reference=gradcam_ref, severity_reference=sev_ref,
        selected_crop=selected, class_id=out["class_id"], crop=out["crop"], predicted_disease=out["disease"],
        confidence=res["confidence"], reliable=res["reliable"], severity_percentage=sev_pct,
        severity_level=sev_level, model_version=model.get_model_version(), language=lang, result_json="{}")
    out["prediction_id"] = row.id
    svc["history"].update_result(row.id, out)
    return out


def _item(r):
    return dict(id=r.id, created_at=r.created_at, crop=r.crop, predicted_disease=r.predicted_disease,
                class_id=r.class_id, confidence=r.confidence, reliable=r.reliable,
                severity_percentage=r.severity_percentage, severity_level=r.severity_level,
                model_version=r.model_version, image_url=f"/media/{r.image_reference}")


@router.get("/predictions", response_model=list[HistoryItem])
def list_predictions(request: Request, limit: int = 50, offset: int = 0):
    return [_item(r) for r in _svc(request)["history"].list(min(limit, 200), offset)]


@router.get("/predictions/{pid}", response_model=HistoryDetail)
def get_prediction(pid: int, request: Request):
    r = _svc(request)["history"].get(pid)
    if not r:
        raise HTTPException(404, "Prediction not found")
    return {**_item(r), "result": json.loads(r.result_json)}


@router.delete("/predictions/{pid}", status_code=204)
def delete_prediction(pid: int, request: Request):
    if not _svc(request)["history"].delete(pid):
        raise HTTPException(404, "Prediction not found")


@router.get("/diseases/{class_id}")
def disease(class_id: str, request: Request, language: str = "en"):
    a = _svc(request)["advisory"].get_advisory(class_id, None, language)
    if not a:
        raise HTTPException(404, "Disease not found in knowledge base")
    return a


@router.get("/advisory/{class_id}")
def advisory(class_id: str, request: Request, language: str = "en", severity_level: str | None = None):
    a = _svc(request)["advisory"].get_advisory(class_id, severity_level, language)
    if not a:
        raise HTTPException(404, "No curated advisory for this class")
    return a
