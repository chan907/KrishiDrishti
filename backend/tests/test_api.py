import io

from PIL import Image

from tests.conftest import leaf_jpeg


def post(client, data, name="leaf.jpg", ctype="image/jpeg", **form):
    return client.post("/api/predict", files={"image": (name, data, ctype)}, data=form)


def test_health(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["model_loaded"] and r["model_version"] == "test-0"


def test_crops_and_diseases(client):
    assert {c["id"] for c in client.get("/api/crops").json()} == {"Tomato", "Potato"}
    assert client.get("/api/crops/Tomato/diseases").status_code == 200
    assert client.get("/api/crops/Banana/diseases").status_code == 404


def test_valid_prediction_and_history_roundtrip(client):
    r = post(client, leaf_jpeg(), crop="", language="hi")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_version"] == "test-0" and 0 <= body["confidence"] <= 1
    assert len(body["top_k"]) == 3 and "disclaimer" in body
    pid = body["prediction_id"]
    assert client.get(f"/api/predictions/{pid}").json()["result"]["prediction_id"] == pid
    assert any(i["id"] == pid for i in client.get("/api/predictions").json())
    assert client.delete(f"/api/predictions/{pid}").status_code == 204
    assert client.get(f"/api/predictions/{pid}").status_code == 404


def test_invalid_file_type(client):
    assert post(client, b"hello", "a.txt", "text/plain").status_code == 400


def test_corrupted_image(client):
    assert post(client, b"\xff\xd8\xff\xe0 broken jpeg").status_code == 400


def test_oversized_file(client):
    assert post(client, b"0" * (2 * 1024 * 1024)).status_code == 413


def test_unsupported_crop(client):
    assert post(client, leaf_jpeg(), crop="banana").status_code == 400


def test_poor_quality_image_rejected(client):
    buf = io.BytesIO(); Image.new("RGB", (300, 300), (3, 3, 3)).save(buf, "JPEG")
    r = post(client, buf.getvalue())
    assert r.status_code == 422 and "Image quality is insufficient" in r.json()["detail"]["message"]


def test_low_confidence_not_a_diagnosis(strict_client):
    r = post(strict_client, leaf_jpeg())
    assert r.status_code == 200
    b = r.json()
    assert b["reliable"] is False and b["disease"] is None and b["advisory"] is None
    assert "Unable to make a reliable prediction" in b["message"]


def test_model_unavailable_returns_503(no_model_client):
    assert no_model_client.get("/api/health").json()["model_loaded"] is False
    assert post(no_model_client, leaf_jpeg()).status_code == 503


def test_advisory_endpoint(client):
    r = client.get("/api/advisory/Tomato___Early_blight", params={"language": "mr"})
    assert r.status_code == 200 and r.json()["language"] == "mr" and r.json()["general_actions"]
    assert client.get("/api/advisory/Nope___x").status_code == 404
