from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.image_inspector import inspect_image_bytes
from app.main import app
from app.schemas import PolicyOutcome


def _create_synthetic_test_image(defect: bool = True) -> bytes:
    img = np.full((320, 320, 3), (200, 200, 200), dtype=np.uint8)
    if defect:
        # Draw dark jagged defect in center
        cv2.line(img, (100, 100), (220, 220), (20, 20, 20), 4)
        cv2.line(img, (220, 220), (250, 180), (10, 10, 10), 3)
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok
    return encoded.tobytes()


def test_inspect_image_bytes_direct():
    image_bytes = _create_synthetic_test_image(defect=True)
    result = inspect_image_bytes(
        image_bytes,
        filename="test_facade.jpg",
        confidence=0.05,
        run_vlm=True,
    )
    assert result.inspection_id.startswith("insp-")
    assert result.dimensions["width"] == 320
    assert result.dimensions["height"] == 320
    assert result.annotated_image.startswith("data:image/jpeg;base64,")
    assert result.health_score >= 0 and result.health_score <= 100
    assert result.primary_recommendation.outcome in (PolicyOutcome.CLEAN, PolicyOutcome.ESCALATE, PolicyOutcome.REVIEW)


def test_api_inspect_image_and_samples():
    client = TestClient(app)

    # 1. Test samples list
    samples_resp = client.get("/api/inspect/samples")
    assert samples_resp.status_code == 200
    samples = samples_resp.json()
    assert isinstance(samples, list)

    # 2. Test upload
    image_bytes = _create_synthetic_test_image(defect=True)
    files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
    data = {"confidence": "0.10", "run_vlm": "true"}
    resp = client.post("/api/inspect/image", files=files, data=data)
    assert resp.status_code == 200
    payload = resp.json()
    assert "inspection_id" in payload
    assert "health_score" in payload
    assert "primary_recommendation" in payload
    assert "annotated_image" in payload


def test_api_inspect_empty_file():
    client = TestClient(app)
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    resp = client.post("/api/inspect/image", files=files)
    assert resp.status_code == 400
