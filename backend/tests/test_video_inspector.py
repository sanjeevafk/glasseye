import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.video_inspector import inspect_video_bytes


def _make_dummy_mp4_bytes(num_frames: int = 15, width: int = 320, height: int = 240) -> bytes:
    """Create a valid in-memory MP4 video file for testing."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_path, fourcc, 10.0, (width, height))
    for i in range(num_frames):
        # Draw a synthetic line/patch to simulate frame content
        frame = np.full((height, width, 3), 120, dtype=np.uint8)
        cv2.line(frame, (20, 20 + i * 5), (100, 100 + i * 5), (20, 20, 20), 3)
        out.write(frame)
    out.release()

    with open(tmp_path, "rb") as f:
        data = f.read()

    import os
    os.remove(tmp_path)
    return data


def test_inspect_video_bytes_basic():
    video_bytes = _make_dummy_mp4_bytes(num_frames=10)
    result = inspect_video_bytes(video_bytes, filename="test_flight.mp4", sample_fps=2.0)
    assert result.filename == "test_flight.mp4"
    assert result.total_video_frames == 10
    assert result.sampled_frames_count > 0
    assert len(result.panel_damage_map) == 12  # 4x3 grid
    assert "P-0-0" in result.panel_damage_map
    assert result.health_score >= 0 and result.health_score <= 100
    assert result.primary_recommendation is not None


def test_api_video_inspection_route():
    client = TestClient(app)
    video_bytes = _make_dummy_mp4_bytes(num_frames=8)

    response = client.post(
        "/api/inspect/video",
        files={"file": ("flight_test.mp4", io.BytesIO(video_bytes), "video/mp4")},
        data={"confidence": "0.15", "sample_fps": "1.0"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["filename"] == "flight_test.mp4"
    assert "panel_damage_map" in data
    assert "frames" in data
    assert "primary_recommendation" in data


def test_api_video_samples_route():
    client = TestClient(app)
    response = client.get("/api/inspect/video/samples")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "filename" in data[0]
    assert "title" in data[0]
