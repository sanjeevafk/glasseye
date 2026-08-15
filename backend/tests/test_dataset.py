from __future__ import annotations

from app.dataset import validate_dataset
from app.synthetic import build_synthetic_dataset


def test_generated_dataset_is_valid_and_split_by_capture_session(tmp_path):
    build = build_synthetic_dataset(tmp_path / "glasseye_v1")
    report = validate_dataset(build.root)

    assert report.valid, report.errors
    assert report.image_count == 256
    assert report.split_counts == {"train": 192, "val": 32, "test": 32}


def test_validator_rejects_invalid_yolo_label(tmp_path):
    build = build_synthetic_dataset(tmp_path / "glasseye_v1")
    label = next((build.root / "labels" / "train").glob("*.txt"))
    label.write_text("7 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    report = validate_dataset(build.root, verify_checksums=False)

    assert not report.valid
    assert any("invalid class ID 7" in error for error in report.errors)
