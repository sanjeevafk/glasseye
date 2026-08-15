"""Dataset-contract validation independent of Ultralytics."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .schemas import DefectClass

EXPECTED_NAMES = {
    0: DefectClass.CLEANABLE.value,
    1: DefectClass.STRUCTURAL.value,
}


@dataclass
class DatasetValidation:
    root: Path
    image_count: int = 0
    label_count: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "valid": self.valid,
            "image_count": self.image_count,
            "label_count": self.label_count,
            "split_counts": self.split_counts,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _parse_names(data: dict[str, object]) -> dict[int, str]:
    names = data.get("names", {})
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {}


def validate_dataset(root: Path, *, verify_checksums: bool = True) -> DatasetValidation:
    result = DatasetValidation(root=root)
    data_path = root / "data.yaml"
    if not data_path.exists():
        result.errors.append("Missing data.yaml.")
        return result
    data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    if _parse_names(data) != EXPECTED_NAMES:
        result.errors.append(f"data.yaml names must equal {EXPECTED_NAMES}; got {_parse_names(data)}.")

    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            result.errors.append(f"Missing image or label directory for {split}.")
            continue
        images = sorted(path for path in image_dir.glob("*.jpg"))
        result.split_counts[split] = len(images)
        result.image_count += len(images)
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                result.errors.append(f"{split}/{image_path.name} is missing its label file.")
                continue
            result.label_count += 1
            for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                parts = raw_line.split()
                if len(parts) != 5:
                    result.errors.append(f"{label_path}:{line_number} must contain five YOLO fields.")
                    continue
                try:
                    class_id = int(parts[0])
                    center_x, center_y, width, height = (float(value) for value in parts[1:])
                except ValueError:
                    result.errors.append(f"{label_path}:{line_number} contains non-numeric label data.")
                    continue
                if class_id not in EXPECTED_NAMES:
                    result.errors.append(f"{label_path}:{line_number} has invalid class ID {class_id}.")
                if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    result.errors.append(f"{label_path}:{line_number} has out-of-range normalized coordinates.")
                if center_x - width / 2 < 0 or center_x + width / 2 > 1:
                    result.errors.append(f"{label_path}:{line_number} extends outside the image horizontally.")
                if center_y - height / 2 < 0 or center_y + height / 2 > 1:
                    result.errors.append(f"{label_path}:{line_number} extends outside the image vertically.")

    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        result.errors.append("Missing manifest.csv.")
        return result
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != result.image_count:
        result.errors.append(f"Manifest has {len(rows)} rows but the dataset has {result.image_count} images.")
    groups: dict[str, set[str]] = {}
    for row in rows:
        source_group = row.get("source_group", "")
        split = row.get("split", "")
        if split not in {"train", "val", "test"}:
            result.errors.append(f"Manifest row {row.get('image_id', '<unknown>')} has invalid split {split!r}.")
            continue
        groups.setdefault(source_group, set()).add(split)
        image_path = root / "images" / split / f"{row.get('image_id', '')}.jpg"
        if not image_path.is_file():
            result.errors.append(f"Manifest image is missing: {image_path}.")
        elif verify_checksums and row.get("checksum") != _sha256(image_path):
            result.errors.append(f"Manifest checksum mismatch: {image_path}.")
    leaked_groups = sorted(group for group, splits in groups.items() if len(splits) > 1)
    if leaked_groups:
        result.errors.append(f"Source-group leakage across splits: {', '.join(leaked_groups)}.")
    return result
