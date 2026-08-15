"""Deterministic, labelled facade media for the hackathon demo.

The generated media is deliberately marked as synthetic. It provides known ground
truth for an end-to-end YOLO and dashboard demonstration without misrepresenting
an unlicensed source dataset as a redistributable training corpus.
"""

from __future__ import annotations

import csv
import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

from .paths import artifacts_root, data_root
from .schemas import DefectClass

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 384
CLASS_TO_ID = {
    DefectClass.CLEANABLE.value: 0,
    DefectClass.STRUCTURAL.value: 1,
}
DATASET_SEED = 20260815
DEMO_SEED = 20260815


@dataclass(frozen=True)
class SyntheticInstance:
    class_name: DefectClass
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class DatasetBuild:
    root: Path
    image_count: int
    dataset_hash: str


@dataclass(frozen=True)
class DemoMedia:
    preinspection_video: Path
    reinspection_video: Path
    fps: float
    frame_count: int


def _facade_base(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    y_gradient = np.linspace(0, 20, IMAGE_HEIGHT, dtype=np.uint8)[:, None, None]
    base = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), (173, 178, 176), dtype=np.uint8)
    grain = rng.integers(-8, 9, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.int16)
    array = np.clip(base.astype(np.int16) + grain + y_gradient, 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    for column in range(5):
        x = round(column * IMAGE_WIDTH / 4)
        draw.line((x, 0, x, IMAGE_HEIGHT), fill=(85, 94, 96), width=3)
        draw.line((x + 3, 0, x + 3, IMAGE_HEIGHT), fill=(205, 210, 208), width=1)
    for row in range(4):
        y = round(row * IMAGE_HEIGHT / 3)
        draw.line((0, y, IMAGE_WIDTH, y), fill=(85, 94, 96), width=3)
        draw.line((0, y + 3, IMAGE_WIDTH, y + 3), fill=(205, 210, 208), width=1)
    # Deliberately neutral facade details so the detector learns the defect forms.
    for x in (48, 335, 510):
        draw.rectangle((x, 28, x + 65, 74), fill=(124, 145, 155), outline=(67, 82, 88), width=2)
        draw.line((x + 32, 28, x + 32, 74), fill=(210, 220, 220), width=1)
    return image


def _draw_cleanable(
    image: Image.Image,
    bbox_xyxy: tuple[int, int, int, int],
    seed: int,
) -> None:
    """Draw a visibly stained region bounded by the supplied annotation rectangle."""

    x1, y1, x2, y2 = bbox_xyxy
    rng = random.Random(seed)
    draw = ImageDraw.Draw(image)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    points: list[tuple[int, int]] = []
    for index in range(18):
        angle = index * (2 * np.pi / 18)
        radius_x = (x2 - x1) * (0.42 + rng.random() * 0.10)
        radius_y = (y2 - y1) * (0.42 + rng.random() * 0.12)
        points.append((int(cx + np.cos(angle) * radius_x), int(cy + np.sin(angle) * radius_y)))
    draw.polygon(points, fill=(126, 76, 34), outline=(85, 48, 22))
    for _ in range(30):
        dot_x = rng.randint(x1 + 4, x2 - 4)
        dot_y = rng.randint(y1 + 4, y2 - 4)
        radius = rng.randint(1, 4)
        draw.ellipse(
            (dot_x - radius, dot_y - radius, dot_x + radius, dot_y + radius),
            fill=(168 + rng.randint(0, 24), 111 + rng.randint(0, 22), 53),
        )


def _draw_structural(
    image: Image.Image,
    bbox_xyxy: tuple[int, int, int, int],
    seed: int,
) -> None:
    """Draw a high-contrast branching crack bounded by the supplied rectangle."""

    x1, y1, x2, y2 = bbox_xyxy
    rng = random.Random(seed)
    draw = ImageDraw.Draw(image)
    center_y = (y1 + y2) // 2
    points = [(x1 + 4, center_y)]
    for x in range(x1 + 18, x2 - 4, 18):
        points.append((x, center_y + rng.randint(-(y2 - y1) // 3, (y2 - y1) // 3)))
    points.append((x2 - 4, center_y + rng.randint(-4, 4)))
    draw.line(points, fill=(228, 216, 205), width=7, joint="curve")
    draw.line(points, fill=(71, 36, 29), width=4, joint="curve")
    for index in (2, max(2, len(points) - 3)):
        px, py = points[index]
        branch = [(px, py), (px + rng.randint(-22, 22), py + rng.choice((-1, 1)) * rng.randint(12, 24))]
        draw.line(branch, fill=(228, 216, 205), width=5)
        draw.line(branch, fill=(71, 36, 29), width=3)


def _draw_instances(image: Image.Image, instances: list[SyntheticInstance], seed: int) -> None:
    for index, instance in enumerate(instances):
        if instance.class_name == DefectClass.CLEANABLE:
            _draw_cleanable(image, instance.bbox_xyxy, seed + index * 97)
        else:
            _draw_structural(image, instance.bbox_xyxy, seed + index * 97)


def _random_instances(seed: int) -> list[SyntheticInstance]:
    rng = random.Random(seed)
    count = rng.choices([0, 1, 2], weights=[0.12, 0.54, 0.34], k=1)[0]
    instances: list[SyntheticInstance] = []
    for position in range(count):
        class_name = DefectClass.CLEANABLE if rng.random() < 0.5 else DefectClass.STRUCTURAL
        if class_name == DefectClass.CLEANABLE:
            width, height = rng.randint(78, 128), rng.randint(48, 82)
        else:
            width, height = rng.randint(100, 150), rng.randint(30, 50)
        x1 = rng.randint(12, IMAGE_WIDTH - width - 12)
        y1 = rng.randint(18, IMAGE_HEIGHT - height - 18)
        candidate = (x1, y1, x1 + width, y1 + height)
        if any(_bbox_overlap(candidate, item.bbox_xyxy) > 0.15 for item in instances):
            x1 = max(12, min(IMAGE_WIDTH - width - 12, x1 + 95 if position else x1))
            y1 = max(18, min(IMAGE_HEIGHT - height - 18, y1 + 71 if position else y1))
            candidate = (x1, y1, x1 + width, y1 + height)
        instances.append(SyntheticInstance(class_name=class_name, bbox_xyxy=candidate))
    return instances


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def _yolo_label(instance: SyntheticInstance) -> str:
    x1, y1, x2, y2 = instance.bbox_xyxy
    center_x = ((x1 + x2) / 2) / IMAGE_WIDTH
    center_y = ((y1 + y2) / 2) / IMAGE_HEIGHT
    width = (x2 - x1) / IMAGE_WIDTH
    height = (y2 - y1) / IMAGE_HEIGHT
    return f"{CLASS_TO_ID[instance.class_name.value]} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(1 << 20):
            digest.update(data)
    return digest.hexdigest()


def _dataset_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "images").rglob("*")) + sorted((root / "labels").rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def build_synthetic_dataset(root: Path | None = None, *, force: bool = False) -> DatasetBuild:
    root = root or data_root()
    image_marker = root / "images" / "train"
    if image_marker.exists() and any(image_marker.iterdir()) and not force:
        dataset_hash_path = root / "dataset_hash.txt"
        return DatasetBuild(
            root=root,
            image_count=sum(1 for path in (root / "images").rglob("*.jpg")),
            dataset_hash=dataset_hash_path.read_text(encoding="utf-8").strip()
            if dataset_hash_path.exists()
            else _dataset_hash(root),
        )
    if force:
        for relative in ("images", "labels"):
            target = root / relative
            if target.exists():
                shutil.rmtree(target)

    root.mkdir(parents=True, exist_ok=True)
    split_sizes = {"train": 192, "val": 32, "test": 32}
    rows: list[dict[str, str]] = []
    offset = 0
    for split, amount in split_sizes.items():
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(amount):
            image_id = f"synthetic_{split}_{index:04d}"
            seed = DATASET_SEED + offset + index
            instances = _random_instances(seed)
            image = _facade_base(seed)
            _draw_instances(image, instances, seed)
            image_path = image_dir / f"{image_id}.jpg"
            label_path = label_dir / f"{image_id}.txt"
            image.save(image_path, quality=96, optimize=False, progressive=False)
            label_path.write_text(
                "\n".join(_yolo_label(instance) for instance in instances) + ("\n" if instances else ""),
                encoding="utf-8",
            )
            rows.append(
                {
                    "source": "glasseye_synthetic_facade_v1",
                    "image_id": image_id,
                    "split": split,
                    "source_group": f"{split}-capture-session-{index // 8:02d}",
                    "classes": "|".join(sorted({item.class_name.value for item in instances})) or "negative",
                    "annotation_status": "generator_ground_truth_reviewed",
                    "checksum": _sha256(image_path),
                }
            )
        offset += amount

    data = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {class_id: class_name for class_name, class_id in CLASS_TO_ID.items()},
    }
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dataset_hash = _dataset_hash(root)
    (root / "dataset_hash.txt").write_text(dataset_hash + "\n", encoding="utf-8")
    (root / "README.md").write_text(
        """# GlassEye synthetic facade v1

This deterministic, synthetic detection dataset is the executable hackathon
demo corpus. Every bounding box is generated alongside its rendered defect and
is therefore known ground truth. It is not represented as field-collected
inspection data.

Classes:

- 0: cleanable_surface_issue
- 1: structural_issue

Splits are grouped by synthetic capture session. A session never spans train,
validation, or test. Use scripts/validate_dataset.py before training.
""",
        encoding="utf-8",
    )
    return DatasetBuild(root=root, image_count=len(rows), dataset_hash=dataset_hash)


def _frame_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _write_video(path: Path, frames: list[Image.Image], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create demo video at {path}")
    try:
        for frame in frames:
            writer.write(_frame_to_bgr(frame))
    finally:
        writer.release()


def create_demo_media(output_dir: Path | None = None, *, seed: int = DEMO_SEED) -> DemoMedia:
    output_dir = output_dir or artifacts_root() / "demo" / "media"
    fps = 4.0
    frame_count = 24
    cleanable = (174, 146, 302, 220)  # Panel B2
    structural = (332, 278, 468, 324)  # Panel C3
    pre_frames: list[Image.Image] = []
    post_frames: list[Image.Image] = []
    for frame_index in range(frame_count):
        jitter_x = (frame_index % 3) - 1
        jitter_y = ((frame_index * 2) % 3) - 1
        adjusted_cleanable = tuple(
            value + (jitter_x if index in (0, 2) else jitter_y) for index, value in enumerate(cleanable)
        )
        adjusted_structural = tuple(
            value + (jitter_x if index in (0, 2) else jitter_y) for index, value in enumerate(structural)
        )
        before = _facade_base(seed + frame_index)
        _draw_instances(
            before,
            [
                SyntheticInstance(DefectClass.CLEANABLE, adjusted_cleanable),
                SyntheticInstance(DefectClass.STRUCTURAL, adjusted_structural),
            ],
            seed + frame_index,
        )
        after = _facade_base(seed + frame_index)
        _draw_instances(
            after,
            [SyntheticInstance(DefectClass.STRUCTURAL, adjusted_structural)],
            seed + frame_index,
        )
        pre_frames.append(before)
        post_frames.append(after)
    preinspection = output_dir / "preinspection.mp4"
    reinspection = output_dir / "reinspection.mp4"
    _write_video(preinspection, pre_frames, fps)
    _write_video(reinspection, post_frames, fps)
    return DemoMedia(
        preinspection_video=preinspection,
        reinspection_video=reinspection,
        fps=fps,
        frame_count=frame_count,
    )
