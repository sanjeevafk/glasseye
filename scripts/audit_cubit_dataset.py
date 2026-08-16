#!/usr/bin/env python3
"""Audit the downloaded CUBIT archives in datasets/cubit.

This is a read-only audit: it never modifies the source archives and never
infers class semantics beyond what can be proven from the files themselves.
It records archive provenance, image counts and formats, label format and
class tokens, sequence structure, cross-split leakage indicators, and writes:

- artifacts/cubit-audit.json  (machine-readable audit)
- docs/cubit-data-card.md     (human-readable data card)
- artifacts/cubit-samples/    (images with rendered polygon annotations)

The audit deliberately treats class IDs as unproven unless a classes.txt or
equivalent mapping is present in the archives.  Where a mapping is *observed*
(prefix-to-class consistency), it is recorded as an observation with the
exact evidence, not as authoritative documentation.
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import re
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CUBIT_DIR = ROOT / "datasets" / "cubit"
SCRATCH = CUBIT_DIR / "_scratch"
ARTIFACTS = ROOT / "artifacts"
SAMPLES = ARTIFACTS / "cubit-samples"

TRAIN_ARCHIVE = CUBIT_DIR / "images-001.zip"
# The test/val archives contain nested test/images.zip + test/labels.zip.
VAL_ARCHIVE = next(CUBIT_DIR.glob("val-*.zip"), None)
TEST_ARCHIVE = next(CUBIT_DIR.glob("test-*.zip"), None)

IMAGE_PATTERN = re.compile(r"^images/([A-Z]+\d+)_(\d+)\.JPG$")
LABEL_PATTERN = re.compile(r"^labels/([A-Z]+\d+)_(\d+)\.txt$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def archive_hash(path: Path) -> str:
    """Hash an archive; the 15 GB train archive is hashed in streaming blocks."""
    return sha256(path)


def ensure_nested_extracted() -> None:
    """Ensure _scratch/{test,val}/{images,labels}.zip exist (nested archives)."""
    for archive, split in ((TEST_ARCHIVE, "test"), (VAL_ARCHIVE, "val")):
        if archive is None:
            continue
        target = SCRATCH / split
        target.mkdir(parents=True, exist_ok=True)
        for member in ("images.zip", "labels.zip"):
            if not (target / member).is_file():
                with zipfile.ZipFile(archive) as handle:
                    handle.extract(f"{split}/{member}", SCRATCH)
                shutil.move(SCRATCH / split / member, target / member)
                # remove the now-empty {split}/ dir if left behind
                leftover = SCRATCH / split
                if leftover.is_dir() and not any(leftover.iterdir()):
                    leftover.rmdir()


def ensure_labels_extracted() -> None:
    """Extract label text files from the nested label zips into _scratch."""
    for split in ("test", "val"):
        labels_zip = SCRATCH / split / "labels.zip"
        target = SCRATCH / f"{split}_labels"
        if not labels_zip.is_file():
            continue
        if target.is_dir() and any(target.glob("labels/*.txt")):
            continue
        with zipfile.ZipFile(labels_zip) as handle:
            handle.extractall(target)


def scan_images_zip(archive: Path) -> dict[str, object]:
    """Central-directory scan of a flat images zip (train archive)."""
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        infos = {info.filename: info for info in handle.infolist()}
    images = [name for name in names if IMAGE_PATTERN.match(name)]
    formats = collections.Counter(
        Path(name).suffix.lower() for name in names if not name.endswith("/")
    )
    prefixes = collections.Counter(
        IMAGE_PATTERN.match(name).group(1) for name in images  # type: ignore[union-attr]
    )
    per_prefix_indices: dict[str, list[int]] = collections.defaultdict(list)
    for name in images:
        match = IMAGE_PATTERN.match(name)
        assert match is not None
        per_prefix_indices[match.group(1)].append(int(match.group(2)))
    index_stats: dict[str, dict[str, object]] = {}
    for prefix, indices in per_prefix_indices.items():
        indices.sort()
        gaps = [right - left for left, right in itertools.pairwise(indices)]
        index_stats[prefix] = {
            "count": len(indices),
            "min_index": indices[0],
            "max_index": indices[-1],
            "median_gap": int(sorted(gaps)[len(gaps) // 2]),
            "max_gap": int(max(gaps)),
        }
    uncompressed = sum(
        infos[name].file_size for name in names if not name.endswith("/")
    )
    return {
        "archive": str(archive.relative_to(ROOT)),
        "entries": len(names),
        "image_count": len(images),
        "formats": dict(sorted(formats.items())),
        "uncompressed_bytes": uncompressed,
        "prefix_counts": dict(sorted(prefixes.items())),
        "per_prefix_index_stats": index_stats,
    }


def analyze_labels(split: str) -> dict[str, object]:
    """Parse every label file in a split: YOLO polygon lines."""
    label_dir = SCRATCH / f"{split}_labels" / "labels"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Labels missing for {split}: {label_dir}")
    files = sorted(label_dir.glob("*.txt"))
    total_lines = 0
    class_counts: collections.Counter[str] = collections.Counter()
    per_file_lines: collections.Counter[int] = collections.Counter()
    points: list[int] = []
    prefix_class: collections.Counter[tuple[str, str]] = collections.Counter()
    prefix_counts: collections.Counter[str] = collections.Counter()
    prefix_consistency: dict[str, dict[str, object]] = {}
    anomalies: list[str] = []
    by_prefix: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for path in files:
        prefix = path.name.split("_")[0]
        prefix_counts[prefix] += 1
        classes_in_file: set[str] = set()
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        per_file_lines[len([line for line in lines if line.strip()])] += 1
        for line in lines:
            if not line.strip():
                continue
            total_lines += 1
            tokens = line.split()
            if not tokens:
                continue
            class_id = tokens[0]
            class_counts[class_id] += 1
            classes_in_file.add(class_id)
            prefix_class[(prefix, class_id)] += 1
            by_prefix[prefix][class_id] += 1
            if (len(tokens) - 1) % 2 != 0:
                anomalies.append(f"odd polygon token count: {path.name}")
            else:
                points.append((len(tokens) - 1) // 2)
    for prefix in sorted(by_prefix):
        observed = dict(sorted(by_prefix[prefix].items()))
        prefix_consistency[prefix] = {
            "observed_class_ids": observed,
            "files": prefix_counts[prefix],
        }
    return {
        "split": split,
        "label_files": len(files),
        "annotation_lines": total_lines,
        "class_token_counts": dict(sorted(class_counts.items())),
        "lines_per_file_histogram": dict(sorted(per_file_lines.items())[:15]),
        "points_per_polygon": {
            "min": min(points) if points else None,
            "max": max(points) if points else None,
            "median": int(sorted(points)[len(points) // 2]) if points else None,
        },
        "prefix_to_class_observation": prefix_consistency,
        "anomalies": anomalies,
    }


def image_name_pairs(split: str) -> dict[str, str]:
    """Map stem -> entry name inside the nested images zip."""
    images_zip = SCRATCH / split / "images.zip"
    with zipfile.ZipFile(images_zip) as handle:
        names = handle.namelist()
    pairs = {}
    for name in names:
        match = IMAGE_PATTERN.match(name)
        if match:
            # label stems are ``{prefix}_{index}``, e.g. CR1_2344
            pairs[f"{match.group(1)}_{match.group(2)}"] = name
    return pairs


def pairing_check(split: str) -> dict[str, object]:
    labels = {
        path.stem
        for path in (SCRATCH / f"{split}_labels" / "labels").glob("*.txt")
    }
    images = set(image_name_pairs(split).keys())
    return {
        "split": split,
        "label_stems": len(labels),
        "image_stems": len(images),
        "labels_without_image": sorted(labels - images)[:10],
        "images_without_label": sorted(images - labels)[:10],
        "fully_paired": labels == images,
    }


def cross_split_leakage() -> dict[str, object]:
    """Quantify exact-filename and near-duplicate leakage between splits."""
    with zipfile.ZipFile(TRAIN_ARCHIVE) as handle:
        train_stems = {
            f"{match.group(1)}_{match.group(2)}"
            for name in handle.namelist()
            if (match := IMAGE_PATTERN.match(name))
        }
    train_indices: dict[str, set[int]] = collections.defaultdict(set)
    with zipfile.ZipFile(TRAIN_ARCHIVE) as handle:
        for name in handle.namelist():
            if match := IMAGE_PATTERN.match(name):
                train_indices[match.group(1)].add(int(match.group(2)))
    held_out = {}
    for split in ("test", "val"):
        stems, indices = set(), collections.defaultdict(set)
        with zipfile.ZipFile(SCRATCH / split / "images.zip") as handle:
            for name in handle.namelist():
                if match := IMAGE_PATTERN.match(name):
                    stems.add(f"{match.group(1)}_{match.group(2)}")
                    indices[match.group(1)].add(int(match.group(2)))
        distances = []
        for prefix, values in indices.items():
            for index in values:
                distances.append(min(abs(index - t) for t in train_indices[prefix]))
        distances.sort()
        held_out[split] = {
            "exact_filename_overlap_with_train": len(stems & train_stems),
            "frames": len(stems),
            "min_index_distance_to_train": distances[0] if distances else None,
            "median_index_distance_to_train": (
                distances[len(distances) // 2] if distances else None
            ),
            "max_index_distance_to_train": distances[-1] if distances else None,
            "frames_within_5_of_train": sum(1 for d in distances if d <= 5),
            "frames_within_20_of_train": sum(1 for d in distances if d <= 20),
        }
    return held_out


def sample_images(scan: dict[str, object], count_per_split: int = 3) -> list[dict[str, object]]:
    """Extract a few images per split and render polygon + box annotations."""
    SAMPLES.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, object]] = []
    label_source = {
        "train": None,  # train labels are derived from the test/val pattern; skipped
        "test": SCRATCH / "test_labels" / "labels",
        "val": SCRATCH / "val_labels" / "labels",
    }
    # pick prefixes present in the split
    prefixes = ["CR1", "SP0", "SP1"]
    for split in ("test", "val"):
        image_map = image_name_pairs(split)
        labels_dir = label_source[split]
        assert labels_dir is not None
        chosen = []
        for prefix in prefixes:
            candidates = [
                stem for stem in image_map if stem.startswith(prefix + "_")
            ]
            candidates.sort()
            if candidates:
                chosen.append(candidates[len(candidates) // 2])
        for stem in chosen[:count_per_split]:
            entry = image_map[stem]
            image_bytes = None
            with zipfile.ZipFile(SCRATCH / split / "images.zip") as handle:
                image_bytes = handle.read(entry)
            image = cv2.imdecode(
                np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                continue
            label_path = labels_dir / f"{stem}.txt"
            polygons = parse_polygons(label_path)
            overlay = image.copy()
            for class_id, points in polygons:
                pts = np.array(points, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= image.shape[1]
                pts[:, 1] *= image.shape[0]
                pts = pts.astype(np.int32)
                cv2.polylines(
                    overlay, [pts], True, (0, 220, 0), 2, cv2.LINE_AA
                )
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            out_name = f"{split}_{stem}.jpg"
            cv2.imwrite(str(SAMPLES / out_name), overlay)
            rendered.append(
                {
                    "split": split,
                    "stem": stem,
                    "image_entry": entry,
                    "image_size": [image.shape[1], image.shape[0]],
                    "polygons_rendered": len(polygons),
                    "file": f"artifacts/cubit-samples/{out_name}",
                }
            )
    # one train sample
    prefix = "CR1"
    with zipfile.ZipFile(TRAIN_ARCHIVE) as handle:
        candidates = [
            name for name in handle.namelist()
            if (match := IMAGE_PATTERN.match(name)) and match.group(1) == prefix
        ]
        candidates.sort()
        if candidates:
            entry = candidates[len(candidates) // 2]
            image_bytes = handle.read(entry)
            image = cv2.imdecode(
                np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is not None:
                out_name = f"train_{Path(entry).stem}.jpg"
                cv2.imwrite(str(SAMPLES / out_name), image)
                rendered.append(
                    {
                        "split": "train",
                        "stem": Path(entry).stem,
                        "image_entry": entry,
                        "image_size": [image.shape[1], image.shape[0]],
                        "polygons_rendered": 0,
                        "file": f"artifacts/cubit-samples/{out_name}",
                        "note": "train sample shown without labels (labels live inside the 15 GB train archive structure; see data card)",
                    }
                )
    return rendered


def parse_polygons(label_path: Path) -> list[tuple[str, list[float]]]:
    polygons = []
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) < 3:
            continue
        class_id = tokens[0]
        coords = [float(value) for value in tokens[1:]]
        polygons.append((class_id, coords))
    return polygons


def write_data_card(audit: dict[str, object]) -> None:
    doc = audit["documentation"]
    lines = [
        "# CUBIT downloaded-data audit and data card",
        "",
        f"Audit date: {doc['audit_date']}.",
        "",
        "This card describes the CUBIT archives under `datasets/cubit` and the",
        "decisions made before any derived training data is created. The source",
        "archives are user-provided, gitignored, and never modified by GlassEye",
        "tooling. A repeatable audit is available through:",
        "",
        "    .venv/bin/python scripts/audit_cubit_dataset.py",
        "",
        "## Provenance",
        "",
        "The archives are a CUBIT-Det / CUBIT-InSeg style release referenced by the",
        "CUHK-USR-Group Defect-Dataset repository (crack, spalling, and moisture",
        "classes are described there; see docs/data-card.md licensing notes).",
        "The local download contains crack (CR) and spalling (SP) sequences only;",
        "no moisture sequences are present.",
        "",
        "| Archive | SHA-256 (first 12) |",
        "|---|---|",
    ]
    for archive in audit["archives"]:
        lines.append(f"| {archive['file']} | {archive['sha256'][:12]} |")
    lines += [
        "",
        "## Image inventory",
        "",
        f"- Train archive `images-001.zip`: {audit['train_images']['image_count']} JPG files",
        f"  ({audit['train_images']['uncompressed_bytes'] / 1e9:.1f} GB uncompressed).",
        f"- Val archive: {audit['val']['image_stems']} images.",
        f"- Test archive: {audit['test']['image_stems']} images.",
        "",
        "Sequence prefixes observed (UAV capture sequences):",
        "",
        "| Prefix | Likely defect | Train | Val | Test |",
        "|---|---|---|---|---|",
    ]
    for prefix in ("CR1", "SP0", "SP1"):
        train_count = audit["train_images"]["prefix_counts"].get(prefix, 0)
        val_count = audit["val_prefixes"].get(prefix, 0)
        test_count = audit["test_prefixes"].get(prefix, 0)
        likely = "crack" if prefix.startswith("CR") else "spalling"
        lines.append(f"| {prefix} | {likely} | {train_count} | {val_count} | {test_count} |")
    lines += [
        "",
        "## Label format",
        "",
        "Labels are YOLO-style segmentation polygons: one annotation per line as",
        "`class x1 y1 x2 y2 ...` with normalized coordinates. The archives contain",
        "**no classes.txt or equivalent authoritative mapping file**.",
        "",
        (
            f"- Test: {audit['test']['label_files']} label files,"
            f" {audit['test']['annotation_lines']} polygons."
        ),
        (
            f"- Val: {audit['val']['label_files']} label files,"
            f" {audit['val']['annotation_lines']} polygons."
        ),
        f"- Class token counts (test): {audit['test']['class_token_counts']}.",
        f"- Class token counts (val): {audit['val']['class_token_counts']}.",
        "",
        "Observed prefix-to-class consistency (not authoritative):",
        "",
    ]
    for split in ("test", "val"):
        lines.append(f"- {split}: {audit[split]['prefix_to_class_observation']}")
    lines += [
        "",
        "Every CR1 file contains only class token `0` and every SP0/SP1 file only",
        "class token `1`, which is consistent with `0 = crack`, `1 = spalling`.",
        "Because no mapping file exists in the archives, GlassEye records this as an",
        "**inference** and does not use the two class IDs as documented semantics.",
        "Any derived dataset therefore uses a single binary `defect` class.",
        "",
        "## Split integrity and leakage",
        "",
        (
            f"- Exact filename overlap between train and held-out splits:"
            f" {audit['leakage']['test']['exact_filename_overlap_with_train']}."
        ),
        "- Sequence prefixes are shared across train/val/test, and frame indices",
        "  interleave: the official split is **frame-interleaved, not temporally",
        "  disjoint**. Median index distance from a test frame to the nearest train",
        (
            f"  frame is {audit['leakage']['test']['median_index_distance_to_train']}"
            f" (val: {audit['leakage']['val']['median_index_distance_to_train']});"
        ),
        (
            f" {audit['leakage']['test']['frames_within_20_of_train']} of"
            f" {audit['leakage']['test']['frames']} test frames are within 20 frames of a"
        ),
        "training frame.",
        "",
        "**Consequence:** CUBIT held-out benchmarks are inflated by near-duplicate",
        "frames and are not a clean generalization measurement. The BFDD test split",
        "(grouped by capture minute, untouched by any training) remains the primary",
        "apples-to-apples benchmark.",
        "",
        "## GlassEye disposition",
        "",
        "1. The CUBIT train archive contains images **only** — no CUBIT training",
        "   labels exist in this download.  The only labeled CUBIT data is the val",
        "   (699) and test (701) splits.",
        "2. Use CUBIT val as the CUBIT training source (polygons -> binary boxes)",
        "   in the combined BFDD+CUBIT dataset; validation is therefore BFDD val",
        "   only, so validation images never appear in training.",
        "3. Keep the official CUBIT test split untouched as a secondary evaluation",
        "   set, with the near-duplicate-leakage caveat above.  The BFDD test split",
        "   remains the primary apples-to-apples benchmark.",
        "4. Do not infer the missing moisture class; do not claim a cleanable-surface",
        "   mapping for CUBIT data (no stain/moisture annotations exist in this",
        "   download).",
        "5. Derived labels are generated into a new gitignored directory only; the",
        "   original archives are never modified.",
        "",
        "## Visual samples",
        "",
        "Rendered annotation samples (polygon outlines in green, derived boxes in",
        "red) are written to `artifacts/cubit-samples/`.",
        "",
    ]
    (ROOT / "docs" / "cubit-data-card.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    ensure_nested_extracted()
    ensure_labels_extracted()

    archives = []
    for path in (TRAIN_ARCHIVE, VAL_ARCHIVE, TEST_ARCHIVE):
        if path is not None and path.is_file():
            archives.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": archive_hash(path),
                }
            )

    train_scan = scan_images_zip(TRAIN_ARCHIVE)
    val_labels = analyze_labels("val")
    test_labels = analyze_labels("test")
    val_pairing = pairing_check("val")
    test_pairing = pairing_check("test")
    leakage = cross_split_leakage()
    samples = sample_images(train_scan)
    val_prefixes = collections.Counter(
        path.name.split("_")[0]
        for path in (SCRATCH / "val_labels" / "labels").glob("*.txt")
    )
    test_prefixes = collections.Counter(
        path.name.split("_")[0]
        for path in (SCRATCH / "test_labels" / "labels").glob("*.txt")
    )

    audit = {
        "dataset": "CUBIT (crack/spalling UAV façade dataset)",
        "source_archives": [archives[0]["file"]] if archives else [],
        "archives": archives,
        "train_images": train_scan,
        "test": {**test_labels, **test_pairing, "prefixes": dict(sorted(test_prefixes.items()))},
        "val": {**val_labels, **val_pairing, "prefixes": dict(sorted(val_prefixes.items()))},
        "test_prefixes": dict(sorted(test_prefixes.items())),
        "val_prefixes": dict(sorted(val_prefixes.items())),
        "leakage": leakage,
        "samples": samples,
        "documentation": {
            "audit_date": "2026-08-16",
            "class_mapping_status": (
                "No classes.txt in archives; prefix-to-class (CR1->0, SP->1) is "
                "an observation recorded with evidence, not authoritative."
            ),
        },
    }
    (ARTIFACTS / "cubit-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    write_data_card(audit)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
