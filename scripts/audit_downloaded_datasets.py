#!/usr/bin/env python3
"""Record a reproducible local audit of the downloaded sources.

Licensing conclusions are based on the primary release pages cited in
docs/data-card.md. This script checks what is actually present locally without
extracting the multi-gigabyte archives into the application tree.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.paths import artifacts_root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def audit_bfdd(archive: Path) -> dict[str, object]:
    if not archive.exists():
        return {"present": False}
    with tarfile.open(archive, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        rgb = [
            member for member in members if member.name.startswith("Dataset_1x/RGB/")
        ]
        labels = [
            member for member in members if member.name.startswith("Dataset_1x/Label/")
        ]
        values: set[int] = set()
        for member in labels:
            stream = tar.extractfile(member)
            assert stream is not None
            with stream:
                image = Image.open(stream)
                values.update(int(value) for value in image.getdata())
    return {
        "present": True,
        "archive_sha256": sha256(archive),
        "rgb_images": len(rgb),
        "label_masks": len(labels),
        "label_values": sorted(values),
        "license": "CC BY 4.0 (primary release page; see docs/data-card.md)",
        "disposition": "eligible_after_label-map_confirmation",
        "notes": [
            "The archive has 838 RGB/mask pairs while the release page states 788 pairs.",
            "Its supplied split files contain frames from shared capture dates; do not use them without a group split.",
            "The numeric mask-to-semantic-class mapping is not supplied in the archive and must be confirmed.",
        ],
    }


def main() -> int:
    datasets = ROOT / "datasets"
    audit = {
        "bfdd": audit_bfdd(datasets / "BFDD Dataset_1x_20260408.tar.gz"),
        "uav2k": {
            "present": (datasets / "UAV2K").is_dir(),
            "archives": sorted(
                path.name for path in (datasets / "UAV2K").glob("*.rar")
            ),
            "license": "not published at release time",
            "disposition": "prototype/evaluation only; do not redistribute or train release artefacts",
        },
        "cubit": {
            "present": (datasets / "Defect-Dataset").is_dir(),
            "license": "no LICENSE file in downloaded repository",
            "disposition": "excluded pending explicit permission",
            "local_contents": "README and sample images only; no CUBIT training corpus was downloaded",
        },
        "bd3": {
            "present": (datasets / "BD3-Dataset").is_dir(),
            "license": "no LICENSE file in downloaded repository",
            "disposition": "excluded pending explicit permission",
            "local_sample_images": len(
                list(
                    (datasets / "BD3-Dataset").glob(
                        "sample images/class_images/**/*.jpg"
                    )
                )
            ),
            "annotation_level": "classification labels, not YOLO bounding boxes",
        },
    }
    output = artifacts_root() / "downloaded-dataset-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
