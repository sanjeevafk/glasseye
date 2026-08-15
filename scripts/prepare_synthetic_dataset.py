#!/usr/bin/env python3
"""Create the deterministic project-specific training dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.synthetic import build_synthetic_dataset

if __name__ == "__main__":
    build = build_synthetic_dataset()
    print(
        json.dumps(
            {
                "dataset_root": str(build.root),
                "image_count": build.image_count,
                "dataset_hash": build.dataset_hash,
            },
            indent=2,
        )
    )
