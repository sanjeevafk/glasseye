#!/usr/bin/env python3
"""Validate the GlassEye YOLO dataset contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.dataset import validate_dataset
from app.paths import artifacts_root, data_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()
    report = validate_dataset(args.data_root, verify_checksums=not args.skip_checksums)
    output = artifacts_root() / "dataset-validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report.as_dict(), indent=2))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
