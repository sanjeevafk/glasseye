"""Stable paths rooted at the GlassEye checkout."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return repo_root() / "data" / "glasseye_v1"


def models_root() -> Path:
    return repo_root() / "models"


def artifacts_root() -> Path:
    return repo_root() / "artifacts"
