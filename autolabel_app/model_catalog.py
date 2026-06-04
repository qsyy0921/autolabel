from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
CONFIG_PATH = ROOT_DIR / "configs" / "models.yaml"


def resolve_local_path(local_path: str) -> Path | str:
    if local_path == "opencv-contours":
        return local_path
    path = Path(local_path)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


@lru_cache(maxsize=1)
def load_model_catalog() -> list[dict[str, Any]]:
    if not CONFIG_PATH.exists():
        return []
    items = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or []
    catalog: list[dict[str, Any]] = []
    for item in items:
        resolved = resolve_local_path(item["local_path"])
        exists = resolved == "opencv-contours" or Path(resolved).exists()
        payload = {
            **item,
            "resolved_path": str(resolved),
            "local_exists": exists,
        }
        catalog.append(payload)
    return catalog


def list_models() -> list[dict[str, Any]]:
    return [dict(item) for item in load_model_catalog()]


def get_model_by_id(model_id: str) -> dict[str, Any] | None:
    for item in load_model_catalog():
        if item["id"] == model_id:
            return dict(item)
    return None


def default_model_for_backend(model_backend: str) -> dict[str, Any] | None:
    candidates = [item for item in load_model_catalog() if item["backend"] == model_backend]
    for item in candidates:
        if item.get("default"):
            return dict(item)
    return dict(candidates[0]) if candidates else None
