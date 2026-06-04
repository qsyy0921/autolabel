from __future__ import annotations

import os
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
SAM3_REPO = ROOT_DIR / "models" / "sam3" / "repo"
SAM31_PT = ROOT_DIR / "models" / "sam3" / "checkpoints" / "sam3.1" / "sam3.1_multiplex.pt"
DEFAULT_SAM31_CONDA_ENV = "sam3.1"
DEFAULT_CONDA_EXE = "/home/panjunhao/miniconda3/bin/conda"


class Sam31Unavailable(RuntimeError):
    pass


def sam31_available() -> bool:
    return SAM3_REPO.exists() and SAM31_PT.exists()


def sam31_conda_env_available() -> bool:
    conda_exe = Path(os.environ.get("AUTOLABEL_CONDA_EXE", DEFAULT_CONDA_EXE))
    env_name = os.environ.get("AUTOLABEL_SAM31_CONDA_ENV", DEFAULT_SAM31_CONDA_ENV)
    env_path = conda_exe.parent.parent / "envs" / env_name
    return conda_exe.exists() and env_path.exists()


def sam31_conda_python() -> Path:
    conda_exe = Path(os.environ.get("AUTOLABEL_CONDA_EXE", DEFAULT_CONDA_EXE))
    env_name = os.environ.get("AUTOLABEL_SAM31_CONDA_ENV", DEFAULT_SAM31_CONDA_ENV)
    return conda_exe.parent.parent / "envs" / env_name / "bin" / "python"


def list_sam31_devices() -> list[dict[str, Any]]:
    devices = [{"id": "cpu", "label": "CPU", "available": True}]
    try:
        import torch
    except ImportError:
        return devices
    if not torch.cuda.is_available():
        return devices
    for index in range(torch.cuda.device_count()):
        devices.append(
            {
                "id": f"cuda:{index}",
                "label": f"GPU {index}: {torch.cuda.get_device_name(index)}",
                "available": True,
            }
        )
    return devices


def normalize_sam31_device(device: str | None = None) -> str:
    requested = (device or os.environ.get("AUTOLABEL_SAM31_DEVICE", "cuda:0")).strip().lower()
    if requested in {"", "auto"}:
        requested = os.environ.get("AUTOLABEL_SAM31_DEVICE", "cuda:0").strip().lower()
    if requested == "cuda":
        requested = "cuda:0"
    if requested == "cpu":
        return requested

    if not requested.startswith("cuda:"):
        raise Sam31Unavailable(f"Unsupported device '{requested}'. Use cpu, cuda, or cuda:N.")

    try:
        index = int(requested.split(":", 1)[1])
    except ValueError as exc:
        raise Sam31Unavailable(f"Unsupported CUDA device '{requested}'. Use cuda:N.") from exc

    import torch

    if not torch.cuda.is_available():
        raise Sam31Unavailable("CUDA is not available on this server")
    if index < 0 or index >= torch.cuda.device_count():
        raise Sam31Unavailable(f"CUDA device {index} is not available; found {torch.cuda.device_count()} GPU(s)")
    return f"cuda:{index}"


def should_use_sam31_conda_worker() -> bool:
    if os.environ.get("AUTOLABEL_SAM31_IN_WORKER") == "1":
        return False
    return os.environ.get("AUTOLABEL_SAM31_USE_CONDA", "1") != "0" and sam31_conda_env_available()


@lru_cache(maxsize=4)
def get_sam31_processor(device: str) -> Any:
    if not sam31_available():
        raise Sam31Unavailable(f"SAM3.1 repo or checkpoint is missing under {ROOT_DIR / 'models' / 'sam3'}")

    if str(SAM3_REPO) not in sys.path:
        sys.path.insert(0, str(SAM3_REPO))

    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    device = normalize_sam31_device(device)
    if device.startswith("cuda:"):
        torch.cuda.set_device(device)
        model_device = "cuda"
    else:
        model_device = device
    model = build_sam3_image_model(
        checkpoint_path=str(SAM31_PT),
        device=model_device,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    return Sam3Processor(model, device=device, confidence_threshold=float(os.environ.get("AUTOLABEL_SAM31_THRESHOLD", "0.35")))


def sam31_inference_context(device: str):
    import contextlib
    import torch

    if device.startswith("cuda:"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def refine_annotation_with_sam31(
    image_path: Path,
    annotation: dict[str, Any],
    category: str,
    device: str | None = None,
) -> list[dict[str, Any]]:
    if should_use_sam31_conda_worker():
        return run_sam31_worker(
            {
                "action": "refine",
                "image_path": str(image_path),
                "annotation": annotation,
                "category": category,
                "device": device,
            }
        )
    return refine_annotation_with_sam31_local(image_path, annotation, category, device=device)


def refine_annotation_with_sam31_local(
    image_path: Path,
    annotation: dict[str, Any],
    category: str,
    device: str | None = None,
) -> list[dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    normalized_device = normalize_sam31_device(device)
    processor = get_sam31_processor(normalized_device)
    with sam31_inference_context(normalized_device):
        state = processor.set_image(image)
        prompt_box = pixel_bbox_to_normalized_cxcywh(annotation.get("bbox") or [0, 0, width, height], width, height)
        output = processor.add_geometric_prompt(prompt_box, True, state)
    candidates = output_to_annotations(output, category=category, source="sam3.1_refined", max_results=6)
    if not candidates:
        return []
    source_bbox = annotation.get("bbox") or [0, 0, width, height]
    candidates.sort(key=lambda item: (bbox_iou(source_bbox, item["bbox"]), item.get("score", 0.0)), reverse=True)
    return [candidates[0]]


def find_similar_with_sam31(
    image_path: Path,
    prompt: str,
    category: str,
    max_results: int = 12,
    threshold: float | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    if should_use_sam31_conda_worker():
        return run_sam31_worker(
            {
                "action": "find",
                "image_path": str(image_path),
                "prompt": prompt,
                "category": category,
                "max_results": max_results,
                "threshold": threshold,
                "device": device,
            }
        )
    return find_similar_with_sam31_local(
        image_path,
        prompt=prompt,
        category=category,
        max_results=max_results,
        threshold=threshold,
        device=device,
    )


def find_similar_with_sam31_local(
    image_path: Path,
    prompt: str,
    category: str,
    max_results: int = 12,
    threshold: float | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    if not prompt.strip():
        raise ValueError("SAM3.1 prompt is empty")

    image = Image.open(image_path).convert("RGB")
    normalized_device = normalize_sam31_device(device)
    processor = get_sam31_processor(normalized_device)
    old_threshold = processor.confidence_threshold
    if threshold is not None:
        processor.confidence_threshold = float(threshold)
    try:
        with sam31_inference_context(normalized_device):
            state = processor.set_image(image)
            output = processor.set_text_prompt(prompt.strip(), state)
        return output_to_annotations(output, category=category or prompt.strip(), source="sam3.1_suggestion", max_results=max_results)
    finally:
        processor.confidence_threshold = old_threshold


def run_sam31_worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    python_exe = sam31_conda_python()
    if not python_exe.exists():
        raise RuntimeError(f"SAM3.1 conda python is missing: {python_exe}")
    timeout = int(os.environ.get("AUTOLABEL_SAM31_WORKER_TIMEOUT", "420"))
    command = [
        str(python_exe),
        "-m",
        "autolabel_app.sam31_worker",
    ]
    env = os.environ.copy()
    env["AUTOLABEL_SAM31_IN_WORKER"] = "1"
    env["PYTHONPATH"] = f"{ROOT_DIR}:{env.get('PYTHONPATH', '')}".rstrip(":")
    result = subprocess.run(
        command,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=str(ROOT_DIR),
        env=env,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        detail = "\n".join(
            part for part in [result.stderr.strip(), result.stdout.strip()] if part
        ) or "SAM3.1 worker failed"
        raise RuntimeError(f"SAM3.1 conda worker failed: {detail[-2000:]}")
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"SAM3.1 conda worker returned invalid JSON: {result.stdout[-1000:]}") from exc
    if "error" in output:
        raise RuntimeError(output["error"])
    return output.get("annotations", [])


def pixel_bbox_to_normalized_cxcywh(bbox: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = [float(value) for value in bbox[:4]]
    cx = (x + w / 2.0) / max(float(width), 1.0)
    cy = (y + h / 2.0) / max(float(height), 1.0)
    return [
        min(max(cx, 0.0), 1.0),
        min(max(cy, 0.0), 1.0),
        min(max(w / max(float(width), 1.0), 0.001), 1.0),
        min(max(h / max(float(height), 1.0), 0.001), 1.0),
    ]


def output_to_annotations(output: dict[str, Any], category: str, source: str, max_results: int) -> list[dict[str, Any]]:
    masks = detach_to_numpy(output.get("masks"))
    boxes = detach_to_numpy(output.get("boxes"))
    scores = detach_to_numpy(output.get("scores"))

    if masks is None or boxes is None:
        return []

    if masks.ndim == 4:
        masks = masks[:, 0]
    if scores is None:
        scores = np.ones((len(masks),), dtype=np.float32)

    order = np.argsort(-scores)[:max_results]
    annotations: list[dict[str, Any]] = []
    for idx in order:
        mask = masks[int(idx)] > 0
        if mask.sum() < 16:
            continue
        polygon = mask_to_polygon(mask)
        if len(polygon) < 6:
            continue
        box = boxes[int(idx)].tolist()
        bbox = xyxy_to_xywh(box)
        annotations.append(
            {
                "id": "",
                "category": category or "object",
                "shape_type": "polygon",
                "bbox": [round(float(value), 2) for value in bbox],
                "points": [[round(polygon[i], 2), round(polygon[i + 1], 2)] for i in range(0, len(polygon), 2)],
                "segmentation": [[round(float(value), 2) for value in polygon]],
                "score": round(float(scores[int(idx)]), 4),
                "source": source,
            }
        )
    return annotations


def detach_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def mask_to_polygon(mask: np.ndarray) -> list[float]:
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(1.5, 0.006 * cv2.arcLength(contour, True))
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(polygon) < 3:
        return []
    return [float(coord) for point in polygon for coord in point]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in a[:4]]
    bx, by, bw, bh = [float(v) for v in b[:4]]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
