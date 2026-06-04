from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from autolabel_app.video_sampling import plan_frame_samples


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
PROJECT_INDEX_DIR = ROOT_DIR / "data" / "projects"
DOWNLOAD_DIR = ROOT_DIR / "data" / "downloads"
MODEL_DIR = ROOT_DIR / "models"

app = FastAPI(title="Autolabel Web Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AnnotationIn(BaseModel):
    id: str
    category: str = "object"
    shape_type: Literal["rectangle", "polygon", "circle"] = "rectangle"
    bbox: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0], min_length=4, max_length=4)
    points: list[list[float]] = Field(default_factory=list)
    segmentation: list[list[float]] = Field(default_factory=list)
    score: float = 1.0
    source: str = "manual"


class AnnotationUpdate(BaseModel):
    annotations: list[AnnotationIn]


class LabelCatalogUpdate(BaseModel):
    labels: list[str]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/projects")
async def create_project(
    video: UploadFile | None = File(None),
    project_name: str = Form(""),
    server_video_path: str = Form(""),
    frame_sampler: Literal["interval", "keyframes", "hybrid", "segmentation_diverse"] = Form("segmentation_diverse"),
    interval_sec: float = Form(2.0),
    max_frames: int = Form(12),
):
    interval_sec = max(0.2, min(float(interval_sec), 60.0))
    max_frames = max(1, min(int(max_frames), 120))
    source_path, source_name, project_dir, storage_mode = prepare_project_source(video, project_name, server_video_path)
    project_id = project_dir.name
    frames_dir = project_dir / "frames"
    annotations_dir = project_dir / "annotations"
    frames_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    frames = extract_frames(
        source_path=source_path,
        frames_dir=frames_dir,
        frame_sampler=frame_sampler,
        interval_sec=interval_sec,
        max_frames=max_frames,
    )

    manifest = {
        "id": project_id,
        "name": project_id,
        "video_name": source_name,
        "video_path": str(source_path),
        "project_dir": str(project_dir),
        "storage_mode": storage_mode,
        "task_type": "segmentation",
        "frame_sampler": frame_sampler,
        "mode": "manual_only",
        "label_catalog": [],
        "interval_sec": interval_sec,
        "frame_count": len(frames),
        "frames": frames,
    }
    register_project(project_id, project_dir)
    save_manifest(project_id, manifest)
    write_frame_annotation_files(project_id, manifest)
    return manifest


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    return load_manifest(project_id)


@app.get("/api/projects/{project_id}/frames/{frame_id}/image")
def get_frame_image(project_id: str, frame_id: str):
    project = load_manifest(project_id)
    frame = next((item for item in project["frames"] if item["id"] == frame_id), None)
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame not found")
    image_path = project_dir(project_id) / "frames" / frame["file_name"]
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Frame image not found")
    return FileResponse(image_path)


@app.put("/api/projects/{project_id}/frames/{frame_id}/annotations")
def update_annotations(project_id: str, frame_id: str, payload: AnnotationUpdate):
    project = load_manifest(project_id)
    for frame in project["frames"]:
        if frame["id"] == frame_id:
            frame["annotations"] = [
                normalize_annotation(item.model_dump(), frame["width"], frame["height"])
                for item in payload.annotations
            ]
            save_manifest(project_id, project)
            write_single_frame_annotation_file(project_id, project, frame)
            return frame
    raise HTTPException(status_code=404, detail="Frame not found")


@app.put("/api/projects/{project_id}/labels")
def update_label_catalog(project_id: str, payload: LabelCatalogUpdate):
    project = load_manifest(project_id)
    project["label_catalog"] = normalize_label_catalog(payload.labels, project)
    save_manifest(project_id, project)
    return {"labels": project["label_catalog"]}


@app.get("/api/projects/{project_id}/export/coco")
def export_coco(project_id: str):
    project = load_manifest(project_id)
    coco = build_coco(project)
    return JSONResponse(coco, headers={"Content-Disposition": 'attachment; filename="autolabel-coco.json"'})


@app.get("/api/projects/{project_id}/export/bundle")
def export_bundle(project_id: str):
    project = load_manifest(project_id)
    bundle_path = build_export_bundle(project_id, project)
    return FileResponse(bundle_path, filename=bundle_path.name, media_type="application/zip")


def extract_frames(
    source_path: Path,
    frames_dir: Path,
    frame_sampler: str,
    interval_sec: float,
    max_frames: int,
) -> list[dict]:
    sample_plan = plan_frame_samples(source_path, frame_sampler, interval_sec, max_frames, model_root=MODEL_DIR)
    if not sample_plan:
        raise HTTPException(status_code=400, detail="No frame samples planned")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open video")

    sampled_frames: list[dict[str, Any]] = []
    for index, sample in enumerate(sample_plan):
        capture.set(cv2.CAP_PROP_POS_FRAMES, sample["frame_index"])
        ok, image = capture.read()
        if not ok:
            continue

        height, width = image.shape[:2]
        frame_id = f"frame_{index + 1:06d}"
        file_name = f"{frame_id}.jpg"
        image_path = frames_dir / file_name
        cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        sampled_frames.append(
            {
                "id": frame_id,
                "index": index,
                "source_frame_index": int(sample["frame_index"]),
                "timestamp_sec": sample["timestamp_sec"],
                "file_name": file_name,
                "width": width,
                "height": height,
                "image_stats": summarize_image(image),
                "annotations": [],
            }
        )

    capture.release()
    if not sampled_frames:
        raise HTTPException(status_code=400, detail="No frames could be extracted")

    if frame_sampler == "segmentation_diverse":
        sampled_frames = rerank_diverse_frames(sampled_frames, max_frames)
    sampled_frames = normalize_selected_frames(sampled_frames[:max_frames], frames_dir)
    return sampled_frames


def summarize_image(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return {
        "brightness": round(float(np.mean(gray)) / 255.0, 4),
        "contrast": round(float(np.std(gray)) / 255.0, 4),
        "edge_density": round(float(np.mean(edges > 0)), 4),
        "saturation": round(float(np.mean(hsv[:, :, 1])) / 255.0, 4),
    }


def rerank_diverse_frames(frames: list[dict[str, Any]], max_frames: int) -> list[dict[str, Any]]:
    if len(frames) <= max_frames:
        return frames

    feature_rows = np.array([frame_feature_vector(frame) for frame in frames], dtype=np.float32)
    if feature_rows.size == 0:
        return frames[:max_frames]

    minima = feature_rows.min(axis=0)
    maxima = feature_rows.max(axis=0)
    denom = np.where(maxima - minima < 1e-6, 1.0, maxima - minima)
    norm = (feature_rows - minima) / denom

    selected: list[int] = []
    base_scores = norm[:, 0] * 0.28 + norm[:, 1] * 0.22 + norm[:, 2] * 0.2 + norm[:, 3] * 0.18 + norm[:, 4] * 0.12
    first_idx = int(np.argmax(base_scores))
    selected.append(first_idx)

    while len(selected) < min(max_frames, len(frames)):
        best_idx = None
        best_score = -1.0
        for idx in range(len(frames)):
            if idx in selected:
                continue
            distances = [float(np.linalg.norm(norm[idx] - norm[chosen])) for chosen in selected]
            diversity_bonus = min(distances) if distances else 0.0
            score = float(base_scores[idx] + 0.35 * diversity_bonus)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)

    chosen = sorted((frames[idx] for idx in selected), key=lambda item: item["timestamp_sec"])
    return chosen


def frame_feature_vector(frame: dict[str, Any]) -> list[float]:
    stats = frame.get("image_stats", {})
    annotations = frame.get("annotations", [])
    width = float(frame.get("width") or 1.0)
    height = float(frame.get("height") or 1.0)
    image_area = max(width * height, 1.0)

    object_count = float(len(annotations))
    object_area = 0.0
    crowding = 0.0
    mask_complexity = 0.0
    for ann in annotations:
        bbox = ann.get("bbox") or [0, 0, 0, 0]
        object_area += float(bbox[2]) * float(bbox[3]) / image_area
        seg = ann.get("segmentation") or []
        if seg and seg[0]:
            mask_complexity += len(seg[0]) / 2.0
    if object_count > 1:
        for idx, ann_a in enumerate(annotations):
            for ann_b in annotations[idx + 1 :]:
                crowding += iou(ann_a.get("bbox", [0, 0, 0, 0]), ann_b.get("bbox", [0, 0, 0, 0]))

    return [
        object_count,
        object_area,
        crowding,
        mask_complexity / max(object_count, 1.0),
        float(stats.get("edge_density", 0.0)),
        float(stats.get("contrast", 0.0)),
        float(stats.get("saturation", 0.0)),
    ]


def normalize_selected_frames(frames: list[dict[str, Any]], frames_dir: Path) -> list[dict[str, Any]]:
    keep_files = set()
    normalized: list[dict[str, Any]] = []
    for index, frame in enumerate(frames, start=1):
        new_id = f"frame_{index:06d}"
        new_file_name = f"{new_id}.jpg"
        old_file = frames_dir / frame["file_name"]
        new_file = frames_dir / new_file_name
        if old_file.exists() and old_file != new_file:
            old_file.rename(new_file)
        keep_files.add(new_file_name)
        normalized.append(
            {
                **frame,
                "id": new_id,
                "index": index - 1,
                "file_name": new_file_name,
            }
        )

    for image_path in frames_dir.glob("*.jpg"):
        if image_path.name not in keep_files:
            image_path.unlink(missing_ok=True)
    return normalized


def detect_and_segment_cv(image: np.ndarray, labels: list[str], task_type: str) -> list[dict]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(gray, 60, 160)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    image_area = width * height
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.004 or area > image_area * 0.65:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 16 or h < 16:
            continue
        candidates.append((area, contour, [float(x), float(y), float(w), float(h)]))

    candidates.sort(key=lambda item: item[0], reverse=True)
    candidates = non_overlapping(candidates[:12])[:5]

    if not candidates:
        fallback = [
            0.0,
            np.array(
                [
                    [[int(width * 0.25), int(height * 0.25)]],
                    [[int(width * 0.72), int(height * 0.25)]],
                    [[int(width * 0.72), int(height * 0.72)]],
                    [[int(width * 0.25), int(height * 0.72)]],
                ],
                dtype=np.int32,
            ),
            [width * 0.25, height * 0.25, width * 0.47, height * 0.47],
        ]
        candidates = [fallback]

    annotations = []
    for index, (area, contour, bbox) in enumerate(candidates):
        segmentation = []
        if task_type in {"segmentation", "both"}:
            epsilon = max(2.0, 0.01 * cv2.arcLength(contour, True))
            polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(polygon) >= 3:
                segmentation = [[float(coord) for point in polygon for coord in point]]
            else:
                segmentation = [bbox_to_polygon(bbox)]

        annotations.append(
            {
                "id": f"pred_{uuid.uuid4().hex[:10]}",
                "category": labels[index % len(labels)],
                "bbox": [round(value, 2) for value in bbox],
                "segmentation": segmentation,
                "score": round(0.72 - index * 0.06, 2),
                "source": "cv_prediction",
            }
        )
    return annotations


def non_overlapping(candidates: list[tuple[float, np.ndarray, list[float]]]) -> list[tuple[float, np.ndarray, list[float]]]:
    kept = []
    for candidate in candidates:
        if all(iou(candidate[2], item[2]) < 0.55 for item in kept):
            kept.append(candidate)
    return kept


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def normalize_annotation(annotation: dict, width: int, height: int) -> dict:
    shape_type = annotation.get("shape_type") or "rectangle"
    points = [[float(p[0]), float(p[1])] for p in annotation.get("points") or [] if len(p) >= 2]

    if shape_type == "polygon" and len(points) >= 3:
        flat = [coord for point in points for coord in point]
        bbox = polygon_bbox(points)
        annotation["points"] = points
        annotation["bbox"] = [round(value, 2) for value in clamp_bbox(bbox, width, height)]
        annotation["segmentation"] = [[round(float(value), 2) for value in flat]]
        return annotation

    if shape_type == "circle" and len(points) >= 2:
        center = points[0]
        edge = points[1]
        radius = float(np.hypot(edge[0] - center[0], edge[1] - center[1]))
        bbox = [center[0] - radius, center[1] - radius, radius * 2.0, radius * 2.0]
        bbox = clamp_bbox(bbox, width, height)
        annotation["points"] = points
        annotation["bbox"] = [round(value, 2) for value in bbox]
        annotation["segmentation"] = [circle_to_polygon(points[0], radius)]
        return annotation

    if len(points) >= 2:
        bbox = points_to_bbox(points[0], points[1])
        annotation["points"] = rect_points_from_bbox(bbox)
    else:
        bbox = annotation.get("bbox") or [0, 0, 1, 1]
        annotation["points"] = rect_points_from_bbox(bbox)

    bbox = clamp_bbox(bbox, width, height)
    annotation["bbox"] = [round(value, 2) for value in bbox]
    annotation["segmentation"] = [bbox_to_polygon(bbox)]
    return annotation


def clamp_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = [float(value) for value in bbox]
    w = max(1.0, min(w, float(width)))
    h = max(1.0, min(h, float(height)))
    x = max(0.0, min(x, float(width) - w))
    y = max(0.0, min(y, float(height) - h))
    return [x, y, w, h]


def bbox_to_polygon(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def rect_points_from_bbox(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y + h]]


def points_to_bbox(a: list[float], b: list[float]) -> list[float]:
    x1, y1 = a
    x2, y2 = b
    return [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]


def polygon_bbox(points: list[list[float]]) -> list[float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min = min(xs)
    y_min = min(ys)
    return [x_min, y_min, max(xs) - x_min, max(ys) - y_min]


def circle_to_polygon(center: list[float], radius: float, sides: int = 24) -> list[float]:
    cx, cy = center
    points: list[float] = []
    for index in range(sides):
        angle = 2.0 * np.pi * index / sides
        points.extend([round(cx + radius * np.cos(angle), 2), round(cy + radius * np.sin(angle), 2)])
    return points


def build_coco(project: dict) -> dict:
    category_names = []
    for frame in project["frames"]:
        for annotation in frame["annotations"]:
            category = annotation.get("category") or "object"
            if category not in category_names:
                category_names.append(category)

    categories = [
        {"id": index + 1, "name": name, "supercategory": "object"}
        for index, name in enumerate(category_names or ["object"])
    ]
    category_ids = {item["name"]: item["id"] for item in categories}
    annotation_id = 1
    annotations = []

    for image_id, frame in enumerate(project["frames"], start=1):
        for item in frame["annotations"]:
            bbox = [round(float(value), 2) for value in item["bbox"]]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[item.get("category") or "object"],
                    "bbox": bbox,
                    "segmentation": item.get("segmentation") or [bbox_to_polygon(bbox)],
                    "area": round(bbox[2] * bbox[3], 2),
                    "iscrowd": 0,
                    "score": item.get("score", 1.0),
                    "source": item.get("source", "manual"),
                }
            )
            annotation_id += 1

    return {
        "info": {
            "description": "Autolabel web demo export",
            "version": "0.2.0",
        },
        "images": [
            {
                "id": index + 1,
                "file_name": frame["file_name"],
                "width": frame["width"],
                "height": frame["height"],
                "timestamp_sec": frame["timestamp_sec"],
                "source_frame_index": frame.get("source_frame_index", index),
            }
            for index, frame in enumerate(project["frames"])
        ],
        "annotations": annotations,
        "categories": categories,
    }


def prepare_project_source(video: UploadFile | None, project_name: str, server_video_path: str) -> tuple[Path, str, Path, str]:
    server_video_path = (server_video_path or "").strip()

    if server_video_path:
        source_path = Path(server_video_path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise HTTPException(status_code=400, detail="Server video path not found")
        project_id = allocate_project_id(project_name or source_path.stem, source_path.parent / "_autolabel")
        project_dir = source_path.parent / "_autolabel" / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        return source_path, source_path.name, project_dir, "server_path"

    if video is None or not video.filename:
        raise HTTPException(status_code=400, detail="Please upload a video or provide server_video_path")

    project_id = allocate_project_id(project_name or Path(video.filename).stem, PROJECT_INDEX_DIR)
    project_dir = PROJECT_INDEX_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename).suffix or ".mp4"
    source_path = project_dir / f"source{suffix}"
    with source_path.open("wb") as output:
        shutil.copyfileobj(video.file, output)
    return source_path, video.filename, project_dir, "uploaded_copy"


def allocate_project_id(name: str, base_dir: Path) -> str:
    slug = slugify(name) or "project"
    candidate = slug
    counter = 2
    while (base_dir / candidate).exists() or project_registry_path(candidate).exists():
        candidate = f"{slug}-{counter}"
        counter += 1
    return candidate


def slugify(value: str) -> str:
    value = re.sub(r"\s+", "-", value.strip().lower())
    value = re.sub(r"[^0-9a-zA-Z\-_一-龥]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-_")
    return value[:64]


def project_dir(project_id: str) -> Path:
    registry_path = project_registry_path(project_id)
    if registry_path.exists():
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        return Path(payload["project_dir"])
    direct_path = PROJECT_INDEX_DIR / project_id
    if direct_path.exists():
        return direct_path
    raise HTTPException(status_code=404, detail="Project not found")


def manifest_path(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"


def project_registry_path(project_id: str) -> Path:
    return PROJECT_INDEX_DIR / f"{project_id}.json"


def load_manifest(project_id: str) -> dict:
    path = manifest_path(project_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(project_id: str, manifest: dict) -> None:
    target = manifest_path(project_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def annotations_dir(project_id: str) -> Path:
    target = project_dir(project_id) / "annotations"
    target.mkdir(parents=True, exist_ok=True)
    return target


def frame_annotation_path(project_id: str, frame_id: str) -> Path:
    return annotations_dir(project_id) / f"{frame_id}.json"


def build_frame_annotation_payload(project: dict, frame: dict) -> dict:
    return {
        "project_id": project["id"],
        "project_name": project.get("name") or project["id"],
        "video_name": project.get("video_name"),
        "video_path": project.get("video_path"),
        "frame_id": frame["id"],
        "file_name": frame["file_name"],
        "source_frame_index": frame.get("source_frame_index"),
        "timestamp_sec": frame.get("timestamp_sec"),
        "width": frame.get("width"),
        "height": frame.get("height"),
        "annotations": frame.get("annotations", []),
    }


def write_single_frame_annotation_file(project_id: str, project: dict, frame: dict) -> None:
    target = frame_annotation_path(project_id, frame["id"])
    payload = build_frame_annotation_payload(project, frame)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_frame_annotation_files(project_id: str, project: dict) -> None:
    keep_files = set()
    for frame in project.get("frames", []):
        keep_files.add(f"{frame['id']}.json")
        write_single_frame_annotation_file(project_id, project, frame)

    for existing in annotations_dir(project_id).glob("*.json"):
        if existing.name not in keep_files:
            existing.unlink(missing_ok=True)


def register_project(project_id: str, actual_project_dir: Path) -> None:
    PROJECT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    registry = {
        "id": project_id,
        "project_dir": str(actual_project_dir.resolve()),
    }
    project_registry_path(project_id).write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def build_export_bundle(project_id: str, project: dict) -> Path:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = DOWNLOAD_DIR / f"{project_id}-annotations.zip"
    coco = build_coco(project)
    actual_project_dir = project_dir(project_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        staging = Path(tmp_dir) / project_id
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
        (staging / "coco.json").write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

        frames_export = staging / "frames"
        frames_export.mkdir(exist_ok=True)
        for frame in project.get("frames", []):
            src = actual_project_dir / "frames" / frame["file_name"]
            if src.exists():
                shutil.copy2(src, frames_export / frame["file_name"])

        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in staging.rglob("*"):
                archive.write(path, arcname=str(path.relative_to(staging.parent)))

    return bundle_path


def normalize_label_catalog(labels: list[str], project: dict | None = None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    for item in labels:
        label = str(item or "").strip()
        if label and label not in seen:
            ordered.append(label)
            seen.add(label)

    if project:
        for frame in project.get("frames", []):
            for annotation in frame.get("annotations", []):
                label = str(annotation.get("category") or "").strip()
                if label and label not in seen:
                    ordered.append(label)
                    seen.add(label)

    return ordered
