from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class CandidateFrame:
    frame_index: int
    timestamp_sec: float
    scene_score: float = 0.0


class AIKeyframePlanner:
    def __init__(self, model_root: Path, device: str | None = None):
        self.model_root = model_root
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dino_dir = model_root / "embedding" / "dino"
        self.seg_model_path = self._pick_first_existing(
            [
                model_root / "segmentation" / "yolo26n-seg.pt",
                model_root / "segmentation" / "yolo26s-seg.pt",
                model_root / "segmentation" / "yolo26m-seg.pt",
            ]
        )
        self.transnet_weights = model_root / "keyframe" / "transnetv2" / "transnetv2-pytorch-weights.pth"
        self.transnet_tf_dir = model_root / "keyframe" / "transnetv2" / "tf_saved_model"
        self.dino_cache_dir = self.dino_dir / "torchhub"

    def plan(self, source_path: Path, fps: float, total_frames: int, interval_sec: float, max_frames: int) -> list[dict]:
        if total_frames <= 0:
            return []

        scenes = self._scene_candidates(source_path, total_frames, fps, max_frames)
        if not scenes:
            return []

        decoded = self._decode_candidates(source_path, scenes)
        if not decoded:
            return []

        with ThreadPoolExecutor(max_workers=2) as executor:
            dino_future = executor.submit(self._dino_branch, [item["image"] for item in decoded])
            yolo_future = executor.submit(self._yolo_branch, [item["image"] for item in decoded])
            dino_result = dino_future.result()
            yolo_result = yolo_future.result()

        embeddings = dino_result["embeddings"]
        dino_ready = dino_result["ready"]
        yolo_scores = yolo_result["scores"]
        yolo_ready = yolo_result["ready"]

        for item, embed, yolo_score in zip(decoded, embeddings, yolo_scores):
            item["embedding"] = embed
            item["yolo_score"] = float(yolo_score)
            item["ai_score"] = self._base_score(item, yolo_ready)

        selected = self._select_diverse(decoded, max_frames, dino_ready)
        selected.sort(key=lambda item: item["frame_index"])
        return [
            {
                "frame_index": int(item["frame_index"]),
                "timestamp_sec": round(float(item["timestamp_sec"]), 3),
            }
            for item in selected
        ]

    def _scene_candidates(self, source_path: Path, total_frames: int, fps: float, max_frames: int) -> list[CandidateFrame]:
        scenes = self._predict_scenes_transnet(source_path)
        if scenes is None:
            scenes = self._predict_scenes_proxy(source_path, total_frames, fps)

        target_candidates = max(max_frames * 4, min(48, total_frames))
        candidates: list[CandidateFrame] = []
        for start, end, scene_score in scenes:
            length = max(end - start + 1, 1)
            pieces = 1 if length < fps * 2 else 2 if length < fps * 5 else 3
            for ratio in np.linspace(0.2, 0.8, num=pieces):
                frame_index = int(round(start + ratio * max(length - 1, 0)))
                frame_index = max(0, min(frame_index, total_frames - 1))
                candidates.append(
                    CandidateFrame(
                        frame_index=frame_index,
                        timestamp_sec=frame_index / fps if fps else 0.0,
                        scene_score=float(scene_score),
                    )
                )

        if not candidates:
            return []

        candidates.append(CandidateFrame(frame_index=0, timestamp_sec=0.0, scene_score=1.0))
        candidates.append(CandidateFrame(frame_index=max(total_frames - 1, 0), timestamp_sec=max(total_frames - 1, 0) / fps if fps else 0.0, scene_score=1.0))

        unique: dict[int, CandidateFrame] = {}
        for candidate in candidates:
            existing = unique.get(candidate.frame_index)
            if existing is None or candidate.scene_score > existing.scene_score:
                unique[candidate.frame_index] = candidate

        ordered = sorted(unique.values(), key=lambda item: (item.scene_score, -item.frame_index), reverse=True)
        return sorted(ordered[:target_candidates], key=lambda item: item.frame_index)

    def _decode_candidates(self, source_path: Path, candidates: list[CandidateFrame]) -> list[dict]:
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            return []

        decoded: list[dict] = []
        for candidate in candidates:
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_index)
            ok, image = capture.read()
            if not ok:
                continue
            decoded.append(
                {
                    "frame_index": candidate.frame_index,
                    "timestamp_sec": candidate.timestamp_sec,
                    "scene_score": candidate.scene_score,
                    "image": image,
                }
            )
        capture.release()
        return decoded

    def _dino_branch(self, images: list[np.ndarray]) -> dict:
        try:
            model = load_dino_model(self.dino_cache_dir, self.device)
            embeddings = infer_dino_embeddings(model, images, self.device)
            return {"ready": True, "embeddings": embeddings}
        except Exception:
            fallback = [fallback_embedding(image) for image in images]
            return {"ready": False, "embeddings": fallback}

    def _yolo_branch(self, images: list[np.ndarray]) -> dict:
        if self.seg_model_path is None:
            return {"ready": False, "scores": [0.0 for _ in images]}

        try:
            model = load_yolo_seg_model(self.seg_model_path)
            results = model.predict(
                source=images,
                device=self.device,
                verbose=False,
                stream=False,
            )
            scores = [score_yolo_result(result, image.shape[1], image.shape[0]) for result, image in zip(results, images)]
            return {"ready": True, "scores": scores}
        except Exception:
            return {"ready": False, "scores": [0.0 for _ in images]}

    def _base_score(self, item: dict, yolo_ready: bool) -> float:
        image = item["image"]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = float(np.std(gray)) / 255.0
        edges = cv2.Canny(gray, 80, 180)
        edge_density = float(np.mean(edges > 0))
        score = 0.38 * float(item["scene_score"]) + 0.24 * contrast + 0.18 * edge_density
        if yolo_ready:
            score += 0.20 * float(item["yolo_score"])
        return float(score)

    def _select_diverse(self, decoded: list[dict], max_frames: int, dino_ready: bool) -> list[dict]:
        if len(decoded) <= max_frames:
            return decoded

        selected: list[dict] = []
        remaining = decoded.copy()
        remaining.sort(key=lambda item: item["ai_score"], reverse=True)
        selected.append(remaining.pop(0))

        while remaining and len(selected) < max_frames:
            best = None
            best_score = -1.0
            for item in remaining:
                diversity = min(
                    embedding_distance(item["embedding"], chosen["embedding"])
                    for chosen in selected
                )
                fused = item["ai_score"] + (0.28 if dino_ready else 0.12) * diversity
                if fused > best_score:
                    best_score = fused
                    best = item
            if best is None:
                break
            selected.append(best)
            remaining.remove(best)

        return selected

    def _predict_scenes_transnet(self, source_path: Path) -> list[tuple[int, int, float]] | None:
        if self.transnet_tf_dir.exists():
            tf_scenes = self._predict_scenes_transnet_tf(source_path)
            if tf_scenes is not None:
                return tf_scenes
        if not self.transnet_weights.exists():
            return None
        try:
            transnet_module = load_transnet_module()
            TransNetV2 = transnet_module.TransNetV2
            model = TransNetV2()
            state_dict = torch.load(self.transnet_weights, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval().to(self.device)

            frames = read_video_rgb(source_path)
            if frames.size == 0:
                return None

            predictions = transnet_predict_frames(model, frames, self.device)
            boundary = predictions["single"]
            scenes = predictions_to_scenes(boundary)
            output: list[tuple[int, int, float]] = []
            for start, end in scenes:
                score = float(np.max(boundary[start : end + 1])) if end >= start else 0.0
                output.append((int(start), int(end), score))
            return output
        except Exception:
            return None

    def _predict_scenes_transnet_tf(self, source_path: Path) -> list[tuple[int, int, float]] | None:
        try:
            import tensorflow as tf  # type: ignore
        except Exception:
            return None

        try:
            model = tf.saved_model.load(str(self.transnet_tf_dir))
            frames = read_video_rgb(source_path)
            if frames.size == 0:
                return None
            predictions = transnet_tf_predict_frames(model, frames)
            boundary = predictions["single"]
            scenes = predictions_to_scenes(boundary)
            output: list[tuple[int, int, float]] = []
            for start, end in scenes:
                score = float(np.max(boundary[start : end + 1])) if end >= start else 0.0
                output.append((int(start), int(end), score))
            return output
        except Exception:
            return None

    def _predict_scenes_proxy(self, source_path: Path, total_frames: int, fps: float) -> list[tuple[int, int, float]]:
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            return [(0, max(total_frames - 1, 0), 1.0)]

        sample_points = min(max(total_frames, 32), 240)
        step = max(1, total_frames // sample_points)
        points: list[tuple[int, float]] = []
        previous = None
        for frame_index in range(0, total_frames, step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                continue
            small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (96, 54))
            if previous is None:
                score = 0.0
            else:
                score = float(np.mean(cv2.absdiff(small, previous)))
            points.append((frame_index, score))
            previous = small
        capture.release()

        if not points:
            return [(0, max(total_frames - 1, 0), 1.0)]

        scores = np.array([score for _, score in points], dtype=np.float32)
        threshold = float(np.percentile(scores, 82)) if len(scores) >= 6 else float(np.mean(scores))
        boundaries = [0]
        for frame_index, score in points:
            if score >= threshold:
                boundaries.append(frame_index)
        boundaries.append(max(total_frames - 1, 0))
        boundaries = sorted(set(boundaries))

        scenes: list[tuple[int, int, float]] = []
        for idx in range(len(boundaries) - 1):
            start = boundaries[idx]
            end = max(boundaries[idx + 1] - 1, start)
            local_scores = [score for frame_index, score in points if start <= frame_index <= end]
            scenes.append((start, end, max(local_scores) / 255.0 if local_scores else 0.0))
        if not scenes:
            scenes = [(0, max(total_frames - 1, 0), 1.0)]
        return scenes

    @staticmethod
    def _pick_first_existing(paths: list[Path]) -> Path | None:
        for path in paths:
            if path.exists():
                return path
        return None


def infer_dino_embeddings(model: torch.nn.Module, images: list[np.ndarray], device: str) -> list[np.ndarray]:
    batch = []
    for image in images:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        batch.append((tensor - mean) / std)
    inputs = torch.stack(batch).to(device)
    with torch.inference_mode():
        features = model(inputs)
        if isinstance(features, dict):
            features = features.get("x_norm_clstoken") or next(iter(features.values()))
        features = torch.nn.functional.normalize(features, dim=-1)
    return [row.detach().cpu().numpy().astype(np.float32) for row in features]


def fallback_embedding(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
    hist = []
    for channel in range(3):
        channel_hist = cv2.calcHist([resized], [channel], None, [16], [0, 256]).flatten()
        hist.append(channel_hist)
    vector = np.concatenate(hist).astype(np.float32)
    norm = np.linalg.norm(vector) or 1.0
    return vector / norm


def score_yolo_result(result, width: int, height: int) -> float:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    image_area = max(float(width * height), 1.0)
    count = 0.0
    area_ratio = 0.0
    complexity = 0.0
    overlap = 0.0
    if boxes is not None and boxes.xyxy is not None:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        count = float(len(xyxy))
        for idx, box in enumerate(xyxy):
            x1, y1, x2, y2 = box[:4]
            area_ratio += max(0.0, (x2 - x1) * (y2 - y1)) / image_area
            for other in xyxy[idx + 1 :]:
                overlap += bbox_iou_xyxy(box, other)
    if masks is not None and getattr(masks, "xy", None) is not None:
        for polygon in masks.xy:
            complexity += float(len(polygon))
    return float(0.4 * min(count / 6.0, 1.0) + 0.28 * min(area_ratio, 1.0) + 0.18 * min(overlap, 1.0) + 0.14 * min(complexity / 200.0, 1.0))


def bbox_iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(0.0, (bx2 - bx1) * (by2 - by1)) - intersection
    return float(intersection / union) if union else 0.0


def embedding_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


@lru_cache(maxsize=1)
def load_yolo_seg_model(path: Path) -> YOLO:
    return YOLO(str(path))


@lru_cache(maxsize=1)
def load_dino_model(cache_dir: Path, device: str):
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_dir))
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
    model.eval().to(device)
    return model


@lru_cache(maxsize=1)
def load_transnet_module():
    module_path = Path(__file__).resolve().parents[1] / "third_party" / "TransNetV2" / "inference-pytorch" / "transnetv2_pytorch.py"
    spec = importlib.util.spec_from_file_location("transnetv2_pytorch_custom", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load TransNetV2 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("transnetv2_pytorch_custom", module)
    spec.loader.exec_module(module)
    return module


def read_video_rgb(source_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        return np.empty((0, 27, 48, 3), dtype=np.uint8)

    frames = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb, (48, 27), interpolation=cv2.INTER_AREA))
    capture.release()
    if not frames:
        return np.empty((0, 27, 48, 3), dtype=np.uint8)
    return np.stack(frames, axis=0).astype(np.uint8)


def transnet_predict_frames(model, frames: np.ndarray, device: str) -> dict[str, np.ndarray]:
    windows = []
    no_padded_frames_start = 25
    no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)
    padded = np.concatenate(
        [frames[0:1]] * no_padded_frames_start + [frames] + [frames[-1:]] * no_padded_frames_end,
        axis=0,
    )
    ptr = 0
    while ptr + 100 <= len(padded):
        windows.append(padded[ptr : ptr + 100][np.newaxis])
        ptr += 50

    singles = []
    with torch.inference_mode():
        for window in windows:
            inputs = torch.from_numpy(window).to(device=device, dtype=torch.uint8)
            one_hot, many_hot = model(inputs)
            singles.append(torch.sigmoid(one_hot)[0, 25:75, 0].detach().cpu().numpy())
    single = np.concatenate(singles, axis=0)[: len(frames)]
    return {"single": single}


def transnet_tf_predict_frames(model, frames: np.ndarray) -> dict[str, np.ndarray]:
    import tensorflow as tf  # type: ignore

    predictions = []
    no_padded_frames_start = 25
    no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)
    padded = np.concatenate(
        [frames[0:1]] * no_padded_frames_start + [frames] + [frames[-1:]] * no_padded_frames_end,
        axis=0,
    )
    ptr = 0
    while ptr + 100 <= len(padded):
        chunk = padded[ptr : ptr + 100][np.newaxis]
        logits, _ = model(tf.cast(chunk, tf.float32))
        predictions.append(tf.sigmoid(logits).numpy()[0, 25:75, 0])
        ptr += 50
    single = np.concatenate(predictions, axis=0)[: len(frames)]
    return {"single": single}


def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    flags = (predictions > threshold).astype(np.uint8)
    scenes = []
    t = -1
    t_prev = 0
    start = 0
    for idx, t in enumerate(flags):
        if t_prev == 1 and t == 0:
            start = idx
        if t_prev == 0 and t == 1 and idx != 0:
            scenes.append([start, idx])
        t_prev = t
    if t == 0:
        scenes.append([start, idx])
    if not scenes:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)
