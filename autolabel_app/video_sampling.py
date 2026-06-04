from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from autolabel_app.ai_keyframe import AIKeyframePlanner


def plan_frame_samples(
    source_path: Path,
    frame_sampler: str,
    interval_sec: float,
    max_frames: int,
    model_root: Path | None = None,
) -> list[dict]:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        return []

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if total_frames else max_frames * interval_sec

    ai_modes = {"keyframes", "hybrid", "segmentation_diverse"}
    if frame_sampler in ai_modes and model_root is not None:
        planner = AIKeyframePlanner(model_root=model_root)
        ai_plan = planner.plan(source_path, fps, total_frames, interval_sec, max_frames)
        if ai_plan:
            capture.release()
            if frame_sampler == "hybrid":
                interval_plan = [
                    {
                        "frame_index": int(index),
                        "timestamp_sec": round(min(index / fps, max(duration_sec - 0.05, 0.0)), 3),
                    }
                    for index in _interval_indices(total_frames, fps, duration_sec, interval_sec, max_frames)
                ]
                merged = {
                    item["frame_index"]: item
                    for item in ai_plan + interval_plan
                }
                return _trim_plan(sorted(merged.values(), key=lambda item: item["frame_index"]), max_frames)
            return ai_plan

    if frame_sampler == "interval":
        frame_indices = _interval_indices(total_frames, fps, duration_sec, interval_sec, max_frames)
    elif frame_sampler in {"keyframes", "hybrid", "segmentation_diverse"}:
        keyframes = _keyframe_indices(capture, total_frames, fps, interval_sec, max_frames)
        if frame_sampler == "keyframes":
            frame_indices = keyframes
        else:
            multiplier = 3 if frame_sampler == "segmentation_diverse" else 1
            interval_indices = _interval_indices(
                total_frames,
                fps,
                duration_sec,
                interval_sec,
                max_frames * multiplier,
            )
            merged = sorted(set(keyframes + interval_indices))
            frame_indices = _trim_indices(merged, max_frames * multiplier)
    else:
        frame_indices = _interval_indices(total_frames, fps, duration_sec, interval_sec, max_frames)

    capture.release()
    return [
        {
            "frame_index": int(frame_index),
            "timestamp_sec": round(min(frame_index / fps, max(duration_sec - 0.05, 0.0)), 3),
        }
        for frame_index in frame_indices
    ]


def _interval_indices(
    total_frames: int,
    fps: float,
    duration_sec: float,
    interval_sec: float,
    max_frames: int,
) -> list[int]:
    if total_frames <= 0:
        return []
    indices: list[int] = []
    for index in range(max_frames):
        timestamp_sec = min(index * interval_sec, max(duration_sec - 0.05, 0.0))
        frame_index = min(int(round(timestamp_sec * fps)), max(total_frames - 1, 0))
        indices.append(frame_index)
        if timestamp_sec >= duration_sec - 0.05:
            break
    return sorted(set(indices))


def _keyframe_indices(
    capture: cv2.VideoCapture,
    total_frames: int,
    fps: float,
    interval_sec: float,
    max_frames: int,
) -> list[int]:
    if total_frames <= 0:
        return []

    sample_points = min(max(total_frames, max_frames), 240)
    sample_step = max(1, total_frames // sample_points)
    min_gap = max(sample_step * 3, int(fps * max(0.5, interval_sec * 0.5)))

    candidates: list[tuple[float, int]] = []
    previous_small: np.ndarray | None = None
    for frame_index in range(0, total_frames, sample_step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        small = cv2.resize(gray, (96, 54))
        if previous_small is None:
            score = 0.0
        else:
            diff = cv2.absdiff(small, previous_small)
            score = float(np.mean(diff))
        candidates.append((score, frame_index))
        previous_small = small

    if not candidates:
        return []

    scores = np.array([score for score, _ in candidates], dtype=np.float32)
    threshold = float(np.percentile(scores, 75)) if len(scores) >= 4 else float(np.mean(scores))
    ranked = sorted(candidates, key=lambda item: item[0], reverse=True)

    selected = [0]
    for score, frame_index in ranked:
        if score < threshold and len(selected) >= max_frames:
            break
        if all(abs(frame_index - existing) >= min_gap for existing in selected):
            selected.append(frame_index)
        if len(selected) >= max_frames:
            break

    selected.append(max(total_frames - 1, 0))
    trimmed = _trim_indices(sorted(set(selected)), max_frames)
    if len(trimmed) < max_frames:
        interval_fill = _interval_indices(total_frames, fps, total_frames / fps if fps else 0.0, interval_sec, max_frames)
        trimmed = _trim_indices(sorted(set(trimmed + interval_fill)), max_frames)
    return trimmed


def _trim_indices(indices: list[int], max_frames: int) -> list[int]:
    if len(indices) <= max_frames:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=max_frames, dtype=int)
    return [indices[pos] for pos in positions]


def _trim_plan(items: list[dict], max_frames: int) -> list[dict]:
    if len(items) <= max_frames:
        return items
    positions = np.linspace(0, len(items) - 1, num=max_frames, dtype=int)
    return [items[pos] for pos in positions]
