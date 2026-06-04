# Balanced Keyframe Sampling Design

## Goal

For the current `project1/dinggan` videos, frame extraction should balance efficiency and instance-segmentation quality. The sampler should favor frames that:

- differ significantly in scene composition
- expose small objects at different positions, scales, and occlusion levels
- contain harder object boundaries
- avoid near-duplicate adjacent frames
- keep project creation fast enough for interactive use

The UI exposes only one strategy: **平衡关键帧**. The internal API name is `segmentation_diverse`.

## Pipeline

```text
video
-> lightweight frame-difference / TransNetV2 candidate discovery
-> decode a limited candidate pool
-> DINOv2 embedding novelty score
-> YOLO26-seg object and coarse-mask complexity score on candidates
-> image-statistics fallback features
-> diversity-aware re-ranking
-> top-k frames for manual instance segmentation
```

## Efficiency Choices

- SAM3.1 is not used during frame extraction.
- YOLO26-seg uses the smallest available segmentation model first, currently `yolo26n-seg.pt`.
- DINO and YOLO run in a limited task pool.
- If a model fails or is unavailable, the sampler falls back to frame difference, contrast, edge density, and saturation.
- The candidate pool is bounded before heavier scoring, so long videos do not trigger full-frame model inference.

## Quality Signals

Each candidate frame is scored using:

- scene-change confidence
- DINO embedding novelty
- object count
- object area ratio
- object crowding / overlap
- coarse mask complexity
- edge density
- grayscale contrast
- saturation

The final selection combines base quality score with a diversity bonus against frames that have already been selected.

## Model Roles

```text
TransNetV2     scene and shot candidate discovery
DINOv2         visual diversity and near-duplicate removal
YOLO26-seg     cheap object / mask-complexity estimate
SAM3.1         annotation-stage mask refinement, not sampling
```

## Why Not Tracking

The current task is not to propagate one object through every frame. It is to build a stronger segmentation dataset by selecting frames that are different enough from one another. Tracking can be added later for label propagation, but it is not part of the default sampler.

## Fallback

If model files are missing or GPU resources are busy, the sampler still returns useful frames using:

- frame-difference peaks
- sparse interval anchors
- first / last frame retention
- edge, contrast, and saturation statistics
