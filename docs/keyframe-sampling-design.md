# Keyframe Sampling Design

## Goal

For object segmentation projects, frame extraction should favor frames that:

- differ significantly in scene composition
- expose different object scales, counts, and overlaps
- contain harder segmentation boundaries
- avoid near-duplicate adjacent frames

## Modes Exposed By The API

- `interval`
  - evenly samples frames by time interval
- `keyframes`
  - prefers large visual changes
- `hybrid`
  - mixes interval and visual-change candidates
- `segmentation_diverse`
  - starts from a larger hybrid candidate pool, runs coarse prediction, then re-ranks for segmentation diversity

## Segmentation-Diverse Pipeline

```text
video
-> TransNetV2 first-pass shot / transition candidates
-> decode candidate frames
-> DINOv2 embedding novelty score
-> YOLO26-seg object and coarse-mask complexity score
-> image statistics fallback features
-> diversity-aware re-ranking
-> top-k frames for manual segmentation annotation
```

## Candidate Planning

The preferred pipeline is model-assisted:

- `TransNetV2` runs first to split the video into scene or shot-level candidates.
- `DINOv2` embeds candidate frames and removes near-duplicates.
- `YOLO26-seg` estimates object count, object area ratio, overlap, and mask complexity.

If a model is not available, the sampler falls back to lightweight CPU-side planning:

- frame-difference peaks
- sparse interval anchors
- first / last frame retention
- edge, contrast, and saturation statistics

## Re-ranking Features

Each candidate frame is scored using:

- object count
- object area ratio
- object crowding / overlap
- mask complexity
- edge density
- grayscale contrast
- saturation
- DINO embedding novelty when available
- TransNetV2 shot-boundary confidence when available

The current demo combines:

- base segmentation complexity score
- diversity bonus against already selected frames

This pushes the final frame set toward:

- more varied object layouts
- more complex boundaries
- less redundancy

## Model Locations

Model files are server-local and are not committed to Git:

```text
models/keyframe/transnetv2/
models/embedding/dino/
models/segmentation/
```

See [../models/README.md](../models/README.md) for the full catalog.

## Future Upgrades

- historical error priors from correction logs
- label-conditioned CLIP or GroundingDINO relevance for prompt-driven selection
- per-project active learning after enough manual labels are collected
