# Server-Local Model Catalog

Model weights and upstream model repositories are intentionally not committed to GitHub. This directory is the expected server-local layout for Autolabel.

## Detection

Path:

```text
models/detection/
```

Expected files:

- `yolo26n.pt`
- `yolo26s.pt`
- `yolo26m.pt`
- `yolo26l.pt`
- `yolo26x.pt`

Purpose:

- future object-detection assisted labeling
- object-count and layout signals for keyframe selection

## Instance Segmentation

Path:

```text
models/segmentation/
```

Expected files:

- `yolo26n-seg.pt`
- `yolo26s-seg.pt`
- `yolo26m-seg.pt`
- `yolo26l-seg.pt`
- `yolo26x-seg.pt`

Purpose:

- coarse object masks for segmentation-oriented keyframe scoring
- future assisted mask proposals

## SAM 2.1

Path:

```text
models/sam/
```

Expected files:

- `sam2.1_t.pt`
- `sam2.1_s.pt`
- `sam2.1_b.pt`
- `sam2.1_l.pt`

Purpose:

- future prompt-based mask refinement from boxes, points, or polygons

## SAM 3 / SAM 3.1

Path:

```text
models/sam3/
```

Expected layout:

```text
models/sam3/
  repo/
  checkpoints/
    sam3/
      sam3.safetensors
    sam3.1/
      sam3.1_multiplex.pt
      sam3.1_multiplex.safetensors
    sam3.1-fp16/
      sam3.1_multiplex_fp16.safetensors
```

Purpose:

- future global/concept segmentation
- future interactive mask refinement

Notes:

- `repo/` is an upstream code checkout kept only on the server.
- SAM 3.1 assets may require accepting the provider license before download.
- The current Web demo does not call SAM by default.

## DINOv2

Path:

```text
models/embedding/dino/
```

Expected files:

```text
models/embedding/dino/checkpoints/dinov2_vits14_pretrain.pth
models/embedding/dino/torchhub/
```

Purpose:

- candidate-frame embedding
- near-duplicate removal
- diversity scoring for keyframe selection

## TransNetV2

Path:

```text
models/keyframe/transnetv2/
```

Expected files:

```text
models/keyframe/transnetv2/transnetv2-pytorch-weights.pth
models/keyframe/transnetv2/tf_saved_model/
```

Purpose:

- shot-boundary detection
- first-pass candidate selection before DINO/YOLO re-ranking

## LocateAnything-3B

Path:

```text
models/grounding/locateanything-3b/
```

Expected files:

- model safetensors shards
- tokenizer files
- processor files
- model configuration files

Purpose:

- future referring-expression and point-conditioned grounding
- future human-AI assisted object localization

## Git Policy

Git should only track this README from `models/`.

Do not commit:

- `*.pt`
- `*.pth`
- `*.safetensors`
- `*.onnx`
- `*.engine`
- upstream model repos
- torch hub caches
- tokenizer or processor assets downloaded with gated models
