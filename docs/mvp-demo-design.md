# Autolabel Web Demo Design

## 1. Current Scope

The current demo focuses on one stable manual-labeling loop:

```text
video input
-> server-side frame extraction
-> browser manual annotation
-> one JSON file per frame
-> project export
```

Automatic model labeling is intentionally not enabled in the review UI yet. YOLO, SAM, DINO, and LocateAnything are kept as server-local model assets for keyframe selection and future assisted labeling.

## 2. Access Pattern

The service is designed for a remote GPU server:

```text
local browser
-> SSH tunnel
-> FastAPI service on 127.0.0.1:<port>
-> server-local videos, frames, annotations, and models
```

Example:

```bash
ssh -p 10022 -N -L 42873:127.0.0.1:42873 mfl@111.0.130.56
uvicorn autolabel_app.main:app --host 127.0.0.1 --port 42873
```

The service should stay bound to `127.0.0.1` on the server, so external users reach it through SSH forwarding instead of exposing a public HTTP port.

## 3. Product Behavior

- Users can create a named project from an uploaded video or a server-local video path.
- The server extracts selected frames into the project directory.
- The selected frame is displayed in the center of the canvas.
- The frame strip behaves like a scrollable wheel.
- Supported manual shapes are rectangle, polygon, and circle.
- `A` switches to the previous frame and `S` switches to the next frame, with wrap-around.
- Switching frames automatically saves current annotations.
- `Ctrl+F` opens a label picker for the currently selected object.
- Labels can be added during annotation and are shared across the current video project.
- Different labels use stable different colors.
- The right panel shows annotations on the current frame and label categories used in the video.

## 4. Data Layout

For uploaded videos, project data is written under:

```text
data/projects/<project_id>/
  manifest.json
  source/<original_video>
  frames/<frame_id>.jpg
  annotations/<frame_id>.json
```

For server-local videos, the target layout follows the original data location:

```text
<video_parent>/_autolabel/<project_id>/
  manifest.json
  frames/<frame_id>.jpg
  annotations/<frame_id>.json
```

This keeps annotation results near the original data while still allowing browser export through the FastAPI service.

## 5. Annotation Schema

Each frame JSON stores only the final editable annotations:

```json
{
  "project_id": "project_name_20260604_120000",
  "frame_id": "frame_000001",
  "file_name": "frames/frame_000001.jpg",
  "timestamp_sec": 2.0,
  "width": 1920,
  "height": 1080,
  "annotations": [
    {
      "id": "ann_000001",
      "category": "object",
      "shape_type": "polygon",
      "points": [[100, 80], [220, 90], [210, 180]],
      "bbox": [100, 80, 120, 100],
      "segmentation": [[100, 80, 220, 90, 210, 180]],
      "source": "manual"
    }
  ]
}
```

Future AI-assisted labels should use the same schema and set `source` to the model backend name until the user confirms or edits them.

## 6. Future Model Entry Points

The next stage can add assisted labeling without changing the annotation schema:

- YOLO26 detection or YOLO26-seg proposes object boxes and coarse masks.
- SAM 2.1 / SAM 3 / SAM 3.1 refines masks from boxes, points, or text concepts.
- LocateAnything provides referring-expression or point-conditioned grounding.
- DINOv2 embeddings support similar-frame retrieval and active learning.

All model assets stay in the server-local `models/` directory. GitHub only tracks `models/README.md`, which documents what should be placed there.
