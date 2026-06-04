from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from autolabel_app.sam31_adapter import (
    find_similar_with_sam31_local,
    refine_annotation_with_sam31_local,
)


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        action = payload.get("action")
        image_path = Path(payload["image_path"])
        device = payload.get("device")
        with redirect_stdout(sys.stderr):
            if action == "refine":
                annotations = refine_annotation_with_sam31_local(
                    image_path,
                    annotation=payload["annotation"],
                    category=payload.get("category") or "object",
                    device=device,
                )
            elif action == "find":
                annotations = find_similar_with_sam31_local(
                    image_path,
                    prompt=payload.get("prompt") or "",
                    category=payload.get("category") or payload.get("prompt") or "object",
                    max_results=int(payload.get("max_results") or 8),
                    threshold=payload.get("threshold"),
                    device=device,
                )
            else:
                raise ValueError(f"Unsupported SAM3.1 worker action: {action}")
        print(json.dumps({"annotations": annotations}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
