"""Serving handlers for the YOLOv8 inference image.

Accepts raw image bytes (image/jpeg, image/png) or a JSON body with a
base64-encoded image, runs detection, and returns detections as JSON:

    {"detections": [{"class_id": 0, "label": "person",
                     "confidence": 0.91, "box_xyxy": [x1, y1, x2, y2]}, ...]}
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parents[1] / "src", _HERE.parents[1]):
    if (_cand / "common").is_dir():
        sys.path.insert(0, str(_cand))
        break

from common.schema import ID_TO_CLASS, MODEL_FILE  # noqa: E402

JPEG = "image/jpeg"
PNG = "image/png"
JSON_TYPE = "application/json"
DEFAULT_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
DEFAULT_IOU = float(os.environ.get("YOLO_IOU", "0.50"))


def resolve_device() -> int | str:
    """Pick CUDA when present; fail on a GPU host whose image cannot see CUDA.

    YOLO_DEVICE=cpu forces CPU (local docker-verify). YOLO_DEVICE=0 forces GPU.
    SageMaker GPU instances set NVIDIA_VISIBLE_DEVICES; if that is set but
    torch cannot see CUDA, serving must not silently fall back to CPU.
    """
    explicit = os.environ.get("YOLO_DEVICE")
    if explicit:
        if explicit.lower() == "cpu":
            return "cpu"
        return int(explicit) if explicit.isdigit() else explicit

    nvidia = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    gpu_host = nvidia not in ("", "void", "none", "-1")
    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False

    if gpu_host and not cuda:
        raise RuntimeError(
            "GPU instance is visible (NVIDIA_VISIBLE_DEVICES) but "
            "torch.cuda.is_available() is False; the inference image is CPU-bound"
        )
    return 0 if cuda else "cpu"


def model_fn(model_dir: str):
    from ultralytics import YOLO

    path = Path(model_dir) / MODEL_FILE
    if not path.is_file():
        found = next(Path(model_dir).rglob(MODEL_FILE), None)
        if found is None:
            raise FileNotFoundError(f"{MODEL_FILE} not found under {model_dir}")
        path = found
    model = YOLO(str(path))
    model.to(resolve_device())
    return model


def input_fn(request_body, request_content_type):
    content_type = (request_content_type or "").split(";")[0].strip().lower()
    from PIL import Image

    if content_type in (JPEG, PNG, "application/x-image", "application/octet-stream"):
        data = request_body if isinstance(request_body, bytes) else request_body.encode()
        return Image.open(io.BytesIO(data)).convert("RGB")

    if content_type == JSON_TYPE:
        payload = json.loads(request_body)
        if "image" not in payload:
            raise ValueError("JSON body must contain a base64 'image' field")
        data = base64.b64decode(payload["image"])
        return Image.open(io.BytesIO(data)).convert("RGB")

    raise ValueError(f"unsupported content type: {request_content_type}")


def predict_fn(image, model):
    results = model.predict(
        image,
        device=resolve_device(),
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        verbose=False,
    )
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append(
                {
                    "class_id": class_id,
                    "label": ID_TO_CLASS.get(class_id, str(class_id)),
                    "confidence": round(float(box.conf[0]), 6),
                    "box_xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
                }
            )
    return {"detections": detections}


def output_fn(prediction, accept):
    accept_type = (accept or JSON_TYPE).split(";")[0].strip().lower()
    if accept_type in ("*/*", "", JSON_TYPE):
        return json.dumps(prediction), JSON_TYPE
    raise ValueError(f"unsupported accept type: {accept}")
