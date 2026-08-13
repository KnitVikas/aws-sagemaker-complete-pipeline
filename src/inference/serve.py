"""Minimal SageMaker BYOC serving app for the YOLOv8 endpoint.

Implements the two endpoints SageMaker requires:
  GET  /ping          -> 200 when the model is loaded
  POST /invocations   -> run detection on the request body

Reuses the model_fn / input_fn / predict_fn / output_fn handlers so the exact
same code path is exercised by unit tests and the live endpoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, Response, request

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import inference  # noqa: E402

# SageMaker extracts model.tar.gz here.
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

app = Flask(__name__)
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = inference.model_fn(MODEL_DIR)
    return _model


@app.route("/ping", methods=["GET"])
def ping() -> Response:
    try:
        _get_model()
        return Response("", status=200, mimetype="application/json")
    except Exception:
        return Response("", status=503, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations() -> Response:
    content_type = request.content_type or "application/octet-stream"
    accept = request.headers.get("Accept", "application/json")
    try:
        data = inference.input_fn(request.get_data(), content_type)
        prediction = inference.predict_fn(data, _get_model())
        body, out_type = inference.output_fn(prediction, accept)
        return Response(body, status=200, mimetype=out_type)
    except ValueError as exc:
        return Response(str(exc), status=415, mimetype="text/plain")
    except Exception as exc:  # noqa: BLE001
        return Response(str(exc), status=500, mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
