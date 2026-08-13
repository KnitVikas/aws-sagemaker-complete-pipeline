from __future__ import annotations

import base64
import json
import struct
import zlib

import pytest

import inference
from common.schema import ID_TO_CLASS


def _png_bytes(width=64, height=48):
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class _FakeBoxTensor:
    def __init__(self, value):
        self._value = value

    def __getitem__(self, idx):
        return self._value

    def tolist(self):
        return self._value


class _FakeBox:
    def __init__(self, cls, conf, xyxy):
        self.cls = _FakeBoxTensor(cls)
        self.conf = _FakeBoxTensor(conf)
        self.xyxy = [_FakeBoxTensor(xyxy)]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, boxes):
        self._boxes = boxes
        self.calls = []

    def predict(self, image, conf, iou, verbose):
        self.calls.append((conf, iou))
        return [_FakeResult(self._boxes)]


def test_input_fn_parses_raw_image_bytes():
    img = inference.input_fn(_png_bytes(), "image/png")
    assert img.size == (64, 48)


def test_input_fn_parses_base64_json():
    payload = json.dumps({"image": base64.b64encode(_png_bytes()).decode()})
    img = inference.input_fn(payload, "application/json")
    assert img.size == (64, 48)


def test_input_fn_rejects_unknown_type():
    with pytest.raises(ValueError, match="unsupported content type"):
        inference.input_fn(b"x", "text/csv")


def test_input_fn_json_requires_image_field():
    with pytest.raises(ValueError, match="base64 'image'"):
        inference.input_fn(json.dumps({"nope": 1}), "application/json")


def test_predict_fn_maps_labels_and_rounds():
    model = _FakeModel([_FakeBox(0.0, 0.912345, [1.111, 2.222, 3.333, 4.444])])
    out = inference.predict_fn("img", model)
    det = out["detections"][0]
    assert det["class_id"] == 0
    assert det["label"] == ID_TO_CLASS[0]
    assert det["confidence"] == pytest.approx(0.912345, abs=1e-6)
    assert det["box_xyxy"] == [1.11, 2.22, 3.33, 4.44]


def test_predict_fn_empty_detections():
    out = inference.predict_fn("img", _FakeModel([]))
    assert out == {"detections": []}


def test_output_fn_defaults_to_json():
    body, ctype = inference.output_fn({"detections": []}, "*/*")
    assert ctype == "application/json"
    assert json.loads(body) == {"detections": []}


def test_output_fn_rejects_unknown_accept():
    with pytest.raises(ValueError, match="unsupported accept type"):
        inference.output_fn({}, "image/png")
