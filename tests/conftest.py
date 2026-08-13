"""Shared fixtures. sys.path mirrors the container layout: each entry script
sits next to the `common` package, so tests import them as top-level modules.
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (
    ROOT / "src",
    ROOT / "src" / "processing",
    ROOT / "src" / "training",
    ROOT / "src" / "inference",
    ROOT / "src" / "data",
    ROOT / "pipelines",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _write_png(path: Path, width: int, height: int) -> None:
    """Write a minimal valid PNG without needing Pillow."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


@pytest.fixture
def png_factory():
    return _write_png


@pytest.fixture
def raw_dataset(tmp_path: Path):
    """A tiny raw tree mimicking data/raw: vehicle/ (YOLO) and person/ (CVAT)."""
    import json

    raw = tmp_path / "raw"

    # vehicle: YOLO image + label with local ids 0 and 1 (both are vehicles).
    vehicle = raw / "vehicle"
    vehicle.mkdir(parents=True)
    for i in range(4):
        _write_png(vehicle / f"veh_{i}.png", 100, 100)
        (vehicle / f"veh_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n")

    # person: CVAT export layout.
    person = raw / "person"
    data_dir = person / "data" / "task"
    data_dir.mkdir(parents=True)
    names = []
    for i in range(4):
        rel = f"task/img_{i}"
        _write_png(person / "data" / f"{rel}.png", 1280, 720)
        names.append(rel)

    manifest_lines = ['{"version":"1.1"}', '{"type":"images"}']
    for rel in names:
        manifest_lines.append(json.dumps({"name": rel, "extension": ".png"}))
    (person / "data" / "manifest.jsonl").write_text("\n".join(manifest_lines))

    shapes = []
    for frame in range(4):
        shapes.append(
            {
                "type": "rectangle",
                "rotation": 0.0,
                "points": [100.0, 100.0, 300.0, 400.0],
                "frame": frame,
                "label": "Person",
            }
        )
    # One rotated box to exercise the AABB path.
    shapes.append(
        {
            "type": "rectangle",
            "rotation": 30.0,
            "points": [500.0, 300.0, 600.0, 500.0],
            "frame": 0,
            "label": "Person",
        }
    )
    (person / "annotations.json").write_text(json.dumps([{"shapes": shapes}]))
    (person / "task.json").write_text(json.dumps({"labels": [{"name": "Person"}]}))
    return raw


@pytest.fixture
def processed_dataset(raw_dataset: Path, tmp_path: Path):
    """Real preprocess output for downstream tests."""
    import preprocess

    out = tmp_path / "processed"
    assert (
        preprocess.main(
            [
                "--input-dir",
                str(raw_dataset),
                "--output-dir",
                str(out),
                "--train-ratio",
                "0.5",
                "--validation-ratio",
                "0.25",
                "--test-ratio",
                "0.25",
                "--seed",
                "0",
            ]
        )
        == 0
    )
    return out
