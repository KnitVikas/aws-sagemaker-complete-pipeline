from __future__ import annotations

import json
import math
from pathlib import Path

import preprocess
import pytest

from common.schema import CLASS_NAMES, CLASS_TO_ID


def test_rotated_aabb_no_rotation():
    assert preprocess._rotated_aabb([10, 20, 30, 40], 0.0) == (10, 20, 30, 40)


def test_rotated_aabb_expands_box():
    # A square rotated 45 degrees has a larger axis-aligned bounding box.
    x1, y1, x2, y2 = preprocess._rotated_aabb([0, 0, 10, 10], 45.0)
    width = x2 - x1
    assert width == pytest.approx(10 * math.sqrt(2), abs=1e-6)


def test_outputs_ultralytics_layout(processed_dataset: Path):
    assert (processed_dataset / "data.yaml").is_file()
    for split in ("train", "val", "test"):
        assert (processed_dataset / "images" / split).is_dir()
        assert (processed_dataset / "labels" / split).is_dir()


def test_every_image_has_a_label(processed_dataset: Path):
    for split in ("train", "val", "test"):
        images = list((processed_dataset / "images" / split).glob("*.*"))
        for img in images:
            label = processed_dataset / "labels" / split / f"{img.stem}.txt"
            assert label.is_file(), f"missing label for {img.name}"


def test_labels_are_normalized_and_valid_classes(processed_dataset: Path):
    for label in (processed_dataset / "labels").rglob("*.txt"):
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            cls, cx, cy, w, h = line.split()
            assert int(cls) in (0, 1)
            for v in (cx, cy, w, h):
                assert 0.0 <= float(v) <= 1.0


def test_vehicle_boxes_remapped_to_single_class(raw_dataset: Path, tmp_path: Path):
    # vehicle-only input: every box must become class 1, never 0.
    (raw_dataset / "person").rename(tmp_path / "person_hidden")
    out = tmp_path / "veh_only"
    preprocess.main(["--input-dir", str(raw_dataset), "--output-dir", str(out), "--seed", "0"])
    ids = set()
    for label in (out / "labels").rglob("*.txt"):
        for line in label.read_text().splitlines():
            if line.strip():
                ids.add(int(line.split()[0]))
    assert ids == {CLASS_TO_ID["vehicle"]}


def test_data_yaml_declares_two_classes(processed_dataset: Path):
    text = (processed_dataset / "data.yaml").read_text()
    assert "nc: 2" in text
    assert "person" in text and "vehicle" in text


def test_split_ratios(processed_dataset: Path):
    counts = {
        split: len(list((processed_dataset / "images" / split).glob("*.*")))
        for split in ("train", "val", "test")
    }
    total = sum(counts.values())
    assert total == 8  # 4 vehicle + 4 person
    assert counts["train"] == 4


def test_class_counts_metadata_written(processed_dataset: Path):
    counts = json.loads((processed_dataset / "metadata" / "class_counts.json").read_text())
    assert set(counts) == {"train", "val", "test"}
    for split in counts.values():
        assert set(split["boxes"]) == set(CLASS_NAMES)


def test_ratios_must_sum_to_one(raw_dataset: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="sum to 1.0"):
        preprocess.main(
            [
                "--input-dir",
                str(raw_dataset),
                "--output-dir",
                str(tmp_path / "bad"),
                "--train-ratio",
                "0.5",
                "--validation-ratio",
                "0.5",
                "--test-ratio",
                "0.5",
            ]
        )
