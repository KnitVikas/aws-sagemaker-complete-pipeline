from __future__ import annotations

import json
from pathlib import Path

import evaluate
import pytest


class _FakeBox:
    map50 = 0.83
    map = 0.61
    mp = 0.78
    mr = 0.72
    ap50 = [0.80, 0.86]


class _FakeMetrics:
    box = _FakeBox()


def test_report_shape_matches_condition_step_json_path():
    report = evaluate.build_report(_FakeMetrics())
    m = report["object_detection_metrics"]
    # The pipeline reads exactly this path; renaming a key breaks the gate.
    assert isinstance(m["map50"]["value"], float)
    assert m["map50"]["value"] == pytest.approx(0.83)
    assert m["map50"]["standard_deviation"] == "NaN"


def test_report_includes_all_metrics():
    m = evaluate.build_report(_FakeMetrics())["object_detection_metrics"]
    for key in ("map50", "map50_95", "precision", "recall"):
        assert key in m
    assert m["map50_95"]["value"] == pytest.approx(0.61)


def test_per_class_ap_recorded():
    per_class = evaluate.build_report(_FakeMetrics())["object_detection_metrics"]["per_class"]
    assert per_class["person"]["ap50"] == pytest.approx(0.80)
    assert per_class["vehicle"]["ap50"] == pytest.approx(0.86)


def test_locate_model_finds_direct_file(tmp_path: Path):
    from common.schema import MODEL_FILE

    (tmp_path / MODEL_FILE).write_bytes(b"stub")
    found = evaluate.locate_model(tmp_path, tmp_path / "work")
    assert found.name == MODEL_FILE


def test_locate_model_extracts_from_tarball(tmp_path: Path):
    import tarfile

    from common.schema import MODEL_FILE

    weights = tmp_path / MODEL_FILE
    weights.write_bytes(b"stub")
    tar_dir = tmp_path / "tarred"
    tar_dir.mkdir()
    with tarfile.open(tar_dir / "model.tar.gz", "w:gz") as archive:
        archive.add(weights, arcname=MODEL_FILE)
    weights.unlink()

    work = tmp_path / "work"
    work.mkdir()
    found = evaluate.locate_model(tar_dir, work)
    assert found.name == MODEL_FILE


def test_locate_model_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        evaluate.locate_model(tmp_path, tmp_path / "work")


def test_report_is_json_serializable():
    report = evaluate.build_report(_FakeMetrics())
    # Must round-trip: it is written as the property file.
    assert json.loads(json.dumps(report)) == report
