from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import train

from common.schema import MODEL_FILE

pytestmark = pytest.mark.slow


def test_one_epoch_smoke_train(processed_dataset: Path, tmp_path: Path):
    """End-to-end 1-epoch train on the tiny synthetic dataset.

    Marked slow: it downloads yolov8n.pt on first run and needs torch. Run with
    `pytest -m slow`. Skipped automatically if ultralytics is unavailable.
    """
    pytest.importorskip("ultralytics")

    model_dir = tmp_path / "model"
    exit_code = train.main(
        [
            "--data-dir",
            str(processed_dataset),
            "--model-dir",
            str(model_dir),
            "--epochs",
            "1",
            "--batch",
            "2",
            "--imgsz",
            "160",
            "--workers",
            "0",
            "--device",
            "cpu",
        ]
    )
    assert exit_code == 0
    assert (model_dir / MODEL_FILE).is_file()

    # Clean any stray ultralytics runs dir created in CWD.
    shutil.rmtree(Path("runs"), ignore_errors=True)


def test_missing_data_yaml_is_reported_as_failure(tmp_path: Path):
    exit_code = train.main(
        [
            "--data-dir",
            str(tmp_path / "nope"),
            "--model-dir",
            str(tmp_path / "model"),
            "--epochs",
            "1",
        ]
    )
    assert exit_code == 1
