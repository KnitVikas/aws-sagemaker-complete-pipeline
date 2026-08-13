"""Pipeline-definition tests: build the graph offline, assert its shape.

No AWS calls: PipelineSession builds request dicts locally, and we feed a fake
digest map plus an explicit role/bucket.
"""

from __future__ import annotations

import json

import pytest
import training_pipeline as tp

FAKE_IMAGES = {
    "preprocess": "111.dkr.ecr.us-east-1.amazonaws.com/pv-yolov8/preprocess@sha256:" + "a" * 64,
    "train": "111.dkr.ecr.us-east-1.amazonaws.com/pv-yolov8/train@sha256:" + "b" * 64,
    "evaluate": "111.dkr.ecr.us-east-1.amazonaws.com/pv-yolov8/evaluate@sha256:" + "c" * 64,
    "inference": "111.dkr.ecr.us-east-1.amazonaws.com/pv-yolov8/inference@sha256:" + "d" * 64,
}


@pytest.fixture(scope="module")
def definition():
    import boto3
    from sagemaker.workflow.pipeline_context import PipelineSession

    from src.common.config import load_config

    config = load_config()
    session = PipelineSession(
        boto_session=boto3.Session(region_name="us-east-1"),
        default_bucket="unit-test-bucket",
    )
    pipeline = tp.build_pipeline(
        config=config,
        images=FAKE_IMAGES,
        role="arn:aws:iam::111111111111:role/unit-test",
        bucket="unit-test-bucket",
        session=session,
    )
    return json.loads(pipeline.definition())


def test_load_images_requires_all_four(tmp_path):
    path = tmp_path / "images.json"
    path.write_text(json.dumps({"images": {"train": "x@sha256:1"}}))
    with pytest.raises(ValueError, match="missing image URIs"):
        tp.load_images(path)


def test_load_images_accepts_full_map(tmp_path):
    path = tmp_path / "images.json"
    path.write_text(json.dumps({"images": FAKE_IMAGES}))
    assert tp.load_images(path) == FAKE_IMAGES


def test_step_names_and_order(definition):
    names = [s["Name"] for s in definition["Steps"]]
    assert names == [tp.STEP_PREPROCESS, tp.STEP_TRAIN, tp.STEP_EVALUATE, tp.STEP_GATE]


def test_gate_uses_map50_threshold(definition):
    gate = next(s for s in definition["Steps"] if s["Name"] == tp.STEP_GATE)
    condition = gate["Arguments"]["Conditions"][0]
    # JsonGet path targets the map50 value the property file exposes.
    left = json.dumps(condition["LeftValue"])
    assert "map50" in left
    assert condition["Type"] == "GreaterThanOrEqualTo"


def test_condition_branches_register_or_reject(definition):
    gate = next(s for s in definition["Steps"] if s["Name"] == tp.STEP_GATE)
    if_steps = json.dumps(gate["Arguments"]["IfSteps"])
    else_steps = json.dumps(gate["Arguments"]["ElseSteps"])
    assert tp.STEP_REGISTER in if_steps
    assert tp.STEP_REJECT in else_steps


def test_images_are_digest_pinned(definition):
    blob = json.dumps(definition)
    for uri in FAKE_IMAGES.values():
        assert uri in blob
    # No mutable tag references leaked in.
    assert ":latest" not in blob


def test_train_receives_single_dataset_channel(definition):
    train = next(s for s in definition["Steps"] if s["Name"] == tp.STEP_TRAIN)
    channels = train["Arguments"]["InputDataConfig"]
    assert [c["ChannelName"] for c in channels] == ["dataset"]


def test_parameters_expose_map50_threshold(definition):
    params = {p["Name"] for p in definition["Parameters"]}
    assert "Map50Threshold" in params
    assert "ModelApprovalStatus" in params
    assert "AucThreshold" not in params  # no XGBoost leftovers
