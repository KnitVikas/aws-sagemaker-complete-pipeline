"""The SageMaker Pipeline: preprocess, train, evaluate, gate, register.

Every step runs one of our own images, referenced by digest. Nothing is passed
as source_dir, so a pipeline definition plus four digests fully determine what
runs.

    python pipelines/training_pipeline.py --definition             # print JSON, no AWS
    python pipelines/training_pipeline.py --upsert                 # create or update
    python pipelines/training_pipeline.py --upsert --start --wait  # and run it

Exit codes from --wait are distinct on purpose: 2 means the quality gate
rejected the model, 1 means something broke. CI should treat those differently.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config  # noqa: E402
from src.common.schema import DATA_YAML, EVALUATION_FILE  # noqa: E402

LOGGER = logging.getLogger("training_pipeline")

STEP_PREPROCESS = "Preprocess"
STEP_TRAIN = "Train"
STEP_EVALUATE = "Evaluate"
STEP_GATE = "QualityGate"
STEP_REGISTER = "RegisterModel"
STEP_REJECT = "RejectModel"

PROPERTY_FILE = "EvaluationReport"
MAP50_JSON_PATH = "object_detection_metrics.map50.value"

REQUIRED_IMAGES = ("preprocess", "train", "evaluate", "inference")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_GATE_REJECTED = 2


# --------------------------------------------------------------------- images


def load_images(path: str | os.PathLike[str]) -> dict[str, str]:
    """Read the digest map written by scripts/resolve_digests.py."""
    with open(path) as handle:
        payload = json.load(handle)
    images = payload.get("images", payload)

    missing = [name for name in REQUIRED_IMAGES if name not in images]
    if missing:
        raise ValueError(f"{path} is missing image URIs for: {missing}")

    for name in REQUIRED_IMAGES:
        if "@sha256:" not in images[name]:
            # Tags are mutable. A pipeline definition frozen today and triggered
            # by the monthly schedule in six months would silently run a
            # different image.
            LOGGER.warning(
                "image %r is pinned by tag, not digest (%s); the monthly "
                "retrain will not be reproducible",
                name,
                images[name],
            )
    return {name: images[name] for name in REQUIRED_IMAGES}


# ------------------------------------------------------------------- pipeline


def build_pipeline(
    config: dict[str, Any],
    images: dict[str, str],
    role: str,
    bucket: str,
    session: PipelineSession,
) -> Pipeline:
    prefix = config["prefix"]
    pipeline_name = config["pipeline"]["name"]
    base = f"s3://{bucket}/{prefix}/{pipeline_name}"

    # Static, not per-execution: step caching keys include these URIs, so
    # interpolating the execution id would guarantee a cache miss every run.
    dataset_base = f"{base}/dataset"
    training_base = f"{base}/training"
    evaluation_base = f"{base}/evaluation"

    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://{bucket}/{prefix}/raw",
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value=config["processing"]["instance_type"],
    )
    train_instance_type = ParameterString(
        name="TrainInstanceType",
        default_value=config["train"]["instance_type"],
    )
    train_instance_count = ParameterInteger(
        name="TrainInstanceCount",
        default_value=config["train"]["instance_count"],
    )
    map50_threshold = ParameterFloat(
        name="Map50Threshold",
        default_value=config["evaluation"]["map50_threshold"],
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value=config["registry"]["approval_status"],
    )

    cache = CacheConfig(
        enable_caching=True,
        expire_after=config["pipeline"]["cache_expire_after"],
    )

    # ---------------------------------------------------------- preprocess
    # One unified Ultralytics dataset (images/, labels/, data.yaml, metadata/)
    # emitted as a single output that both train and evaluate consume.
    split = config["split"]
    preprocessor = Processor(
        image_uri=images["preprocess"],
        role=role,
        instance_count=config["processing"]["instance_count"],
        instance_type=processing_instance_type,
        volume_size_in_gb=config["processing"]["volume_size_gb"],
        sagemaker_session=session,
        # No entrypoint override: the image's ENTRYPOINT is preprocess.py.
    )
    preprocess_step = ProcessingStep(
        name=STEP_PREPROCESS,
        step_args=preprocessor.run(
            inputs=[
                ProcessingInput(
                    source=input_data,
                    destination="/opt/ml/processing/input",
                    input_name="raw",
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="dataset",
                    source="/opt/ml/processing/output",
                    destination=dataset_base,
                )
            ],
            arguments=[
                "--input-dir",
                "/opt/ml/processing/input",
                "--output-dir",
                "/opt/ml/processing/output",
                "--train-ratio",
                str(split["train"]),
                "--validation-ratio",
                str(split["validation"]),
                "--test-ratio",
                str(split["test"]),
                "--seed",
                str(split["seed"]),
            ],
        ),
        cache_config=cache,
    )

    dataset_uri = preprocess_step.properties.ProcessingOutputConfig.Outputs[
        "dataset"
    ].S3Output.S3Uri

    # ------------------------------------------------------------- training
    tcfg = config["train"]
    hyperparameters = {
        "model": tcfg["model"],
        "imgsz": tcfg["imgsz"],
        "epochs": tcfg["epochs"],
        "batch": tcfg["batch"],
        "workers": tcfg["workers"],
        "patience": tcfg["patience"],
        # data.yaml sits at the root of the dataset channel.
        "data-yaml": f"/opt/ml/input/data/dataset/{DATA_YAML}",
    }
    estimator = Estimator(
        image_uri=images["train"],
        role=role,
        instance_count=train_instance_count,
        instance_type=train_instance_type,
        volume_size=tcfg["volume_size_gb"],
        max_run=tcfg["max_runtime_seconds"],
        output_path=training_base,
        hyperparameters=hyperparameters,
        # train.py prints these; the regexes lift them into CloudWatch and the
        # training job's FinalMetricDataList.
        metric_definitions=[
            {"Name": "validation:mAP50", "Regex": r"metrics/mAP50:([0-9\.]+)"},
            {"Name": "validation:mAP50-95", "Regex": r"metrics/mAP50-95:([0-9\.]+)"},
        ],
        sagemaker_session=session,
    )
    train_step = TrainingStep(
        name=STEP_TRAIN,
        step_args=estimator.fit(
            inputs={
                "dataset": TrainingInput(
                    s3_data=dataset_uri,
                    # A whole directory tree of images + labels, not a record set.
                    s3_data_type="S3Prefix",
                    input_mode="File",
                )
            }
        ),
        cache_config=cache,
    )

    # ----------------------------------------------------------- evaluation
    evaluator = Processor(
        image_uri=images["evaluate"],
        role=role,
        instance_count=config["processing"]["instance_count"],
        instance_type=processing_instance_type,
        volume_size_in_gb=config["processing"]["volume_size_gb"],
        sagemaker_session=session,
    )
    evaluate_step = ProcessingStep(
        name=STEP_EVALUATE,
        step_args=evaluator.run(
            inputs=[
                ProcessingInput(
                    # Arrives as model.tar.gz; evaluate.py unpacks it.
                    source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model",
                    input_name="model",
                ),
                ProcessingInput(
                    source=dataset_uri,
                    destination="/opt/ml/processing/dataset",
                    input_name="dataset",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                    destination=evaluation_base,
                )
            ],
            arguments=[
                "--model-dir",
                "/opt/ml/processing/model",
                "--data-dir",
                "/opt/ml/processing/dataset",
                "--output-dir",
                "/opt/ml/processing/evaluation",
                "--conf",
                str(config["evaluation"]["conf"]),
                "--iou",
                str(config["evaluation"]["iou"]),
            ],
        ),
        property_files=[
            PropertyFile(
                name=PROPERTY_FILE,
                output_name="evaluation",
                path=EVALUATION_FILE,
            )
        ],
        cache_config=cache,
    )

    # ------------------------------------------------------------- register
    model = Model(
        image_uri=images["inference"],
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        sagemaker_session=session,
    )
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    evaluate_step.properties.ProcessingOutputConfig.Outputs[
                        "evaluation"
                    ].S3Output.S3Uri,
                    EVALUATION_FILE,
                ],
            ),
            content_type="application/json",
        )
    )
    register_step = ModelStep(
        name=STEP_REGISTER,
        step_args=model.register(
            content_types=["image/jpeg", "image/png", "application/json"],
            response_types=["application/json"],
            inference_instances=[config["endpoint"]["instance_type"]],
            transform_instances=["ml.g4dn.xlarge"],
            model_package_group_name=config["registry"]["model_package_group"],
            approval_status=model_approval_status,
            model_metrics=model_metrics,
            description="YOLOv8 person/vehicle detector",
            customer_metadata_properties={
                # Lets anyone holding a package ARN find the exact commit and
                # the images that produced it.
                "git_sha": os.environ.get("CODEBUILD_RESOLVED_SOURCE_VERSION", "local"),
                "train_image": images["train"],
                "preprocess_image": images["preprocess"],
            },
        ),
    )

    reject_step = FailStep(
        name=STEP_REJECT,
        error_message=Join(
            on="",
            values=[
                "Model rejected: test mAP50 below the ",
                map50_threshold,
                " threshold. Nothing was registered.",
            ],
        ),
    )

    gate_step = ConditionStep(
        name=STEP_GATE,
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluate_step.name,
                    property_file=PROPERTY_FILE,
                    json_path=MAP50_JSON_PATH,
                ),
                right=map50_threshold,
            )
        ],
        if_steps=[register_step],
        else_steps=[reject_step],
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[
            input_data,
            processing_instance_type,
            train_instance_type,
            train_instance_count,
            map50_threshold,
            model_approval_status,
        ],
        steps=[preprocess_step, train_step, evaluate_step, gate_step],
        sagemaker_session=session,
    )


# ------------------------------------------------------------------ plumbing


def resolve_session(region: str, bucket: str | None) -> tuple[PipelineSession, str]:
    boto_session = boto3.Session(region_name=region)
    session = PipelineSession(boto_session=boto_session, default_bucket=bucket or None)
    return session, bucket or session.default_bucket()


def resolve_role(explicit: str | None, session: PipelineSession) -> str:
    if explicit:
        return explicit
    if os.environ.get("SAGEMAKER_ROLE_ARN"):
        return os.environ["SAGEMAKER_ROLE_ARN"]
    return sagemaker.get_execution_role(sagemaker_session=session)


def failed_steps(client: Any, execution_arn: str) -> list[dict[str, Any]]:
    steps = client.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn)[
        "PipelineExecutionSteps"
    ]
    return [step for step in steps if step.get("StepStatus") == "Failed"]


def wait(client: Any, execution_arn: str, poll_seconds: int = 30) -> int:
    """Block until the execution settles, then report why it settled that way."""
    while True:
        execution = client.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
        status = execution["PipelineExecutionStatus"]
        if status in ("Succeeded", "Failed", "Stopped"):
            break
        LOGGER.info("execution %s ...", status.lower())
        time.sleep(poll_seconds)

    if status == "Succeeded":
        LOGGER.info("pipeline succeeded")
        return EXIT_OK

    # Without this, the only signal in a CI log is "pipeline failed".
    rejected = False
    for step in failed_steps(client, execution_arn):
        reason = step.get("FailureReason", "no FailureReason reported")
        LOGGER.error("step %s failed: %s", step["StepName"], reason)
        if step["StepName"] == STEP_REJECT:
            rejected = True

    if rejected:
        LOGGER.error("the quality gate rejected this model; the build itself is fine")
        return EXIT_GATE_REJECTED
    LOGGER.error("pipeline %s: %s", status.lower(), execution.get("FailureReason", ""))
    return EXIT_ERROR


def describe_registered_model(client: Any, execution_arn: str, group: str) -> dict[str, Any]:
    """Collect the metadata the deploy stage logs (but does not trust)."""
    packages = client.list_model_packages(
        ModelPackageGroupName=group,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )["ModelPackageSummaryList"]
    if not packages:
        return {"pipeline_execution_arn": execution_arn}

    arn = packages[0]["ModelPackageArn"]
    package = client.describe_model_package(ModelPackageName=arn)
    container = package["InferenceSpecification"]["Containers"][0]
    return {
        "model_package_arn": arn,
        "model_package_version": package.get("ModelPackageVersion"),
        "approval_status": package.get("ModelApprovalStatus"),
        "model_data_url": container.get("ModelDataUrl"),
        "image_uri": container.get("Image"),
        "pipeline_execution_arn": execution_arn,
        "git_sha": package.get("CustomerMetadataProperties", {}).get("git_sha"),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="build/images.json", help="digest map to pin steps to")
    parser.add_argument("--role-arn", default=None)
    parser.add_argument("--bucket", default=config.get("bucket") or None)
    parser.add_argument("--region", default=config["region"])
    parser.add_argument("--model-package-group", default=None)
    parser.add_argument("--map50-threshold", type=float, default=None)
    parser.add_argument("--git-sha", default=None)
    parser.add_argument("--definition", action="store_true", help="print the JSON and exit")
    parser.add_argument("--upsert", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--output", default=None, help="write model package metadata here")
    args = parser.parse_args(argv)

    if args.model_package_group:
        config["registry"]["model_package_group"] = args.model_package_group
    if args.map50_threshold is not None:
        config["evaluation"]["map50_threshold"] = args.map50_threshold
    if args.git_sha:
        os.environ["CODEBUILD_RESOLVED_SOURCE_VERSION"] = args.git_sha

    images = load_images(args.images)
    session, bucket = resolve_session(args.region, args.bucket)
    role = resolve_role(args.role_arn, session)
    pipeline = build_pipeline(config, images, role, bucket, session)

    if args.definition:
        print(json.dumps(json.loads(pipeline.definition()), indent=2))
        return EXIT_OK

    if args.upsert:
        LOGGER.info("upserting pipeline %s", pipeline.name)
        pipeline.upsert(role_arn=role, tags=[{"Key": "project", "Value": config["project"]}])

    if not args.start:
        return EXIT_OK

    execution = pipeline.start()
    LOGGER.info("started %s", execution.arn)

    exit_code = EXIT_OK
    if args.wait:
        exit_code = wait(session.sagemaker_client, execution.arn, args.poll_seconds)

    if args.output:
        metadata: dict[str, Any] = {"pipeline_execution_arn": execution.arn}
        if exit_code == EXIT_OK:
            metadata = describe_registered_model(
                session.sagemaker_client,
                execution.arn,
                config["registry"]["model_package_group"],
            )
        metadata["images"] = images
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(metadata, indent=2))
        LOGGER.info("wrote %s", args.output)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
