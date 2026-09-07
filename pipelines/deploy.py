"""Deploy the latest *Approved* model package to the real-time endpoint.

Re-resolves the registry (never trusts an upstream artifact), then deploys
infra/endpoint.yaml via CloudFormation. Safe to run locally or from CI; the
CodePipeline deploy stage runs the equivalent aws cloudformation deploy.

    python pipelines/deploy.py --role-arn ... --bucket ...
"""



from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config  # noqa: E402

LOGGER = logging.getLogger("deploy")


def latest_approved_package(client, group: str) -> str:
    packages = client.list_model_packages(
        ModelPackageGroupName=group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )["ModelPackageSummaryList"]
    if not packages:
        raise SystemExit(f"no Approved model package in group {group}")
    return packages[0]["ModelPackageArn"]


def deploy_stack(template: str, stack: str, params: dict[str, str], region: str) -> int:
    overrides = [f"{k}={v}" for k, v in params.items() if v is not None and v != ""]
    cmd = [
        "aws", "cloudformation", "deploy",
        "--template-file", template,
        "--stack-name", stack,
        "--capabilities", "CAPABILITY_IAM", "CAPABILITY_NAMED_IAM",
        "--no-fail-on-empty-changeset",
        "--region", region,
        "--parameter-overrides", *overrides,
    ]
    LOGGER.info("deploying stack %s", stack)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=config["region"])
    parser.add_argument("--bucket", default=config.get("bucket") or None)
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--model-package-group", default=config["registry"]["model_package_group"])
    parser.add_argument("--model-package-arn", default=None, help="override auto-resolution")
    parser.add_argument("--alarm-topic-arn", default="")
    parser.add_argument("--template", default="infra/endpoint.yaml")
    args = parser.parse_args(argv)

    if not args.bucket:
        raise SystemExit("--bucket is required (data capture destination)")

    session = boto3.Session(region_name=args.region)
    sm = session.client("sagemaker")
    arn = args.model_package_arn or latest_approved_package(sm, args.model_package_group)
    LOGGER.info("deploying %s", arn)

    project = config["project"]
    ecfg = config["endpoint"]
    params = {
        "Project": project,
        "ModelPackageArn": arn,
        "ModelVersion": arn.rsplit("/", 1)[-1],
        "SageMakerExecutionRoleArn": args.role_arn,
        "InstanceType": ecfg["instance_type"],
        "InitialInstanceCount": str(ecfg["initial_instance_count"]),
        "MinCapacity": str(ecfg["min_capacity"]),
        "MaxCapacity": str(ecfg["max_capacity"]),
        "InvocationsPerInstanceTarget": str(ecfg["invocations_per_instance_target"]),
        "DataCaptureS3Uri": f"s3://{args.bucket}/{config['prefix']}/datacapture",
        "DataCaptureSamplingPercent": str(ecfg["data_capture_sampling_percent"]),
        "AlarmTopicArn": args.alarm_topic_arn,
    }
    rc = deploy_stack(args.template, f"{project}-endpoint", params, args.region)
    if rc != 0:
        return rc

    # Wire the drift alarm + dashboard now that the endpoint exists.
    import pipelines.monitoring as monitoring

    monitoring.main(
        [
            "--region", args.region,
            "--endpoint-name", f"{project}-endpoint",
            *(["--topic-arn", args.alarm_topic_arn] if args.alarm_topic_arn else []),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
