"""Model Monitor + drift signals for the YOLOv8 endpoint.

Raw images are not tabular, so the classic DefaultModelMonitor data-quality
baseline does not apply directly. Instead we monitor two things:

  1. Operational health: latency, invocations, 4xx/5xx (endpoint template alarms).
  2. Detection-statistics drift: a scheduled job compares live per-request
     detection counts / class mix (parsed from captured responses) against the
     training class distribution written by preprocess (metadata/class_counts.json).

This module wires the CloudWatch alarm on the detection-drift metric and builds
a dashboard. The detection-statistics job itself is a lightweight scheduled
Processing job (image reuses the evaluate container) that emits the metric.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config  # noqa: E402

LOGGER = logging.getLogger("monitoring")

DRIFT_NAMESPACE = "pv-yolov8/Monitoring"
DRIFT_METRIC = "DetectionRateDrift"


def put_drift_alarm(cw: Any, project: str, endpoint: str, topic_arn: str | None) -> None:
    """Alarm on the detection-rate drift metric; feeds the retraining Lambda."""
    kwargs: dict[str, Any] = dict(
        AlarmName=f"{project}-detection-drift",
        AlarmDescription="Live detection rate diverged from the training baseline",
        Namespace=DRIFT_NAMESPACE,
        MetricName=DRIFT_METRIC,
        Dimensions=[{"Name": "EndpointName", "Value": endpoint}],
        Statistic="Average",
        Period=3600,
        EvaluationPeriods=3,
        Threshold=0.30,  # >30% relative divergence over 3 hours
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
    )
    if topic_arn:
        kwargs["AlarmActions"] = [topic_arn]
    cw.put_metric_alarm(**kwargs)
    LOGGER.info("put alarm %s", kwargs["AlarmName"])


def dashboard_body(project: str, endpoint: str, region: str, pipeline_name: str) -> str:
    variant = "AllTraffic"

    def row(namespace: str, metric: str, *extra) -> list:
        return [namespace, metric, "EndpointName", endpoint, "VariantName", variant, *extra]

    sm = "AWS/SageMaker"
    ep = "/aws/sagemaker/Endpoints"
    widgets = [
        {
            "type": "metric",
            "width": 12,
            "height": 6,
            "properties": {
                "title": "Invocations & latency",
                "region": region,
                "metrics": [
                    row(sm, "Invocations"),
                    row(sm, "ModelLatency", {"stat": "p99", "yAxis": "right"}),
                ],
                "period": 60,
            },
        },
        {
            "type": "metric",
            "width": 12,
            "height": 6,
            "properties": {
                "title": "Errors",
                "region": region,
                "metrics": [
                    row(sm, "Invocation4XXErrors"),
                    row(sm, "Invocation5XXErrors"),
                ],
                "period": 60,
                "stat": "Sum",
            },
        },
        {
            "type": "metric",
            "width": 12,
            "height": 6,
            "properties": {
                "title": "GPU utilization",
                "region": region,
                "metrics": [
                    row(ep, "GPUUtilization"),
                    row(ep, "GPUMemoryUtilization"),
                ],
                "period": 60,
            },
        },
        {
            "type": "metric",
            "width": 12,
            "height": 6,
            "properties": {
                "title": "Detection-rate drift vs training baseline",
                "region": region,
                "metrics": [[DRIFT_NAMESPACE, DRIFT_METRIC, "EndpointName", endpoint]],
                "period": 3600,
                "stat": "Average",
                "annotations": {"horizontal": [{"label": "threshold", "value": 0.30}]},
            },
        },
        {
            "type": "metric",
            "width": 24,
            "height": 6,
            "properties": {
                "title": "Training mAP history",
                "region": region,
                "metrics": [
                    ["/aws/sagemaker/TrainingJobs", "validation:mAP50"],
                    ["/aws/sagemaker/TrainingJobs", "validation:mAP50-95"],
                ],
                "period": 300,
                "stat": "Maximum",
            },
        },
    ]
    return json.dumps({"widgets": widgets})


def put_dashboard(cw: Any, project: str, body: str) -> None:
    cw.put_dashboard(DashboardName=f"{project}-mlops", DashboardBody=body)
    LOGGER.info("put dashboard %s-mlops", project)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=config["region"])
    parser.add_argument("--endpoint-name", default=config["endpoint"]["name"])
    parser.add_argument("--topic-arn", default=None)
    parser.add_argument("--dashboard-only", action="store_true")
    parser.add_argument("--alarm-only", action="store_true")
    args = parser.parse_args(argv)

    session = boto3.Session(region_name=args.region)
    cw = session.client("cloudwatch")
    project = config["project"]

    if not args.dashboard_only:
        put_drift_alarm(cw, project, args.endpoint_name, args.topic_arn)
    if not args.alarm_only:
        body = dashboard_body(
            project, args.endpoint_name, args.region, config["pipeline"]["name"]
        )
        put_dashboard(cw, project, body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
