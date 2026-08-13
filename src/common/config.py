"""Loader for config/config.yaml.

Kept dependency-light: the driver scripts and the tests use it, and the
containers can import it without needing boto3.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    config_path = Path(path or os.environ.get("MLOPS_CONFIG", DEFAULT_CONFIG_PATH))
    with open(config_path) as handle:
        config: dict[str, Any] = yaml.safe_load(handle)

    # Environment wins so CodeBuild can retarget a region or bucket without a
    # code change.
    config["region"] = os.environ.get("AWS_REGION") or os.environ.get(
        "AWS_DEFAULT_REGION", config.get("region")
    )
    if os.environ.get("ARTIFACT_BUCKET"):
        config["bucket"] = os.environ["ARTIFACT_BUCKET"]
    if os.environ.get("MODEL_PACKAGE_GROUP"):
        config["registry"]["model_package_group"] = os.environ["MODEL_PACKAGE_GROUP"]
    return config


def s3_uri(config: dict[str, Any], *parts: str) -> str:
    """Build s3://<bucket>/<prefix>/<parts...>."""
    bucket = config["bucket"]
    if not bucket:
        raise ValueError(
            "config.bucket is empty; set ARTIFACT_BUCKET or let the caller "
            "substitute the SageMaker default bucket first"
        )
    joined = "/".join(p.strip("/") for p in parts if p)
    return f"s3://{bucket}/{config['prefix']}/{joined}".rstrip("/")
