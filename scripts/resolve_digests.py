#!/usr/bin/env python3
"""Resolve pushed image tags to immutable digests and write build/images.json.

The SageMaker Pipeline references every step image by digest (repo@sha256:...),
never by tag, so a definition frozen at upsert time (and re-run months later by
the monthly EventBridge schedule) always runs the exact same image.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.config import load_config  # noqa: E402

IMAGES = ("preprocess", "train", "evaluate", "inference")


def ecr_digest(repo: str, tag: str, region: str) -> str:
    out = subprocess.check_output(
        [
            "aws",
            "ecr",
            "describe-images",
            "--repository-name",
            repo,
            "--image-ids",
            f"imageTag={tag}",
            "--region",
            region,
            "--query",
            "imageDetails[0].imageDigest",
            "--output",
            "text",
        ],
        text=True,
    ).strip()
    if not out or out == "None":
        raise RuntimeError(f"no digest for {repo}:{tag}")
    return out


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=os.environ.get("TAG", "dev"))
    parser.add_argument("--account-id", default=os.environ.get("ACCOUNT_ID"))
    parser.add_argument("--region", default=cfg["region"])
    parser.add_argument("--project", default=cfg["project"])
    parser.add_argument("--output", default="build/images.json")
    args = parser.parse_args(argv)

    if not args.account_id:
        args.account_id = subprocess.check_output(
            ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
            text=True,
        ).strip()

    registry = f"{args.account_id}.dkr.ecr.{args.region}.amazonaws.com"
    images = {}
    for name in IMAGES:
        repo = f"{args.project}/{name}"
        digest = ecr_digest(repo, args.tag, args.region)
        images[name] = f"{registry}/{repo}@{digest}"
        print(f"{name}: {images[name]}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tag": args.tag, "images": images}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
