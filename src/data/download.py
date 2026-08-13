"""Unpack person.zip (CVAT) and vehicle.zip (YOLO) into data/raw/.

Does not download from the internet — the zips are expected at the paths in
config (default: data/zips/{person,vehicle}.zip). Optionally uploads the
extracted raw tree to S3 for the SageMaker pipeline.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parents[1] / "src", _HERE.parents[1]):
    if (_cand / "common").is_dir():
        sys.path.insert(0, str(_cand))
        break

from common.config import load_config  # noqa: E402


def _extract(zip_path: Path, dest: Path) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(
            f"Missing {zip_path}. Place person.zip and vehicle.zip under data/zips/ "
            f"(or symlink from /data-mount/ops/tp_ops)."
        )
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    print(f"extracted {zip_path.name} -> {dest} ({len(list(dest.rglob('*')))} entries)")


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    data_cfg = cfg["data"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-zip", default=data_cfg["person_zip"])
    parser.add_argument("--vehicle-zip", default=data_cfg["vehicle_zip"])
    parser.add_argument("--output-dir", default=data_cfg["raw_dir"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--upload-s3", default=None, help="s3://bucket/prefix for raw upload")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    person_dest = out / "person"
    vehicle_dest = out / "vehicle"

    if args.force and out.exists():
        shutil.rmtree(out)

    _extract(Path(args.person_zip), person_dest)
    _extract(Path(args.vehicle_zip), vehicle_dest)

    if args.upload_s3:
        import boto3

        s3 = boto3.client("s3")
        uri = args.upload_s3.rstrip("/")
        assert uri.startswith("s3://")
        bucket, _, prefix = uri[5:].partition("/")
        for path in out.rglob("*"):
            if path.is_file():
                key = f"{prefix}/{path.relative_to(out)}".lstrip("/")
                s3.upload_file(str(path), bucket, key)
        print(f"uploaded raw data to {uri}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
