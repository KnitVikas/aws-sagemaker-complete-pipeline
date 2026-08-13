"""Ultralytics YOLOv8 training entry point.

Honors the SageMaker training contract, so the same file runs unchanged as the
container entry point and as a local CLI:

    data.yaml       /opt/ml/input/data/dataset/data.yaml   (SM_CHANNEL_DATASET)
    artifact        /opt/ml/model/best.pt                   (SM_MODEL_DIR)
    failure reason  /opt/ml/output/failure

Both --hyphen and --underscore hyperparameter spellings are accepted because
the SageMaker training toolkit renders hyperparameters as --<key> verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parents[1] / "src", _HERE.parents[1]):
    if (_cand / "common").is_dir():
        sys.path.insert(0, str(_cand))
        break

from common.config import load_config  # noqa: E402
from common.schema import DATA_YAML, MODEL_FILE  # noqa: E402

FAILURE_FILE = Path("/opt/ml/output/failure")
IN_CONTAINER = Path("/opt/ml").exists()
DEFAULT_DATA_DIR = (
    os.environ.get("SM_CHANNEL_DATASET", "/opt/ml/input/data/dataset")
    if IN_CONTAINER
    else "data/processed"
)
DEFAULT_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model") if IN_CONTAINER else "artifacts"


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    tcfg = cfg["train"]
    parser = argparse.ArgumentParser(description=__doc__)

    def add(name: str, **kwargs) -> None:
        parser.add_argument(f"--{name.replace('_', '-')}", f"--{name}", dest=name, **kwargs)

    add("model", default=tcfg["model"])
    add("imgsz", type=int, default=tcfg["imgsz"])
    add("epochs", type=int, default=tcfg["epochs"])
    add("batch", type=int, default=tcfg["batch"])
    add("workers", type=int, default=tcfg["workers"])
    add("patience", type=int, default=tcfg["patience"])
    add("seed", type=int, default=cfg["split"]["seed"])
    add("data_dir", default=DEFAULT_DATA_DIR)
    add("data_yaml", default=None, help="explicit path to data.yaml (else <data-dir>/data.yaml)")
    add("model_dir", default=DEFAULT_MODEL_DIR)
    add("device", default=None, help="cuda device index or 'cpu' (default: auto)")
    return parser


def resolve_data_yaml(args) -> Path:
    """Return a data.yaml whose `path` points at the actual dataset location.

    preprocess bakes an absolute `path:` for wherever it ran; the training
    container mounts that dataset somewhere else, so rewrite it to the real root.
    """
    import yaml

    src = Path(args.data_yaml) if args.data_yaml else Path(args.data_dir) / DATA_YAML
    if not src.is_file():
        return src

    spec = yaml.safe_load(src.read_text())
    root = src.parent.resolve()
    if spec.get("path") == str(root):
        return src
    spec["path"] = str(root)
    # Input channels are read-only in SageMaker, so write to a temp path. The
    # `path` field is absolute, so the yaml itself can live anywhere.
    import tempfile

    fixed = Path(tempfile.gettempdir()) / "data.local.yaml"
    fixed.write_text(yaml.safe_dump(spec, sort_keys=False))
    return fixed


def train(args) -> dict:
    from ultralytics import YOLO

    data_yaml = resolve_data_yaml(args)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    # Absolute so Ultralytics does not nest it under its global runs/ dir.
    runs_dir = model_dir / "runs"

    device = args.device
    if device is None:
        try:
            import torch

            device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        device=device,
        project=str(runs_dir),
        name="train",
        exist_ok=True,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"training finished but {best} is missing")
    shutil.copy2(best, model_dir / MODEL_FILE)

    # Pull final validation metrics out of results for the metric_definitions regex.
    metrics = getattr(results, "results_dict", {}) or {}
    map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
    map5095 = float(metrics.get("metrics/mAP50-95(B)", 0.0))
    print(f"metrics/mAP50:{map50:.6f}")
    print(f"metrics/mAP50-95:{map5095:.6f}")

    metadata = {
        "model": args.model,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "device": str(device),
        "map50": map50,
        "map50_95": map5095,
        "data_yaml": str(data_yaml),
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"saved {model_dir / MODEL_FILE}")
    return metadata


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    args = build_parser(cfg).parse_args(argv)
    try:
        train(args)
    except Exception:
        detail = traceback.format_exc()
        print(detail, file=sys.stderr)
        if FAILURE_FILE.parent.exists():
            FAILURE_FILE.write_text(detail)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
