"""Evaluate a trained YOLOv8 model on the held-out test split.

Writes the SageMaker property-file JSON the pipeline's ConditionStep reads:

    JsonGet(property_file=..., json_path="object_detection_metrics.map50.value")

In a Processing job the model arrives as model.tar.gz under
/opt/ml/processing/model, so the tarball is unpacked before loading.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parents[1] / "src", _HERE.parents[1]):
    if (_cand / "common").is_dir():
        sys.path.insert(0, str(_cand))
        break

from common.config import load_config  # noqa: E402
from common.schema import CLASS_NAMES, DATA_YAML, EVALUATION_FILE, MODEL_FILE  # noqa: E402

IN_CONTAINER = Path("/opt/ml/processing").exists()
DEFAULT_MODEL = "/opt/ml/processing/model" if IN_CONTAINER else "artifacts"
DEFAULT_DATA = "/opt/ml/processing/dataset" if IN_CONTAINER else "data/processed"
DEFAULT_OUTPUT = "/opt/ml/processing/evaluation" if IN_CONTAINER else "artifacts/evaluation"


def locate_model(model_dir: Path, workdir: Path) -> Path:
    direct = model_dir / MODEL_FILE
    if direct.is_file():
        return direct
    tarballs = sorted(model_dir.glob("*.tar.gz"))
    if tarballs:
        with tarfile.open(tarballs[0]) as archive:
            archive.extractall(workdir)  # noqa: S202 - our own training artifact
        found = next(workdir.rglob(MODEL_FILE), None)
        if found:
            return found
    found = next(model_dir.rglob(MODEL_FILE), None)
    if found:
        return found
    raise FileNotFoundError(f"{MODEL_FILE} not found under {model_dir}")


def _rewrite_data_path(src: Path) -> Path:
    """Point data.yaml's `path` at its own directory (the mounted dataset root)."""
    import yaml

    spec = yaml.safe_load(src.read_text())
    root = src.parent.resolve()
    if spec.get("path") == str(root):
        return src
    spec["path"] = str(root)
    import tempfile

    fixed = Path(tempfile.gettempdir()) / "data.eval.yaml"
    fixed.write_text(yaml.safe_dump(spec, sort_keys=False))
    return fixed


def build_report(metrics) -> dict:
    box = metrics.box

    def metric(value) -> dict:
        return {"value": float(value), "standard_deviation": "NaN"}

    per_class = {}
    try:
        for idx, name in enumerate(CLASS_NAMES):
            ap50 = float(box.ap50[idx]) if idx < len(box.ap50) else 0.0
            per_class[name] = {"ap50": ap50}
    except Exception:
        per_class = {}

    return {
        "object_detection_metrics": {
            "map50": metric(box.map50),
            "map50_95": metric(box.map),
            "precision": metric(box.mp),
            "recall": metric(box.mr),
            "per_class": per_class,
        }
    }


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    ecfg = cfg["evaluation"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", "--model_dir", dest="model_dir", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default=DEFAULT_DATA)
    parser.add_argument("--data-yaml", "--data_yaml", dest="data_yaml", default=None)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--conf", type=float, default=ecfg["conf"])
    parser.add_argument("--iou", type=float, default=ecfg["iou"])
    parser.add_argument("--imgsz", type=int, default=cfg["train"]["imgsz"])
    args = parser.parse_args(argv)

    # Absolute so Ultralytics does not nest val_runs under its global runs/ dir.
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data_yaml) if args.data_yaml else Path(args.data_dir) / DATA_YAML
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
    data_yaml = _rewrite_data_path(data_yaml)

    from ultralytics import YOLO

    with tempfile.TemporaryDirectory() as workdir:
        model_path = locate_model(Path(args.model_dir), Path(workdir))
        model = YOLO(str(model_path))
        metrics = model.val(
            data=str(data_yaml),
            split="test",
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            project=str(output_dir / "val_runs"),
            name="test",
            exist_ok=True,
            verbose=False,
        )
        # Copy the plots Ultralytics generates into the evaluation output.
        for plot in Path(metrics.save_dir).glob("*.png"):
            shutil.copy2(plot, output_dir / plot.name)

    report = build_report(metrics)
    (output_dir / EVALUATION_FILE).write_text(json.dumps(report, indent=2))

    m = report["object_detection_metrics"]
    print("evaluation:")
    for key in ("map50", "map50_95", "precision", "recall"):
        print(f"  {key:10s} {m[key]['value']:.4f}")
    print(f"wrote {output_dir / EVALUATION_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
