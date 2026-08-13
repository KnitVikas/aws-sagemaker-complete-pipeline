"""Unify person.zip (CVAT) and vehicle.zip (YOLO) into one Ultralytics dataset.

Runs identically as a SageMaker Processing job (paths under /opt/ml/processing)
and on a laptop (paths under data/). Output layout (Ultralytics standard):

    <output-dir>/images/{train,val,test}/*.jpg
    <output-dir>/labels/{train,val,test}/*.txt      class cx cy w h  (normalized)
    <output-dir>/data.yaml                           nc=2, names=[person, vehicle]
    <output-dir>/metadata/class_counts.json          per-split box counts (baseline)

Two source formats are merged:
  vehicle/  already YOLO; every box remapped to the unified 'vehicle' id (1).
  person/   CVAT export (annotations.json); rectangles (possibly rotated) are
            reduced to axis-aligned boxes, normalized, class 'person' id (0).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parents[1] / "src", _HERE.parents[1]):
    if (_cand / "common").is_dir():
        sys.path.insert(0, str(_cand))
        break

from common.config import load_config  # noqa: E402
from common.schema import CLASS_NAMES, CLASS_TO_ID, DATA_YAML  # noqa: E402

IN_CONTAINER = Path("/opt/ml/processing").exists()
DEFAULT_INPUT = "/opt/ml/processing/input" if IN_CONTAINER else "data/raw"
DEFAULT_OUTPUT = "/opt/ml/processing/output" if IN_CONTAINER else "data/processed"

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# (image_path, [(class_id, cx, cy, w, h), ...]) with normalized YOLO boxes.
Box = tuple[int, float, float, float, float]
Sample = tuple[Path, list[Box]]


# --------------------------------------------------------------- vehicle (YOLO)


def collect_vehicle(vehicle_dir: Path) -> list[Sample]:
    """Return (image_path, [(cls, cx, cy, w, h), ...]) with cls forced to vehicle."""
    vehicle_id = CLASS_TO_ID["vehicle"]
    samples = []
    images = [p for p in vehicle_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    for img in sorted(images):
        label = img.with_suffix(".txt")
        boxes: list[Box] = []
        if label.is_file():
            for line in label.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, w, h = parts
                # Remap any local class id (0/1 sub-types) to the unified vehicle class.
                boxes.append((vehicle_id, float(cx), float(cy), float(w), float(h)))
        samples.append((img, boxes))
    return samples


# ----------------------------------------------------------------- person (CVAT)


def _rotated_aabb(points: list[float], rotation_deg: float) -> tuple[float, float, float, float]:
    """Axis-aligned bbox (xtl,ytl,xbr,ybr) of a possibly-rotated CVAT rectangle."""
    x1, y1, x2, y2 = points
    if not rotation_deg:
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    rotated = []
    for px, py in corners:
        dx, dy = px - cx, py - cy
        rotated.append((cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t))
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    return min(xs), min(ys), max(xs), max(ys)


def collect_person(person_dir: Path) -> list[Sample]:
    """Parse CVAT annotations.json, map frame->image, convert to YOLO boxes."""
    person_id = CLASS_TO_ID["person"]

    annotations_path = next(person_dir.rglob("annotations.json"), None)
    manifest_path = next(person_dir.rglob("manifest.jsonl"), None)
    if annotations_path is None or manifest_path is None:
        raise FileNotFoundError(
            f"CVAT annotations.json / manifest.jsonl not found under {person_dir}"
        )

    # Frame order comes from manifest lines that carry a "name" (skip headers).
    frame_names: list[str] = []
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "name" in obj:
            ext = obj.get("extension", ".jpg")
            frame_names.append(f"{obj['name']}{ext}")

    data_root = manifest_path.parent  # images live under here in CVAT exports
    ann = json.loads(annotations_path.read_text())
    if isinstance(ann, list):
        ann = ann[0]

    # Group shapes by frame index.
    by_frame: dict[int, list[dict]] = {}
    for shape in ann.get("shapes", []):
        by_frame.setdefault(shape["frame"], []).append(shape)

    samples = []
    for frame_idx, rel_name in enumerate(frame_names):
        img = data_root / rel_name
        if not img.is_file():
            # Fall back to basename search (CVAT nests under a task dir).
            matches = list(data_root.rglob(Path(rel_name).name))
            if not matches:
                continue
            img = matches[0]

        width, height = _image_size(img)
        boxes: list[Box] = []
        for shape in by_frame.get(frame_idx, []):
            if shape.get("type") != "rectangle":
                continue
            xtl, ytl, xbr, ybr = _rotated_aabb(shape["points"], shape.get("rotation", 0.0) or 0.0)
            xtl, xbr = sorted((max(0.0, xtl), min(float(width), xbr)))
            ytl, ybr = sorted((max(0.0, ytl), min(float(height), ybr)))
            bw, bh = xbr - xtl, ybr - ytl
            if bw <= 1 or bh <= 1:
                continue
            cx = (xtl + xbr) / 2.0 / width
            cy = (ytl + ybr) / 2.0 / height
            boxes.append((person_id, cx, cy, bw / width, bh / height))
        samples.append((img, boxes))
    return samples


def _image_size(path: Path) -> tuple[int, int]:
    """Return (width, height). Uses PIL if available, else the JPEG SOF marker."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        # Minimal JPEG dimension parser to avoid a hard PIL dependency in tests.
        data = path.read_bytes()
        i = 2
        while i < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                h = int.from_bytes(data[i + 5 : i + 7], "big")
                w = int.from_bytes(data[i + 7 : i + 9], "big")
                return w, h
            length = int.from_bytes(data[i + 2 : i + 4], "big")
            i += 2 + length
        raise ValueError(f"could not determine image size for {path}") from None


# --------------------------------------------------------------------- writing


def split_samples(samples, ratios, seed):
    import random

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["validation"])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def write_split(name, samples, output_dir):
    img_dir = output_dir / "images" / name
    lbl_dir = output_dir / "labels" / name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    counts = {cls: 0 for cls in CLASS_NAMES}
    for idx, (img, boxes) in enumerate(samples):
        # Unique, collision-free stem (source images share basenames across dirs).
        stem = f"{name}_{idx:06d}"
        shutil.copy2(img, img_dir / f"{stem}{img.suffix.lower()}")
        lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for (c, cx, cy, w, h) in boxes]
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        for c, *_ in boxes:
            counts[CLASS_NAMES[c]] += 1
    return counts


def write_data_yaml(output_dir: Path) -> None:
    content = (
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {list(CLASS_NAMES)}\n"
    )
    (output_dir / DATA_YAML).write_text(content)


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--train-ratio", type=float, default=cfg["split"]["train"])
    parser.add_argument("--validation-ratio", type=float, default=cfg["split"]["validation"])
    parser.add_argument("--test-ratio", type=float, default=cfg["split"]["test"])
    parser.add_argument("--seed", type=int, default=cfg["split"]["seed"])
    args = parser.parse_args(argv)

    ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {sum(ratios.values())}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    # Clear contents rather than the directory itself: in a Processing job the
    # output dir is a bind mount and cannot be removed.
    if output_dir.exists():
        for child in output_dir.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    person_dir = input_dir / "person"
    vehicle_dir = input_dir / "vehicle"

    samples = []
    if vehicle_dir.exists():
        v = collect_vehicle(vehicle_dir)
        print(f"vehicle: {len(v)} images")
        samples += v
    if person_dir.exists():
        p = collect_person(person_dir)
        print(f"person: {len(p)} images")
        samples += p
    if not samples:
        raise FileNotFoundError(f"no person/ or vehicle/ data found under {input_dir}")

    splits = split_samples(samples, ratios, args.seed)
    all_counts = {}
    for name, split in splits.items():
        counts = write_split(name, split, output_dir)
        all_counts[name] = {"images": len(split), "boxes": counts}
        print(f"  {name:5s} {len(split):4d} images  boxes={counts}")

    write_data_yaml(output_dir)
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "class_counts.json").write_text(json.dumps(all_counts, indent=2))
    print(f"wrote {output_dir / DATA_YAML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
