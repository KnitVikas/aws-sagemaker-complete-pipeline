"""Schema constants shared by preprocess, train, evaluate and inference."""

from __future__ import annotations

CLASS_NAMES = ("person", "vehicle")
CLASS_TO_ID = {"person": 0, "vehicle": 1}
ID_TO_CLASS = {0: "person", 1: "vehicle"}
NUM_CLASSES = len(CLASS_NAMES)

# Canonical Ultralytics dataset layout produced by preprocess.py
DATA_YAML = "data.yaml"
MODEL_FILE = "best.pt"
MODEL_WEIGHTS_DIR = "weights"
EVALUATION_FILE = "evaluation.json"

# SageMaker channel / processing layout
IMAGES_DIR = "images"
LABELS_DIR = "labels"
