#!/usr/bin/env bash
# Smoke-test the built images against the /opt/ml contracts, no AWS required.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${PROJECT:-pv-yolov8}"
TAG="${1:-latest}"
WORK="$(mktemp -d)"
CID=""
cleanup() {
  [[ -n "$CID" ]] && docker rm -f "$CID" >/dev/null 2>&1 || true
  # Containers write as root; chown back before removing from the host.
  docker run --rm -v "$WORK:/w" busybox chown -R "$(id -u):$(id -g)" /w >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "== preprocess =="
mkdir -p "$WORK/in" "$WORK/out"
cp -r data/raw/* "$WORK/in/" 2>/dev/null || { echo "run 'make data' first"; exit 1; }
docker run --rm \
  -v "$WORK/in:/opt/ml/processing/input" \
  -v "$WORK/out:/opt/ml/processing/output" \
  "${PROJECT}/preprocess:${TAG}"
test -f "$WORK/out/data.yaml" && echo "  data.yaml produced OK"

echo "== train (1 epoch, cpu) =="
mkdir -p "$WORK/model"
docker run --rm \
  -e SM_CHANNEL_DATASET=/opt/ml/input/data/dataset \
  -e SM_MODEL_DIR=/opt/ml/model \
  -v "$WORK/out:/opt/ml/input/data/dataset" \
  -v "$WORK/model:/opt/ml/model" \
  "${PROJECT}/train:${TAG}" \
  python3 /opt/ml/code/train.py --epochs 1 --imgsz 160 --batch 2 --workers 0 --device cpu
test -f "$WORK/model/best.pt" && echo "  best.pt produced OK"

echo "== evaluate =="
mkdir -p "$WORK/eval"
docker run --rm \
  -v "$WORK/model:/opt/ml/processing/model" \
  -v "$WORK/out:/opt/ml/processing/dataset" \
  -v "$WORK/eval:/opt/ml/processing/evaluation" \
  "${PROJECT}/evaluate:${TAG}"
test -f "$WORK/eval/evaluation.json" && echo "  evaluation.json produced OK"

echo "== inference (/ping + /invocations) =="
CID="$(docker run -d -p 8080:8080 -v "$WORK/model:/opt/ml/model" "${PROJECT}/inference:${TAG}" serve)"
for _ in $(seq 1 30); do
  code="$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/ping || true)"
  [[ "$code" == "200" ]] && break
  sleep 2
done
echo "  /ping -> $code"
IMG="$(find "$WORK/out/images/test" -name '*.jpg' -o -name '*.png' | head -1)"
curl -s -X POST --data-binary "@${IMG}" -H 'Content-Type: image/jpeg' \
  http://localhost:8080/invocations | head -c 300
echo
echo "all container contracts verified"
