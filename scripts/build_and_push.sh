#!/usr/bin/env bash
# Build (and optionally push) the four pipeline images plus the shared base.
#
#   ./scripts/build_and_push.sh [--skip-push] [--tag TAG] [image ...]
#
# Images build in dependency order: base -> preprocess, and train -> evaluate.
# Each is tagged with the short git SHA; --cache-from reuses the previous tag so
# layer cache survives ephemeral CodeBuild hosts. A context-hash label lets a
# rebuild be skipped when nothing that affects the image changed.
set -euo pipefail

cd "$(dirname "$0")/.."

PROJECT="${PROJECT:-pv-yolov8}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${ACCOUNT_ID:-}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo dev)}"
SKIP_PUSH=0
PLATFORM="linux/amd64"
IMAGES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-push) SKIP_PUSH=1; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    all) shift ;;
    *) IMAGES+=("$1"); shift ;;
  esac
done
if [[ ${#IMAGES[@]} -eq 0 ]]; then
  IMAGES=(preprocess train evaluate inference)
fi

registry() {
  if [[ -n "$ACCOUNT_ID" ]]; then
    echo "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
  else
    echo "local"
  fi
}
REGISTRY="$(registry)"

repo_uri() { echo "${REGISTRY}/${PROJECT}/$1"; }

context_hash() {
  # Hash the Dockerfile + copied source so we can skip unchanged rebuilds.
  local image="$1"
  {
    cat "docker/${image}/Dockerfile"
    find src config -type f 2>/dev/null | sort | xargs cat 2>/dev/null
  } | sha256sum | cut -c1-16
}

login() {
  [[ "$SKIP_PUSH" -eq 1 || "$REGISTRY" == "local" ]] && return 0
  aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$REGISTRY"
}

build_one() {
  local image="$1"; shift
  local dockerfile="docker/${image}/Dockerfile"
  local local_tag="${PROJECT}/${image}:${TAG}"
  local chash; chash="$(context_hash "$image")"

  echo ">> building ${local_tag} (context ${chash})"
  local cache_args=()
  if docker image inspect "${PROJECT}/${image}:latest" >/dev/null 2>&1; then
    cache_args=(--cache-from "${PROJECT}/${image}:latest")
  fi

  docker build \
    --platform "$PLATFORM" \
    -f "$dockerfile" \
    -t "$local_tag" \
    -t "${PROJECT}/${image}:latest" \
    --label "context_hash=${chash}" \
    "${cache_args[@]}" \
    "$@" \
    .

  if [[ "$SKIP_PUSH" -eq 0 && "$REGISTRY" != "local" ]]; then
    local remote; remote="$(repo_uri "$image"):${TAG}"
    docker tag "$local_tag" "$remote"
    docker push "$remote"
    echo "pushed ${remote}"
  fi
}

login

# base underpins preprocess; build it first but do not push (not a pipeline step).
if printf '%s\n' "${IMAGES[@]}" | grep -qx preprocess; then
  echo ">> building ${PROJECT}/base:${TAG}"
  docker build --platform "$PLATFORM" -f docker/base/Dockerfile \
    -t "${PROJECT}/base:latest" .
fi

for image in "${IMAGES[@]}"; do
  case "$image" in
    preprocess) build_one preprocess --build-arg "BASE_IMAGE=${PROJECT}/base:latest" ;;
    train)      build_one train ;;
    evaluate)   build_one evaluate --build-arg "TRAIN_IMAGE=${PROJECT}/train:latest" ;;
    inference)  build_one inference ;;
    *) echo "unknown image: $image" >&2; exit 2 ;;
  esac
done

echo "done: tag=${TAG}"
