# pv-yolov8 — SageMaker MLOps for a person/vehicle detector

An end-to-end, reproducible MLOps project that trains a **YOLOv8** two-class
object detector (`person`, `vehicle`), gates it on **mAP@0.5**, registers it,
serves it from an autoscaled GPU endpoint with data capture, and retrains it
monthly. Every pipeline step runs one of our own container images pinned by
**digest**, so a definition frozen today runs identically months later.

```
Source ─▶ Test ─▶ Build 4 images (ECR) ─▶ SageMaker Pipeline ─▶ Model Registry
                                              │                        │ approve
                              preprocess ▶ train ▶ evaluate ▶ mAP gate  ▼
                              (CVAT+YOLO)  (YOLOv8)  (mAP@0.5)      Deploy ▶ GPU Endpoint
                                                                        │ data capture
                                              EventBridge (monthly) ────┘ Model Monitor ▶ drift ▶ retrain
```

## The data

Two local zips (symlinked into `data/zips/`) with **different formats**, unified
by the preprocess step:

| source | format | class handling |
| --- | --- | --- |
| `vehicle.zip` | YOLO (`.jpg` + `.txt`), 256 imgs | all boxes remapped to `vehicle` (id 1) |
| `person.zip` | CVAT export (`annotations.json`) | rectangles (incl. rotated) → axis-aligned YOLO boxes, `person` (id 0) |

Unified class map everywhere downstream: **`person=0`, `vehicle=1`**.

## Repository layout

```
config/config.yaml              region, class map, instance types, mAP threshold, schedule
src/data/download.py            unpack person.zip + vehicle.zip -> data/raw
src/processing/preprocess.py    CVAT->YOLO, class remap, split, Ultralytics data.yaml
src/training/train.py           Ultralytics YOLOv8 entry point -> best.pt
src/processing/evaluate.py      mAP metrics -> evaluation.json property file
src/inference/{inference,serve}.py  handlers + Flask /ping,/invocations server
src/common/                     shared class map + config loader
docker/{base,preprocess,train,evaluate,inference}/  one image per step
scripts/build_and_push.sh       build/push images (context hash, --cache-from)
scripts/resolve_digests.py      tag -> repo@sha256 map -> build/images.json
scripts/verify_containers.sh    docker-run each image against the /opt/ml contract
pipelines/training_pipeline.py  the 5-step pipeline (mAP gate + RegisterModel)
pipelines/deploy.py             deploy approved package to the endpoint
pipelines/monitoring.py         drift alarm + CloudWatch dashboard
infra/ecr.yaml                  4 repositories
infra/endpoint.yaml             model + endpoint + autoscaling + alarms
infra/codepipeline.yaml         Source->Test->Images->Train->Deploy + IAM
infra/eventbridge.yaml          monthly retrain + approval deploy + drift Lambda
buildspecs/{test,images,pipeline,deploy}.yml
tests/                          unit + offline pipeline-definition tests
Makefile                        make data, images, train-local, test, pipeline, deploy, destroy
```

## Local workflow (no AWS needed)

```bash
make setup          # create .venv, install ultralytics/torch + sagemaker SDK
make data           # unpack the two zips into data/raw
make test           # fast unit tests (ruff + pytest, no GPU)
make train-local    # preprocess -> train -> evaluate on this machine
make test-all       # also run the slow 1-epoch train test
```

`make train-local` writes `artifacts/best.pt` and `artifacts/evaluation/evaluation.json`.

### Containers (needs Docker)

```bash
make images         # build base + 4 step images locally
make docker-verify  # run each image against its /opt/ml contract, incl. /ping + /invocations
```

`make docker-verify` proves the full chain: preprocess produces `data.yaml`,
train (via `SM_CHANNEL_DATASET`/`SM_MODEL_DIR`) produces `best.pt`, evaluate
produces `evaluation.json`, and the inference server answers `/ping` (200) and
`/invocations` (detections JSON).

## AWS deploy order

Set once:

```bash
export AWS_REGION=us-east-1
export ARTIFACT_BUCKET=your-bucket
export MODEL_PACKAGE_GROUP=pv-yolov8-models
```

1. **ECR + repos**: `make ecr`
2. **Images**: `ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text) \
   ARTIFACT_BUCKET=$ARTIFACT_BUCKET ./scripts/build_and_push.sh` then
   `python scripts/resolve_digests.py --account-id $ACCOUNT_ID` (writes `build/images.json`).
3. **Upload raw data**: `python src/data/download.py --upload-s3` (or point `InputDataUrl` at existing S3).
4. **Train + gate + register**: `make pipeline` (upserts, starts, waits). Exit code
   `2` means the mAP gate rejected the model — that is a data/model problem, not a build failure.
5. **Approve** the model package in the SageMaker Model Registry (console or CLI).
6. **Deploy**: approval fires the EventBridge rule → CodePipeline deploy stage,
   or run manually: `SAGEMAKER_ROLE_ARN=... make deploy`.
7. **Monitoring**: `make monitor` (drift alarm + dashboard; deploy also does this).

### One-shot CI/CD

`make cicd` deploys `infra/codepipeline.yaml` (needs a GitHub CodeConnections ARN
and repo id as parameters). Thereafter every push runs Test → Images → Train →
Deploy. `make events` deploys the monthly schedule, the approval-triggered
deploy, and the drift-triggered retrain Lambda.

## The quality gate

`evaluate.py` runs `model.val()` on the held-out `test` split and writes a
property file. The pipeline's `ConditionStep` reads
`object_detection_metrics.map50.value` and registers only when it is
`>= evaluation.map50_threshold` (default `0.40`, override with
`--map50-threshold` or the `Map50Threshold` pipeline parameter). Otherwise a
`FailStep` fires and nothing is registered.

## Reproducibility notes

- **Digests, not tags.** `build/images.json` maps each step to `repo@sha256:...`.
  The monthly EventBridge schedule re-runs a frozen definition, so a mutable
  `:latest` would silently drift. Changing an image is an explicit code change.
- **The model package is the deployable unit.** It records the model artifact,
  the inference image digest, and the evaluation metrics. Deploy re-resolves the
  latest **Approved** package from the registry rather than trusting an artifact.
- **EndpointConfig names carry the model version** (`pv-yolov8-<version>`), because
  the resource is immutable — otherwise CloudFormation reports "no changes".

## Verification checklist (AWS)

- Set `Map50Threshold` deliberately high and confirm the gate blocks registration (exit 2).
- Approve a package and confirm blue/green canary rollout on the endpoint.
- POST sample images and watch `SageMakerVariantInvocationsPerInstance` autoscale.
- Confirm data capture lands in `s3://$ARTIFACT_BUCKET/pv-yolov8/datacapture/`.
- Trip the detection-drift alarm and confirm the retrain Lambda starts the pipeline.

## Teardown

```bash
make destroy        # deletes endpoint / events / cicd stacks
```

ECR (`pv-yolov8-ecr`) is intentionally retained: the inference image referenced
by any model package must outlive every endpoint built from it. Delete it
manually once no package needs it.

## Requirements

- Python 3.10+ (local driver); images use Ultralytics' Python 3.10 base.
- A GPU is used automatically when present (`train.py` picks `cuda:0`, else CPU).
- Docker with the daemon running for `make images` / `make docker-verify`.
- AWS credentials with the permissions in `infra/codepipeline.yaml` for the cloud steps.
