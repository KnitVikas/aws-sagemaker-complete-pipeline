# pv-yolov8: Technical Interview Guide

## Purpose

This document explains the project as it is implemented and provides detailed
technical interview questions and model answers. The answers are written in the
first person so they can be adapted directly during an interview.

The project is an end-to-end MLOps system for a two-class object detector:

- Classes: `person` and `vehicle`
- Model family: Ultralytics YOLOv8
- Cloud: AWS
- Orchestration: SageMaker Pipelines and CodePipeline
- Registry: SageMaker Model Registry
- Serving: SageMaker real-time endpoint
- Infrastructure: CloudFormation
- Monitoring: CloudWatch plus SageMaker data capture

> Interview rule: distinguish between **implemented**, **configured**, and
> **planned** functionality. For example, operational endpoint monitoring is
> implemented, while automated detection-drift calculation is described but not
> yet implemented.

---

## 1. Executive explanation

### 30-second summary

I built a reproducible AWS MLOps pipeline for detecting people and vehicles in
images. It unifies CVAT and YOLO datasets, trains and evaluates YOLOv8 through a
SageMaker Pipeline, applies an mAP quality gate, and registers successful models
with `PendingManualApproval`. After approval, a deployment workflow provisions
an autoscaled SageMaker endpoint through CloudFormation. The system uses
digest-pinned containers, model registry versioning, data capture, CloudWatch
alarms, and event-driven retraining hooks.

### 90-second summary

The source data comes from two annotation formats: person annotations exported
from CVAT and vehicle labels already in YOLO format. A preprocessing container
converts both into one deterministic Ultralytics dataset with the class mapping
`person=0` and `vehicle=1`, then creates train, validation, and test splits.

A SageMaker Pipeline runs preprocessing, training, and held-out test evaluation.
The training job writes `best.pt` into the SageMaker model directory, which
SageMaker packages as `model.tar.gz` in S3. Evaluation produces
`evaluation.json`, including mAP@0.5, mAP@0.5:0.95, precision, recall, and
per-class AP. A condition step registers the model only when mAP@0.5 passes the
configured threshold; otherwise it deliberately fails through a rejection step.

CodePipeline validates the repository, builds four container images, pushes
them to ECR, resolves mutable tags to immutable digests, and executes the
SageMaker Pipeline. Registered models require manual approval. The intended
approval path starts a separate Source-and-Deploy pipeline, which resolves the
latest approved package and updates a SageMaker endpoint with CloudFormation.
The endpoint has request/response capture, target-tracking autoscaling, latency
and error alarms, and blue/green rollback controls.

### Architecture

```text
GitHub push
    |
    v
CodePipeline
    |
    +--> Test: Ruff + Pytest
    |
    +--> Images: build four containers --> ECR --> resolve image digests
    |
    +--> Train: upsert and execute SageMaker Pipeline
           |
           +--> Preprocess
           +--> Train YOLOv8 --> S3 model.tar.gz
           +--> Evaluate held-out test split --> evaluation.json
           +--> Quality gate
                    |
                    +--> pass: register PendingManualApproval
                    +--> fail: reject without registration

Manual approval
    |
    v
EventBridge --> approval deploy pipeline --> CloudFormation
                                            |
                                            v
                                  SageMaker real-time endpoint
                                  + data capture
                                  + autoscaling
                                  + CloudWatch alarms
```

---

## 2. Repository map

| Area | Important files | Responsibility |
|---|---|---|
| Configuration | `config/config.yaml` | Region, split, model, instances, gate, endpoint and schedule |
| Shared contracts | `src/common/schema.py`, `src/common/config.py` | Class IDs, artifact names and configuration loading |
| Data | `src/data/download.py`, `src/processing/preprocess.py` | Extract raw archives and create a unified YOLO dataset |
| Training | `src/training/train.py` | Train YOLOv8 and export `best.pt` |
| Evaluation | `src/processing/evaluate.py` | Evaluate on test data and write registry metrics |
| Pipeline | `pipelines/training_pipeline.py` | Build, upsert, run and inspect the SageMaker Pipeline |
| Inference | `src/inference/inference.py`, `src/inference/serve.py` | Decode requests, run YOLO and implement `/ping` and `/invocations` |
| Deployment | `pipelines/deploy.py`, `infra/endpoint.yaml` | Resolve approved model and deploy endpoint |
| CI/CD | `infra/codepipeline.yaml`, `buildspecs/*.yml` | Test, image build, training and deployment workflows |
| Events | `infra/eventbridge.yaml` | Approval deploy, scheduled retraining and drift-triggered retraining |
| Monitoring | `pipelines/monitoring.py` | Dashboard and drift alarm configuration |
| Images | `docker/*`, `scripts/build_and_push.sh` | Build step-specific containers |
| Reproducibility | `scripts/resolve_digests.py` | Convert ECR tags to immutable digest URIs |
| Tests | `tests/` | Data, evaluation, inference and offline pipeline-definition tests |

---

## 3. Current configuration worth knowing

The values below come from `config/config.yaml` and should be described as demo
configuration rather than universal best practices.

| Setting | Current value | Interview implication |
|---|---:|---|
| Region | `us-east-1` | All resources must be checked in the same region |
| Base model | `yolov8n.pt` | Small model optimized for speed and demo cost |
| Input size | 640 | Standard YOLO image size |
| Epochs | 5 | Intentionally small; insufficient for production convergence |
| Training instance | `ml.m5.xlarge` | CPU training because GPU training quota was unavailable |
| Processing instance | `ml.m5.xlarge` | Preprocessing and evaluation run on CPU |
| Split | 70/15/15 | Train/validation/test |
| Split seed | 42 | Deterministic split |
| Gate metric | mAP@0.5 | Registration decision |
| Gate threshold | `0.005` | Demo-only and too low for a production release gate |
| Approval | `PendingManualApproval` | Human controls production promotion |
| Endpoint instance | `ml.g4dn.xlarge` | NVIDIA T4 GPU endpoint |
| Endpoint capacity | 1–4 | Target-tracking horizontal scaling |
| Scaling target | 50 invocations/instance/minute | Current target value |
| Data capture | 100% | Useful for testing, expensive and sensitive in production |

### Known code/documentation inconsistencies

These are useful interview discussion points rather than details to hide:

| Topic | Documentation/intention | Current implementation |
|---|---|---|
| Quality threshold | README describes a default mAP@0.5 gate of `0.40` | `config/config.yaml` currently sets `0.005` |
| Deployment strategy | README verification text refers to a canary rollout | Endpoint uses blue/green infrastructure with `ALL_AT_ONCE`, not gradual canary traffic |
| Main CI Deploy stage | Promotion is conceptually approval-driven | The main Git-push pipeline still has a Deploy stage, which selects the latest previously approved package; on the first run it can fail if none is approved |
| Drift monitoring | README depicts Model Monitor → drift → retrain | No Model Monitor schedule or drift metric producer exists; only capture, dashboard, alarm, and alarm-response wiring exist |
| Gate rejection | Exit code `2` distinguishes quality rejection from runtime failure | CodeBuild still treats any nonzero exit as a failed stage unless the buildspec handles `2` specially |
| Branch default | CloudFormation parameter defaults to `main` | The deployed repository has used `master`; deployment must override the parameter |
| Monitoring schedule | `config.yaml` declares an hourly monitoring schedule | No code currently consumes that setting |
| Container smoke test | Inference supports `YOLO_DEVICE=cpu` | `verify_containers.sh` does not explicitly set it |

---

# Technical interview questions and answers

## 4. Project and architecture

### Q1. What problem does this project solve?

It automates the lifecycle of a person-and-vehicle object detector: heterogeneous
data preparation, training, evaluation, registration, approval, deployment,
scaling, and operational monitoring. The main value is not only the YOLO model;
it is repeatability and controlled promotion of a specific model artifact and
inference environment.

### Q2. Why use SageMaker Pipelines and CodePipeline together?

They solve different orchestration problems:

- CodePipeline is the software delivery workflow. It reacts to source changes,
  runs tests, builds containers, and starts ML work.
- SageMaker Pipelines is the ML workflow. It understands Processing jobs,
  Training jobs, model artifacts, property files, conditions, and registration.

Keeping those responsibilities separate makes the ML graph reusable for manual,
scheduled, CI-triggered, or drift-triggered training.

### Q3. What are the main design principles?

1. One canonical class schema across all steps.
2. Separate containers for preprocessing, training, evaluation, and inference.
3. Immutable ECR digests in the SageMaker Pipeline definition.
4. Held-out evaluation before registration.
5. Manual approval before production promotion.
6. Infrastructure as code for repeatable deployment.
7. CloudWatch metrics and rollback alarms around serving.
8. Local and cloud execution paths using the same Python entry points.

### Q4. Why not build one large container for every step?

Separate images reduce coupling and let each step evolve independently. A
preprocessing image does not need Torch, while training and evaluation do. The
inference image includes only the serving contract. This improves fault
isolation and makes image digests part of the lineage. The tradeoff is more
image storage and CI build time.

### Q5. What is the unit of production deployment?

The SageMaker Model Package is the deployable unit. It binds:

- S3 model artifact
- inference image digest
- accepted content and response types
- supported inference instance type
- evaluation metrics
- metadata such as source commit
- approval status

Deploying by Model Package ARN prevents accidentally combining weights from one
run with an image from another.

### Q6. Is the system fully event driven?

Partly. Git pushes start CodePipeline. Model approval is designed to start a
deploy-only pipeline. CloudWatch alarm state changes can invoke a retraining
Lambda, and a scheduler can start monthly training. However, the job that should
calculate and publish the custom detection-drift metric is currently missing,
so that particular event chain is incomplete.

### Q7. What happens when a component fails?

- Test failure stops before image builds.
- Image build or ECR failure stops before training.
- Processing or training failure fails the SageMaker Pipeline.
- A quality failure intentionally enters `RejectModel`, with a separate CLI
  exit code (`2`) from infrastructure/runtime failure (`1`).
- Endpoint rollout watches 5xx and p99 latency alarms and can roll back.
- Pipeline failure can publish an SNS notification through EventBridge.

### Q8. How would you explain this project on a whiteboard?

I draw two planes:

- **Build plane:** source → tests → images → preprocess → train → evaluate →
  quality gate → registry.
- **Serve plane:** approval → endpoint deployment → inference → capture →
  metrics → retraining signal.

I then show S3 as the artifact/data backbone, ECR as the runtime backbone, and
CloudFormation as the endpoint control plane.

---

## 5. Data engineering and preprocessing

### Q9. What data formats are supported?

The vehicle source is already YOLO text annotation format. The person source is
a CVAT export containing `annotations.json`, `manifest.jsonl`, and image files.
`preprocess.py` converts both to one Ultralytics directory structure:

```text
images/{train,val,test}/
labels/{train,val,test}/
data.yaml
metadata/class_counts.json
```

### Q10. How is class consistency enforced?

`src/common/schema.py` defines one mapping:

```text
person  = 0
vehicle = 1
```

Vehicle input class IDs are deliberately discarded and remapped to `vehicle=1`.
CVAT rectangles are mapped to `person=0`. Training, evaluation, and inference
all import the same constants, preventing label drift between steps.

### Q11. How are CVAT rectangles converted?

The preprocessor:

1. maps CVAT frame indices to filenames using `manifest.jsonl`;
2. groups annotations by frame;
3. processes rectangle shapes;
4. converts rotated rectangles into their axis-aligned bounding box;
5. clamps the result to image dimensions;
6. drops boxes of one pixel or less;
7. converts coordinates to normalized YOLO `(cx, cy, width, height)`.

### Q12. What information is lost for rotated boxes?

Rotation is lost because the implementation converts an oriented rectangle into
an axis-aligned bounding box. This is appropriate for a standard YOLO detection
model but may include more background and reduce localization quality. If
orientation is important, I would use oriented bounding-box training instead.

### Q13. How are train, validation, and test splits generated?

All collected samples are shuffled with a local `random.Random(seed)` instance
and split according to the configured ratios. The seed is 42, so repeated runs
with the same ordered inputs produce the same split.

### Q14. Is there any risk of data leakage?

Yes. Splitting is currently image-level. Frames from the same source video may
appear in train and test, which can inflate metrics due to near-duplicate
backgrounds and temporal neighbors. A production implementation should group by
video, camera, site, or time window before splitting.

### Q15. How are filename collisions handled?

Output files use a split-specific sequential stem such as `train_000001`,
rather than preserving the raw basename. This prevents sources with identical
filenames from overwriting each other and guarantees each image has a
corresponding label file.

### Q16. Are negative images supported?

Yes. Every image is written with a label file, even when it contains no boxes.
An empty YOLO label file represents a valid negative example. This is useful for
reducing false positives, provided the negatives are correctly sampled.

### Q17. What is `class_counts.json` for?

It records image and box counts by split and class. It can support validation,
lineage, class-imbalance analysis, and a future detection-distribution drift
baseline. The file is generated, but the production drift processor that
consumes it is not implemented.

### Q18. How would you improve data quality?

I would add:

- group-aware train/test splitting;
- duplicate and near-duplicate detection;
- corrupt-image validation;
- minimum/maximum box-size checks;
- annotation visualization samples;
- class and camera distribution reports;
- label-version metadata;
- schema checks for unexpected CVAT labels;
- a dataset manifest with source checksums.

---

## 6. Model training

### Q19. Why YOLOv8?

YOLOv8 offers a practical accuracy/latency tradeoff, straightforward transfer
learning, mature tooling, and easy export/deployment. It is suitable for a
real-time security-camera detector where throughput matters. The code uses
`yolov8n.pt`, the smallest variant, to reduce training and serving cost.

### Q20. What training parameters are used?

The current defaults are 640-pixel input, 5 epochs, batch 8, two workers, and
patience 15. Training begins from `yolov8n.pt`. Those are demo values; five
epochs and a tiny dataset are not enough evidence of production quality.

### Q21. How does the training code support both local and SageMaker execution?

It detects SageMaker paths and environment variables:

- `SM_CHANNEL_DATASET` for the dataset channel
- `SM_MODEL_DIR` for model output
- `/opt/ml/output/failure` for failure details

Outside SageMaker it defaults to `data/processed` and `artifacts`. The same
entry point therefore works locally, in Docker, and in a SageMaker Training job.

### Q22. Why is `data.yaml` rewritten?

The preprocessor writes an absolute dataset path valid inside the preprocessing
container. SageMaker mounts the resulting S3 dataset at another path in the
training and evaluation jobs. Since input channels are read-only, the code
writes a temporary `data.yaml` whose `path` points to the actual mount.

### Q23. How is the best model artifact selected?

Ultralytics writes its best checkpoint under
`runs/train/weights/best.pt`. The training entry point copies that checkpoint to
`SM_MODEL_DIR/best.pt`. SageMaker automatically archives the model directory
into `model.tar.gz` and uploads it to the configured S3 training output path.

### Q24. Where is the cloud artifact stored?

The estimator output base is:

```text
s3://<bucket>/pv-yolov8/pv-yolov8-training/training/
```

SageMaker appends the training job name and
`output/model.tar.gz`. The registered model package records the exact
`ModelDataUrl`.

### Q25. How is the training device selected?

Training checks `torch.cuda.is_available()` and chooses GPU device `0` if
available, otherwise CPU. The current configuration uses `ml.m5.xlarge`, so the
cloud training job is CPU-based. This was a quota/cost decision, not an optimal
training architecture.

### Q26. What metrics does training emit?

The entry point prints final validation mAP@0.5 and mAP@0.5:0.95 in a format
matched by the Estimator's metric regexes. SageMaker records these in the
Training job's final metrics and CloudWatch.

### Q27. How would you make experiments comparable?

I would persist:

- dataset manifest/version and checksum;
- source commit;
- all hyperparameters;
- image digests;
- random seeds;
- instance and library versions;
- training and evaluation metrics;
- model package ARN.

Most runtime lineage already exists through digest URIs and package metadata,
but formal experiment tracking and dataset versioning should be added.

---

## 7. Evaluation and quality gates

### Q28. Which dataset is used for the release decision?

The evaluation container explicitly runs `model.val(..., split="test")` on the
held-out test split. Validation metrics produced during training are useful for
optimization, but the pipeline condition reads the separate test report.

### Q29. What does `evaluation.json` contain?

It contains:

- mAP@0.5
- mAP@0.5:0.95
- mean precision
- mean recall
- per-class AP@0.5

The structure is deliberately compatible with the SageMaker property-file JSON
path used by the condition step.

### Q30. What is mAP@0.5?

For each class, predictions are matched to ground-truth boxes at an Intersection
over Union threshold of 0.5. The precision-recall curve produces Average
Precision, then AP is averaged across classes. mAP@0.5 is relatively lenient
about localization compared with mAP@0.5:0.95.

### Q31. Why use mAP@0.5 as the gate?

It is intuitive and suitable for a first detector gate. However, a production
security system should combine it with per-class recall, false positives per
camera-hour, small-object performance, day/night slices, and latency.

### Q32. What is the current threshold, and is it acceptable?

The threshold is `0.005`, or 0.5% mAP@0.5. It is only useful to demonstrate the
conditional workflow and is not a meaningful production standard. I would
calibrate a threshold against a validated baseline and business error costs.

### Q33. What happens when the gate fails?

The `ConditionStep` follows `RejectModel`, a SageMaker `FailStep`. Nothing is
registered. The waiting CLI recognizes that step and returns exit code `2` so
CI can distinguish a model-quality rejection from a broken build.

### Q34. Why register evaluation metrics with the model?

Model Registry then contains the artifact, image, supported interfaces, and
quality evidence in one governed object. Reviewers can inspect the metrics
before changing approval status, and deployment can be traced back to the
evaluation report.

### Q35. What evaluation weaknesses remain?

- image-level split leakage risk;
- no confidence-threshold tuning study;
- no confusion matrix gate;
- no per-camera/day-night slices;
- no false-positive-rate requirement;
- no robustness or adversarial tests;
- no latency/throughput release gate;
- no statistical comparison with the currently deployed model.

---

## 8. SageMaker Pipeline

### Q36. What are the steps?

The top-level graph is:

1. `Preprocess`
2. `Train`
3. `Evaluate`
4. `QualityGate`
   - `RegisterModel` on pass
   - `RejectModel` on failure

`RegisterModel` and `RejectModel` are conditional branches rather than
top-level sequential steps.

### Q37. What pipeline parameters can be overridden at runtime?

- input S3 URI;
- processing instance type;
- training instance type;
- training instance count;
- mAP@0.5 threshold;
- model approval status.

This allows scheduled or manual runs to change operational values without
rebuilding the graph.

### Q38. How is data passed between steps?

The preprocessing output is uploaded to S3 and referenced through the pipeline
property `S3Output.S3Uri`. Training consumes it as an `S3Prefix` channel.
Evaluation consumes both the same dataset URI and the Training step's
`S3ModelArtifacts`. Registration references the same model artifact and the
evaluation report URI.

### Q39. How does step caching work?

Preprocessing, training, and evaluation use SageMaker `CacheConfig` with a
30-day expiry. Static output prefixes allow SageMaker's step signature to match
when inputs and arguments are unchanged. Cache reuse saves cost but requires
careful digest and data URI management.

### Q40. What could make caching unsafe?

Mutable container tags, mutable data at the same S3 URI, hidden external state,
or incomplete step arguments can produce stale reuse. This project mitigates
the image problem by digest-pinning containers, but raw S3 data still needs
formal versioning or immutable prefixes/checksums.

### Q41. Why use a `PropertyFile`?

It makes `evaluation.json` available to pipeline expressions without writing
custom glue code. `JsonGet` reads
`object_detection_metrics.map50.value`, and the condition compares it with the
runtime gate threshold.

### Q42. Why use `PipelineSession`?

`PipelineSession` records SDK calls as pipeline step arguments rather than
running jobs immediately. This lets the code generate a declarative SageMaker
Pipeline definition.

### Q43. How is the pipeline tested without AWS jobs?

The tests construct the graph with fake digest URIs, a fake role, and a
`PipelineSession`, serialize the definition, and assert:

- expected step names and order;
- quality-gate branches;
- exposed parameters;
- digest references;
- absence of `:latest`;
- one dataset training channel.

### Q44. What is a weakness in package metadata?

`customer_metadata_properties.git_sha` is read from an environment variable
when the graph is constructed. Depending on the execution path, observed model
packages have contained `local`, reducing lineage quality. The commit should be
an explicit pipeline parameter or immutable execution metadata.

---

## 9. Containers and reproducibility

### Q45. Which images are built?

Four ECR images:

1. preprocessing;
2. training;
3. evaluation;
4. inference.

A small local base image supports preprocessing but is not an independent
pipeline step.

### Q46. Why build tags and then resolve digests?

Tags are convenient for building and identifying a commit, but mutable.
`resolve_digests.py` asks ECR for each pushed image's SHA-256 digest and writes
`build/images.json` using `repo@sha256:...`. The SageMaker Pipeline consumes
those immutable URIs.

### Q47. What lineage guarantee does digest pinning provide?

Re-running an unchanged pipeline definition later uses the exact same
container bytes, even if a tag such as `latest` or a commit tag has been moved.
It does not by itself version input data, external base-image provenance before
build, or AWS service behavior.

### Q48. How does the build script reduce rebuild work?

It computes a context hash over the Dockerfile plus copied source/config and
adds it as an image label. It also passes the existing local `latest` image as
Docker cache input. In ephemeral CodeBuild environments, however, local cache
benefits may be limited unless registry or CodeBuild caching is configured.

### Q49. How are ECR images retained?

Preprocess, train, and evaluate repositories expire untagged images after 14
days. The inference repository intentionally retains untagged images because
model packages reference inference images by digest and a deployed or historical
package must not lose its runtime.

### Q50. How are container contracts tested?

`verify_containers.sh` runs the full local sequence:

- preprocess and verify `data.yaml`;
- one-epoch CPU train and verify `best.pt`;
- evaluate and verify `evaluation.json`;
- start the inference server;
- check `/ping`;
- send a test image to `/invocations`.

This is stronger than unit testing handlers alone because it validates mount
paths and entry points.

---

## 10. CI/CD, registry, and deployment

### Q51. What happens after a Git push?

The main CodePipeline runs:

```text
Source → Test → Images → Train → Deploy
```

The Train CodeBuild project upserts and starts the SageMaker Pipeline, waits for
completion, and writes model package metadata to a build artifact.

### Q52. Why is the new package not immediately production-ready?

Registration defaults to `PendingManualApproval`. Training quality gates answer
“does it meet automated criteria?” Manual approval answers “has an authorized
person accepted the release risk?” Those are separate controls.

### Q53. How is a model approved?

An authorized principal changes `ModelApprovalStatus` from
`PendingManualApproval` to `Approved` in SageMaker Model Registry or through
`UpdateModelPackage`. This requires `sagemaker:UpdateModelPackage`.

### Q54. How is approval supposed to trigger deployment?

An EventBridge rule listens for a SageMaker Model Package state-change event
whose group is `pv-yolov8-models` and status is `Approved`. It targets the
approval deployment pipeline, designed as:

```text
Source (DetectChanges=false) → Deploy
```

The source stage supplies `infra/endpoint.yaml` and the buildspec; it should not
retrain.

### Q55. Why was a separate approval pipeline needed?

Calling `codepipeline:StartPipelineExecution` always starts a pipeline from its
first stage. EventBridge cannot directly jump to the Deploy stage of the full
pipeline. Targeting the original pipeline caused approval to rebuild, retrain,
and potentially supersede another execution. A deploy-only pipeline separates
promotion from model creation.

### Q56. How does deployment choose a package?

`deploy.py` and `buildspecs/deploy.yml` list model packages filtered by
`Approved`, sorted by creation time descending, and select the first result.
This protects deployment from a stale upstream ARN but means “latest approved”
is global to the group rather than explicitly bound to the approval event.

### Q57. What race condition exists in “latest approved” resolution?

If two versions are approved close together, a deployment triggered for the
older event may resolve the newer package. For exact event semantics, the
EventBridge event should pass the Model Package ARN into a Lambda, Step
Functions workflow, or pipeline variable, and deployment should verify that
exact package is still approved.

### Q58. How does CloudFormation update the endpoint?

It creates:

- `AWS::SageMaker::Model` named with the package version;
- immutable EndpointConfig named with the version;
- a stable Endpoint name;
- scaling target and policy;
- latency and 5xx alarms.

Versioned model/config names force a real endpoint update when the package
changes. Re-running with the same template and parameters produces an empty
change set and safely succeeds.

### Q59. What rollout strategy is used?

The endpoint uses SageMaker blue/green deployment with `ALL_AT_ONCE` traffic
routing, a 60-second termination wait, and a 30-minute maximum execution
timeout. It is operationally blue/green but not a gradual canary. A canary would
require enough initial capacity and an explicit traffic split.

### Q60. What causes automatic rollback?

CloudFormation supplies the endpoint update with rollback alarms for:

- one or more 5xx errors in a one-minute period;
- p99 model latency greater than two seconds for three periods.

These alarms protect rollout health, not model prediction quality.

### Q61. What deployment failure occurred with the GPU image?

The v8 image failed `/ping` because its PyTorch CUDA build required a newer
NVIDIA driver than the `ml.g4dn.xlarge` serving host exposed. CloudWatch logged:

```text
CUDA initialization: NVIDIA driver ... too old
torch.cuda.is_available() is False
```

The endpoint update rolled back to the prior configuration. The lesson is to
validate CUDA, PyTorch, container, and SageMaker instance compatibility before
promotion—not merely force `device=0`.

### Q62. How should that CUDA issue be fixed?

I would select or build an inference image whose CUDA/PyTorch runtime is
compatible with the SageMaker host driver, then test it on the target instance
family. I would add startup logs for Torch/CUDA versions, device name, and
`torch.cuda.is_available()`, plus an integration test endpoint before production
approval.

---

## 11. Inference and API contract

### Q63. What request formats are accepted?

- raw JPEG;
- raw PNG;
- `application/octet-stream` image bytes;
- JSON containing a base64 `image` field.

The response is JSON:

```json
{
  "detections": [
    {
      "class_id": 0,
      "label": "person",
      "confidence": 0.91,
      "box_xyxy": [10.0, 20.0, 100.0, 200.0]
    }
  ]
}
```

### Q64. Can the endpoint accept a video file directly?

No. The current contract processes one image per invocation. A video client or
stream-processing layer must sample/decode frames, invoke the detector, and
optionally add tracking. Sending MP4 bytes as an image is unsupported.

### Q65. How is the server implemented?

SageMaker starts the custom image with the `serve` argument. A small launcher
starts Gunicorn with one worker on port 8080. Flask implements:

- `GET /ping` to lazily load/check the model;
- `POST /invocations` to decode, predict, and serialize.

One worker avoids loading multiple copies of the model into GPU memory.

### Q66. How does device selection work?

The current inference code supports:

- `YOLO_DEVICE=cpu` for explicit local CPU execution;
- `YOLO_DEVICE=0` for explicit GPU;
- automatic CUDA selection when Torch reports availability;
- failure instead of silent CPU fallback when a GPU appears visible but CUDA
  cannot initialize.

The fail-fast behavior is valuable because paying for a GPU while serving on
CPU is an expensive hidden degradation.

### Q67. What was observed during load testing?

A 15-minute sequential client run issued 10,179 requests with zero client
errors, approximately 11.3 requests per second. Before the strict GPU update,
model latency was roughly 82 ms average and 121 ms p99, with GPU utilization at
zero—evidence that the process was CPU-bound despite using a GPU endpoint.

### Q68. What input validation is missing?

The service should add:

- maximum body and decoded image dimensions;
- decompression-bomb protection;
- malformed image handling with clearer 4xx errors;
- request IDs and structured logs;
- configurable confidence/IoU policy rather than untrusted per-request values;
- schema/version field in responses;
- batch behavior if required.

### Q69. How is the endpoint authenticated?

The Flask server is not directly internet-facing. Calls go through SageMaker
Runtime and require AWS IAM permission such as `sagemaker:InvokeEndpoint`.
Network isolation, VPC endpoints, KMS keys, and resource-scoped IAM should be
added for a stricter production environment.

---

## 12. Autoscaling and performance

### Q70. How does the endpoint scale?

Application Auto Scaling changes
`sagemaker:variant:DesiredInstanceCount` between 1 and 4. Target tracking uses
`SageMakerVariantInvocationsPerInstance` with target 50, 60-second scale-out
cooldown, and 300-second scale-in cooldown.

### Q71. Does it scale processes or instances?

Instances. The container intentionally has one Gunicorn worker because each
worker would hold another model copy. When load increases, SageMaker adds
endpoint instances and distributes requests across them.

### Q72. Is 50 requests per instance a good target?

It is not yet proven. It must be calibrated through concurrent load tests using
the representative image size, desired latency SLO, GPU utilization, queue
time, and error rate. A sequential load generator does not fully exercise
concurrency or scaling behavior.

### Q73. Why can autoscaling still produce latency spikes?

Target tracking is reactive. Metrics arrive in periods, then a new GPU instance
must launch and load the model. Sudden bursts can exceed one instance before
scale-out completes. Mitigations include higher minimum capacity, scheduled
scaling, faster model startup, asynchronous inference, or request buffering.

### Q74. What is the maximum throughput?

The code does not establish a production maximum. The observed sequential rate
is a client-side measurement, not a capacity test. I would run staged concurrent
tests, record p50/p95/p99 latency and errors, identify saturation per instance,
then choose the autoscaling target and quota.

### Q75. What quota affects scaling?

The regional SageMaker real-time endpoint quota for the chosen instance family.
`MaxCapacity=4` is only the application's ceiling. Effective capacity is the
minimum of the scaling maximum, Service Quota, and available regional capacity.

---

## 13. Monitoring and retraining

### Q76. What operational metrics are monitored?

The dashboard includes:

- invocations;
- p99 model latency;
- 4xx and 5xx errors;
- GPU and GPU-memory utilization;
- training mAP history;
- a placeholder custom detection-drift metric.

CloudFormation separately creates 5xx and latency alarms.

### Q77. What data is captured?

The endpoint configuration captures inputs and outputs to:

```text
s3://<artifact-bucket>/pv-yolov8/datacapture/
```

The current sampling percentage is 100%. That is useful during validation but
should be reconsidered for privacy, retention, S3 cost, and throughput.

### Q78. Is SageMaker Model Monitor fully implemented?

No. The project does not create a SageMaker Monitoring Schedule. It creates
data capture, a CloudWatch dashboard, and a custom drift alarm. The module
documentation says a scheduled processor should parse captured responses and
compare detection counts/class mix with `class_counts.json`, but no such
processor or schedule exists.

### Q79. What happens to the missing drift metric?

The drift alarm uses `TreatMissingData="notBreaching"`. Therefore, no datapoints
leave the alarm in `OK`; it will not trigger retraining. This is safe from false
retraining loops but can hide a broken monitoring pipeline. A separate
“metric missing” alarm is needed.

### Q80. How is drift-triggered retraining intended to work?

1. A processor publishes `DetectionRateDrift`.
2. Three hourly averages above 0.30 put the alarm into `ALARM`.
3. EventBridge matches that state change.
4. A Lambda starts the SageMaker training pipeline with
   `PendingManualApproval`.
5. The new model must still pass evaluation and receive approval.

Steps 2–5 are configured; step 1 is missing.

### Q81. Is detection-rate drift enough?

No. Prediction distribution drift can indicate a change but cannot prove model
quality degradation. With delayed ground truth, I would also monitor precision,
recall, false-positive rate, miss rate, and performance by camera/time slice.
I would monitor input characteristics such as brightness, blur, resolution,
camera angle, and embedding distribution.

### Q82. What is the periodic retraining policy?

`infra/eventbridge.yaml` defines a monthly SageMaker Pipeline schedule at 03:00
UTC on the first day, using `PendingManualApproval`. There is also a config
schedule value, so one source of truth should be enforced to avoid drift.

### Q83. What monitoring implementation detail is risky?

`buildspecs/deploy.yml` runs `monitoring.py ... || true`. Dashboard or alarm
creation can fail while deployment still reports success. In production, I
would make required alarms fail deployment or split optional dashboard setup
from mandatory rollback controls.

---

## 14. Security and IAM

### Q84. Which IAM roles are used?

- CodeBuild role: logs, S3, ECR, SageMaker pipeline/model operations,
  CloudFormation deployment, endpoint operations, autoscaling, CloudWatch, and
  pass-role.
- SageMaker execution role: assumed by jobs and endpoint, with S3/ECR access.
- CodePipeline role: artifact access, CodeBuild invocation, and source
  connection use.
- EventBridge invocation role: starts the approval pipeline.
- Scheduler role: starts the SageMaker Pipeline.
- Drift Lambda role: logs and starts retraining.

### Q85. Is least privilege fully achieved?

No. Several statements use `Resource: "*"`, the SageMaker execution role uses
`AmazonSageMakerFullAccess`, and application autoscaling is broad. This is
acceptable for a tutorial but not a least-privilege production posture.

### Q86. How would you tighten IAM?

- scope model package actions to the package group ARN;
- scope pipeline actions to the pipeline ARN;
- scope ECR to the four repository ARNs;
- scope S3 to exact prefixes;
- separate build, training, and deployment roles;
- scope CloudFormation and endpoint actions by naming convention/tags;
- constrain `iam:PassRole` with service conditions;
- use permission boundaries and SCPs where appropriate.

### Q87. What is wrong with `sagemaker-model-registry-policy.json`?

It contains a trailing comma after `sagemaker:CreateModelPackage`, which makes
it invalid strict JSON. It also uses `Resource: "*"`. It should be validated and
scoped before use.

### Q88. What data-protection controls are missing?

- explicit KMS keys for S3, model artifacts, logs, and endpoint storage;
- VPC-only endpoint access/private networking;
- S3 lifecycle and retention for captured images;
- PII/privacy controls for security-camera imagery;
- audit/reporting around who approved model packages;
- image vulnerability policy enforcement after ECR scanning;
- signed image/provenance controls.

### Q89. Does ECR scanning block vulnerable images?

No. Repositories have scan-on-push enabled, but the CI workflow does not inspect
scan findings or fail on severity thresholds. Scanning is configured, not
enforced.

---

## 15. Testing strategy

### Q90. What tests run in CI?

Ruff and Pytest tests not marked `slow` or `docker`. The suite covers:

- class conversion and split output;
- normalized YOLO labels;
- class-count metadata;
- evaluation report shape;
- model tar extraction;
- inference input/output handlers and device selection;
- offline SageMaker Pipeline structure.

### Q91. What tests do not run in normal CI?

Slow training tests and Docker contract tests are excluded. The standard CI
therefore does not prove that newly built images start successfully on the
actual SageMaker target instance.

### Q92. Which test would have caught the GPU deployment failure?

An integration test that launches the inference image on a SageMaker
`ml.g4dn.xlarge` staging endpoint, invokes `/ping` and one prediction, then
checks GPU utilization or startup device logs. A local CPU Docker smoke test
cannot validate host driver compatibility.

### Q93. How would you structure test levels?

1. Unit tests for transforms and handlers.
2. Dataset contract tests.
3. Offline pipeline-definition tests.
4. Docker contract tests.
5. Small cloud integration jobs.
6. Temporary staging endpoint test on the target instance.
7. Load and rollback tests.
8. Shadow/canary validation before production.

---

## 16. Cost and operations

### Q94. What are the largest cost drivers?

- always-on GPU endpoint instances;
- training and evaluation compute;
- repeated Docker builds;
- retained inference images;
- 100% data capture;
- possible Studio/DataZone networking such as NAT gateways and interface
  endpoints.

The GPU endpoint dominates when left running.

### Q95. How does the project control cost?

- tiny YOLO model;
- CPU training due quota/cost constraints;
- one initial endpoint instance;
- target-tracking scale-in;
- step caching;
- untagged ECR expiration for non-inference images;
- explicit endpoint teardown through CloudFormation.

### Q96. What cost controls should be added?

- AWS Budgets and anomaly alerts;
- mandatory tags and cost allocation;
- development endpoint schedules or serverless/asynchronous alternatives;
- S3 lifecycle rules;
- ECR retention by count for tagged images;
- data-capture sampling/retention policy;
- CodeBuild caching;
- concurrency controls on retraining;
- automatic temporary endpoint cleanup.

### Q97. Why not scale the endpoint to zero?

Standard SageMaker real-time endpoint autoscaling uses a minimum instance count
of at least one in this configuration. For intermittent workloads, I would
consider Serverless Inference if compatible, Asynchronous Inference with
scale-to-zero support, batch processing, or explicit create/delete scheduling.

### Q98. How do you safely stop billing?

Delete the CloudFormation endpoint stack so it removes the endpoint, endpoint
configuration, model resource, alarms, and scaling resources consistently.
Direct endpoint deletion may leave stack drift and supporting resources. ECR,
S3, logs, CI/CD, schedules, and Studio networking must be reviewed separately.

---

## 17. Troubleshooting scenarios

### Q99. A model is registered but not deployed. What do you check?

1. Is `ModelPackageStatus=Completed`?
2. Is `ModelApprovalStatus=Approved`?
3. Did the EventBridge rule match?
4. Is its target pipeline present and enabled?
5. Can the EventBridge role call `StartPipelineExecution`?
6. Did Deploy resolve the intended package?
7. Did CloudFormation create/update the stack?
8. Did `/ping` pass?
9. Did rollback alarms fire?

### Q100. Why can `ModelPackageStatus=Completed` still show pending approval?

They represent different states. `Completed` means the package resource was
created successfully. `PendingManualApproval` means it has not been authorized
for deployment.

### Q101. A CodePipeline execution is `Superseded`. Was approval lost?

No. The model package remains approved. `Superseded` means a newer pipeline
execution replaced an older one under CodePipeline's execution mode. Separating
approval deployment from the full source pipeline prevents unrelated pushes
from cancelling promotion work.

### Q102. CloudFormation Deploy succeeds but the endpoint does not update. Why?

If the latest approved package, model version, and template parameters are
unchanged, CloudFormation creates an empty change set. Because deployment uses
`--no-fail-on-empty-changeset`, the job correctly succeeds without replacing
the endpoint.

### Q103. The endpoint is on a GPU instance but GPU utilization is zero. Why?

Possible causes:

- Torch is CPU-only;
- CUDA runtime and host driver are incompatible;
- device was not selected;
- traffic is too low or metric timing is misleading;
- preprocessing dominates CPU;
- the model silently fell back to CPU.

I inspect startup logs, Torch/CUDA versions, `torch.cuda.is_available()`, device
name, model device, latency, and CloudWatch GPU metrics.

### Q104. Why do training images detect objects but a new camera video does not?

This indicates distribution shift rather than endpoint failure. Camera angle,
night illumination, object scale, compression, and scene composition may differ
from the small training set. I verify the API on known test images, inspect
confidence scores, create a representative camera dataset, relabel, retrain,
and evaluate by camera/domain.

### Q105. Why did CloudFormation need `GetTemplateSummary`?

`aws cloudformation deploy` performs template/parameter analysis before creating
the change set. The CodeBuild deployment role originally lacked
`cloudformation:GetTemplateSummary`, so deployment failed before endpoint
creation. The fix was adding that permission to the deployment role.

### Q106. What if two training runs happen concurrently?

They can compete for quota and both register versions. “Latest approved”
deployment may become ambiguous. I would add execution concurrency policy,
idempotency tokens, explicit package-ARN promotion, and possibly a release
queue.

---

## 18. Senior-level critical assessment

### Q107. What parts are production-oriented?

- infrastructure as code;
- digest-pinned step images;
- separate held-out evaluation;
- conditional registration;
- manual approval;
- model package deployment;
- blue/green rollback alarms;
- autoscaling;
- request/response capture;
- offline graph tests and container contract tests.

### Q108. What parts remain tutorial-grade?

- tiny dataset and five training epochs;
- extremely low quality threshold;
- possible frame-level data leakage;
- broad IAM permissions;
- incomplete drift calculation;
- no staging endpoint integration test;
- CUDA image compatibility not pinned;
- full data capture without retention/privacy policy;
- no exact package ARN propagation from approval event;
- no encryption/network configuration;
- no automatic ECR scan enforcement.

### Q109. What would you prioritize next?

1. Fix and pin the CUDA-compatible inference image.
2. Add a staging endpoint smoke test on the target GPU family.
3. Raise and justify quality gates using representative camera data.
4. Replace image-level split with video/camera-group splitting.
5. Implement detection/input drift processing and missing-metric alarms.
6. Make approval deploy the exact package ARN.
7. Tighten IAM, encryption, network, and retention controls.
8. Benchmark concurrency and calibrate autoscaling.

### Q110. How would you evolve this into a surveillance platform?

I would keep this detector as the perception service, then add:

- RTSP/Kinesis Video ingestion;
- frame sampling and batching;
- ByteTrack for within-camera tracking;
- OCR/ANPR and Re-ID where legally permitted;
- an event schema and event bus;
- S3 clip evidence with hashes and retention;
- OpenSearch for time/camera/object queries;
- a policy engine for zones, schedules, loitering, and alerts;
- an investigation interface grounded in event and clip evidence.

LLM agents should operate on structured events and evidence, not sit in the
real-time frame path.

---

## 19. Questions to ask the interviewer

1. What are the target cameras, image resolutions, FPS, and day/night mix?
2. Which error is more expensive: missed detections or false alarms?
3. What is the required p95/p99 alert latency?
4. Is traffic continuous, bursty, or scheduled?
5. When and how does ground truth become available?
6. What are the retention, privacy, and regional-compliance requirements?
7. Is manual approval mandatory for every production model?
8. What is the expected recovery strategy after a bad model release?
9. What GPU quota and cost envelope are available?
10. Should deployment select the latest approved model or an explicit release?
11. Is cross-account training and production separation required?
12. What evidence must be retained for audit or investigation?

---

## 20. Useful operational commands

### Check package statuses

```bash
aws sagemaker list-model-packages \
  --region us-east-1 \
  --model-package-group-name pv-yolov8-models \
  --sort-by CreationTime \
  --sort-order Descending
```

### Approve one package

```bash
aws sagemaker update-model-package \
  --region us-east-1 \
  --model-package-name "<MODEL_PACKAGE_ARN>" \
  --model-approval-status Approved
```

### Check endpoint

```bash
aws sagemaker describe-endpoint \
  --region us-east-1 \
  --endpoint-name pv-yolov8-endpoint
```

### Invoke with one image

```bash
aws sagemaker-runtime invoke-endpoint \
  --region us-east-1 \
  --endpoint-name pv-yolov8-endpoint \
  --content-type image/jpeg \
  --accept application/json \
  --body fileb://sample.jpg \
  /tmp/prediction.json

cat /tmp/prediction.json
```

### Check main pipeline

```bash
aws codepipeline get-pipeline-state \
  --region us-east-1 \
  --name pv-yolov8
```

### Delete endpoint safely

```bash
aws cloudformation delete-stack \
  --region us-east-1 \
  --stack-name pv-yolov8-endpoint
```

---

## 21. Final interview closing statement

This project demonstrates that I can connect computer vision with production
software delivery rather than treating training as an isolated notebook. Its
strongest aspects are the canonical data contract, containerized pipeline,
held-out quality gate, digest-based reproducibility, model-registry approval,
and infrastructure-driven serving. I can also identify where the current
tutorial must improve: representative data and stronger gates, CUDA-compatible
serving tests, complete drift computation, exact release binding, and tighter
security controls. That distinction between a functioning demo and a reliable
production system is central to how I approach MLOps.
