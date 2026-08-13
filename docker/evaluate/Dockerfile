# Evaluation reuses the train image so best.pt is loaded by the exact
# ultralytics/torch build that produced it (no version skew).
ARG TRAIN_IMAGE=pv-yolov8/train:latest
FROM ${TRAIN_IMAGE}

COPY src/processing/evaluate.py /opt/ml/code/evaluate.py

# Processing jobs invoke the entrypoint directly (not via the training toolkit).
ENTRYPOINT ["python3", "/opt/ml/code/evaluate.py"]
