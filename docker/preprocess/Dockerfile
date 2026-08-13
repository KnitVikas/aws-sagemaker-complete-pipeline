# Preprocess: converts person.zip (CVAT) + vehicle.zip (YOLO) into one
# Ultralytics dataset. Runs as a SageMaker Processing job.
ARG BASE_IMAGE=pv-yolov8/base:latest
FROM ${BASE_IMAGE}

COPY src/common/ /opt/ml/code/common/
COPY src/processing/preprocess.py /opt/ml/code/preprocess.py
COPY config/config.yaml /opt/ml/code/config/config.yaml

ENV MLOPS_CONFIG=/opt/ml/code/config/config.yaml
ENTRYPOINT ["python3", "/opt/ml/code/preprocess.py"]
