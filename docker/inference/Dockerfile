# Inference image: Ultralytics runtime + a small Flask/gunicorn server that
# implements the SageMaker /ping and /invocations contract on port 8080.
#
# CUDA 11.7 works with the CUDA 11.4-era NVIDIA driver currently exposed by
# SageMaker ml.g4dn.xlarge. Newer Ultralytics GPU images use CUDA 12.x and fail
# during container startup with "NVIDIA driver ... is too old".
ARG PYTORCH_IMAGE=pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM ${PYTORCH_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/program:${PATH}"

RUN pip install --no-cache-dir \
    "ultralytics==8.4.118" \
    "flask>=3.0" \
    "gunicorn>=21.2"

COPY src/common/ /opt/ml/code/common/
COPY src/inference/inference.py /opt/ml/code/inference.py
COPY src/inference/serve.py /opt/ml/code/serve.py

# `serve` must be on PATH: SageMaker starts the container with `serve`.
COPY docker/inference/serve /opt/program/serve
RUN chmod +x /opt/program/serve

EXPOSE 8080
WORKDIR /opt/ml/code
