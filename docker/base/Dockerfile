# Shared slim base for the CPU-only data steps (preprocess).
# No torch/ultralytics here: preprocessing is pure Python + PIL.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# libGL/glib are pulled in by opencv/Pillow image codecs in some environments.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY docker/base/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/ml/code
