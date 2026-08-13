# Inference image: Ultralytics runtime + a small Flask/gunicorn server that
# implements the SageMaker /ping and /invocations contract on port 8080.
ARG ULTRALYTICS_IMAGE=ultralytics/ultralytics:8.4.118
FROM ${ULTRALYTICS_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/program:${PATH}"

RUN pip install --no-cache-dir flask>=3.0 gunicorn>=21.2

COPY src/common/ /opt/ml/code/common/
COPY src/inference/inference.py /opt/ml/code/inference.py
COPY src/inference/serve.py /opt/ml/code/serve.py

# `serve` must be on PATH: SageMaker starts the container with `serve`.
COPY docker/inference/serve /opt/program/serve
RUN chmod +x /opt/program/serve

EXPOSE 8080
WORKDIR /opt/ml/code
