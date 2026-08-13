# Training image: Ultralytics + CUDA torch, driven by the SageMaker training
# toolkit so hyperparameters arrive as CLI flags and SM_CHANNEL_* env vars.
ARG ULTRALYTICS_IMAGE=ultralytics/ultralytics:8.4.118
FROM ${ULTRALYTICS_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SAGEMAKER_PROGRAM=train.py \
    SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

# The toolkit provides the `train` entrypoint SageMaker invokes for BYOC.
RUN pip install --no-cache-dir sagemaker-training

COPY src/common/ /opt/ml/code/common/
COPY src/training/train.py /opt/ml/code/train.py
COPY config/config.yaml /opt/ml/code/config/config.yaml
ENV MLOPS_CONFIG=/opt/ml/code/config/config.yaml

WORKDIR /opt/ml/code
