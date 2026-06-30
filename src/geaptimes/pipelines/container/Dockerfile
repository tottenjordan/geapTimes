# geapTimes runtime image — serves double duty:
#   1. the TimesFM custom **serving container** (CMD below honours the Vertex prediction contract);
#   2. the **base_image** for the KFP pipeline components (KFP overrides the entrypoint, so the
#      serving CMD is inert when the image runs as a component executor).
#
# Built by Cloud Build (see cloudbuild.yaml) to avoid local Docker + the corp airlock. The build
# pulls from **public PyPI** with **CPU-only torch** on purpose: the uv.lock pins a CUDA torch from
# an internal mirror that Cloud Build workers can't reach, and CPU is all the serving/components need.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    AIP_HTTP_PORT=8080 \
    AIP_HEALTH_ROUTE=/health \
    AIP_PREDICT_ROUTE=/predict

WORKDIR /app

# CPU-only torch first, from the PyTorch CPU index. The project's `timesfm[torch]` only needs
# `torch>=2.0.0`, so this satisfies it without the multi-GB CUDA stack PyPI would otherwise pull.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the geapTimes package + remaining runtime deps from public PyPI (torch already satisfied).
# `uv_build` (the PEP 517 backend) is fetched from PyPI automatically by pip's isolated build.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Bake the TimesFM checkpoint into the image so serving nodes never download at cold start (safer
# behind VPC-SC). The HF cache lives at $HF_HOME; HF_HUB_OFFLINE is flipped on *after* the download.
ARG TIMESFM_CHECKPOINT=google/timesfm-2.5-200m-pytorch
ENV TIMESFM_CHECKPOINT=${TIMESFM_CHECKPOINT}
RUN python -c "import timesfm; timesfm.TimesFM_2p5_200M_torch.from_pretrained('${TIMESFM_CHECKPOINT}')"
ENV HF_HUB_OFFLINE=1

LABEL solution=geaptimes

EXPOSE 8080
# Shell form so ${AIP_HTTP_PORT} expands; timeout 0 disables the worker timeout for the lazy model
# load on the first request. Single worker keeps the model resident once (and memory predictable).
CMD exec gunicorn -b 0.0.0.0:${AIP_HTTP_PORT} --workers 1 --threads 4 --timeout 0 \
    geaptimes.pipelines.serving.app:app
