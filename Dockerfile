# syntax=docker/dockerfile:1.2
# GlassEye production image: single container serving the built SPA and API.
# The deterministic demo (synthetic dataset, YOLO checkpoint, scenario video,
# mission events) is regenerated at boot so the container is self-contained.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/app/.ultralytics \
    YOLO_SETTINGS_DISABLED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MPLCONFIGDIR=/app/.matplotlib \
    PYTHONPATH=/app/backend

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg && rm -rf /var/lib/apt/lists/*

# Backend + deps. CPU-only torch: the deterministic demo runs fine on CPU
# and this avoids multi-GB CUDA wheels.
COPY backend/pyproject.toml backend/pyproject.toml
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install ./backend

# Frontend build.
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install
COPY frontend/ .
# Same-origin API: the SPA calls the backend at its own origin.
ENV VITE_BACKEND_URL=""
RUN npm run build

# Runtime image.
FROM base
COPY backend/ /app/backend/
COPY scripts/ /app/scripts/
COPY Makefile /app/Makefile

# Checkpoints ship in the image so the build skips retraining
COPY models/ /app/models/

# Advisory VLM config for the demo bake. Render injects service env vars as
# build args automatically; local builds without them fall back to fixture
# mode (no network, no key). GLASSEYE_VLM_API_KEY is only referenced inside
# the RUN shell below (never persisted via ENV); the key is set as a Render
# service env var, which Render forwards to the build as an ARG. Hackathon
# tradeoff: the key value is visible in the image's build layer, so rotate it
# after the event if it matters.
ARG DEMO_VLM_MODE=fixture
ARG GLASSEYE_VLM_API_KEY=
ARG GLASSEYE_VLM_BASE_URL=https://integrate.api.nvidia.com/v1
ARG GLASSEYE_VLM_MODEL=meta/llama-3.2-11b-vision-instruct
ARG GLASSEYE_VLM_TIMEOUT_SECONDS=60
ENV DEMO_VLM_MODE=${DEMO_VLM_MODE} \
    GLASSEYE_VLM_BASE_URL=${GLASSEYE_VLM_BASE_URL} \
    GLASSEYE_VLM_MODEL=${GLASSEYE_VLM_MODEL} \
    GLASSEYE_VLM_TIMEOUT_SECONDS=${GLASSEYE_VLM_TIMEOUT_SECONDS}

# Generate the deterministic demo at build time so the image ships with the
# dataset, checkpoint, scenario video, and mission events baked in. Boot then
# starts instantly instead of retraining on every cold start.
#
# NOTE: this RUN sits BEFORE the frontend dist COPY on purpose. The demo only
# depends on backend/ + scripts/, so frontend-only changes reuse this cached
# layer and deploys stay fast (~3-4 min) instead of retraining the model.
#
# The demo bake runs with the real VLM when GLASSEYE_VLM_API_KEY is present
# (injected by Render as a build arg); otherwise the fixture provider is used.
RUN sh -c 'python scripts/prepare_synthetic_dataset.py \
    && python scripts/validate_dataset.py \
    && python scripts/train_yolo.py --if-missing \
    && python scripts/run_demo.py'

# Frontend dist copied LAST so frontend-only changes never invalidate the
# demo-generation layer above.
COPY --from=frontend /frontend/dist /app/frontend/dist

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --limit-concurrency 10 --timeout-keep-alive 5 --backlog 128"]
