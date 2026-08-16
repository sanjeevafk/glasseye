# GlassEye production image: single container serving the built SPA and API.
# The deterministic demo (synthetic dataset, YOLO checkpoint, scenario video,
# mission events) is regenerated at boot so the container is self-contained.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/app/.ultralytics \
    MPLCONFIGDIR=/app/.matplotlib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl ffmpeg && rm -rf /var/lib/apt/lists/*

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

# The deterministic demo checkpoint ships in the image so the build skips
# retraining (train_yolo.py --if-missing). 6 MB vs 5-10 min of CPU training
# on every deploy.
COPY models/glasseye-yolo-v1/ /app/models/glasseye-yolo-v1/

# Generate the deterministic demo at build time so the image ships with the
# dataset, checkpoint, scenario video, and mission events baked in. Boot then
# starts instantly instead of retraining on every cold start.
#
# NOTE: this RUN sits BEFORE the frontend dist COPY on purpose. The demo only
# depends on backend/ + scripts/, so frontend-only changes reuse this cached
# layer and deploys stay fast (~3-4 min) instead of retraining the model.
RUN python scripts/prepare_synthetic_dataset.py \
    && python scripts/validate_dataset.py \
    && python scripts/train_yolo.py --if-missing \
    && python scripts/run_demo.py

# Frontend dist copied LAST so frontend-only changes never invalidate the
# demo-generation layer above.
COPY --from=frontend /frontend/dist /app/frontend/dist

CMD ["sh", "-c", "PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
