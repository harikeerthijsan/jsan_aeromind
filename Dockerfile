# AeroMind web service for Railway: the API and the built frontend in one container.
# GPU work (YOLO, Qwen3-VL via Ollama, COLMAP) is not done here: GPU_WORKER=remote queues it for the GPU worker
# (backend/run-worker.ps1 on the machine with the GPU), which shares this service's PostgreSQL database and bucket.

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GPU_WORKER=remote AEROMIND_SKIP_DOTENV=1 ENABLE_DEMO_SEED=false \
    UPLOAD_DIR=/data/uploads RECON_DIR=/data/reconstructions VENDOR_REFERENCE_DIR=/data/vendor_refs \
    FRONTEND_DIST=/app/frontend/dist COLMAP_PATH=off YOLO_MODEL_PATH=
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install -r requirements.txt
COPY backend/app ./app
COPY --from=web /web/dist /app/frontend/dist
RUN mkdir -p /data/uploads /data/reconstructions /data/vendor_refs
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*' --timeout-keep-alive 75"]
