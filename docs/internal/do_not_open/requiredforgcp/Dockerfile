# Gateway image. Base torch wheel from PyPI is CUDA-enabled and uses the host
# GPU through the NVIDIA container runtime (already set up on the DL VM image).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Yield LightGBM model baked INTO the image (not via /artifacts mount, which
# is read-only and can shadow image paths). Staged from
# outputs/output_files/kaggle/working/saved_models/ by scripts/stage_models_gcp.py.
COPY artifacts/yield/ /app/models/yield/
ENV YIELD_MODEL_PATH=/app/models/yield/lightgbm_tuned.txt

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8000

# One worker: the models are big and load once. Scale by adding VM/GPU, not workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
