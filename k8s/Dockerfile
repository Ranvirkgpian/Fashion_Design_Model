# Dockerfile — GPU-enabled container for AI Fashion Design Assistant
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY models/ ./models/

ENV HF_HOME=/app/models/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV NVIDIA_VISIBLE_DEVICES=all

EXPOSE 8000

CMD ["python3", "scripts/fastapi_server.py"]
