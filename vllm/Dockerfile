FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python-is-python3 \
    build-essential git curl ca-certificates supervisor \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN python3 -m pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN python3 -m pip install vllm==0.8.5 fastapi uvicorn[standard] huggingface_hub

WORKDIR /app

COPY supervisord.llm.conf /etc/supervisord.conf

RUN python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda build", torch.version.cuda)
PY

EXPOSE 8000

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisord.conf"]