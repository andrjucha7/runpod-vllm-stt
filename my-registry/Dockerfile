FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps:
# - ffmpeg: common for audio decoding
# - supervisor: process manager
# - python: runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ffmpeg supervisor ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pip settings (optional but helps stability)
ENV PIP_NO_CACHE_DIR=1
RUN python3 -m pip install --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

COPY stt_server.py /app/stt_server.py
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Internal ports (for documentation; Runpod exposure is configured in the Pod)
EXPOSE 8000 8001

CMD ["/usr/bin/supervisord", "-n"]
