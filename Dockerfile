# OpenHeart — Dockerfile
# v4.5.0: Python backend with CUDA 12.4, CosyVoice TTS, vLLM inference
# Build: docker build -t openheart:latest .
# Run:   docker run --gpus all -p 9876:9876 -p 8080:8080 openheart:latest
#
# Tradeoff: pip-only vs conda (environment.yml)
#   The project's canonical environment (.yaml) pins hundreds of packages including
#   CUDA-specific ones (cupy-cuda12x, cuda-toolkit) tied to the host GPU setup.
#   A full conda env import would conflict with the container's CUDA 12.4 runtime,
#   massively bloat image size (~12 GB+), and take hours to resolve.
#   We use pip instead — faster builds, smaller images, and avoids CUDA version skew
#   between host and container. For a conda-based build, use a multi-stage approach:
#   build the env on a host with matching CUDA, export, then COPY --from stage.

FROM nvidia/cuda:12.4-base-ubuntu22.04

LABEL maintainer="OpenHeart Team"
LABEL description="OpenHeart multi-modal AI companion — Python backend"

# ── System dependencies ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m pip install --no-cache-dir --upgrade pip

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────
# v4.5.0: torch with CUDA 12.4, vLLM for fast inference
# All packages from pyproject.toml + extras
RUN pip install --no-cache-dir \
    torch \
    vllm \
    ultralytics \
    transformers \
    aiohttp \
    websockets \
    lancedb \
    "redis[hiredis]" \
    networkx \
    numpy \
    pillow \
    openai \
    funasr \
    soundfile \
    pyyaml \
    "spacy>=3.7" \
    snownlp \
    textblob \
    opencv-python-headless \
    pynput \
    pyautogui \
    pywhispercpp \
    silero-vad \
    scikit-learn \
    scipy \
    sentence-transformers \
    accelerate \
    bitsandbytes \
    ten-vad

# ── CosyVoice (local package from deps/) ─────────────────────────────
COPY deps/CosyVoice /app/deps/CosyVoice
ENV PYTHONPATH="/app/deps/CosyVoice:${PYTHONPATH}"

# ── Application code ─────────────────────────────────────────────────
COPY src/ /app/src/
COPY config/ /app/config/
COPY scripts/ /app/scripts/
COPY pyproject.toml /app/
COPY 雪奈.json /app/

RUN mkdir -p /app/data/cold_memory

# ── Runtime environment defaults ─────────────────────────────────────
ENV OPENHEART_MODE=mock
ENV OPENHEART_VRAM_TIER=auto
ENV OPENHEART_CONTEXT_LIMIT=2048
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# ── Build-time API key configuration ─────────────────────────────────
ARG DEEPSEEK_API_KEY=""
ENV DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
ARG VLM_API_KEY=""
ENV VLM_API_KEY=${VLM_API_KEY}

# ── Ports ────────────────────────────────────────────────────────────
# 9876: Live2D WebSocket bridge (§7.3)
# 8080: Frontend control panel
EXPOSE 9876 8080

# ── Healthcheck ──────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9876/health')"

# ── Entry point ──────────────────────────────────────────────────────
CMD ["python", "scripts/demo_full.py"]
