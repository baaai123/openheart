# OpenHeart

**有温度的 AI 虚拟伙伴 — A Virtual Companion That Cares**

OpenHeart is a multi-modal AI companion that integrates voice dialogue, real-time screen perception, a 5-tier memory system, and a Live2D interactive avatar. She runs on local GPU, sees your screen, hears your voice, and remembers your shared history.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA RTX 3070 8GB | NVIDIA RTX 3080 Ti 16GB |
| **VRAM** | 7.5 GB (Low tier) | ≥ 15.5 GB (High tier) |
| **RAM** | 16 GB | 32 GB |
| **OS** | Linux (Ubuntu 22.04+) | Linux (Ubuntu 22.04+) |
| **CUDA** | 12.1+ | 12.4+ |
| **Python** | 3.11 | 3.11 |
| **Redis** | 7.2+ | 7.2+ |

VRAM tiers are auto-detected at startup:
- **Low** (≥7.5 GB): No shadow verification, no YOLO-World, CosyVoice on CPU
- **Medium** (≥11.5 GB): No shadow verification, Whisper medium
- **High** (≥15.5 GB): All models + shadow verification enabled

---

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/yourusername/openheart.git
cd openheart
conda create -n openheart python=3.11 -y
conda activate openheart
```

### 2. Install dependencies

```bash
# Core dependencies
pip install -r requirements.txt  # if available, or use environment.yml
# Alternative: conda environment
conda env update -f environment.yml
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys:
#   DEEPSEEK_API_KEY=sk-your-deepseek-key
#   DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
#   DEEPSEEK_MODEL=deepseek-v4-flash
#   VLM_API_KEY=sk-your-vlm-key (optional, for visual prompt learning)
```

### 4. Start Redis

```bash
redis-server
```

### 5. Download models

Model files are too large to bundle. Run the setup script to download:

```bash
bash scripts/setup_env.sh
```

This will download:
- SenseVoice ASR model (~1.5 GB)
- CosyVoice3-0.5B TTS model (~4.8 GB)
- YOLOE detection models (~1.5 GB total)
- bge-small-zh-v1.5 embedding model

### 6. Run OpenHeart

**Voice mode** (mic → ASR → LLM → TTS → speaker):

```bash
python scripts/demo_full.py
```

**Text mode** (keyboard input, no mic needed):

```bash
python scripts/demo_full.py --voice-mode text
```

### 7. (Optional) Live2D Avatar + Control Panel

```bash
# Terminal 1: Live2D avatar (requires Electron)
cd electron-l2d && npm install && npm start

# Terminal 2: Web control panel
python frontend/server.py
# Open http://localhost:8000 in browser
```

---

## Features

### Voice Dialogue
- **ASR**: SenseVoice (funasr, iic/SenseVoiceSmall) — real-time Chinese/English speech recognition
- **LLM**: DeepSeek API (streaming) — conversational intelligence
- **TTS**: CosyVoice3-0.5B (vLLM, ~4.8 GB VRAM) — natural voice synthesis
- Pipeline: `Mic → VAD → ASR → DeepSeek → CosyVoice → Speaker`

### Screen Vision
Two-stage YOLOE detection pipeline + VLM concept learning:
- **RegionProposer**: YOLOE-small PF mode (~0.3 GB VRAM, ~17ms), bounding box proposals
- **ConceptClassifier**: YOLOE-large SAVPE/text-prompt (~0.7 GB VRAM), concept classification
- **PromptLearner**: MiniCPM-V-4.6 (cloud API) — learns new visual concepts
- **OCR**: EasyOCR on window regions
- **SpatialGraph + EntityGraph**: spatial relationship tracking across frames

### 5-Tier Memory

| Tier | Backend | Role |
|------|---------|------|
| HOT (T0) | Redis Stream | Raw session data (30s TTL) |
| WARM (T1) | Redis Hash | Active context (24h TTL) |
| CORE (T2) | LanceDB | Important episodes (promotion ≥0.7) |
| COLD (T3) | LanceDB | Long-term archive (24h sync) |
| DEEP (T4) | LanceDB | Pattern insights (min 3 occurrences) |

### Live2D Avatar
- Electron + PixiJS + pixi-live2d-display (Cubism 4)
- WebSocket bridge (port 9876) to Python
- Mouth sync, eye tracking, expression control
- Transparent window with click-through mode

---

## Configuration

All configuration files are in `config/`:

| File | Purpose |
|------|---------|
| `audio.yaml` | Audio pipeline settings |
| `baseline.json` | Personality baseline (雪奈) |
| `easter_eggs.json` | Hidden trigger responses |
| `emotion_params.yaml` | Emotion model parameters |
| `endpoints.yaml` | API endpoint configurations |
| `fast_path_rules.yaml` | Fast-path decision rules |
| `live2d.yaml` | Live2D avatar settings |
| `memory.yaml` | Memory tier configurations |
| `model_paths.yaml` | Model file locations |
| `sentiment.yaml` | Sentiment analysis settings |
| `thresholds.yaml` | System thresholds |
| `transcript_overlay.yaml` | Transcript overlay settings |

---

## Troubleshooting

**Out of VRAM error**
→ Lower VRAM tier is auto-selected. Close other GPU applications. Set `COSYVOICE_CPU=1` to run TTS on CPU.

**No audio input**
→ Check `parec` is available: `parec --list-sources`. On PulseAudio systems, install `pulseaudio-utils`.

**Live2D not showing**
→ The avatar runs as a separate Electron app in `electron-l2d/`. Run `cd electron-l2d && npm install && npm start`. Fallback text bubble mode activates automatically if L2D is unavailable.

**Redis connection refused**
→ Start Redis: `redis-server`. Default port is 6379.

---

*OpenHeart — 不只是 AI，而是有温度的陪伴。Not just AI, but a companion with warmth.*
