# OpenHeart

**有温度的 AI 虚拟伙伴 / A Virtual Companion That Cares**

OpenHeart 是一款多模态 AI 虚拟伙伴，集成语音对话、实时屏幕视觉感知、五层记忆系统和 Live2D 可交互形象。她在本地 GPU 上运行（RTX 3080 Ti 16GB），能看见你的屏幕、听见你的声音、记住你们的过去。

OpenHeart is a multi-modal AI companion with voice dialogue, real-time screen perception, a 5-tier memory system, and a Live2D interactive avatar. She runs on local GPU (RTX 3080 Ti 16GB), sees your screen, hears your voice, and remembers your shared history.

---

## 语音闭环 / Voice Closed Loop (Working)

```
Mic (parec) → SenseVoice ASR (funasr) → DeepSeek API → CosyVoice3-0.5B TTS (vLLM) → Speaker (paplay)
```

一键启动：`python scripts/demo_full.py`

Single command to launch the full pipeline: `python scripts/demo_full.py`

## 屏幕视觉 / Screen Vision (Working)

Two-stage YOLOE detection pipeline + VLM concept learning:

- **RegionProposer**: YOLOE-small PF mode (~0.3 GB VRAM, ~17ms), outputs raw bounding boxes
- **ConceptClassifier**: YOLOE-large SAVPE/text-prompt (~0.7 GB VRAM, ~100ms), classifies with learned visual prompts
- **PromptLearner**: MiniCPM-V-4.6 (cloud API at api.modelbest.cn) learns new concepts into PromptMemory (68+ concepts stored)
- **OCR**: EasyOCR on window crops
- **SpatialGraph + EntityGraph**: track spatial relationships across frames via networkx
- **ReflectionEngine**: background 5s loop, discovers patterns from EntityGraph → stored in deep memory

## 五层记忆 / 5-Tier Memory

| Tier | Backend | Role |
|------|---------|------|
| HOT (T0) | Redis Stream | Raw session data (30s TTL) |
| WARM (T1) | Redis Hash | Active context (24h TTL) |
| CORE (T2) | LanceDB | Important episodes (promotion ≥0.7) |
| COLD (T3) | LanceDB | Long-term archive (24h sync) |
| DEEP (T4) | LanceDB | Pattern insights (min 3 occurrences) |

Extra: **RetrievalGate** (composite scoring: recency 0.4, relevance 0.4, importance 0.2), **EntityGraph**, **ReflectionEngine**, **query_visual** tool for LLM.

## Live2D 虚拟形象 / Live2D Avatar

Electron app at `electron-l2d/`. WebSocket bridge (port 9876) to Python.

- **Mouth sync**: start/finish signals from TTS
- **Eye tracking**: global cursor polling (33ms)
- **Expressions**: smile, sad, surprised, blush, glasses, elf_ears, dark_face (xiaoyue model)
- **Transparent window**, click-through mode (Ctrl+Shift+P)

## 控制面板 / Control Panel

`frontend/index.html` — API config, persona editor, module toggles, status panel, one-click start + progress bar.

## 技术栈 / Tech Stack

| Category | Components |
|----------|-----------|
| **GPU** | NVIDIA RTX 3080 Ti 16GB |
| **ASR** | SenseVoice (funasr, iic/SenseVoiceSmall) |
| **LLM** | DeepSeek API (stream_decide) |
| **TTS** | CosyVoice3-0.5B (vLLM, ~4.8 GB) |
| **Vision** | YOLOE-v8s (PF + SAVPE, ~1.5 GB total) + MiniCPM-V-4.6 (cloud API) |
| **OCR** | EasyOCR (PaddleOCR-ONNX backend) |
| **Memory** | Redis 7.2 (Stream/Hash) + LanceDB |
| **Avatar** | Electron + PixiJS + pixi-live2d-display (Cubism 4) |
| **Personality** | 雪奈 — tsundere/gremlin/哥们 style (prompt_modules.json) |

## VRAM 占用 / VRAM Usage

| Component | VRAM |
|-----------|------|
| CosyVoice3 (vLLM) | ~4.8 GB |
| YOLOE (RegionProposer + ConceptClassifier) | ~1.5 GB total |
| SenseVoice | ~250 MB |
| VLM (MiniCPM-V-4.6) | 0 GB (cloud API) |

Startup auto-detects VRAM tier (HIGH ≥12 GB / LOW <12 GB).

## 快速开始 / Quick Start

```bash
# 1. 安装依赖 / Install dependencies
conda create -n openheart python=3.11 -y
bash scripts/setup_env.sh

# 2. 启动 Redis / Start Redis
redis-server

# 3. 设置 API 密钥 / Set API keys
export DEEPSEEK_API_KEY="sk-..."
export VLM_API_KEY="sk-..."

# 4. 运行语音 Demo / Run voice demo
python scripts/demo_full.py

# 5. （可选）启动 Live2D + 控制面板 / (Optional) Start L2D + panel
cd electron-l2d && npm start    # Live2D avatar
python frontend/server.py       # Control panel at http://localhost:8000
```

---

*OpenHeart — 不只是 AI，而是有温度的陪伴。 / Not just AI, but a companion with warmth.*
