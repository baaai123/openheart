# OpenHeart for Windows

**有温度的 AI 虚拟伙伴 — A Virtual Companion That Cares**

OpenHeart is a multi-modal AI companion that integrates voice dialogue, real-time screen perception, a 5-tier memory system, and a Live2D interactive avatar. She runs on local GPU, sees your screen, hears your voice, and remembers your shared history.

This guide covers the **Release Zero** Windows package — a one-click launcher that runs the Python backend in **WSL2 (Ubuntu)** with an **Electron Live2D avatar** on the Windows side.

---

## Prerequisites

Before you begin, make sure your system meets these requirements.

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA RTX 3070 8GB | NVIDIA RTX 3080 Ti 16GB+ |
| **VRAM** | 7.5 GB | 15.5 GB+ |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 30 GB free | 50 GB+ (for model files) |

### Windows

- **Windows 10** (build 19041+) or **Windows 11**
- **WSL2** with **Ubuntu 22.04** installed
  - *Not sure how?* Follow the [official Microsoft guide](https://learn.microsoft.com/en-us/windows/wsl/install)
- **NVIDIA GPU drivers** with **CUDA 12.1+** installed on Windows
  - Download from [NVIDIA Driver Downloads](https://www.nvidia.com/download.aspx)
  - Run `nvidia-smi` in a terminal to verify CUDA version

### Inside WSL2 (Ubuntu)

- **Python 3.11** — check with `python3 --version`
- **pip** — Python package manager
- **Redis 7.2+** — the memory backend
  ```bash
  sudo apt update && sudo apt install redis-server -y
  ```
- **Conda** (recommended for managing Python environments)
  - Install [Miniforge](https://github.com/conda-forge/miniforge) in WSL2

### On Windows Side

- **Node.js** (required for the Live2D avatar)
  - Download from [nodejs.org](https://nodejs.org/) (LTS version recommended)
- **npm** (comes with Node.js)

---

## Quick Start

### Step 1: Extract the package

Unzip `OpenHeart-release-zero.zip` somewhere easy to find, like `C:\OpenHeart\` or your Desktop.

You should see this structure after extraction:

```
OpenHeart/
├── launch.bat              # ← Double-click to start!
├── dist/OpenHeart.exe      # PyInstaller backend executable
├── electron-l2d/           # Live2D avatar + config panel
├── config/                 # Runtime configuration files
├── models/                 # Model files (download in Step 4)
├── scripts/                # Setup and utility scripts
└── README-Windows.md       # This file
```

### Step 2: Configure your API key

1. In the `OpenHeart/` folder, find `.env.example`
2. Copy it and rename to `.env`
3. Open `.env` in Notepad and enter your DeepSeek API key:

```
DEEPSEEK_API_KEY=sk-your-actual-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash
```

> **Where do I get a key?** Sign up at [platform.deepseek.com](https://platform.deepseek.com) and create an API key.

### Step 3: Set up the Python environment (inside WSL2)

Open a WSL2 terminal (type "Ubuntu" in the Start Menu) and run:

```bash
cd /mnt/c/Path/To/OpenHeart    # replace with your actual extraction path
bash scripts/setup_env.sh
```

This will:
- Create a conda environment called `openheart`
- Install Python 3.11 and all required packages
- Configure CUDA support for your GPU

### Step 4: Download the model files

Models are too large to bundle in the ZIP. Download them from inside WSL2:

```bash
python scripts/download_models.py --download
```

This downloads all required models to the `models/` folder:
- **Whisper large-v3** (~3 GB) — speech recognition
- **CosyVoice-300M** (~4.8 GB) — voice synthesis
- **Qwen2.5-3B** (~4 GB) — the main AI brain
- **YOLO-World-Small** (~1.5 GB) — screen object detection
- **bge-small-zh-v1.5** (~133 MB) — text embedding
- **TinyCLIP-ViT** (~100 MB) — scene understanding
- And more...

> **Total download:** ~15 GB. Make sure you have a stable internet connection.
>
> **Need to save space?** Use `python scripts/download_models.py --download --model qwen_3b --model cosyvoice --model faster_whisper` to pick only the essentials.
>
> **Stuck on download?** Add `--hf-mirror` if you're in a region with slow HuggingFace access: `python scripts/download_models.py --download --hf-mirror`

### Step 5: Start Redis

Open a WSL2 terminal and run:

```bash
redis-server
```

Keep this terminal window open. Redis must stay running for OpenHeart to work.

### Step 6: Launch OpenHeart

Double-click **`launch.bat`** in the `OpenHeart/` folder.

This does two things at once:
1. Starts the Python backend inside WSL2 (logs go to `/tmp/openheart.log`)
2. Launches the Live2D avatar + configuration panel

**Wait 30–60 seconds** on the first launch while the AI models load into GPU memory.

When ready, you'll see:
- A **Live2D avatar window** (a transparent anime character on your desktop)
- A **configuration panel** for adjusting settings

### Step 7: Start talking

- **Voice mode (default):** Speak into your microphone. OpenHeart will hear you, think, and respond with spoken voice.
- **Text mode:** If you don't have a microphone, start with `bash run_backend.sh --voice-mode text` from WSL2 instead.

---

## Directory Layout

```
OpenHeart/                          # Root folder (same as the ZIP contents)
│
├── launch.bat                      # [Windows] One-click launcher
├── .env                            # Your API keys and settings
├── .env.example                    # Template for .env
│
├── dist/
│   └── OpenHeart.exe               # PyInstaller bundle (optional backend)
│
├── electron-l2d/                   # Live2D avatar application
│   ├── main.js                     # Electron main process
│   ├── config_renderer.js          # Settings panel UI
│   ├── renderer.js                 # Avatar renderer logic
│   ├── run_server.py               # WebSocket server (port 9876)
│   ├── start.bat                   # Standalone L2D launcher
│   ├── package.json                # Node.js dependencies
│   └── models/                     # Live2D model files
│
├── config/                         # Runtime settings (YAML/JSON)
│   ├── audio.yaml                  # Microphone and audio pipeline
│   ├── baseline.json               # Personality profile
│   ├── emotion_params.yaml         # Emotion model tuning
│   ├── endpoints.yaml              # API endpoint URLs
│   ├── live2d.yaml                 # Avatar appearance and behavior
│   ├── memory.yaml                 # Memory system settings
│   ├── model_paths.yaml            # Where models are stored
│   ├── thresholds.yaml             # Sensitivity and timing
│   └── ...                         # (see README.md for full list)
│
├── models/                         # AI model files (downloaded separately)
│   ├── faster_whisper_large_v3/    # Speech-to-text
│   ├── cosyvoice-300m/             # Voice synthesis
│   ├── qwen2.5-3b-gptq/            # Main AI (GPTQ 4bit)
│   ├── qwen2.5-1.5b-int8/          # Shadow verifier (optional)
│   ├── yolo_world_nano/            # Screen object detection
│   ├── yolov11n.pt                 # Lightweight detection
│   ├── bge-small-zh-v1.5/          # Text embeddings
│   └── ...                         # Additional models
│
├── scripts/                        # Utility scripts
│   ├── demo_full.py                # Main Python entry point
│   ├── download_models.py          # Model downloader
│   ├── setup_env.sh                # Environment setup (conda + pip)
│   └── validate_env.py             # Checks your setup is correct
│
├── run_backend.sh                  # Start backend from WSL2 terminal
├── pyinstaller.spec                # Build config for .exe
├── README.md                       # Full Linux documentation
└── README-Windows.md               # This file
```

---

## Known Issues (Release Zero)

This is the first Windows release. Some rough edges:

### Window detection is heuristic-based
OpenHeart can see your screen, but on Windows it identifies windows using **coordinate heuristics** (position, size, title patterns) rather than native Windows API. Some UI elements may be misattributed or missed entirely.

### First launch is slow (30–60s)
Models must load into VRAM from disk. Subsequent launches are faster. If you close the backend but keep WSL2 running, models stay cached.

### Microphone setup requires extra steps
Voice input needs **microphone passthrough to WSL2**. See the Troubleshooting section below.

### Live2D requires npm install
The first time you launch, the Electron app needs to install its Node.js dependencies. `launch.bat` handles this automatically, but it may add a few seconds.

### Redis must be running
If the backend can't connect, check that `redis-server` is running in WSL2.

### No automatic updates
Model files and code updates require manual re-download. Watch the project repository for release announcements.

---

## How to Update Models

When new model versions are released, update them from inside WSL2:

```bash
cd /mnt/c/Path/To/OpenHeart
python scripts/download_models.py --download --model faster_whisper   # for example
```

Or re-download everything:

```bash
python scripts/download_models.py --download
```

You can also check which models are already downloaded:

```bash
python scripts/download_models.py --list
```

> **Note:** If you're in China or have slow access to HuggingFace, use the mirror:
> `python scripts/download_models.py --download --hf-mirror`

---

## Troubleshooting

### "No backend connection" error

**Most likely:** Redis isn't running.

1. Open a WSL2 terminal
2. Run `redis-server` and leave it running
3. Re-launch OpenHeart

**Still broken?** Check if Redis is on the right port:
```bash
redis-cli ping   # should reply "PONG"
```

### "Out of VRAM" error

The system auto-selects a VRAM tier, but it may not be aggressive enough. Try:

1. **Close other GPU apps** (browsers with hardware acceleration, games, etc.)
2. **Run TTS on CPU** — set this environment variable before launching:
   ```bash
   export COSYVOICE_CPU=1
   ```
   Then run `bash run_backend.sh` from WSL2.
3. **Force low VRAM mode** — edit `config/model_paths.yaml` and set the paths you want to skip.

### Live2D not showing up

If the avatar window doesn't appear:

1. **Launch it manually** — open a Command Prompt in the `OpenHeart/electron-l2d/` folder and run:
   ```
   start.bat
   ```
2. **Check Node.js** — make sure Node.js is installed on Windows:
   ```
   node --version
   npm --version
   ```
3. **Reinstall dependencies** — in the `electron-l2d/` folder:
   ```
   npm install && npm start
   ```

If nothing works, the backend still runs fine — you just won't see the avatar. Voice and text interaction will still work.

### Microphone not working

This is a known WSL2 limitation. The microphone needs explicit passthrough:

1. **Install PulseAudio on Windows** — download from [pulseaudio.org](https://www.pulseaudio.org/)
2. **Or use Windows-native audio capture** — see the [Microsoft WSL2 audio guide](https://learn.microsoft.com/en-us/windows/wsl/tutorials/audio)
3. **Check WSL2 mic access** — in your WSL2 terminal:
   ```bash
   arecord -l
   ```
   If no devices appear, audio passthrough isn't configured.

**Quick workaround:** Use text mode instead.
```bash
bash run_backend.sh --voice-mode text
```

### Backend won't start

Check the log file for errors:

```bash
cat /tmp/openheart.log
```

Look for:
- **"ModuleNotFoundError"** — a Python package is missing. Run `bash scripts/setup_env.sh` again.
- **"CUDA out of memory"** — see the "Out of VRAM" section above.
- **"Connection refused"** — Redis isn't running.
- **"API key not found"** — your `.env` file is missing or misconfigured.

### API errors

All conversation goes through the DeepSeek API. If you see errors:

- **401 Unauthorized** — your API key is invalid. Double-check `.env`
- **429 Too Many Requests** — you've hit a rate limit. Wait a minute.
- **Empty responses** — check your DeepSeek account balance.

### "redis-server not found"

Install Redis in WSL2:

```bash
sudo apt update
sudo apt install redis-server -y
```

### WSL2 not starting

Make sure WSL2 is properly set up:

```bash
wsl --set-default-version 2
wsl --install -d Ubuntu-22.04
```

See the [Microsoft WSL2 installation docs](https://learn.microsoft.com/en-us/windows/wsl/install) for detailed help.

---

## Architecture (for the curious)

```
Windows Side                    WSL2 (Ubuntu)
┌─────────────────────┐        ┌──────────────────────────────┐
│                     │        │                              │
│  launch.bat ────────────────│──→ python scripts/demo_full.py│
│                     │        │         ↓                    │
│  Electron L2D ◄──── WS ────│──→ WebSocket Server (:9876)   │
│  (avatar + panel)   │        │         ↓                    │
│                     │        │  ┌─ ASR (Whisper)           │
│                     │        │  ├─ LLM (DeepSeek API)      │
│                     │        │  ├─ TTS (CosyVoice)         │
│                     │        │  ├─ Vision (YOLO-World)     │
│                     │        │  └─ Memory (Redis/LanceDB)  │
│                     │        │                              │
└─────────────────────┘        └──────────────────────────────┘
```

- **Python backend** runs entirely in WSL2 (Ubuntu) — this does all the heavy lifting (speech recognition, AI thinking, voice synthesis, screen analysis)
- **Live2D avatar** runs as a native Windows Electron app — it connects to the backend via WebSocket (port 9876) to receive expressions, mouth movements, and voice audio
- **Configuration panel** is built into the Electron app — adjust settings without touching config files
- **Redis** runs inside WSL2 and handles short-term memory and state management

---

*OpenHeart — 不只是 AI，而是有温度的陪伴。Not just AI, but a companion with warmth.*
