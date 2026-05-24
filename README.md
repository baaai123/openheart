# OpenHeart

有温度的虚拟伙伴。A multi-modal virtual companion system with emotion-aware dialogue, visual perception, personality evolution, and Live2D avatar rendering.

## Quick Start

```bash
# 1. Environment setup
bash scripts/setup_env.sh   # creates conda env 'openheart', installs deps

# 2. Activate and download models
conda activate openheart
python scripts/download_models.py --download

# 3. Start infrastructure (Redis + LanceDB)
docker compose up -d

# 4. Run in mock mode (no GPU required)
python run.py --mode=mock

# Or run with full GPU pipeline
python run.py --mode=real --vram-tier=auto
```

For VRAM constrained setups, use `--vram-tier=low` or `--vram-tier=mid` to disable compute-heavy components automatically.

## Requirements

| Component | Minimum |
|---|---|
| Python | >= 3.11 |
| GPU VRAM (high tier) | >= 15.5 GB |
| GPU VRAM (medium tier) | >= 11.5 GB |
| GPU VRAM (low tier) | >= 7.5 GB |
| RAM | >= 16 GB |
| Redis | >= 7.2 (with Stream) |
| Docker | recommended for infra services |

VRAM tier is auto-detected at startup. See [VRAM tiers](#) for which models run at each tier.

## Architecture

The system follows a 7-layer pipeline:

```
Input                      ┌─────────────┐
  │ Screen + Audio + Mouse │ Perception  │  Visual: YOLO-World, YOLOv11n, PaddleOCR, TinyCLIP
  ▼                        │ (4-lane)    │  Audio: VAD, ASR (Whisper), emotion, ring buffer
┌───────────┐              └──────┬──────┘
│ Perception│                     │ vision_snapshot + audio_event
└─────┬─────┘                     │
      │              ┌────────────▼──────┐
      ├──────────────│ Fusion            │  Time-window fusion, event classification
      │              │ (time_window)     │  Entity alignment, scene synthesis
      │              └────────┬──────────┘
      │                       │ Scene
      │              ┌────────▼──────────┐
      ├──────────────│ Memory            │  Hot: Redis + networkx graph (session)
      │              │ (hot + cold)      │  Cold: LanceDB (long-term, filtered)
      │              └────────┬──────────┘
      │                       │ context + user_model
      │              ┌────────▼──────────┐
      │              │ Personality       │  Baseline + preference shift + emotion adj.
      │              │ (dynamic fusion)  │  PersonaAuditor guards against drift
      │              └────────┬──────────┘
      │                       │ dynamic persona
      │              ┌────────▼──────────┐
      ├──────────────│ Decision          │  Qwen2.5-3B (GPTQ 4bit) + optional shadow
      │              │ (main + shadow)   │  Fast-path matcher [已废弃 v5.0], reflex rules, cloud fallback [已废弃 v5.0]
      │              └────────┬──────────┘
      │                       │ command + confidence
      │              ┌────────▼──────────┐
      │              │ Prediction        │  Gentle reminder, proactive suggestions
      │              │ (proactive)       │
      │              └────────┬──────────┘
      │                       │ action_sequence
      │              ┌────────▼──────────┐
      │              │ Execution         │  ActionSequenceScheduler channels:
      │              │ (scheduler)       │  avatar (Live2D / FallbackTextBubble)
      ▼              │                   │  voice (CosyVoice TTS)
Action               │                   │  mouse (keyboard + mouse control)
                     └───────────────────┘
```

Each layer communicates via a unified message envelope (`trace_id`, `source_layer`, `version`, `metadata.degraded`). Every capability has a local fallback -- no hard cloud dependency.

## Configuration

All config files live in `config/` and must match the spec exactly:

| File | Purpose |
|---|---|
| `model_paths.yaml` | Local paths to all model weights |
| `thresholds.yaml` | Detection and confidence thresholds |
| `emotion_params.yaml` | Emotion classifier parameters |
| `fast_path_rules.yaml` | Reflex rule definitions |
| `audio.yaml` | Audio pipeline settings |
| `sentiment.yaml` | Sentiment provider (default / structbert) |
| `transcript_overlay.yaml` | Transcript display settings |
| `live2d.yaml` | Live2D renderer config |
| `endpoints.yaml` | Cloud service endpoints |
| `memory.yaml` | Hot/cold memory parameters |
| `easter_eggs.json` | Easter egg triggers and responses |

## Testing

```bash
# All tests
pytest tests/ -v

# Contract tests only (module completeness gate)
pytest tests/contracts/ -v
bash tests/run_all_contracts.sh
```

Contract tests are the primary completeness gate. A module is done only when its contract test passes. See `tests/contracts/` for per-layer contracts.

## Project Structure

```
├── config/               # YAML/JSON configuration files
├── models/               # Downloaded model weights (gitignored)
├── rules/                # Reflex rules (template, core, interactive, user-taught)
├── scripts/              # setup_env.sh, download_models.py, validate_env.py
├── src/
│   ├── perception/       # 4-lane visual + audio sensing
│   ├── fusion/           # Time-window, event classification, scene synthesis
│   ├── memory/           # Hot (Redis) + Cold (LanceDB) memory
│   ├── personality/      # Baseline, preference shift, dynamic fusion
│   ├── decision/         # Main 3B + shadow 1.5B verifier, fast-path, reflex
│   ├── prediction/       # Proactive gentle reminders
│   └── execution/        # Action scheduler, avatar/voice/mouse channels
├── tests/
│   ├── contracts/        # Contract tests (one per module)
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── mocks/            # Mock implementations
│   ├── fixtures/         # Test data
│   └── performance/      # Performance benchmarks
└── run.py                # Entry point
```

## Development

See [AGENTS.md](AGENTS.md) for detailed development conventions, naming rules, and the full specification document index.

Key development rules:
- Contract-driven: write tests first, then implement
- Vertical-slice: build end-to-end feature chains, not horizontal layers
- All try/except blocks must document what exception is caught and why it is safe
- All errors log at WARNING level with `trace_id`
- Emotion output is restricted to `joy`, `sadness`, `neutral` (`anger`/`surprise` are placeholders)
- Context truncation happens in `ContextAssembler` at message boundaries, never in tokenization
- Default context limit: 2048 tokens (4096 in performance mode with re-validation)

## License

Proprietary. See license file for details.
