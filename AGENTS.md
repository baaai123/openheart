# OpenHeart — Agent Instructions

## Project Status

**Pre-implementation.** No source code exists yet. The repo contains only three specification documents that define the entire system architecture. All code generation must strictly follow these specs.

## Core Documents

| File | Role |
|---|---|
| `多模态智能体架构工程规格书v4.5.0.md` | **Primary spec.** Single source of truth for all interfaces, data formats, constraints, degradation matrices, and directory structure (Chapter 13). |
| `项目宪法.md` | **Hard rules.** Zero-tolerance naming conventions, forbidden behaviors, VRAM constraints, degradation matrix, development strategy. Overrides any assumption. |
| `实施方案.md` | **Implementation plan.** Phase-by-phase execution order, prompt templates for AI code generation, contract test structure. |

## Non-Negotiable Rules

### Naming (zero tolerance)

| Correct | Wrong (DO NOT USE) |
|---|---|
| `FallbackTextBubble` | `FallbackAvatarChannel` |
| `degraded` | `downgraded` / `fallback_mode` |
| `emotion.category` | `emotion.type` |
| `voice_channel` | `tts_channel` |
| `avatar_channel` | `live2d_channel` |
| `mouse_channel` | `input_channel` |

### Emotion categories

Only `joy`, `sadness`, `neutral` are reliable outputs. `anger` and `surprise` are **placeholder enums** — downstream modules **must not** branch on them unless `config/sentiment.yaml` has `provider: "structbert"`.

### Other hard constraints

- All layer-to-layer communication uses the unified message envelope (spec §0.3): `trace_id`, `source_layer`, `version` (monotonic), `metadata.degraded`, etc.
- Context truncation **must** happen in `ContextAssembler` at message boundaries — never in tokenization.
- Default context hard limit: **2048 tokens** (performance mode up to 4096, needs re-validation).
- Live2D rendering **must** run in a dedicated thread, never on the asyncio event loop.
- All `try/except` blocks must have comments explaining what exception is caught and why it's safe.
- All errors must be logged at WARNING level with `trace_id`.
- Environment variables are read **once** at startup into `RuntimeConfig`. Modules get config via DI or singleton — **never** `os.environ` inside modules.
- `librosa` is not a hard dependency — lazy-import only when `VoiceFeatureExtractor.enabled = True`.
- Sensitive data defaults to local-only. No cloud upload unless user explicitly says "记住这个".

### VRAM tiers (auto-selected at startup)

| Tier | Available VRAM | Enabled |
|---|---|---|
| High | ≥ 15.5 GB | All models + shadow verification |
| Medium | ≥ 11.5 GB | No shadow, Whisper medium |
| Low | ≥ 7.5 GB | No shadow, no YOLO-World, no fast path, CosyVoice on CPU |

## Directory Structure (spec §13)

```
src/
├── perception/          # 4-lane visual + audio
│   ├── visual/
│   ├── audio/           # ring_buffer, onset_detector, asr_stream, ten_vad, silero_vad, vad_factory, emotion, audio_pipeline
│   ├── sync_vision_query.py
│   └── perception_bus.py
├── fusion/              # time_window, event_classifier, entity_fusion, scene_synthesis
├── memory/
│   ├── hot/             # Redis-backed session memory
│   ├── cold/            # LanceDB long-term memory
│   ├── sync/            # Hot→Cold sync
│   ├── decay/           # Memory decay
│   ├── user_model.py
│   └── memory_service.py
├── personality/         # baseline, preference_shift, emotion_adj, dynamic_fusion, persona_auditor
├── decision/            # main_decision (3B) [已废弃], shadow_verifier (1.5B) [已废弃], cloud_fallback [已废弃 v5.0], fast_path_matcher [已废弃 v5.0], reflex/, learning/, teaching
├── prediction/          # gentle_reminder
└── execution/
    ├── action_scheduler.py
    ├── channels/         # avatar_channel, mouse_channel, voice_channel
    ├── tts_service/      # cosyvoice_adapter, char_duration_predictor, stream_handler
    ├── fallback_text_bubble.py
    ├── mouse_controller.py
    └── state_bus.py
config/                  # baseline.json, live2d.yaml, emotion_params.yaml, thresholds.yaml, endpoints.yaml, easter_eggs.json, memory.yaml, audio.yaml, fast_path_rules.yaml, sentiment.yaml, transcript_overlay.yaml, model_paths.yaml
models/
rules/                   # template.py/json, core_rules.json, interactive_rules.json, user_taught_rules.json
tests/
├── unit/
├── integration/
├── fixtures/
└── contracts/           # Contract tests — each module must pass its corresponding contract test
```

## Development Strategy

- **Contract-driven**: Write contract tests in `tests/contracts/` first, then implement. Module = done only when its contract test passes.
- **Vertical-slice first**: Build end-to-end feature chains (e.g., "pure voice dialogue loop"), not horizontal layers.
- **Mock-first**: Unimplemented dependencies use mocks in `tests/mocks/` that follow `src/` interface contracts.
- Run all contract tests before each commit: `tests/run_all_contracts.sh`

## Config Files

All config file keys and defaults must match the spec **exactly**. Do not invent names.

Key configs: `model_paths.yaml`, `thresholds.yaml`, `emotion_params.yaml`, `fast_path_rules.yaml`, `audio.yaml`, `sentiment.yaml`, `transcript_overlay.yaml`

## Degradation Philosophy

Every capability must have a **local fallback**. No "cloud unavailable = feature dead" hard dependencies. All degradation paths must log with `degraded: true` metadata. Crash is preferred over silent data corruption.

## Key v4.5.0 Changes from Prior Versions

1. Default context reduced to 2048 tokens (was 4096).
2. Shadow verification is **optional** — controlled by `RuntimeConfig.enable_shadow`; only enabled on high-VRAM tier.
3. Live2D rendering runs in a **separate thread** (spec §7.3.4).
4. Three VRAM tiers instead of two.
5. CosyVoice falls back to CPU ONNX on low-VRAM tier (fast path forced off).
6. `FallbackAvatarChannel` renamed to `FallbackTextBubble`.
7. Context truncation by `ContextAssembler` at message boundaries only.
8. Cold memory Level 2 summary uses 3B model, not Louvain community detection.
9. User model correction interface (§5.7.5) allows natural-language adjustments.
10. `SyncVisionQuery` is now async-safe with timeout fallback and caching.

## Tech Stack

- Python ≥ 3.11, asyncio + uvloop
- Models on local GPU: Qwen2.5-3B (GPTQ 4bit), Qwen2.5-1.5B (INT8), Qwen2.5-0.5B (FP16), YOLO-World-Small (FP16), YOLOv11n (INT8/TensorRT), Whisper large-v3, CosyVoice-300M (FP16), TinyCLIP-ViT, bge-small-zh-v1.5
- Local CPU: spaCy ≥3.7, SnowNLP, PaddleOCR-ONNX, ResNet18/34, networkx ≥3.0
- Storage: Redis ≥7.2 (with Stream), LanceDB ≥0.6
- Rendering: live2d-py ≥0.3.0 (fallback: FallbackTextBubble)
- TTS gRPC priority, WebSocket fallback

## Prompt Template (for AI code generation)

When generating code, always include these sections in the prompt:

```
[角色] 资深 Python 工程师，严格实现规格书接口，不做简化。
[任务] One-sentence task description.
[输入] Spec chapter text + interface files + config files.
[输出] Complete Python files, type-annotated, with # v4.5.0 §X.Y.Z annotations.
[限制] No unspecified deps. joy/sadness/neutral only. ContextAssembler truncation. All try/except annotated.
```

## Agent skills

### Issue tracker

Local markdown in `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context (`CONTEXT.md` + `docs/adr/` at repo root). See `docs/agents/domain.md`.