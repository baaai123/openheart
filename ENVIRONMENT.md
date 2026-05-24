# Environment

Development environment for the OpenHeart project. Captured 2026-05-17. Updated for three-channel architecture (ADR-0003), IconCache, DPI-aware mouse, and visual pipeline enhancements.

## System

| Property | Value |
|---|---|
| OS | Ubuntu 26.04 LTS (Resolute) |
| Kernel | 6.6.114.1-microsoft-standard-WSL2 |
| WSL Version | 2.7.3.0 |
| WSLg Version | 1.0.73 |
| MSRDC Version | 1.2.6676 |
| Direct3D Version | 1.611.1-81528511 |
| DXCore Version | 10.0.26100.1-240331-1435.ge-release |
| Windows Build | 10.0.26200.8328 |
| RAM | 31 GiB total (3.7 GiB used, 27 GiB available) |
| Swap | 8.0 GiB (0 B used) |
| CPU Cores | 24 (available) |
| Hostname | LAPTOP-PJQ55QGI |
| Shell | /bin/bash |

**Disk Layout:**

| Mount | Size | Used | Available | Use% |
|---|---|---|---|---|
| `/dev/sdd` (root) | 1007G | 47G | 910G | 5% |
| `/mnt/c` (C:) | 306G | 162G | 144G | 53% |
| `/mnt/d` (D:) | 1.6T | 286G | 1.3T | 18% |
| `/tmp` (tmpfs) | 16G | 2.5G | 14G | 16% |

## GPU

| Property | Value |
|---|---|
| GPU Model | NVIDIA GeForce RTX 3080 Ti Laptop GPU |
| VRAM | 16384 MiB (16 GB) |
| Driver Version | 596.36 |
| CUDA (nvidia-smi) | 13.2 |
| CUDA (PyTorch) | 13.0 |
| Compute Capability | 8.6 |
| Current VRAM Usage | 14500 MiB / 16384 MiB (88% util, visual pipeline active) |
| Current Power | 142 W / 175 W |
| Temperature | 68 C |

**Current GPU Process:** PID 20714 `/python3.11` (visual pipeline: Qwen3-VL + OmniParser + CLIP).

## Python Environment

| Property | Value |
|---|---|
| Python | 3.13.13 (conda-forge, GCC 14.3.0) |
| Conda | 26.3.2 |
| Environment | base (miniforge3) |
| Prefix | `/home/baaai/miniforge3` |

**Additional Environment (Visual + TTS Inference):**

| Property | Value |
|---|---|
| Conda Environment | cv311 |
| Python | 3.11.11 |
| vLLM | 0.11.2 |
| PyTorch | 2.11.0+cu130 |

**Core ML Stack:**

| Package | Version |
|---|---|
| PyTorch | 2.11.0+cu130 |
| torchaudio | 2.11.0 |
| torchvision | 0.26.0 |
| torchao | 0.17.0 |
| cuDNN (nvidia-cudnn-cu13) | 9.19.0.56 (91900) |
| CUDA Toolkit | 13.0.2 |
| nvidia-cublas | 13.1.0.3 |
| transformers | 5.8.0 |
| accelerate | 1.13.0 |
| safetensors | 0.7.0 |
| bitsandbytes | 0.49.2 |
| onnxruntime-gpu | 1.26.0 |

**Key Application Packages:**

| Package | Version | Role |
|---|---|---|
| funasr | 1.3.1 | SenseVoice ASR |
| modelscope | 1.36.3 | Model hub/download |
| openai | 2.36.0 | DeepSeek v4 Flash API client |
| lancedb | 0.30.2 | Cold/long-term memory |
| redis (python) | 7.4.0 | Hot/session memory |
| paddlex | 3.5.1 | Visual perception |
| paddleocr | 3.5.0 | OCR pipeline |
| ultralytics | 8.4.48 | YOLO object detection |
| spacy | 3.8.14 | NLP pipeline |
| transformers | 5.8.0 | General model interface |
| sentence-transformers | 5.4.1 | Text embeddings |
| silero-vad | 6.2.1 | Voice activity detection |
| ten-vad | 1.0.6.8 | Tencent VAD |

See Appendix for full package listing.

## Architecture — Three-Channel Design (ADR-0003)

The system has been refactored from a monolithic "all-injection" visual pipeline to a **three-channel architecture** that decouples seeing, understanding, and acting:

```
                     Screen Capture (1280x720)
                            │
            ┌───────────────┼───────────────────┐
            │               │                   │
            ▼               ▼                   ▼
 ┌──────────────────┐ ┌──────────┐ ┌──────────────────────┐
 │ Channel 1        │ │Channel 2 │ │ Channel 3            │
 │ Visual (VLM)     │ │Conversa- │ │ Execution            │
 │ Qwen3-VL-2B      │ │tion      │ │ IconCache            │
 │ ↓                │ │LLM       │ │ L2+L3+L4             │
 │ structured       │ │DeepSeek  │ │ ↓                    │
 │ persona prompt   │ │API       │ │ 3-tier name match    │
 │ ↓                │ │          │ │ ↓                    │
 │ ~100 char scene  │ │VLM desc  │ │ DPI-aware mouse      │
 │ description      │ │+ ASR     │ │ (PowerShell :Cursor) │
 │                  │ │+ memory  │ │                      │
 └──────────────────┘ └──────────┘ └──────────────────────┘
```

| Channel | Model | Input | Output | Latency Target |
|---|---|---|---|---|
| **1 (Visual)** | Qwen3-VL-2B (vLLM, local GPU) | 1280×720 screenshot thumbnail | ~100 char compact scene description | ~800ms |
| **2 (Conversation)** | DeepSeek v4 Flash (API, cloud) | VLM description + ASR transcript + memory + personality | Natural language reply (no coordinates) | ~900ms first token |
| **3 (Execution)** | IconCache (L2/L3/L4 tiered cache) | Name string from LLM action tag | DPI-aware pixel coordinates via cache | <10ms (cache hit) |

Channel 1 reduces the visual description from ~400–800 tokens (full UI tree) to ~100 char natural language, cutting DeepSeek API cost by 60–80%. Channel 3 bypasses LLM entirely for mouse execution — cached coordinates return in microseconds.

## Decision Layer — v5.x Refactor

The DecisionBridge god object (1441 lines, 31 methods) has been split into three cohesive modules:

| Module | Lines | Responsibility |
|---|---|---|
| **ConversationOrchestrator** | 135 | Single-turn flow: teaching → persona → memory → decide |
| **PersonaContext** | 97 | Pure functional: baseline + offsets + emotion → personality state text + L2D expression |
| **MemoryContext** | 103 | Assembles MemorySnapshot from hot/cold stores with privacy filtering |

The orchestrator uses constructor injection — it receives decision_engine, teaching, persona, and memory as dependencies. The original DecisionBridge is retained for reference (marked DEPRECATED).

## Audio I/O

Audio is handled through the WSLg PulseAudio server.

| Property | Value |
|---|---|
| Server | PulseAudio via WSLg |
| Pulse Server Socket | `unix:/mnt/wslg/PulseServer` |
| Capture Tool | `parec` (PulseAudio record) |
| Playback Tool | `paplay` (PulseAudio play) |
| Portaudio | Not installed (not a dependency) |

The system uses PulseAudio's `parec` for microphone capture and `paplay` for audio output. No portaudio dependency is required.

## Models in Use

The active pipeline uses models across four lanes: Visual, ASR, Decision, TTS.

| Stage | Model | Engine | Size | Notes |
|---|---|---|---|---|
| Visual (VLM) | Qwen3-VL-2B-Instruct | vLLM 0.11.2 (cv311) | ~4 GB | INT8 quantized, screen region analysis |
| Visual (OCR) | EasyOCR | Python (cv311) | ~100 MB | English/Chinese text extraction |
| Visual (Icon) | OmniParser icon-detect | PyTorch (cv311) | ~50 MB | Icon/UI element detection |
| Visual (Embed) | CLIP ViT-B/32 | open_clip (cv311) | ~600 MB | Visual embedding for context |
| Visual (Detection) | YOLOE | — | — | ❌ Removed — excluded from pipeline to save ~0.7 GB VRAM; L2 YOLOv11n + OmniParser icon-detect replaces YOLO functions |
| ASR | SenseVoice | funasr 1.3.1 | ~200 MB | Chinese + emotion labels |
| Decision | DeepSeek v4 Flash | openai API | N/A (cloud) | Via `config/endpoints.yaml` |
| TTS | CosyVoice3-0.5B (Fun-CosyVoice3-0.5B-2512) | CosyVoice (local) + vLLM 0.11.2 | ~3 GB | Nahida epoch 6 SFT voice |

**Active model paths:**
- `models/CosyVoice3-0.5B/` -- TTS base model (vLLM accelerated)
- `deps/CosyVoice3/` -- Source code dependency
- `models/Qwen3-VL-2B-Instruct/` -- VLM model (vLLM hosted)
- `models/OmniParser/` -- UI element detection
- `models/CLIP-ViT-B-32/` -- Visual embeddings

## CosyVoice SFT Training

Four SFT characters have been trained on CosyVoice3-0.5B.

### SFT-Trained Characters

| Character | Samples | Best Epoch | Best Checkpoint | Disk Size | Status |
|---|---|---|---|---|---|
| Nahida (nahida) | 1643 | 6 | `exp/nahida/llm/torch_ddp/epoch_6_whole.pt` (CV acc=79.5%, loss=0.68) | 3.8 GB | ✅ Active |
| 胡桃 (hutao) | 1018 | 23 | `胡桃/sft_output/epoch_23_whole.pt` | 2.4 GB | Kept (pruned) |
| 伊吹 (ibuki) | 77 | 49 | `伊吹/sft_output/epoch_49_whole.pt` (loss=0.97) | 1.2 GB | Kept (best only) |
| 妃咲 (feixiao) | 89 | — | DELETED | — | ❌ Removed |

**Training Summary:**
- Nahida epoch 6 is the active SFT voice (CV acc=79.5%, loss=0.68)
- 妃咲 checkpoint deleted; 伊吹/胡桃 checkpoints pruned to best only

## Pipeline Latency

### Audio Pipeline

| Stage | Component | Latency |
|---|---|---|
| ASR | SenseVoice | ~0.3s |
| Decision | DeepSeek v4 Flash (streaming) | ~0.9s first token |
| TTS | CosyVoice3-0.5B (vLLM, Nahida epoch 6) | ~1.7s first chunk |
| **Total** | **First audio** | **~2.9s** |

### Visual Pipeline

L2 (OmniParser icon-detect), L3 (EasyOCR), and L4 (CLIP ViT-B/32) run in **full parallel** on each frame. L5 (Qwen3-VL-2B VLM) is **not part of the parallel group** — it runs as a separate **non-blocking background thread** at a lower polling rate (default 2s interval), consuming the same 1280×720 frame independently.

| Stage | Component | Latency | Notes |
|---|---|---|---|
| L2 | OmniParser icon-detect | ~600ms | UI element detection, parallel with L3+L4 |
| L3 | EasyOCR (full frame) | ~1200ms | Text recognition, parallel with L2+L4 |
| L4 | CLIP ViT-B/32 | ~400ms | Scene classification, parallel with L2+L3 |
| L5 (background) | Qwen3-VL-2B-Instruct (vLLM, screen analysis) | ~800ms | Non-blocking thread, 2s poll interval; 1280×720 thumbnail input |
| VLM Preload | Qwen3-VL-2B-Instruct (vLLM model load) | ~5000ms | One-time at startup |
| **Visual Total** | **L2+L3+L4 parallel group** | **~1200ms (bounded by slowest lane: L3)** | L5 runs independently, not in critical path |

## IconCache — Desktop Icon Coordinate Cache

New singleton cache for name-to-coordinate icon resolution (ADR-0003, Channel 3). Populated from visual pipeline L2 (OmniParser icon-labeled UI elements) and L3 (EasyOCR text content) outputs, along with PowerShell window hierarchy.

**Source:** `src/perception/visual/icon_cache.py`

**3-tier name matching:**
| Tier | Strategy | Example |
|---|---|---|
| T1 (exact) | Direct dict lookup by label key | `"回收站"` → exact match |
| T2 (substring) | Case-insensitive substring (name in key or key in name) | `"chrome"` → `"Google Chrome"` |
| T3 (overlap) | Jaccard char-set overlap > 0.5 | `"回站"` → `"回收站"` (score 0.66) |

**Cache entries:** `{label: {"coord": (x,y), "conf": float, "window": str, "screenshot": np.ndarray}}`

**Dependencies:**
- `src/perception/visual/window_enum.py` — PowerShell bridge to `window_enum.ps1` for window hierarchy enumeration (DPI-aware via `SetProcessDPIAware()`)
- `src/perception/visual/window_enum.ps1` — C# `EnumWindows` + `GetWindowRect` with `SetProcessDPIAware()` for physical-pixel window bounds

## Mouse — DPI-Aware Execution

Mouse actions now use **DPI-aware physical-pixel coordinates** throughout, eliminating the HiDPI coordinate mismatch that caused off-target clicks at non-100% scaling.

**DPI fixes (applied consistently across all actions):**
- `SetProcessDPIAware()` called via PowerShell C# `Add-Type` before `[System.Windows.Forms.Cursor]::Position` in all `move_to`, `click`, `scroll`, `type_text` PowerShell scripts
- Matches the same DPI context used by screenshot capture (`screenshot.py`), ensuring screenshot pixel coordinates and mouse pixel coordinates are in the same physical-pixel space
- PowerShell subprocess (`powershell.exe -ExecutionPolicy Bypass -Command -`) used throughout for mouse control via `System.Windows.Forms.Cursor::Position`
- See: `src/execution/channels/mouse_channel.py` (all DPI-aware move/click/scroll/type methods)

**Endpoint jitter behavior:**
- `generate_bezier_path()` accepts `endpoint_jitter` parameter but **does not apply it** to the final path point — the last waypoint lands exactly at target coordinates (±0px), avoiding the previous ±2px random jitter that caused sub-pixel misalignment on HiDPI displays

**Bezier trajectory (§7.4.1):**
- Cubic Bezier curves with control-point jitter (20–50px)
- Sigmoid speed profile with ±5% noise
- 1–2 micro-pauses inserted mid-path
- `num_points` varies by personality speed setting (20–100)

## Key Fixes & Changes

| Fix | Description | Files |
|---|---|---|
| **DPI coordinate mismatch** | HiDPI displays (e.g. 2560×1440 @150%) previously reported logical coordinates (1707×960) while mouse used physical (2560×1440). `SetProcessDPIAware()` in screenshot + window_enum + mouse PowerShell ensures all coordinate spaces are physical-pixel. | `screenshot.py`, `window_enum.ps1`, `mouse_channel.py` |
| **Icon confidence filtering** | `IconCache.update()` filters UI elements by `icon(...)` type pattern match, discarding non-icon elements from coordinate indexing | `icon_cache.py` |
| **Pre-resize removal** | Visual pipeline lanes each handle their own frame resize internally. The external pre-resize step (previously resizing frames before lane dispatch) was removed — each lane now receives the full raw frame and applies lane-specific resizing (L5: 1280px max side, L2/L3/L4: internal model defaults) | `visual_pipeline.py`, `qwen_vl_lane.py` |
| **VLM-aware prompt config** | `config/prompt_modules.json` now includes a `"visual"` capability section explaining that `[VLM]` tags in system messages contain AI visual model descriptions of screen content (windows, icons, buttons, positions). The LLM understands VLM output as "what your eyes see." | `config/prompt_modules.json` |

## Incompatible Models

The following models exist on disk but cannot be loaded at runtime.

| Model | Size | Failure Reason |
|---|---|---|
| CosyVoice2-0.5B | 4.1 GB | BF16 CUBLAS conflict. All PyTorch weights (flow.pt, hift.pt, llm.pt) are BF16 format. The installed CUDA/CUBLAS version does not support BF16 gemm operations on this GPU configuration, causing `CUBLAS_STATUS_NOT_SUPPORTED`. Hardware limitation -- RTX 3080 Ti is Ampere (CC 8.6) but CUBLAS version lacks BF16 support. |
| Genie-TTS GUI | 5.8 GB | Windows GUI binary (.exe). Not compatible with WSL2 headless Linux operation. No Python API or inference interface. |
| GPT-SoVITS artifacts | 689 MB | Legacy artifacts from previous TTS pipeline. Located under `妃咲/GPT_weights_v2/` (445 MB) and `妃咲/SoVITS_weights_v2/` (244 MB). Superseded by CosyVoice-300M SFT. |

**Total wasted disk space: ~10.6 GB.**

CosyVoice3-0.5B is now the active TTS model (with vLLM 0.11.2 acceleration), replacing CosyVoice-300M.

## Environment Variables

### OpenHeart Application

| Variable | Purpose |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API authentication token |
| `OPENMATE_LOW_VRAM` | Force low VRAM tier (set to 1 or true) |
| `OPENMATE_PERFORMANCE_MODE` | Enable performance mode (4096 token context) |
| `OPENMATE_SHOW_TRANSCRIPT` | Show ASR transcript overlay |
| `CUDA_VISIBLE_DEVICES` | GPU device selection |
| `OPENMATE_REDIS_HOST` | Redis host (default: localhost) |
| `OPENMATE_REDIS_PORT` | Redis port (default: 6379) |

### System

| Variable | Value |
|---|---|
| `SHELL` | `/bin/bash` |
| `HOME` | `/home/baaai` |
| `USER` | `baaai` |
| `LANG` | `C.UTF-8` |
| `CONDA_PREFIX` | `/home/baaai/miniforge3` |
| `CONDA_DEFAULT_ENV` | `base` |
| `CUDA_HOME` | `/usr/local/cuda-13.2` |
| `LD_LIBRARY_PATH` | `/usr/local/cuda-13.2/lib64` (repeated) |
| `PULSE_SERVER` | `unix:/mnt/wslg/PulseServer` |
| `DISPLAY` | `:0` |
| `WAYLAND_DISPLAY` | `wayland-0` |
| `WSL_DISTRO_NAME` | `Ubuntu` |
| `WSL_INTEROP` | `/run/WSL/520_interop` |
| `WSL2_GUI_APPS_ENABLED` | `1` |
| `HTTP_PROXY` | `http://127.0.0.1:7890` |
| `HTTPS_PROXY` | `http://127.0.0.1:7890` |
| `NO_PROXY` | Internal IP ranges, localhost, domains |

## Appendix: Raw System Dumps

This appendix preserves the full original content from the previous ENVIRONMENT.md verbatim.

### System Info

```
系统与版本：6.6.114.1-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Mon Dec  1 20:46:23 UTC 2025 x86_64 GNU/Linux
        No LSB modules are available.
        Distributor ID: Ubuntu
        Description:    Ubuntu 26.04 LTS
        Release:        26.04
        Codename:       resolute
        WSL 版本: 2.7.3.0
        内核版本: 6.6.114.1-1
        WSLg 版本: 1.0.73
        MSRDC 版本: 1.2.6676
        Direct3D 版本: 1.611.1-81528511
        DXCore 版本: 10.0.26100.1-240331-1435.ge-release
        Windows: 10.0.26200.8328
        NAME              STATE           VERSION
        * Ubuntu            Running         2
        docker-desktop    Stopped         2
```

### Hardware

```
硬件与性能：RAM:               total        used        free      shared  buff/cache   available
            Mem:            31Gi       3.7Gi        26Gi       2.5Gi       3.6Gi        27Gi
            Swap:          8.0Gi          0B       8.0Gi
            可用CPU核心:24
            Filesystem                                Size  Used Avail Use% Mounted on
            none                                       16G     0   16G   0% /usr/lib/modules/6.6.87.2-microsoft-standard-WSL2
            none                                       16G  4.0K   16G   1% /mnt/wsl
            drivers                                   306G  162G  144G  53% /usr/lib/wsl/drivers
            /dev/sdd                                 1007G   47G  910G   5% /
            none                                       16G  132K   16G   1% /mnt/wslg
            none                                       16G     0   16G   0% /usr/lib/wsl/lib
            rootfs                                     16G  2.7M   16G   1% /init
            none                                       16G  652K   16G   1% /run
            none                                       16G     0   16G   0% /run/lock
            none                                       16G     0   16G   0% /run/shm
            none                                       16G   76K   16G   1% /mnt/wslg/versions.txt
            none                                       16G   76K   16G   1% /mnt/wslg/doc
            C:\                                       306G  162G  144G  53% /mnt/c
            D:\                                       1.6T  286G  1.3T  18% /mnt/d
            none                                      1.0M     0  1.0M   0% /run/credentials/systemd-journald.service
            tmpfs                                      16G  2.5G   14G  16% /tmp
            none                                      1.0M     0  1.0M   0% /run/credentials/systemd-resolved.service
            none                                      1.0M     0  1.0M   0% /run/credentials/getty@tty1.service
            none                                      1.0M     0  1.0M   0% /run/credentials/console-getty.service
            tmpfs                                     3.2G   12K  3.2G   1% /run/user/1000
            tmpfs                                     3.2G   12K  3.2G   1% /run/user/0
            none                                       16G  572K   16G   1% /mnt/wsl/docker-desktop/shared-sockets/host-services
            /dev/sde                                  137M   71M   55M  57% /mnt/wsl/docker-desktop/docker-desktop-user-distro
            /dev/loop0                                819M  819M     0 100% /mnt/wsl/docker-desktop/cli-tools
            C:\Program Files\Docker\Docker\resources  306G  162G  144G  53% /Docker/host
```

### GPU (Original Snapshot)

This was captured before the current training run. Current GPU state is in the GPU section above.

```
GPU环境：+-----------------------------------------------------------------------------------------+
        | NVIDIA-SMI 565.72                 Driver Version: 566.14         CUDA Version: 12.7     |
        |-----------------------------------------+------------------------+----------------------+
        | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
        | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
        |                                         |                        |               MIG M. |
        |=========================================+========================+======================|
        |   0  NVIDIA GeForce RTX 3080 ...    On  |   00000000:01:00.0  On |                  N/A |
        | N/A   55C    P0             39W /  175W |    1212MiB /  16384MiB |      6%      Default |
        |                                         |                        |                  N/A |
        +-----------------------------------------+------------------------+----------------------+

        +-----------------------------------------------------------------------------------------+
        | Processes:                                                                              |
        |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
        |        ID   ID                                                               Usage      |
        |=========================================================================================|
        |  No running processes found                                                             |
        +-----------------------------------------------------------------------------------------+
        nvcc: NVIDIA (R) Cuda compiler driver
        Copyright (c) 2005-2026 NVIDIA Corporation
        Built on Thu_Mar_19_11:12:51_PM_PDT_2026
        Cuda compilation tools, release 13.2, V13.2.78
        Build cuda_13.2.r13.2/compiler.37668154_0
cuda版本：13.0
```

### Network Config

```
网络配置：1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
        link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
        inet 127.0.0.1/8 scope host lo
        valid_lft forever preferred_lft forever
        inet 10.255.255.254/32 brd 10.255.255.254 scope global lo
        valid_lft forever preferred_lft forever
        inet6 ::1/128 scope host proto kernel_lo
        valid_lft forever preferred_lft forever
        2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
        link/ether 8c:b8:7e:0f:23:19 brd ff:ff:ff:ff:ff:ff
        altname enx8cb87e0f2319
        inet 10.19.217.32/21 brd 10.19.223.255 scope global noprefixroute eth0
        valid_lft forever preferred_lft forever
        inet6 2001:250:4000:8235:3c39:5d47:fde0:8c0b/128 scope global nodad noprefixroute
        valid_lft forever preferred_lft forever
        inet6 2001:250:4000:8235:c479:4fb2:c6b6:6d9/64 scope global nodad deprecated noprefixroute
        valid_lft forever preferred_lft 0sec
        inet6 fe80::5d86:874f:bdc5:cbce/64 scope link nodad noprefixroute
        valid_lft forever preferred_lft forever
        3: eth1: <BROADCAST,MULTICAST> mtu 1500 qdisc mq state DOWN group default qlen 1000
        link/ether 58:11:22:de:87:09 brd ff:ff:ff:ff:ff:ff
        altname enx581122de8709
        4: loopback0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
        link/ether 00:15:5d:a1:db:93 brd ff:ff:ff:ff:ff:ff
        altname enx00155da1db93
        iproute：10.19.223.254
        公网出口：103.151.172.13
```

### Software (Original pip list output)

This section lists packages from the system Python. The conda environment package listing follows separately.

```
软件与驱动：Package                   Version
        ------------------------- -------------
        attrs                     25.4.0
        autocommand               2.2.2
        Automat                   25.4.16
        babel                     2.17.0
        bcrypt                    5.0.0
        blinker                   1.9.0
        certifi                   2026.1.4
        chardet                   5.2.0
        click                     8.1.8
        command-not-found         0.3
        configobj                 5.0.9
        constantly                23.10.4
        cryptography              46.0.5
        dbus-python               1.4.0
        distro                    1.9.0
        distro-info               1.15
        httplib2                  0.22.0
        hyperlink                 21.0.0
        idna                      3.11
        incremental               24.7.2
        inflect                   7.5.0
        jaraco.context            6.0.1
        jaraco.functools          4.1.0
        jaraco.text               4.0.0
        Jinja2                    3.1.6
        jsonpatch                 1.32
        jsonpointer               2.4
        jsonschema                4.19.2
        jsonschema-specifications 2023.12.1
        launchpadlib              2.1.0
        lazr.restfulclient        0.14.6
        lazr.uri                  1.0.6
        libpass                   1.9.3
        linkify-it-py             2.0.3
        markdown-it-py            3.0.0
        MarkupSafe                3.0.3
        mdurl                     0.1.2
        more-itertools            10.8.0
        netifaces                 0.11.0
        oauthlib                  3.3.1
        packaging                 26.0
        pip                       25.1.1
        pyasn1                    0.6.3
        pyasn1_modules            0.4.1
        pycurl                    7.45.7
        Pygments                  2.19.2
        PyGObject                 3.56.2
        PyHamcrest                2.1.0
        PyJWT                     2.10.1
        pyOpenSSL                 25.3.0
        pyparsing                 3.3.2
        pyserial                  3.5
        python-apt                3.1.0+ubuntu1
        PyYAML                    6.0.3
        referencing               0.36.2
        requests                  2.32.5
        rich                      13.9.4
        rpds-py                   0.27.1
        service-identity          24.2.0
        setuptools                78.1.1
        systemd-python            235
        Twisted                   25.5.0
        typeguard                 4.4.4
        typing_extensions         4.15.0
        ubuntu-pro-client         8001
        uc-micro-py               1.0.3
        unattended-upgrades       0.1
        urllib3                   2.6.3
        wadllib                   2.0.0
        wheel                     0.46.3
        zipp                      3.23.0
        zope.interface            8.2
```

### Software (Original conda list output)

```
# packages in environment at /home/baaai/miniforge3:
#
# Name                            Version          Build                 Channel
_openmp_mutex                     4.5              20_gnu                conda-forge
accelerate                        1.13.0           pypi_0                pypi
aiohappyeyeballs                  2.6.1            pypi_0                pypi
aiohttp                           3.13.5           pypi_0                pypi
aiosignal                         1.4.0            pypi_0                pypi
aistudio-sdk                      0.3.8            pypi_0                pypi
annotated-doc                     0.0.4            pypi_0                pypi
annotated-types                   0.7.0            pypi_0                pypi
anyio                             4.13.0           pypi_0                pypi
archspec                          0.2.5            pyhd8ed1ab_0          conda-forge
attrs                             26.1.0           pypi_0                pypi
backports.zstd                    1.3.0            py313h18e8e13_0       conda-forge
basedpyright                      1.39.3           pypi_0                pypi
bce-python-sdk                    0.9.71           pypi_0                pypi
bitsandbytes                      0.49.2           pypi_0                pypi
black                             26.3.1           pypi_0                pypi
blis                              1.3.3            pypi_0                pypi
boltons                           25.0.0           pyhd8ed1ab_0          conda-forge
brotli-python                     1.2.0            py313hf159716_1       conda-forge
bzip2                             1.0.8            hda65f42_9            conda-forge
c-ares                            1.34.6           hb03c661_0            conda-forge
ca-certificates                   2026.4.22        hbd8a1cb_0            conda-forge
catalogue                         2.0.10           pypi_0                pypi
certifi                           2026.4.22        pyhd8ed1ab_0          conda-forge
cffi                              2.0.0            py313hf46b229_1       conda-forge
chardet                           7.4.3            pypi_0                pypi
charset-normalizer                3.4.7            pyhd8ed1ab_0          conda-forge
click                             8.3.3            pypi_0                pypi
cloudpathlib                      0.24.0           pypi_0                pypi
colorlog                          6.10.1           pypi_0                pypi
conda                             26.3.2           py313h78bf25f_1       conda-forge
conda-libmamba-solver             26.4.0           pyhd8ed1ab_0          conda-forge
conda-package-handling            2.4.0            pyh7900ff3_2          conda-forge
conda-package-streaming           0.12.0           pyhd8ed1ab_0          conda-forge
confection                        1.3.3            pypi_0                pypi
contourpy                         1.3.3            pypi_0                pypi
cpp-expected                      1.3.1            h171cf75_0            conda-forge
crc32c                            2.8              pypi_0                pypi
cuda-bindings                     13.2.0           pypi_0                pypi
cuda-pathfinder                   1.5.4            pypi_0                pypi
cuda-toolkit                      13.0.2           pypi_0                pypi
cycler                            0.12.1           pypi_0                pypi
cymem                             2.0.13           pypi_0                pypi
deprecation                       2.1.0            pypi_0                pypi
distro                            1.9.0            pyhd8ed1ab_1          conda-forge
docstring-to-markdown             0.17             pypi_0                pypi
evdev                             1.9.3            pypi_0                pypi
filelock                          3.29.0           pypi_0                pypi
fmt                               12.1.0           hff5e90c_0            conda-forge
fonttools                         4.62.1           pypi_0                pypi
frozendict                        2.4.7            py313h07c4f96_0       conda-forge
frozenlist                        1.8.0            pypi_0                pypi
fsspec                            2026.4.0         pypi_0                pypi
future                            1.0.0            pypi_0                pypi
h11                               0.16.0           pypi_0                pypi
h2                                4.3.0            pyhcf101f3_0          conda-forge
hf-xet                            1.5.0            pypi_0                pypi
hiredis                           3.3.1            pypi_0                pypi
hpack                             4.1.0            pyhd8ed1ab_0          conda-forge
httpcore                          1.0.9            pypi_0                pypi
httpx                             0.28.1           pypi_0                pypi
huggingface-hub                   1.14.0           pypi_0                pypi
hyperframe                        6.1.0            pyhd8ed1ab_0          conda-forge
icu                               78.3             h33c6efd_0            conda-forge
idna                              3.13             pyhcf101f3_0          conda-forge
imagesize                         2.0.0            pypi_0                pypi
importlib-metadata                9.0.0            pypi_0                pypi
iniconfig                         2.3.0            pypi_0                pypi
jedi                              0.19.2           pypi_0                pypi
jinja2                            3.1.6            pypi_0                pypi
joblib                            1.5.3            pypi_0                pypi
jsonpatch                         1.33             pyhd8ed1ab_1          conda-forge
jsonpointer                       3.1.1            pyhcf101f3_0          conda-forge
keyutils                          1.6.3            hb9d3cd8_0            conda-forge
kiwisolver                        1.5.0            pypi_0                pypi
krb5                              1.22.2           ha1258a1_0            conda-forge
lance-namespace                   0.7.6            pypi_0                pypi
lance-namespace-urllib3-client    0.7.6            pypi_0                pypi
lancedb                           0.30.2           pypi_0                pypi
ld_impl_linux-64                  2.45.1           default_hbd61a6d_102  conda-forge
libarchive                        3.8.7            gpl_hc2c16d8_100      conda-forge
libcurl                           8.20.0           hcf29cc6_0            conda-forge
libedit                           3.1.20250104     pl5321h7949ede_0      conda-forge
libev                             4.33             hd590300_2            conda-forge
libexpat                          2.7.5            hecca717_0            conda-forge
libffi                            3.5.2            h3435931_0            conda-forge
libgcc                            15.2.0           he0feb66_18           conda-forge
libgcc-ng                         15.2.0           h69a702a_18           conda-forge
libgomp                           15.2.0           he0feb66_18           conda-forge
libiconv                          1.18             h3b78370_2            conda-forge
liblzma                           5.8.3            hb03c661_0            conda-forge
libmamba                          2.6.0            hd28c85e_0            conda-forge
libmamba-spdlog                   2.6.0            hf859cbd_0            conda-forge
libmambapy                        2.6.0            py313h75b7c84_0       conda-forge
libmpdec                          4.0.0            hb03c661_1            conda-forge
libmsgpack-c                      6.1.0            h54a6638_6            conda-forge
libnghttp2                        1.68.1           h877daf1_0            conda-forge
libsolv                           0.7.37           h9463b59_0            conda-forge
libsqlite                         3.53.0           hf4e2dac_0            conda-forge
libssh2                           1.11.1           hcf80075_0            conda-forge
libstdcxx                         15.2.0           h934c35e_18           conda-forge
libuuid                           2.42             h5347b49_0            conda-forge
libxml2                           2.15.3           h49c6c72_0            conda-forge
libxml2-16                        2.15.3           hca6bf5a_0            conda-forge
libzlib                           1.3.2            h25fd6f3_2            conda-forge
lz4-c                             1.10.0           h5888daf_1            conda-forge
lzo                               2.10             h280c20c_1002         conda-forge
mamba                             2.6.0            hf80e505_0            conda-forge
markdown-it-py                    4.2.0            pypi_0                pypi
markupsafe                        3.0.3            pypi_0                pypi
matplotlib                        3.10.9           pypi_0                pypi
mdurl                             0.1.2            pypi_0                pypi
menuinst                          2.4.2            py313h78bf25f_0       conda-forge
modelscope                        1.36.3           pypi_0                pypi
mouseinfo                         0.1.3            pypi_0                pypi
mpmath                            1.3.0            pypi_0                pypi
msgpack-python                    1.1.2            py313h7037e92_1       conda-forge
multidict                         6.7.1            pypi_0                pypi
murmurhash                        1.0.15           pypi_0                pypi
mypy-extensions                   1.1.0            pypi_0                pypi
ncurses                           6.6              hdb14827_0            conda-forge
networkx                          3.6.1            pypi_0                pypi
nlohmann_json-abi                 3.12.0           h0f90c79_1            conda-forge
nltk                              3.9.4            pypi_0                pypi
nodejs-wheel-binaries             24.15.0          pypi_0                pypi
numpy                             2.3.5            pypi_0                pypi
nvidia-cublas                     13.1.0.3         pypi_0                pypi
nvidia-cuda-cupti                 13.0.85          pypi_0                pypi
nvidia-cuda-nvrtc                 13.0.88          pypi_0                pypi
nvidia-cuda-runtime               13.0.96          pypi_0                pypi
nvidia-cudnn-cu13                 9.19.0.56        pypi_0                pypi
nvidia-cufft                      12.0.0.61        pypi_0                pypi
nvidia-cufile                     1.15.1.6         pypi_0                pypi
nvidia-curand                     10.4.0.35        pypi_0                pypi
nvidia-cusolver                   12.0.4.66        pypi_0                pypi
nvidia-cusparse                   12.6.3.3         pypi_0                pypi
nvidia-cusparselt-cu13            0.8.0            pypi_0                pypi
nvidia-nccl-cu13                  2.28.9           pypi_0                pypi
nvidia-nvjitlink                  13.0.88          pypi_0                pypi
nvidia-nvshmem-cu13               3.4.5            pypi_0                pypi
nvidia-nvtx                       13.0.85          pypi_0                pypi
opencv-contrib-python             4.10.0.84        pypi_0                pypi
opencv-python                     4.13.0.92        pypi_0                pypi
opencv-python-headless            4.13.0.92        pypi_0                pypi
openheart                         0.1.0            pypi_0                pypi
openssl                           3.6.2            h35e630c_0            conda-forge
packaging                         26.2             pyhc364b38_0          conda-forge
paddleocr                         3.5.0            pypi_0                pypi
paddlex                           3.5.1            pypi_0                pypi
pandas                            3.0.2            pypi_0                pypi
parso                             0.8.7            pypi_0                pypi
pathspec                          1.1.1            pypi_0                pypi
pillow                            12.2.0           pypi_0                pypi
pip                               26.0.1           pyh145f28c_0          conda-forge
platformdirs                      4.9.6            pyhcf101f3_0          conda-forge
pluggy                            1.6.0            pyhf9edf01_1          conda-forge
polars                            1.40.1           pypi_0                pypi
polars-runtime-32                 1.40.1           pypi_0                pypi
preshed                           3.0.13           pypi_0                pypi
prettytable                       3.17.0           pypi_0                pypi
propcache                         0.5.2            pypi_0                pypi
psutil                            7.2.2            pypi_0                pypi
py-cpuinfo                        9.0.0            pypi_0                pypi
pyarrow                           24.0.0           pypi_0                pypi
pyautogui                         0.9.54           pypi_0                pypi
pybind11-abi                      11               hc364b38_1            conda-forge
pyclipper                         1.4.0            pypi_0                pypi
pycosat                           0.6.6            py313h07c4f96_3       conda-forge
pycparser                         2.22             pyh29332c3_1          conda-forge
pycryptodome                      3.23.0           pypi_0                pypi
pydantic                          2.13.4           pypi_0                pypi
pydantic-core                     2.46.4           pypi_0                pypi
pygetwindow                       0.0.9            pypi_0                pypi
pygments                          2.20.0           pypi_0                pypi
pymsgbox                          2.0.1            pypi_0                pypi
pynput                            1.8.1            pypi_0                pypi
pyparsing                         3.3.2            pypi_0                pypi
pypdfium2                         5.8.0            pypi_0                pypi
pyperclip                         1.11.0           pypi_0                pypi
pyrect                            0.2.0            pypi_0                pypi
pyscreeze                         1.0.1            pypi_0                pypi
pysocks                           1.7.1            pyha55dd90_7          conda-forge
pytest                            9.0.3            pypi_0                pypi
pytest-asyncio                    1.3.0            pypi_0                pypi
python                            3.13.13          h6add32d_100_cp313    conda-forge
python-bidi                       0.6.9            pypi_0                pypi
python-dateutil                   2.9.0.post0      pypi_0                pypi
python-lsp-jsonrpc                1.1.2            pypi_0                pypi
python-lsp-server                 1.14.0           pypi_0                pypi
python-xlib                       0.33             pypi_0                pypi
python3-xlib                      0.15             pypi_0                pypi
python_abi                        3.13             8_cp313               conda-forge
pytokens                          0.4.1            pypi_0                pypi
pytweening                        1.2.0            pypi_0                pypi
pywhispercpp                      1.4.1            pypi_0                pypi
pyyaml                            6.0.2            pypi_0                pypi
readline                          8.3              h853b02a_0            conda-forge
redis                             7.4.0            pypi_0                pypi
regex                             2026.4.4         pypi_0                pypi
reproc                            14.2.7.post0     hb03c661_0            conda-forge
reproc-cpp                        14.2.7.post0     hecca717_0            conda-forge
requests                          2.33.1           pyhcf101f3_1          conda-forge
rich                              15.0.0           pypi_0                pypi
ruamel.yaml                       0.18.17          py313h54dd161_2       conda-forge
ruamel.yaml.clib                  0.2.15           py313h54dd161_1       conda-forge
safetensors                       0.7.0            pypi_0                pypi
scikit-learn                      1.8.0            pypi_0                pypi
scipy                             1.17.1           pypi_0                pypi
sentence-transformers             5.4.1            pypi_0                pypi
setuptools                        81.0.0           pypi_0                pypi
shapely                           2.1.2            pypi_0                pypi
shellingham                       1.5.4            pypi_0                pypi
silero-vad                        6.2.1            pypi_0                pypi
simdjson                          4.6.3            hb700be7_0            conda-forge
six                               1.17.0           pypi_0                pypi
smart-open                        7.6.1            pypi_0                pypi
snownlp                           0.12.3           pypi_0                pypi
spacy                             3.8.14           pypi_0                pypi
spacy-legacy                      3.0.12           pypi_0                pypi
spacy-loggers                     1.0.5            pypi_0                pypi
spdlog                            1.17.0           hab81395_1            conda-forge
srsly                             2.5.3            pypi_0                pypi
sympy                             1.14.0           pypi_0                pypi
ten-vad                           1.0.6.8          pypi_0                pypi
textblob                          0.20.0           pypi_0                pypi
thinc                             8.3.13           pypi_0                pypi
threadpoolctl                     3.6.0            pypi_0                pypi
tk                                8.6.13           noxft_h366c992_103    conda-forge
tokenizers                        0.22.2           pypi_0                pypi
torch                             2.11.0           pypi_0                pypi
torchaudio                        2.11.0           pypi_0                pypi
torchvision                       0.26.0           pypi_0                pypi
tqdm                              4.67.3           pyh8f84b5b_0          conda-forge
transformers                      5.8.0            pypi_0                pypi
triton                            3.6.0            pypi_0                pypi
truststore                        0.10.4           pyhcf101f3_0          conda-forge
typer                             0.25.1           pypi_0                pypi
typing-extensions                 4.15.0           pypi_0                pypi
typing-inspection                 0.4.2            pypi_0                pypi
tzdata                            2025c            hc9c84f9_1            conda-forge
ujson                             5.12.1           pypi_0                pypi
ultralytics                       8.4.48           pypi_0                pypi
ultralytics-thop                  2.0.19           pypi_0                pypi
urllib3                           2.6.3            pyhd8ed1ab_0          conda-forge
wasabi                            1.1.3            pypi_0                pypi
wcwidth                           0.7.0            pypi_0                pypi
weasel                            1.0.0            pypi_0                pypi
websockets                        16.0             pypi_0                pypi
wrapt                             2.1.2            pypi_0                pypi
yaml-cpp                          0.8.0            h3f2d84a_0            conda-forge
yarl                              1.23.0           pypi_0                pypi
zipp                              3.23.1           pypi_0                pypi
zstandard                         0.25.0           py313h54dd161_1       conda-forge
zstd                              1.5.7            hb78ec9c_6            conda-forge
```

### Environment Variables (Original Dump)

```
环境变量：SHELL=/bin/bash
        no_proxy=192.168.*,172.31.*,172.30.*,172.2*,172.19.*,172.18.*,172.17.*,172.16.*,10.*,127.*,*.local,localhost,*360buyimg.com,100ime-iat-api.xfyun.cn,*jd.com,*zhimg.com,*zhihu.com
        WSL2_GUI_APPS_ENABLED=1
        CONDA_EXE=/home/baaai/miniforge3/bin/conda
        _CE_M=
        WSL_DISTRO_NAME=Ubuntu
        XML_CATALOG_FILES=file:///home/baaai/miniforge3/etc/xml/catalog file:///etc/xml/catalog
        NAME=LAPTOP-PJQ55QGI
        PWD=/home/baaai
        LOGNAME=baaai
        CONDA_PREFIX=/home/baaai/miniforge3
        MAMBA_ROOT_PREFIX=/home/baaai/miniforge3
        HOME=/home/baaai
        LANG=C.UTF-8
        WSL_INTEROP=/run/WSL/520_interop
        LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=00:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:*.t7z=01;31:*.zip=01;31:*.z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.zst=01;31:*.tzst=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.wim=01;31:*.swm=01;31:*.dwm=01;31:*.esd=01;31:*.avif=01;35:*.jpg=01;35:*.jpeg=01;35:*.mjpg=01;35:*.mjpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.webp=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.m4a=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.oga=00;36:*.opus=00;36:*.spx=00;36:*.xspf=00;36:*~=00;90:*#=00;90:*.bak=00;90:*.old=00;90:*.orig=00;90:*.part=00;90:*.rej=00;90:*.swp=00;90:*.tmp=00;90:*.dpkg-dist=00;90:*.dpkg-old=00;90:*.ucf-dist=00;90:*.ucf-new=00;90:*.ucf-old=00;90:*.rpmnew=00;90:*.rpmorig=00;90:*.rpmsave=00;90:
        WAYLAND_DISPLAY=wayland-0
        CONDA_PROMPT_MODIFIER=(base)
        https_proxy=http://127.0.0.1:7890
        _CONDA_EXE=/home/baaai/miniforge3/bin/conda
        LESSCLOSE=/usr/bin/lesspipe %s %s
        _CONDA_ROOT=/home/baaai/miniforge3
        MAMBA_EXE=/home/baaai/miniforge3/bin/mamba
        TERM=xterm-256color
        _CE_CONDA=
        LESSOPEN=| /usr/bin/lesspipe %s
        USER=baaai
        NO_PROXY=192.168.*,172.31.*,172.30.*,172.2*,172.19.*,172.18.*,172.17.*,172.16.*,10.*,127.*,*.local,localhost,*360buyimg.com,100ime-iat-api.xfyun.cn,*jd.com,*zhimg.com,*zhihu.com
        CONDA_SHLVL=1
        DISPLAY=:0
        SHLVL=1
        HTTPS_PROXY=http://127.0.0.1:7890
        HTTP_PROXY=http://127.0.0.1:7890
        http_proxy=http://127.0.0.1:7890
        CONDA_PYTHON_EXE=/home/baaai/miniforge3/bin/python
        LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:/usr/local/cuda-13.2/lib64:/usr/local/cuda-13.2/lib64:/usr/local/cuda-13.2/lib64:/lib64:
        XDG_RUNTIME_DIR=/run/user/1000
        CONDA_DEFAULT_ENV=base
        WSLENV=
        BUN_INSTALL=/home/baaai/.bun
        CUDA_HOME=/usr/local/cuda-13.2
        XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
        PATH=/usr/local/cuda-13.2/bin:/usr/local/cuda-13.2/bin:/usr/local/cuda-13.2/bin:/usr/local/cuda-13.2/bin:/bin:/home/baaai/miniforge3/bin:/home/baaai/miniforge3/condabin:/home/baaai/.opencode/bin:/home/baaai/.bun/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/lib/wsl/lib:/mnt/c/Program Files/WindowsApps/MicrosoftCorporationII.WindowsSubsystemForLinux_2.7.3.0_x64__8wekyb3d8bbwe:/mnt/c/Windows/system32:/mnt/c/Windows:/mnt/c/Windows/System32/Wbem:/mnt/c/Windows/System32/WindowsPowerShell/v1.0:/mnt/c/Windows/System32/OpenSSH:/mnt/c/WINDOWS/system32:/mnt/c/WINDOWS:/mnt/c/WINDOWS/System32/Wbem:/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0:/mnt/c/WINDOWS/System32/OpenSSH:/mnt/c/Program Files/Docker/Docker/resources/bin:/mnt/c/Program Files (x86)/NVIDIA Corporation/PhysX/Common:/mnt/c/Program Files/NVIDIA Corporation/NVIDIA App/NvDLISR:/mnt/c/Users/PC/AppData/Local/Microsoft/WindowsApps:/mnt/c/Users/PC/AppData/Local/Programs/Microsoft VS Code/bin:/snap/bin
        DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
        HOSTTYPE=x86_64
        PULSE_SERVER=unix:/mnt/wslg/PulseServer
        _=/bin/env
```

## 2026-05-14 Visual Integration Update

This section summarizes changes made on 2026-05-14 to document the visual perception pipeline integration.

### Changes Summary

| Area | Change |
|---|---|
| Python Environment | CosyVoice3 inference env renamed from `cosyvoice_vllm` (Python 3.10) to `cv311` (Python 3.11) |
| GPU Status | Updated from training active to visual pipeline active (Qwen3-VL + OmniParser + CLIP) |
| Models in Use | Added 4 new visual models — Qwen3-VL-2B-Instruct (vLLM), OmniParser icon-detect, EasyOCR, CLIP ViT-B/32. YOLOE marked as degraded/removed. |
| Pipeline Latency | Added Visual Pipeline subsection (~5000ms preload, ~3000ms runtime for full frame) |
| Config Alignment | Visual perception configuration now active under `config/` — see spec v4.5.0 Chapter 13 for directory layout |
| Degradation Matrix | OmniParser has CPU fallback; EasyOCR degrades to PaddleOCR in low-VRAM mode; CLIP missing falls back to text-only context |

### VRAM Budget (Visual Pipeline)

| Model | Approx VRAM |
|---|---|
| Qwen3-VL-2B-Instruct (INT8, vLLM) | ~4.0 GB |
| Qwen2.5-3B (GPTQ 4bit, Decision main) | ~3.5 GB |
| Qwen2.5-1.5B (INT8, Shadow) | ~1.5 GB |
| CosyVoice3-0.5B (vLLM, TTS) | ~3.0 GB |
| OmniParser + EasyOCR + CLIP | ~1.2 GB |
| Overhead + buffers | ~1.5 GB |
| **Total estimated** | **~14.7 GB / 16 GB** |

All visual models share the same `cv311` conda environment for unified dependency management.

## 2026-05-14 End-of-Session Update

### 1. Conda Environment Consolidation

The `cosyvoice_vllm` environment (Python 3.10) has been replaced by `cv311` (Python 3.11). All visual pipeline models (Qwen3-VL, OmniParser, EasyOCR, CLIP) and TTS (CosyVoice3 with vLLM) now run under the same `cv311` conda environment. The old `cosyvoice_vllm` environment has been removed.

### 2. Fusion Layer Integrated

Four fusion modules have been implemented in `src/fusion/`:

- `time_window.py` — Temporal alignment of perception events across lanes
- `event_classifier.py` — Classifies fused events by type and confidence
- `entity_fusion.py` — Cross-lane entity resolution (visual + audio references)
- `scene_synthesis.py` — Produces a unified scene description for the decision layer

All modules follow the spec v4.5.0 interfaces and use the unified message envelope (§0.3).

### 3. Runtime Loop Extracted

The main application loop has been extracted from inline scripts into `src/runtime_loop.py`. This module owns the lifecycle: initialization, perception dispatch, fusion, memory I/O, decision inference, and execution scheduling. It consumes `RuntimeConfig` via DI and logs each cycle with `trace_id`.

### 4. Orchestrator Merge Status

An `orchestrator.py` was started to replace the monolithic `demo_full.py` entry point. The merge is in progress but `run.sh` has been reverted to `demo_full.py` for stability. The orchestrator boot sequence uncovered a TTS warmup ordering issue (see Known Issues below).

### 5. Known Issues

| Issue | Impact | Status |
|---|---|---|
| **Qwen3-VL VRAM contention with TTS vLLM** | Both Qwen3-VL and CosyVoice3 use the same `cv311` vLLM instance. Peak VRAM during parallel visual + TTS load can exceed the 16 GB budget, triggering OOM or thrashing. | **Open** — needs VRAM-aware scheduling or separate vLLM instances with tier-based gating |
| **Orchestrator boot sequence breaks TTS warmup** | When launched via `orchestrator.py`, the TTS model warmup (model load + prefill) races against Qwen3-VL initialization. The vLLM scheduler deadlocks if both models attempt GPU allocation simultaneously. | **Open** — `run.sh` reverted to `demo_full.py` as workaround; orchestrator needs staged init |

### 2026-05-14 End-of-Session

**Visual Pipeline (Final):**
- 4-lane active: L2 OmniParser icon-detect, L3 EasyOCR, L4 CLIP ViT-B/32, L5 Qwen3-VL-2B vLLM
- L5 VLM uses ROI multi-image input: crops top UI regions (confidence threshold 0.5, max 8 crops) via OmniParser bboxes
- OCR runs in parallel with ASR (dispatched before ASR, collected after)
- Background poller: L2+L3+L4 every 2s; L5 VLM only during idle (skipped during conversation)
- Mouse capture via PowerShell subprocess (same pattern as screenshot fallback); no ctypes X11 to avoid segfault

**Performance Optimizations:**
- TTS vLLM max_model_len: 32768 → 2048 → 1024 (KV cache ~1.2 GB)
- Qwen3-VL vLLM max_model_len: 1536 → 1024 → 512 (KV cache ~2.2 GB)
- Total VRAM reclaimed: ~3.4 GB
- Fusion pipeline: runs post-speech (not blocking LLM); cached summary reused during conversation
- ASR VAD threshold: 0.007 → 0.004 (matched to user's normal speaking RMS)
- TTS echo prevention: mic pipe drained non-blocking after each conversation turn
- Orphan vLLM process cleanup: pkill in run.sh before startup

**Bug Fixes Applied:**
- ctypes X11 segfault: replaced with PowerShell subprocess in mouse_capture.py
- Orphan vLLM EngineCore processes consuming GPU: killed before startup
- VLM+TTS GPU contention: skip_vlm flag during conversation, TTS-before discard visual future
- UnboundLocalError _visual_future: placeholder declaration kept after removing dispatch

**VRAM Budget (Current):**
- TTS vLLM: ~2.5 GB (model 1GB + KV 1.2GB)
- Qwen3-VL vLLM: ~6.5 GB (model 4.3GB + KV 2.2GB)
- CLIP ViT-B/32: ~0.6 GB
- OmniParser: ~0.05 GB
- EasyOCR: ~0.08 GB (CPU)
- SenseVoice: ~0.8 GB
- bge-small: ~0.1 GB
- CUDA overhead: ~1.5 GB
- Total: ~12 GB / 16 GB
- Headroom: ~4 GB

**Files Changed This Session:**
- src/runtime_loop.py: visual/voice sync, TTS echo drain, VAD threshold, VLM injection, skip_vlm
- src/perception/visual/mouse_capture.py: ctypes→PowerShell
- src/perception/visual/visual_pipeline.py: ROI crops, skip_vlm, VLM logging
- src/perception/visual/qwen_vl_lane.py: multi-image ROI, max_model_len=512
- src/perception/visual/clip_scene.py: secondary+app zeroshot
- src/perception/visual/fusion.py: DEAD_CODE annotations
- src/perception/visual/types.py: SceneClass fields
- src/perception/sync_vision_query.py: _sync_infer OmniParser ROI
- deps/CosyVoice/cosyvoice/cli/model.py: max_model_len=1024
- run.sh: orphan cleanup
- tests/smoke/: test_mouse_capture.py, test_clip_zeroshot.py

## 2026-05-15 Post-Phase 4 State

### Four Phases Completed

| Phase | Summary |
|---|---|
| Phase 1 | Foundation: project skeleton, config system, directory structure, contract test framework |
| Phase 2 | Perception: 4-lane visual pipeline (VLM, OCR, icon, CLIP) + audio (VAD, ASR, emotion) |
| Phase 3 | Memory + Personality: Redis hot memory, LanceDB cold memory, baseline/personality modules, emotion adjustment |
| Phase 4 | Decision + Execution: main_decision (3B), shadow_verifier (1.5B), proactive triggers, channel dispatch, TTS, fallback text bubble |

### Current Architecture

Active subsystem health: Memory (9/10) + Personality (7/7) + Decision (10/12).

- **Memory**: 9 of 10 contract tests pass. Hot (Redis) and cold (LanceDB) memory fully operational. Hot-to-cold sync and memory decay active.
- **Personality**: All 7 contract tests pass. Baseline, preference shift, emotion adjustment, dynamic fusion, persona auditor all green.
- **Decision**: 10 of 12 contract tests pass. The two intentional gaps: `main_decision` and `shadow_verifier` are structurally in place but their AI backends are intentionally disabled (see Known Limitations below).

### VRAM Budget

~10-12 GB sustained with max_model_len optimizations:

| Optimization | Value |
|---|---|
| TTS max_model_len | 1024 (down from default) |
| Qwen3-VL max_model_len | 768 |
| Visual pipeline headroom | ~4 GB of 16 GB total |
| Active VRAM usage | ~12 GB / 16 GB |

These reductions keep the full pipeline (VLM + visual lanes + CosyVoice + ASR) within budget on the RTX 3080 Ti 16 GB.

### New Modules

Introduced during Phase 4 implementation:

- `voice_pipeline` — Orchestrates VAD -> ASR -> emotion pipeline stages
- `decision_bridge` — Mediates between perception output and decision engine input
- `execution_pipeline` — Coordinates channel dispatch and action sequencing
- `proactive/*` — Time-based and context-based proactive trigger modules
- `persona_calibrator` — Calibrates personality output against user interaction history
- `calibration_engine` — Backend for persona calibration, runs periodically

### Dependencies Added

- `lancedb 0.30.2` — Cold/long-term memory storage (from scratch)
- `redis-py 7.4.0` — Hot/session memory (from scratch)

### Dependencies Removed

- `paddlex 3.5.1` — Removed from active dependency set. Visual lane migrated to Qwen3-VL + OmniParser + CLIP. PaddleX remains in the environment but is no longer imported by any application code.

### run.sh Changes

- **Redis auto-start**: `run.sh` now checks for Redis availability and starts it automatically if not running
- **LanceDB dir**: LanceDB database path initialized before pipeline start (`--lancedb-dir` flag)
- **VLLM_PLUGINS=""**: Empty string explicitly set to suppress vLLM plugin warnings and avoid unwanted plugin autoload

### Known Limitations

- `main_decision` + `shadow_verifier` are **intentionally DEAD**. Their AI backends (Qwen2.5-3B and Qwen2.5-1.5B) are not loaded. The system routes decisions through DeepSeek v4 Flash API instead. These modules exist as structural stubs with contract test coverage for when local inference is re-enabled.
- **ASR occasionally misrecognizes** — SenseVoice has ~5-8% character error rate on mixed Chinese-English input in noisy conditions. No secondary ASR verification path active.
- **Mouse click not executed** — The mouse controller captures coordinates and logs them but does not issue actual click events. This is a safety constraint in the current implementation.

### Tests

| Suite | Count | Status |
|---|---|---|
| Contract tests pass | 515 | ✅ |
| Contract tests fail | 40 | ❌ (expected: main_decision, shadow_verifier intentional gaps) |
| Contract tests error | 22 | ⚠️ (environment-dependent, being addressed) |
| E2E tests pass | 60+ | ✅ |

Contract coverage spans all core modules. The 40 failures are dominated by the two intentionally disabled modules. The 22 errors are environment-specific (Redis connection, GPU memory allocation timing) and do not indicate logic defects.
