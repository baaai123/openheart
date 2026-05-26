# OpenHeart

<p align="center"><strong>有温度的 AI 虚拟伙伴 — A Virtual Companion That Cares</strong></p>

<p align="center">
  <em>本地 GPU 运行 · 语音对话 · 屏幕感知 · 主动话题 · 桌面前端 · 可教学</em>
</p>

<p align="center">
  <a href="https://github.com/baaai/openheart"><img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" alt="Python"></a>
  <a href="https://github.com/baaai/openheart"><img src="https://img.shields.io/badge/Platform-Linux%20%7C%20WSL2-orange" alt="Platform"></a>
  <a href="https://github.com/baaai/openheart/blob/main/LICENSE"><img src="https://img.shields.io/github/license/baaai/openheart" alt="License"></a>
</p>

---

## ✨ 项目特色

OpenHeart 不只是又一个 AI 聊天机器人。她是一个**本地运行、有记忆、会主动开口、能看你的屏幕**的虚拟伙伴。

- 🎙️ **完整语音闭环** — SenseVoice ASR → DeepSeek 决策 → CosyVoice3-0.5B SFT 自定义音色 TTS，全部本地 GPU 推理
- 👁️ **屏幕视觉感知** — YOLOE 双阶段检测流水线 + VLM 概念学习，她能"看到"你在看什么
- 🧠 **五层记忆系统** — Redis 热记忆 → LanceDB 持久化冷记忆 → 用户画像 → 记忆衰减 → 自然语言回忆
- 🔥 **主动话题** — 内心独白 + 退火状态机，她会在你不说话时主动开口，且懂得控制频率
- 🎭 **Live2D 虚拟形象** — Electron + PixiJS 渲染，嘴型同步、视线跟随、表情控制、点击穿透
- 🖥️ **桌面前端** — Electron 桌面应用，Live2D 形象渲染 + 后台管理 + 配置面板，一键启停
- 🎓 **可教学** — 用自然语言教她新行为：「记住，以后我说X你就做Y」
- ⚡ **VRAM 自适应** — 自动检测显存，低/中/高三个档位自动调整模型加载策略

---

## 📋 当前实现进度

### 🧠 大脑 — 决策与推理
- [x] DeepSeek API 对话推理（streaming）
- [x] 角色人格系统（雪奈 — 毒舌傲娇 + 雌小鬼 + 好兄弟）
- [x] 人格动态融合（DynamicFusion）
- [x] 情绪调节（EmotionAdj）
- [x] 人格审计器（PersonaAuditor）
- [x] 人格校准器（PersonaCalibrator）
- [x] 偏好偏移（PreferenceShift）
- [x] 上下文组装器（ContextAssembler，消息边界截断，默认 2048 tokens）
- [x] 安全分类器（SafetyClassifier）
- [x] 规则引擎（RuleEngine）+ 快速路径匹配
- [x] 彩蛋系统（EasterEggSystem）
- [x] 决策桥接器（DecisionBridge）

### 👂 耳朵 — 语音识别
- [x] SenseVoice（funasr）实时中文/英文识别
- [x] mic 采集（parec / PulseAudio 16kHz mono）
- [x] VAD 静音检测

### 👀 眼睛 — 屏幕视觉
- [x] 截图采集 + 窗口注意力追踪
- [x] YOLOE 双阶段检测流水线（RegionProposer + ConceptClassifier）
- [x] VLM 概念学习（MiniCPM-V）
- [x] PaddleOCR 文本识别
- [x] 空间图谱（SpatialGraph）+ 语义匹配
- [x] 视觉编排器（VisualOrchestrator）
- [x] SyncVisionQuery 异步安全查询

### 👄 嘴巴 — 语音合成
- [x] CosyVoice3-0.5B SFT 自定义音色
- [x] 支持多角色音色（妃咲、伊吹、胡桃）
- [x] gRPC 优先 / WebSocket 回退

### 🧠 记忆
- [x] HOT 层（Redis Stream，30s TTL）
- [x] WARM 层（Redis Hash，24h TTL）
- [x] CORE 层（LanceDB，≥0.7 晋升分）
- [x] COLD 层（LanceDB，24h 同步）
- [x] DEEP 层（LanceDB，≥3 次出现模式）
- [x] 用户画像生成与修正
- [x] 记忆召回（RecallHandler）
- [x] 记忆衰减

### 🔥 主动话题（Proactive Speaking）
- [x] 静默心跳检测（SilenceHeartbeat）
- [x] 退火状态机控制开口频率
- [x] 内心独白（thinking_persona）→ 场景感知 → 自主判断
- [x] mic 独立线程 + asyncio 2s 心跳

### 🖥️ 桌面前端（electron-l2d/）

基于 Electron 构建的完整桌面应用，位于 `electron-l2d/`：

**Live2D 形象窗口**
- [x] PixiJS v7 + pixi-live2d-display + Cubism 4 渲染
- [x] 500×900 透明悬浮窗 + 点击穿透模式
- [x] WebSocket（端口 9876）实时桥接 Python 后端
- [x] 嘴型同步（根据语音音量）
- [x] 视线跟随（全局鼠标追踪，33ms 轮询）
- [x] 表情控制
- [x] GPU 软件回退（SwiftShader）

**后台管理面板**
- [x] 后端进程一键启停
- [x] API 密钥 / 模型 / 端点配置
- [x] 后端状态实时监控（running/stopped）
- [x] Voice / Visual 开关
- [x] 持久化配置（JSON 文件保存）
- [x] WebSocket 断线自动重连
- [x] 暗色主题 UI

### 🎓 教学系统
- [x] 自然语言教学 → 规则创建（OBSERVATION）
- [x] 3 次命中 → 自动晋升 CORE 规则
- [x] 教学模块（TeachingModule）+ 规则学习器（RuleLearner）

---

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│               🖥️  Electron 桌面前端 (electron-l2d/)           │
│        Live2D 渲染 · 管理面板 · 后端管理 · WebSocket IPC        │
├──────────────────────────────────────────────────────────────┤
│                     OpenHeart Runtime                         │
├───────────┬──────────┬──────────┬───────────┬───────────────┤
│ Perception│  Fusion  │  Memory  │ Decision  │  Execution    │
│           │          │          │           │               │
│ • ASR     │ • Scene  │ • HOT    │ • LLM     │ • TTS         │
│ • Vision  │   → Text │ • WARM   │ • Rules   │ • Mouse       │
│ • OCR     │ • Fusion │ • CORE   │ • Safety  │ • Overlay     │
│           │   Pipe   │ • COLD   │ • Teach   │               │
│           │          │ • DEEP   │ • Easter  │               │
├───────────┴──────────┴──────────┴───────────┴───────────────┤
│  Personality  │  Proactive   │  Prediction  │  Heartbeat     │
└──────────────────────────────────────────────────────────────┘
```

核心数据流：`Mic → VAD → ASR → DecisionBridge → CosyVoice → Speaker`

---

## 🚀 快速开始

### 环境要求

| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| **GPU** | NVIDIA RTX 3070 8GB | RTX 3080 Ti 16GB+ |
| **VRAM** | 7.5 GB（低档） | ≥ 15.5 GB（高档） |
| **RAM** | 16 GB | 32 GB |
| **OS** | Ubuntu 22.04 / WSL2 | Ubuntu 22.04 |
| **CUDA** | 12.1+ | 12.4+ |
| **Python** | 3.11 | 3.11 |
| **Redis** | 7.2+ | 7.2+ |

### VRAM 自动分档

| 档位 | 显存 | 策略 |
|------|------|------|
| 低 | ≥ 7.5 GB | 无 Shadow 验证，无 YOLO-World，CosyVoice CPU |
| 中 | ≥ 11.5 GB | 无 Shadow 验证，Whisper medium |
| 高 | ≥ 15.5 GB | 全部模型 + Shadow 验证 |

### 1. 克隆并配置环境

```bash
git clone https://github.com/baaai/openheart.git
cd openheart

# Conda 环境
conda create -n openheart python=3.11 -y
conda activate openheart
pip install -e ".[dev]"
```

### 2. 配置 API 密钥

```bash
cp .env.example .env
# 编辑 .env：
#   DEEPSEEK_API_KEY=sk-your-key
#   DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
#   DEEPSEEK_MODEL=deepseek-v4-flash
```

### 3. 启动 Redis

```bash
redis-server
```

### 4. 下载模型

```bash
bash scripts/setup_env.sh
```

下载内容包括：
- SenseVoice ASR 模型（~1.5 GB）
- CosyVoice3-0.5B TTS 模型（~4.8 GB）
- YOLOE 检测模型（~1.5 GB）
- bge-small-zh-v1.5 嵌入模型

### 5. 运行 OpenHeart

**方式一：桌面前端（推荐）**

直接用 `electron-l2d/` 桌面应用一键启动：

```bash
# 在 electron-l2d/ 目录下
npm install
npm start
```

桌面应用会自动管理 Python 后端进程的启停，无需手动运行脚本。

**方式二：命令行运行**

语音模式（mic → ASR → LLM → TTS → 扬声器）：

```bash
python scripts/demo_full.py
```

文本模式（键盘输入，不需要麦克风）：

```bash
python scripts/demo_full.py --voice-mode text
```

**方式三：Web 控制面板**

```bash
python frontend/server.py
# 浏览器打开 http://localhost:8000
```

### Docker 部署

```bash
docker-compose up -d
```

---

## 🎛️ 配置说明

所有配置文件位于 `config/` 目录：

| 文件 | 用途 |
|------|------|
| `baseline.json` | 角色人格基线 |
| `model_paths.yaml` | 模型文件路径 |
| `endpoints.yaml` | API 端点配置 |
| `audio.yaml` | 音频流水线参数 |
| `memory.yaml` | 记忆各层配置 |
| `thresholds.yaml` | 系统阈值（VAD、情绪等） |
| `emotion_params.yaml` | 情绪模型参数 |
| `sentiment.yaml` | 情感分析配置 |
| `live2d.yaml` | Live2D 形象设置 |
| `fast_path_rules.yaml` | 快速路径规则 |
| `transcript_overlay.yaml` | 字幕叠加配置 |
| `easter_eggs.json` | 彩蛋触发词/回复 |
| `prompt_modules.json` | 提示词模块 |
| `ui_settings.json` | UI 设置 |

---

## 🧠 角色系统

OpenHeart 默认角色为 **雪奈（Yukina）** —— 一个毒舌傲娇 + 雌小鬼 + 好兄弟式的聊天搭子。

```json
{
  "name": "雪奈",
  "tone": "毒舌、傲娇、小恶魔，但内心关心",
  "style": "生活化网络语言，玩梗像呼吸一样自然",
  "length_limit": 100
}
```

- 人格由 `雪奈.json` 驱动，支持自定义角色
- 人格审计器确保回复不偏离基线
- 情绪调节动态调整语气和风格
- 偏好偏移自动适配用户习惯

---

## 📂 项目结构

```
openheart/
├── electron-l2d/          # 🖥️ 桌面前端（Electron 应用）
│   ├── main.js            #   主进程：窗口管理 + WebSocket
│   ├── renderer.js        #   Live2D PixiJS 渲染引擎
│   ├── config.html        #   后台管理面板
│   ├── index.html         #   Live2D 形象窗口
│   └── l2d_client.py      #   Python WebSocket 客户端
├── frontend/              # Web 控制面板（备用）
│   ├── server.py          #   FastAPI 后端
│   └── index.html         #   控制面板页面
├── src/
│   ├── perception/       # 感知层：ASR + 视觉 + OCR
│   │   └── visual/       #   YOLOE 检测、VLM、空间图谱
│   ├── fusion/           # 融合层：场景→文本、融合流水线
│   ├── memory/           # 五层记忆：Redis + LanceDB
│   │   ├── hot/          #   Redis Stream 热记忆
│   │   └── adapters/     #   存储适配器
│   ├── personality/      # 人格系统：基线、融合、校准
│   ├── decision/         # 决策层：LLM、规则、安全、教学
│   │   ├── reflex/       #   规则引擎
│   │   └── learning/     #   规则学习器
│   ├── proactive/        # 主动话题：心跳 + 退火状态机
│   ├── execution/        # 执行层：TTS、鼠标、形象
│   │   ├── channels/     #   输出通道
│   │   └── tts_service/  #   CosyVoice 适配器
│   ├── prediction/       # 预测层
│   ├── runtime/          # 运行时状态管理
│   ├── config/           # 配置加载
│   └── infra/            # 基础设施：追踪、校验
├── config/               # YAML/JSON 配置文件（15 个）
├── rules/                # 规则定义（core/interactive/user_taught）
├── scripts/              # 启动脚本 & 工具
│   └── demo_full.py      # 主入口
├── tests/                # 测试（unit/integration/contracts/smoke）
├── models/               # 下载的模型文件
├── docker-compose.yml    # Docker 部署
└── pyproject.toml        # 项目元数据 & 依赖
```

---

## 🔌 模型与依赖

### 本地 GPU 模型

| 模型 | 用途 | 大小 |
|------|------|------|
| SenseVoiceSmall | ASR 语音识别 | ~200 MB |
| CosyVoice3-0.5B | TTS 语音合成（SFT） | ~4.8 GB |
| YOLOE-small | 区域检测 | ~0.3 GB |
| YOLOE-large | 概念分类 | ~0.7 GB |
| bge-small-zh-v1.5 | 文本嵌入 | ~130 MB |
| PaddleOCR-ONNX | 文字识别 | ~100 MB |

### 云 API

| 服务 | 用途 |
|------|------|
| DeepSeek API | 对话推理（deepseek-v4-flash） |
| MiniCPM-V API | 视觉概念学习 |

---

## 📝 开发

```bash
# 运行测试
pytest tests/ -v

# 运行合约测试
bash tests/run_all_contracts.sh

# 代码格式化
ruff format src/ tests/
black src/ tests/

# 类型检查
mypy src/
```

开发详细指南见 [`CONTEXT.md`](./CONTEXT.md) 和 [`AGENTS.md`](./AGENTS.md)。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。

---

## 📄 许可证

本项目仅供学习和研究使用。
