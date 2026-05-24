# OpenHeart

一个"有温度的虚拟伙伴"系统，提供情绪价值与陪伴感。当前实现聚焦语音闭环（SenseVoice ASR → DeepSeek API 决策 → CosyVoice-300M SFT 自定义音色 TTS），运行于单块 RTX 3080 Ti Laptop (16GB) 的 WSL2 环境。

## Language

**Spec Gap** (规格空白):
架构规格书未覆盖实际硬件场景或实现路径的地方。不是代码错误——规格书是冻结参照，gap 记录了实际做法与规格假设的差异。
_Avoid_: Divergence, violation, deviation, non-compliance

**有意的工程决策**:
基于实际硬件约束和验证结果做出的实现选择。所有 Spec Gap 都属于此类——不存在"违规"。
_Avoid_: Workaround, hack, shortcut

**DEAD CODE**:
`src/` 中存在的源文件，但在工作流水线（demo）中未被接线或调用。
_Avoid_: Unused code, legacy code

**demo** (最终测试程序):
`scripts/demo_full.py` —— 实际运行的全链路语音闭环。作为审计和验证的主要参照实现。
_Avoid_: Prototype, proof-of-concept (demo 是当前的实际产品)

**voice_closed_loop** (语音闭环):
完整的 "听 → 想 → 说" 流水线：麦克风采集 → SenseVoice ASR → DeepSeek API 决策 → CosyVoice-300M SFT TTS → 扬声器播放。
_Avoid_: Voice pipeline, audio chain

**SFT character** (SFT 角色):
经过 CosyVoice-300M SFT 微调训练的特定音色。当前已训练：妃咲、伊吹、胡桃。
_Avoid_: Speaker, voice model, TTS voice

## Relationships

- 一个 **Spec Gap** 对应一次 **有意的工程决策**
- **demo** 是 **voice_closed_loop** 的唯一实现载体
- `src/` 中的 **DEAD CODE** 与 **demo** 无接线关系
- 每个 **SFT character** 有独立的训练数据（sft_data/）和检查点（sft_output/）

## Example dialogue

> **Dev:** "规格书说用 whisper.cpp 做 ASR，但 demo 里用的是 SenseVoice——这是偏离吗？"
> **Domain expert:** "不是偏离。规格书没有考虑到 RTX 3080 Ti Laptop 的实际显存约束——whisper.cpp 要 2.9GB，SenseVoice 只要 200MB 且中文更准。这是 **Spec Gap**，用 **有意的工程决策** 填补了。"
>
> **Dev:** "那 `src/perception/audio/asr_stream.py` 里还包了 faster-whisper 的代码呢？"
> **Domain expert:** "那是 **DEAD CODE**。规格书时期的实现路径，但 demo 不走它。审计里直接标 DEAD CODE。"
>
> **Dev:** "如果将来换一块 24GB 的卡，要回到 whisper.cpp 吗？"
> **Domain expert:** "不需要——SenseVoice 在任何硬件上都更优。这个 Spec Gap 应该反向写回规格书，把 SenseVoice 列为正式 ASR 选项。"

## Flagged ambiguities

- "shadow verification" 在规格书 §5.2 定义但 RuntimeConfig 永久禁用 —— 当前场景不需要，但未来多 GPU 场景可能需要重新讨论
- "VRAM tier" 在规格书是 3 档但代码是 2 档 —— 16GB 单卡不存在中档场景，但如果接入双卡或多用户场景需重新评估

## Phase 2 Terms

**proactive_speaking** (主动话题):
AI 在无用户语音输入时，通过内心独白(inner_thought) + 场景感知 + 人格状态自主判断是否开口。由 mic 独立线程 + asyncio 2s 心跳驱动，受退火状态机调控频率。
_Avoid_: Auto-talk, unprompted speech, self-triggered conversation

**inner_thought** (内心独白):
思考人格(thinking_persona)基于场景、静默时长、人格状态和最近记忆生成的 1-2 句中文观察性文本。放在 LLM 的 user turn 位置传入对话人格，明确区分"她观察到什么"和"她说了什么"——不伪装成 assistant 自言自语。
_Avoid_: Internal monologue, self-talk, stream of consciousness

**thinking_persona** (思考人格):
独立于对话人格的 LLM 配置分支：复用同一 deepseek-v4-flash API，但 max_tokens=100、temperature 偏高(0.9)。仅输出 inner_thought 观察性文本，不生成面向用户的对话回复。与对话人格共享系统提示词中的人格基线，但不参与对话轮次计数。
_Avoid_: Observer persona, inner voice model, silent persona

**user_teaching** (用户教学):
用户通过自然语言教 AI 新行为："记住，以后我说X你就做Y" → teaching.py 解析意图 → learner.py 创建 OBSERVATION 规则 → 3 次成功命中后晋升为 CORE 规则。涉及删除/发送/支付的规则自动升级为 NEEDS_CONFIRM。相邻规则冲突检测（SAFE+SAFE 组合可能产生危险→升级）。
_Avoid_: Rule learning, user training, behavior programming

**proactive_annealing** (主动退火):
主动话题的 4 级退火状态机：0=主动(心跳 2s)、1=克制(8s)、2=极少(30s)、3=仅响应(不主动)。每级需连续 2 次未回应才退一级；用户连续 3 次主动说话恢复一级。防止 AI 过于频繁地主动开口。
_Avoid_: Frequency decay, cooldown, throttling

**orchestrator** (决策编排器):
v5.x 架构新增。ConversationOrchestrator 替代 DecisionBridge 的单轮对话编排：teaching→persona→memory→decide 的顺序流程。构造函数注入 decision_engine/teaching/persona/memory，不拥有数据获取能力。
_Avoid_: DecisionBridge, bridge, god object

**persona_context** (人格上下文):
纯函数式人格状态生成器。接收 baseline + preference_offsets + emotion_label → 输出 PersonalityState(prompt_text, l2d_expression)。不访问配置、内存或网络。
_Avoid_: Personality fusion, personality engine

**memory_context** (记忆上下文):
从 Hot/Cold MemoryService 组装 LLM 记忆上下文。输出结构化 MemorySnapshot(historical_summary, memory_drawer) + to_prompt_text()。recent_dialog 由 SessionState 另取。内部处理隐私过滤。
_Avoid_: Memory builder, context assembler

**conversation_orchestrator** → alias for orchestrator

**DecisionBridge** (决策桥接器) — DEPRECATED:
v5.x DEPRECATED: 已拆分为 ConversationOrchestrator + PersonaContext + MemoryContext
_Avoid_: (use orchestrator instead)
