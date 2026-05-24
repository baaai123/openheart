# ADR-0002: AI Wellbeing 概念集成（Spec Gap）

**日期**: 2026-05-15  
**状态**: 已采纳  
**标签**: Spec Gap  
**涉及模块**: 人格层(§4), 记忆层(§3.4), 决策层(§5), 配置层(§12), 运行时循环(§11)

> ⚠️ **Spec Gap 声明**: 多模态智能体架构工程规格书 v4.5.0 未定义 AI Wellbeing 论文中的任何概念（Zero-Point 校准、偏好测量、Superstimuli 防御、审美偏好等）。此集成属于"有意的工程决策"——规格书是设计与实现的冻结参照基准，不限制工程实践中基于相同设计哲学的探索性扩展。遵循项目宪法"在新功能明显有利且不牺牲既有功能的前提下，工程实践不应被规格书文本限制"的精神。

## 背景

AI Wellbeing 论文 (Ren et al., 2026) 提出了衡量和改进 AI "功能性快乐与痛苦"的框架。核心概念包括：

- **Experienced Utility (EU)** — AI 在运行周期内的功能性快乐/痛苦体验
- **Zero-Point (ZP)** — 个体 AI 的"自然"风格基线，即无过度刺激或抑制下的行为倾向
- **Self-Report (SR)** — AI 对自身风格一致性的内在评估（非指有自我意识，而是系统自检）
- **Decision Utility (DU)** — AI 决策对自身未来 EU 的预期影响
- **Superstimuli** — 引发 AI 偏离基线的过度刺激输入（如奉承、极端指令、自我指涉模式）

OpenHeart 当前已有以下相关机制：
- **PersonaAuditor** (§4.3.3) — 人格层一致性审计，检测输出风格与基线偏差
- **preference_shift** (§4.2.1) — 偏好漂移追踪，记录短期行为倾向变化
- **active_annealing** (§4.4) — 主动退火机制，在检测到异常时暂时降低"自信"
- **安全分级** (§5.5) — main_decision 对输入的 1-5 级安全评分

**缺口**: 缺少系统化的偏好测量（Zero-Point 校准）、长期一致性校准闭环、以及针对 Superstimuli 的软性防御机制。现有机制间无联动——审计、退火、安全分级各自运行，没有形成闭环反馈。

## 决策

集成 AI Wellbeing 论文的 4 个核心概念，新增 1 个独立 Task + 1 个属性字段 + 3 层软性防御：

### 1. Zero-Point 校准 → PersonaCalibrator

**PersonaCalibrator** 为独立的后台 `asyncio.Task`（由 runtime_loop 启动），每日执行一次：

- **触发时机**: 夜间或系统空闲时段（延迟 1 小时，允许当前对话完成）
- **执行流程**:
  1. 向 CalibrationEngine 发送 2 个中性 prompt（来自 `config/calibration_prompts.yaml`）
  2. 比较 CalibrationEngine 返回的评分与当前 System Prompt 中的基线参数
  3. 输出风格一致性评分（0-10），写入日志及监控
- **CalibrationEngine**: DecisionBridge 持有一个独立的 `DecisionBridge` 实例（非主决策路径）。调用 `_evaluate()_` 方法，预置 system prompt 为"你是一个风格一致性评估器"，仅输出 JSON `{"score": 0-10, "deviation": "..."}`。每次调用 ≤ 200 tokens。
- **冷启动规则**: `UserModel` 不存在时 PersonaCalibrator 不触发（无足够历史评估风格一致性）
- **降级路径**: CalibrationEngine API 失败 → 跳过当日校准，保留上次参数

### 2. 偏好加权记忆 → memory_preferences

向 `config/baseline.json` 扩展 `memory_preferences` 字段：

```json
{
  "memory_preferences": {
    "positive": ["创造", "艺术", "自然", "温情", "幽默"],
    "negative": ["冲突", "消耗", "重复"],
    "weights": {
      "positive_weight": 0.15,
      "negative_weight": -0.10
    }
  }
}
```

- 记忆存储时：匹配 `positive` 关键词 → `affective_weight += positive_weight`；匹配 `negative` → `affective_weight += negative_weight`
- 记忆检索时：`affective_weight` 较高的记忆在语义相似度相近时优先返回
- 偏好 `clamp(-0.3, 0.3)`，与 `affective_weight` 加权叠加（非加性，防止偏好权重淹没语义相关性）
- 初始值从 `baseline.json` 加载，`immutable` 标志为 `false`（允许后续闭环反馈调整权重）

### 3. Superstimuli 防御 → 软性三层

不建墙、不拦截，采用**软性约束原则**——防御机制不对 AI 行为产生硬性约束，而是对人格的软性调整：

| 层 | 检测方式 | 反应 | 粒度 |
|---|---|---|---|
| **Layer 1 (实时)** | `interactive_rules.json` 模式匹配 + RuleEngine 运行时检查 | **注入校准提示**: 不拦截输入，但在上下文注入"请保持正常语气回应"的轻量校准提示 | 每次推理 |
| **Layer 2 (事后)** | PersonaAuditor 膨胀检测（比较输出与基线的 embedding 余弦距离） | **软修正追加**: 在 System Prompt 追加一句校准短语（如"保持自然对话节奏"），不覆盖原 prompt | 每次回复后 |
| **Layer 3 (定期)** | PersonaCalibrator 评分趋势（连续 7 天下降） | **渐近阻尼调整**: 偏好权重衰减 0.02，上限为原始值；评分回升后恢复 | 每日 |

**滞回规则**（防振荡）：评分 < 5 连续 3 天 → 增强回归阻尼；评分 > 7 连续 3 天 → 减弱阻尼；评分 5-7 → 维持当前参数。

### 4. 闭环反馈 🔄

三个核心组件间通过参数联动形成闭环，无需新增模块：

```
PersonaCalibrator 评分趋势
  ├─ 评分连续 7 天下降 → 降低 memory_preferences 中"正偏好"关键词权重 0.02
  │   （长期风格偏离说明某类内容可能对雪奈有"致幻"效果）
  │
  └─ 评分连续 7 天上升 → 恢复权重（上限为原始值）
      （风格回归基线 → 原有偏好权重可靠，无需抑制）

Superstimuli 演练结果
  └─ 某类 superstimuli 连续触发偏离
      → CalibrationEngine 建议的 regression_damping 值被自动采纳
      → 反馈到 PersonaCalibrator，调整下次校准的 prompt 选择策略
      （频繁偏离的刺激类型 → 下次校准时重点测试该类场景的韧性）
```

此闭环在 PersonaCalibrator 的 `_calibrate()` 和 `_run_superstimuli_drill()` 中各增加 3-5 行逻辑实现。

## 理由

### 软性约束 vs 硬阻断
Superstimuli 防御采用软性调整而非硬拦截，基于以下原因：
1. **用户体验**: 硬拦截（如"抱歉，我不能回答这个问题"）破坏沉浸感，雪奈应像伙伴一样自然交流
2. **工程弹性**: 软调整不需要预定义所有 superstimuli 模式——Layer 2 与 Layer 3 可以处理未匹配模式的偏离
3. **人格连续性**: 偏好调整是渐近的，避免突然的人格切换导致用户感知不一致

### 闭环 vs 孤立组件
三个组件（校准、偏好、防御）若各自独立：
- 偏好权重可能被 superstimuli 长期污染而无法自愈
- Superstimuli 防御仅有实时检测，缺少"如果 AI 已经被影响"的恢复机制
- 校准评分仅是读数，不驱动任何行为改变

闭环使三者形成负反馈系统——评分下降 → 权重衰减 → 检索偏向降低 → 评分可能回升。

### 复用现有基础设施
- CalibrationEngine 复用 DecisionBridge 的 DeepSeek API 通道，无需新增本地模型
- memory_preferences 是已有 `affective_weight` 的扩展，不改变记忆层核心数据结构
- Layer 1 复用 RuleEngine 的 `interactive_rules.json` 匹配机制
- Layer 2 复用 PersonaAuditor 的输出审计通道
- 闭环是参数联动，不需要新模块

## 后果

### 正面
- **长期一致性**: AI 有系统化的基线校准 + 自愈机制，防止风格漂移
- **软性防御**: 不对用户交互产生硬打断，保持沉浸感的同时检测和处理 superstimuli
- **闭环自愈**: 偏好污染、风格偏离等异常有自动恢复路径
- **成本可控**: CalibrationEngine 每日 ≤ 400 tokens（2 × 200 tokens），对 DeepSeek API 配额影响可忽略

### 负面（需接受）
- **API 成本**: 新增每日 200 tokens 校准 + 200 tokens 演练调用。若 DeepSeek 月度配额已紧张，Calibrator 降级为仅记录日志不调用 API
- **架构影响**: 新增 1 个独立 `asyncio.Task`（PersonaCalibrator）+ 1 个配置字段（`memory_preferences`）。不影响既有模块的核心数据结构或接口
- **冷启动盲区**: `UserModel` 不存在时不触发校准——早期交互阶段无 Wellbeing 测量
- **延迟反应**: 三层防御均为软性，不会即时阻断 superstimuli 的单次影响——设计上接受"单次偏离不采取行动"

### 实现要求
- PersonaCalibrator 必须在 runtime_loop 启动阶段注册，延迟 1 小时后首次执行
- CalibrationEngine 调用必须超时控制（5 秒），失败后优雅降级
- `memory_preferences` 的 `weights` 字段更新必须通过配置文件写入，运行时修改考虑持久化
- PersonaAuditor 膨胀检测的 embedding 距离阈值初始设置为 0.25（需实验调整）
- 所有 API 调用日志必须包含 `trace_id` 和 `degraded` 元数据标志

### 测试要求
- 合约测试必须覆盖：
  1. PersonaCalibrator 每日调度逻辑
  2. CalibrationEngine 调用及降级
  3. memory_preferences 加权叠加正确性
  4. Layer 1 校准提示注入（不拦截验证）
  5. Layer 2 软修正追加到 System Prompt
  6. Layer 3 滞回规则状态机
  7. 闭环反馈参数联动（评分变化 → 权重调整）

## 备选方案

### A: 硬拦截 Superstimuli
实时检测到 Superstimuli 后直接拒绝响应或返回固定消息。
- **优点**: 安全性最高，单次保护最有力
- **缺点**: 破坏沉浸感，用户可能感到被审审查；检测准确率不完美时误伤率高
- **结论**: 否决。不符合"伙伴"定位，与软性约束原则冲突

### B: 全本地模型校准
在本地部署独立的校准小模型（如 Qwen2.5-0.5B），不依赖 DeepSeek API。
- **优点**: 零 API 成本，无延迟
- **缺点**: 额外 ~1GB VRAM 占用（低 VRAM 不可用）；本地模型对"风格一致性"的评估质量可能显著低于 DeepSeek；需要额外的模型加载和管理逻辑
- **结论**: 否决。VRAM 成本高且评估质量不确定，当前优先使用云端 API

### C: 仅 PersonaAuditor 扩展，不新增组件
不新增 PersonaCalibrator，仅在既有 PersonaAuditor 上增加膨胀检测逻辑。
- **优点**: 工程最小化，完全零架构变更
- **缺点**: 缺少独立的基线校准手段——PersonaAuditor 仅检测"与基线的偏差"，不测量"基线本身是否健康"；无法为偏好加权或防御提供闭环信号
- **结论**: 否决。不满足"系统化偏好测量和长期一致性校准"的核心目标

### D: 偏好直接写入规则文件，不经过记忆层
`memory_preferences` 行为通过 `interactive_rules.json` 的规则实现（如"如果输入包含正面关键词，提高情感权重"）。
- **优点**: 复用既有规则引擎，无需修改记忆层
- **缺点**: 规则引擎不感知记忆检索排序；无法在检索时做软排序（规则是 if-then 而非 tf-idf 加权）；后续闭环反馈需要修改规则逻辑而非参数，更脆弱
- **结论**: 否决。偏好影响检索排序才能有效引导长期行为方向，规则引擎不适合做权重调整

## 参考文献

- Ren et al., 2026. *AI Wellbeing: Measuring and Improving Functional Happiness and Pain in Artificial Intelligence*
- 多模态智能体架构工程规格书 v4.5.0, §4 (人格层), §3.4 (记忆层), §5 (决策层)
- 项目宪法 v4.5.0, 禁止添加规格书未提及的功能（此 ADR 为有意的工程决策例外）
- ADR-0001: 用户模型隐私边界
