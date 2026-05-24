# OpenHeart 硬约束检查清单 (Constraint Checklist)

基于《多模态智能体架构工程规格书 v4.5.0》和《项目宪法 v1.1》提取的所有"不得"、"必须"、"禁止"规则。每次代码审查时逐条核对。

---

## 0. 全局命名规范 (零容忍) — 宪法 §2

- [ ] 0.1 FallbackTextBubble 而非 FallbackAvatarChannel
- [ ] 0.2 degraded 而非 downgraded / fallback_mode
- [ ] 0.3 emotion.category 而非 emotion.type
- [ ] 0.4 voice_channel 而非 tts_channel
- [ ] 0.5 avatar_channel 而非 live2d_channel
- [ ] 0.6 mouse_channel 而非 input_channel
- [ ] 0.7 配置文件中所有键名与规格书原文完全一致，禁止自行发明

## 1. 情绪类别硬约束 (最高优先级) — 宪法 §2.2 / 规格书 §1.4.6

- [ ] 1.1 当前版本仅可靠输出 joy / sadness / neutral 三类
- [ ] 1.2 anger 和 surprise 为扩展占位枚举值，下游模块禁止编写依赖这两个值的分支逻辑
- [ ] 1.3 唯一例外：仅当 config/sentiment.yaml 中 provider 设置为 "structbert" 时，约束自动解除
- [ ] 1.4 感知层 metadata.emotion 为客观用户情绪；人格层 0.5B 输出的情绪为主观响应情绪；两者作用域不同

## 2. 全局消息信封 — 规格书 §0.3

- [ ] 2.1 所有层间通信必须遵循统一消息信封格式
- [ ] 2.2 trace_id 在一次完整用户交互链路中保持不变，贯穿所有层级
- [ ] 2.3 version 在同一 trace_id 内单调递增，下游模块消费前必须检查版本号
- [ ] 2.4 degraded 为 true 时下游应降低对该消息的置信度期望
- [ ] 2.5 emotion 字段为感知层填写的用户情绪预估值，是全链路唯一用户情绪信源
- [ ] 2.6 affective_flag 为 true 时，下游可适当提升该消息优先级

## 3. 禁止行为 — 宪法 §1.2

- [ ] 3.1 禁止自行简化任何模块
- [ ] 3.2 禁止添加规格书未提及的功能
- [ ] 3.3 禁止在模块内部直接读取环境变量（必须通过 RuntimeConfig 单例获取）
- [ ] 3.4 禁止在 tokenization 阶段直接截断原始 token 序列（必须由 ContextAssembler 在消息边界完成）
- [ ] 3.5 禁止将用户数据上传至任何云端服务
- [ ] 3.6 禁止创建规格书未列出的文件或目录

## 4. 必须行为 — 宪法 §1.3

- [ ] 4.1 所有核心能力必须具备本地降级路径，不得存在"云端不可用则功能完全丧失"的硬依赖
- [ ] 4.2 所有 try/except 必须注释说明捕获的预期异常及安全性
- [ ] 4.3 所有错误必须通过 WARNING 级别日志输出，并包含 trace_id
- [ ] 4.4 所有降级路径必须有日志记录
- [ ] 4.5 Live2D 渲染必须在独立子线程中执行，禁止占用 asyncio 主事件循环

## 5. 听觉感知约束 — 规格书 §1.4

- [ ] 5.1 VoiceFeatureExtractor 默认 enabled=False，voice_feature_weight=0
- [ ] 5.2 librosa 不作为项目强制依赖，仅在 enabled=True 时延迟导入 (import librosa)
- [ ] 5.3 若 enabled=False，extract() 应立即返回 None 字典，不触发任何导入
- [ ] 5.4 SnowNLP 不可用时回退 spacytextblob；两者均不可用则默认 neutral，degraded=true
- [ ] 5.5 TEN VAD 不可用时自动切换为 Silero VAD（不降级，false）；Silero 也不可用则降级为持续 ASR，degraded=true
- [ ] 5.6 whisper.cpp 加载失败时，听觉通道完全不可用，仅视觉，degraded=true
- [ ] 5.7 中文起音检测触发后，必须强制 Silero VAD 保持激活至少 min_speech_ms=250ms

## 6. 视觉感知约束 — 规格书 §1.6 / §1.7

- [ ] 6.1 YOLO-World 不可用时关闭路1，仅用路2+路3+路4，degraded=true
- [ ] 6.2 YOLOv11n 不可用时由 YOLO-World 粗粒度替代，degraded=true
- [ ] 6.3 PaddleOCR 不可用时降级为"文本跳过"，保留坐标，degraded=true
- [ ] 6.4 场景分类器不可用，默认为 other，degraded=true
- [ ] 6.5 低配档位直接关闭 YOLO-World，视为已降级，不等待加载失败事件
- [ ] 6.6 SyncVisionQuery 异步安全：调用方必须检查返回结果中的 metadata.stale 和 metadata.failed

## 7. 记忆层约束 — 规格书 §3.2 / §3.3

- [ ] 7.1 敏感信息过滤：同步至冷记忆的 Scene 必须经过敏感信息过滤（正则匹配手机号、身份证、密码等）
- [ ] 7.2 涉及个人敏感信息的记忆，除非用户明确要求"记住这个"，否则默认不写入冷记忆
- [ ] 7.3 冷记忆首次成功同步后，必须设置 cold_memory:initialized 哨兵键（无 TTL）
- [ ] 7.4 冷记忆 Level 2 摘要必须使用 Qwen2.5-3B 模型，而非 Louvain 社区检测
- [ ] 7.5 冷记忆 Level 2 摘要质量分 < 0.6 时，保留原始记忆并标记"待重试"
- [ ] 7.6 MemoryService 图查询接口必须透明合并热、冷记忆结果
- [ ] 7.7 热记忆数据仅会话级（TTL: 当前会话），不得持久化到冷记忆（除非经同步流程）

## 8. 用户模型约束 — 规格书 §3.4

- [ ] 8.1 relationship_meta 必须包含 model_confidence 和 user_verified_fields 字段
- [ ] 8.2 可推断字段必须增加可选的同级 _confidence 版本（如 personality_confidence）
- [ ] 8.3 新用户冷记忆为空时，user_model 必须使用预置模板生成（不是 3B 模型输出）
- [ ] 8.4 新用户初始化时必须检查 cold_memory:initialized 哨兵键
- [ ] 8.5 用户模型 version < 2（仅含预置模板数据）时，System Prompt 使用 new_user_fallback 模板
- [ ] 8.6 用户模型修正接口：必须识别修正意图、更新对应字段、设置 user_verified 和 user_verified_fields

## 9. 人格层约束 — 规格书 §4

- [ ] 9.1 基线文件 immutable: true，一经创建不可修改
- [ ] 9.2 冷启动时（cold_memory:initialized 不存在或为 false），长期偏好偏移向量所有数值字段置零
- [ ] 9.3 动态人格文件：数值型字段必须被钳制在基线 min/max 范围内
- [ ] 9.4 分类型字段仅可在 allowed 枚举值间按步长迁移
- [ ] 9.5 布尔型字段直接继承基线值
- [ ] 9.6 PersonaAuditor 必须检查：边界越界、安全约束、偏移速率
- [ ] 9.7 每个数值参数单次偏移不超过基线值的 ±15%，累计偏移不超过 min/max 范围的 80%
- [ ] 9.8 长期偏好偏移不得触发用户模型更新（循环避免），反之亦然；1 小时内最多各更新一次

## 10. 决策层约束 — 规格书 §5.2 / §5.4.0

- [ ] 10.1 影子验证由 RuntimeConfig.enable_shadow 控制；仅高配启用，中配和低配自动禁用
- [ ] 10.2 enable_shadow=False 时，影子验证模型不加载，直接跳过验证
- [ ] 10.3 影子验证连续 3 次冲突（相似度 < 0.7 且 safety_level 不一致）→ 触发静默回退
- [ ] 10.4 静默回退：降低主决策 temperature（不含低于 0.4）+ 加权平均，非模型健康检查/重启
- [ ] 10.5 静默回退 temperature 地板 0.4；若已 ≤0.4，改 top_p 收窄 0.05（不低于 0.6）
- [ ] 10.6 主决策（3B）默认上下文 2048 tokens
- [ ] 10.7 性能模式下 3B 上下文可提升至 3072/4096 tokens，启动时必须重新校验显存预算
- [ ] 10.8 影子验证（1.5B）默认上下文 1024 tokens
- [ ] 10.9 实时情绪（0.5B）默认上下文 512 tokens

## 11. 上下文截断规则 — 规格书 §5.4.0 / 宪法 §3.2

- [ ] 11.1 截断必须由 ContextAssembler 在高层上下文组装阶段完成，禁止在 tokenization 阶段直接截断
- [ ] 11.2 System Prompt 必须完整保留
- [ ] 11.3 对话轮次必须在消息边界处裁剪，禁止在单条 message 中间截断
- [ ] 11.4 截断后必须仍遵循 chat_template 的合法结构
- [ ] 11.5 优先保留当前 Scene 和最近对话轮次；冷记忆摘要按重要性排序后从尾部丢弃
- [ ] 11.6 截断发生时必须记录 OPENMATE_OOM_PREVENTION 日志
- [ ] 11.7 torch.cuda.mem_get_info() 剩余 < 1.0 GB 时自动截断上下文至当前长度的 50%

## 12. 规则安全约束 — 规格书 §5.7.2 / §5.7.3

- [ ] 12.1 涉及删除/发送/支付/系统设置类的操作规则必须经过安全审查
- [ ] 12.2 safety_level = DANGEROUS_AUTO_BLOCK：自动阻止
- [ ] 12.3 safety_level = NEEDS_CONFIRM：向用户口头确认后才执行
- [ ] 12.4 NEEDS_CONFIRM 规则暂存 Redis pending_rules:{trace_id}（TTL 120s），等待用户确认
- [ ] 12.5 用户模型修正后，关联的预测层触发条件必须重新评估

## 13. 预测层约束 — 规格书 §6.3

- [ ] 13.1 预防性安抚（依赖 emotional_pattern 字段）仅在 emotional_pattern_confidence ≥ 0.6 或字段已 user_verified 时才允许触发
- [ ] 13.2 每次触发前必须实时检查阈值
- [ ] 13.3 预测层提醒优先级低于用户触发的决策动作（冲突时提醒延迟或取消）
- [ ] 13.4 健康提醒一天内不重复

## 14. 执行层约束 — 规格书 §7

- [ ] 14.1 Live2D 渲染必须运行在独立子线程，严禁占用 asyncio 主事件循环
- [ ] 14.2 主线程通过 queue.Queue 下发指令；渲染线程通过 asyncio.Queue 回传状态
- [ ] 14.3 渲染线程心跳检测必须通过 asyncio.ensure_future 定期执行
- [ ] 14.4 渲染线程异常退出时，主线程必须调用 close() 清理残留资源
- [ ] 14.5 Live2D 初始化 5 秒内未收到就绪信号 → 视为初始化失败，走降级路径
- [ ] 14.6 所有 FallbackAvatarChannel 引用必须统一更名为 FallbackTextBubble
- [ ] 14.7 FallbackTextBubble 不可用时，avatar 动作完全跳过，仅语音和键鼠正常
- [ ] 14.8 Transcript Overlay 崩溃时，音频播放完全不受影响，每 60s 尝试重建
- [ ] 14.9 视觉闭环验证（7.4.2）：调用方必须检查 VisionSnapshot 的 stale 和 failed 状态

## 15. 快路径约束 — 规格书 §9.2

- [ ] 15.1 CosyVoice 低配 CPU 部署时，快路径强制关闭
- [ ] 15.2 快路径关闭时，所有反射决策（即使置信度 ≥0.9）均走普通路径
- [ ] 15.3 快路径必须跳过：冷记忆同步、人格微调、预测层、影子验证
- [ ] 15.4 快路径生效条件：反射规则命中且置信度 ≥ 0.9，且该规则未要求影子验证

## 16. 显存与 VRAM 分档 — 规格书 §12.1 / 宪法 §3.1

- [ ] 16.1 系统启动时必须根据可用显存自动选择三档之一（高≥15.5 / 中≥11.5 / 低≥7.5 GB）
- [ ] 16.2 低配档位必须主动关闭 YOLO-World（不等待加载失败事件）
- [ ] 16.3 低配档位必须关闭影子验证
- [ ] 16.4 低配档位 CosyVoice 转 CPU（ONNX Runtime）
- [ ] 16.5 中配档位必须关闭影子验证，Whisper 降级为 medium
- [ ] 16.6 高配档位启用影子验证，快路径支持
- [ ] 16.7 可用显存 < 7.5 GB 时直接抛出 SystemRequirementError，拒绝启动
- [ ] 16.8 每次模型加载后必须执行 del model + torch.cuda.empty_cache()
- [ ] 16.9 每次生成前检查 torch.cuda.mem_get_info()，剩余 < 1.0 GB 时记录 OPENMATE_OOM_PREVENTION 日志

## 17. 模型热加载 — 规格书 §12.2 / 宪法 §4.5

- [ ] 17.1 热加载必须获取 model_load_semaphore（信号量初始值 1）
- [ ] 17.2 旧模型引用置空后，必须显式调用 torch.cuda.synchronize() 和 torch.cuda.empty_cache()
- [ ] 17.3 若旧模型有独立 CUDA 上下文，必须显式销毁
- [ ] 17.4 每次热加载后执行 torch.cuda.memory_summary() 记录增量
- [ ] 17.5 连续 2 次增量 > 100MB，直接终止并重启模型服务进程，而非仅警告

## 18. 运行时配置管理 — 宪法 §3.3

- [ ] 18.1 所有模式开关（LOW_VRAM、PERFORMANCE_MODE、NO_SHADOW、ENABLE_SHADOW 等）在系统启动时一次性解析为 RuntimeConfig 对象
- [ ] 18.2 各模块通过依赖注入或全局单例获取 RuntimeConfig，禁止在模块内部直接读取环境变量
- [ ] 18.3 enable_shadow 控制影子验证模型的加载与使用，仅在高配下为 true

## 19. 降级矩阵完整性 — 宪法 §4 / 规格书 §11

- [ ] 19.1 3B 模型崩溃/超时：1.5B 立即接管主决策通道（若可用），否则降级为模板匹配
- [ ] 19.2 1.5B 影子验证崩溃：关闭影子验证，仅依靠 3B，直到 1.5B 恢复
- [ ] 19.3 CosyVoice 服务崩溃：切换为云端 TTS 备用；低配 CPU ONNX 崩溃则静默重启
- [ ] 19.4 live2d-py 初始化/运行失败：avatar 通道降级为 FallbackTextBubble；语音和键鼠正常
- [ ] 19.5 所有本地组件每 5 秒心跳至 Redis health:{component}，超时 15 秒判定不健康
- [ ] 19.6 崩溃优于静默错误：降级可能导致关键逻辑错误时，宁可崩溃重启

## 20. 日志规范 — 宪法 §9.2 / 规格书 §10.1

- [ ] 20.1 所有错误必须通过 WARNING 级别日志输出
- [ ] 20.2 日志必须包含 trace_id
- [ ] 20.3 日志格式必须包含：timestamp、trace_id、layer、component、level、operation、span_id
- [ ] 20.4 所有降级路径必须有日志记录，标记 degraded=true

## 21. 开发策略 — 宪法 §7

- [ ] 21.1 合约驱动：先写测试，再写实现
- [ ] 21.2 每个模块必须以通过 tests/contracts/ 中对应的合约测试为完成标准
- [ ] 21.3 每次提交前运行 tests/run_all_contracts.sh
- [ ] 21.4 Mock 必须严格遵循 src/ 下对应 interface.py 的合约
- [ ] 21.5 按功能链路纵向开发（竖切优先），每个竖切链路必须是完整可运行的闭环

## 22. 代码输出要求 — 宪法 §9.1

- [ ] 22.1 类型标注完整（Python 3.11+ 语法）
- [ ] 22.2 关键逻辑处注明规格书约束编号（如 # v4.5.0 1.3.1）
- [ ] 22.3 降级路径必须实现并用日志标记
- [ ] 22.4 所有 try/except 必须注释说明捕获的预期异常及安全性

## 23. v4.5.0 关键版本差异 — 宪法 §8

- [ ] 23.1 决策默认上下文降至 2048 tokens（原 4096）
- [ ] 23.2 影子验证改为可选，由 RuntimeConfig.enable_shadow 控制，仅高配开启
- [ ] 23.3 显存配置改为三档，启动时自动选择
- [ ] 23.4 CosyVoice 低配转为 CPU ONNX，快路径强制关闭
- [ ] 23.5 新增用户模型修正接口（§5.7.5）
- [ ] 23.6 SyncVisionQuery 改为异步安全，增加超时回退与缓存
- [ ] 23.7 FallbackAvatarChannel 已统一更名为 FallbackTextBubble
- [ ] 23.8 上下文截断由 ContextAssembler 在消息边界完成，禁止直接截断 token 序列

---

**统计：共 23 大类，100+ 条可验证约束。**
