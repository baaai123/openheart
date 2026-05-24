"""
UserModelGenerator — 3B-model-driven user model generation & incremental updates.
v4.5.0 §3.4

Reuses Qwen2.5-3B (GPTQ) from MainDecisionEngine — NO separate model instance.
Triggers:
  - Periodic (every 24h, after cold-memory sync)
  - Session end (incremental behavioral update)
  - Emotion/topic shift (keyword-frequency change > 30 %)
  - Key event (user explicitly reflects — e.g. "我最近一直在想...")
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — v4.5.0 §3.4.2
# ---------------------------------------------------------------------------

# Default cold-memory lookback window for user model generation
_COLD_MEMORY_LOOKBACK_DAYS: int = 30

# Minimum importance threshold for scenes to be included in model input
_COLD_MEMORY_MIN_IMPORTANCE: float = 0.2

# Maximum key memories to extract
_MAX_KEY_MEMORIES: int = 10

# Maximum cold memory summaries to feed into the prompt (avoid context blow-up)
_MAX_SCENE_SUMMARIES: int = 20

# Keyword-frequency change threshold for emotion/topic trigger (§3.4.2)
_TOPIC_CHANGE_THRESHOLD: float = 0.30

# Diff threshold — more than 30 % field changes → mark "needs review" (§3.4.3)
_DIFF_NEEDS_REVIEW_THRESHOLD: float = 0.30

# Emotion label patterns — used for emotion-shift detection
_EMOTION_LABELS: list[str] = ["joy", "sadness", "neutral"]
_EMOTION_CN: list[str] = ["开心", "高兴", "快乐", "难过", "悲伤", "焦虑", "紧张", "愤怒"]


# ---------------------------------------------------------------------------
# UserModel data structure helper — v4.5.0 §3.4.1
# ---------------------------------------------------------------------------

def _make_new_user_model() -> dict[str, Any]:
    """Build the pre-populated new-user template — v4.5.0 §3.4.3."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "user_model_id": str(uuid.uuid4()),
        "version": 1,
        "generated_at": now,
        "inferred_traits": {
            "personality": "尚未形成稳定认知，有待进一步了解",
            "personality_confidence": 0.3,
            "communication_style": "未知",
            "communication_style_confidence": 0.3,
            "emotional_pattern": "暂无数据",
            "emotional_pattern_confidence": 0.3,
        },
        "knowledge_profile": {
            "topics_of_interest": [],
            "topics_to_avoid": [],
            "expertise_level": {},
        },
        "behavioral_insights": {
            "active_hours": [],
            "avg_session_length_min": 0,
            "preferred_interaction_mode": "mixed",
        },
        "key_memories": [],
        "relationship_meta": {
            "first_interaction_date": now,
            "total_interaction_hours": 0.0,
            "relationship_stage": "new",
            "nickname_preference": "",
            "model_confidence": 0.3,
            "user_verified_fields": [],
        },
    }


# ---------------------------------------------------------------------------
# Prompt template — v4.5.0 §3.4.3
# ---------------------------------------------------------------------------

_USER_MODEL_SYSTEM_PROMPT: str = (
    "你是一个用户认知总结助手。请根据以下关于用户的历史交互记录，"
    "生成或更新用户模型（UserModel）。\n\n"
    "输出一个完整的 JSON 对象，格式如下：\n"
    "{\n"
    '  "inferred_traits": {\n'
    '    "personality": "对用户性格的推断（如：偏内向、喜欢幽默、对技术有热情）",\n'
    '    "personality_confidence": 0.7,\n'
    '    "communication_style": "沟通风格（如：直接、喜欢用梗、偶尔自嘲）",\n'
    '    "communication_style_confidence": 0.7,\n'
    '    "emotional_pattern": "情绪模式（如：工作日下午容易焦虑，周末较为放松）",\n'
    '    "emotional_pattern_confidence": 0.7\n'
    "  },\n"
    '  "knowledge_profile": {\n'
    '    "topics_of_interest": ["话题1", "话题2"],\n'
    '    "topics_to_avoid": ["敏感话题1"],\n'
    '    "expertise_level": {"编程": "intermediate"}\n'
    "  },\n"
    '  "behavioral_insights": {\n'
    '    "active_hours": ["weekday_evening", "weekend_morning"],\n'
    '    "avg_session_length_min": 45,\n'
    '    "preferred_interaction_mode": "voice_heavy|text_heavy|mixed"\n'
    "  },\n"
    '  "key_memories": [\n'
    '    {"summary": "事件简述", "emotional_significance": "high|medium|low", '
    '"category": "achievement|loss|humor|conflict|其他"}\n'
    "  ],\n"
    '  "relationship_meta": {\n'
    '    "first_interaction_date": "ISO8601",\n'
    '    "relationship_stage": "new|familiar|close|trusted",\n'
    '    "nickname_preference": "用户允许的称呼"\n'
    "  }\n"
    "}\n\n"
    "要求：\n"
    "1. confidence 值在 [0, 1] 之间，反映对该字段推断的确信程度。\n"
    "2. 如果某字段缺乏数据支撑，confidence 应低于 0.4。\n"
    "3. 不要杜撰不存在的信息。\n"
    "4. 只输出 JSON，不要包含任何其他文本。\n"
)


# ---------------------------------------------------------------------------
# UserModelGenerator
# ---------------------------------------------------------------------------

class UserModelGenerator:
    """Generate and incrementally update the user model using the 3B model.

    v4.5.0 §3.4:
      - Reuses Qwen2.5-3B from MainDecisionEngine (no separate instance).
      - Queries cold memory for scenes within the last 30 days, importance > 0.2.
      - Produces updated UserModel JSON with version bump and per-field confidence.
      - Falls back to pre-populated template for new users with no cold memory data.

    Triggers:
      1. Periodic (24h, after cold-memory sync cycle)
      2. Session end → incremental behavioral update
      3. Emotion/topic shift detection (keyword-frequency change > 30 %)
      4. Key event (user reflects explicitly)

    Consumer layers — v4.5.0 §3.4.4:
      - Decision layer: system prompt injection
      - Personality layer: communication_style → long-term preference shift
      - Memory layer: topics_of_interest → weighted retrieval filter
      - Prediction layer: active_hours / avg_session_length → reminder timing
    """

    def __init__(
        self,
        decision_engine: Any,  # MainDecisionEngine — v4.5.0 §3.4.3: reuse 3B model
        memory_service: Any,  # MemoryService — for cold memory queries
        runtime_config: Any,  # RuntimeConfig — DI
    ) -> None:
        self._decision = decision_engine
        self._memory = memory_service
        self._config = runtime_config
        self._last_keyword_freq: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prior_model: Optional[dict[str, Any]] = None,
        skip_model: bool = False,
        scenes: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Main generation entry point — v4.5.0 §3.4.3.

        Queries cold memory for recent scenes (or uses *scenes* if provided),
        builds a prompt, invokes the 3B model, parses the output, and returns
        an updated UserModel dict.

        Args:
            prior_model: Existing user model to use as prior. If None, attempts
                         to load from memory service or uses the new-user template.
            skip_model: If True, skip LLM invocation and return the new-user
                        template directly (for testing / degraded paths).
            scenes: Optional pre-fetched scene dicts from cold memory. When
                    provided, these are used directly instead of querying
                    cold memory via ``_query_recent_scenes()``.  v4.5.0 §3.4.3

        Returns:
            Updated UserModel dict (version bumped).
        """
        # Resolve prior model
        if prior_model is None:
            prior_model = await self._load_existing_model()

        if scenes is not None:
            cold_scenes = scenes
        else:
            cold_scenes = await self._query_recent_scenes()

        if not cold_scenes:
            # No cold memory data → new-user template (§3.4.3)
            logger.info(
                "UserModelGenerator: No cold memory scenes found (importance > %.2f, "
                "last %d days). Using new-user template. v4.5.0 §3.4.3",
                _COLD_MEMORY_MIN_IMPORTANCE,
                _COLD_MEMORY_LOOKBACK_DAYS,
            )
            return _make_new_user_model()

        if skip_model:
            logger.info(
                "UserModelGenerator: skip_model=True, returning new-user template."
            )
            return _make_new_user_model()

        # Build prompt
        prompt = self._build_prompt(cold_scenes, prior_model)

        # Invoke 3B model
        try:
            raw_output = await self._invoke_model(prompt)
        except Exception as exc:
            # Catches: model not loaded, CUDA OOM, generation timeout.
            # Safe: return prior model (or template) with degraded metadata,
            # log at WARNING with trace context.
            logger.warning(
                "UserModelGenerator: 3B model invocation failed: %s. "
                "Returning prior model. v4.5.0 §3.4.3",
                exc,
            )
            if prior_model is not None and prior_model.get("version", 0) > 0:
                result = deepcopy(prior_model)
                result.setdefault("relationship_meta", {})["model_confidence"] = (
                    prior_model.get("relationship_meta", {}).get("model_confidence", 0.5) or 0.5
                )
                return result
            return _make_new_user_model()

        # Parse output
        new_model = self._parse_output(raw_output, prior_model)

        # Quality control — diff check (§3.4.3)
        self._check_diff(prior_model or {}, new_model)

        # Update keyword frequency tracker for emotion/topic trigger
        self._update_keyword_freq(cold_scenes)

        return new_model

    async def update_on_session_end(
        self,
        session_minutes: float,
        prior_model: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Incremental update on session end — v4.5.0 §3.4.2.

        Updates behavioral_insights (active_hours, avg_session_length_min,
        preferred_interaction_mode) and relationship_meta.total_interaction_hours.

        Args:
            session_minutes: Duration of the just-ended session in minutes.
            prior_model: Existing user model. If None, loads from memory.

        Returns:
            Updated UserModel dict.
        """
        if prior_model is None:
            prior_model = await self._load_existing_model()

        if prior_model is None or prior_model.get("version", 0) < 1:
            model = _make_new_user_model()
        else:
            model = deepcopy(prior_model)

        # Bump version
        model["version"] = int(model.get("version", 0)) + 1
        model["generated_at"] = datetime.now(timezone.utc).isoformat()

        # Update relationship_meta
        rel = model.setdefault("relationship_meta", {})
        old_hours = float(rel.get("total_interaction_hours", 0.0))
        rel["total_interaction_hours"] = old_hours + (session_minutes / 60.0)

        # Update first_interaction_date if not set
        if not rel.get("first_interaction_date"):
            rel["first_interaction_date"] = datetime.now(timezone.utc).isoformat()

        # Update relationship stage heuristic
        total_h = float(rel.get("total_interaction_hours", 0.0))
        if total_h < 5.0:
            rel["relationship_stage"] = "new"
        elif total_h < 50.0:
            rel["relationship_stage"] = "familiar"
        elif total_h < 200.0:
            rel["relationship_stage"] = "close"
        else:
            rel["relationship_stage"] = "trusted"

        # Update behavioral_insights
        bhv = model.setdefault("behavioral_insights", {})
        old_avg = float(bhv.get("avg_session_length_min", 0))
        old_count = float(bhv.get("_session_count", b"0"))
        # Weighted running average
        new_count = old_count + 1.0
        bhv["avg_session_length_min"] = round(
            (old_avg * old_count + session_minutes) / new_count, 1
        )
        bhv["_session_count"] = new_count  # internal tracker, stripped on output

        # Update active_hours heuristic
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()  # 0=Mon, 6=Sun
        if weekday < 5:
            if 6 <= hour < 12:
                slot = "weekday_morning"
            elif 12 <= hour < 18:
                slot = "weekday_afternoon"
            else:
                slot = "weekday_evening"
        else:
            if 6 <= hour < 12:
                slot = "weekend_morning"
            elif 12 <= hour < 18:
                slot = "weekend_afternoon"
            else:
                slot = "weekend_evening"

        active = list(bhv.get("active_hours", []))
        if slot not in active:
            active.append(slot)
            active.sort()
            bhv["active_hours"] = active

        # Persist via memory service if available
        if self._memory is not None and hasattr(self._memory, "save_user_model"):
            try:
                await self._memory.save_user_model(model)
            except Exception as exc:
                # Catches: storage backend errors (Redis/LanceDB unavailable).
                # Safe: model update is best-effort; log and continue.
                logger.warning(
                    "UserModelGenerator: failed to persist session-end update: %s",
                    exc,
                )

        logger.info(
            "UserModelGenerator: session-end update — version=%d, "
            "total_hours=%.1f, stage=%s",
            model["version"],
            rel.get("total_interaction_hours", 0.0),
            rel.get("relationship_stage", "new"),
        )

        return model

    def check_emotion_trigger(
        self,
        recent_scenes: list[dict[str, Any]],
    ) -> bool:
        """Check if emotion/topic keyword-frequency change exceeds threshold.

        v4.5.0 §3.4.2: If keyword-frequency change > 30 % in newly synced
        scenes, trigger an incremental model update.

        Args:
            recent_scenes: List of recently synced cold memory scene dicts.

        Returns:
            True if trigger threshold exceeded, False otherwise.
        """
        if not recent_scenes:
            return False

        # Build keyword frequency from new scenes
        new_freq: dict[str, int] = {}
        for scene in recent_scenes:
            summary = scene.get("scene_summary", scene.get("summary", ""))
            text = str(summary)
            # Simple keyword extraction: Chinese/English words >= 2 chars
            keywords = re.findall(r"[\u4e00-\u9fff\w]{2,}", text.lower())
            for kw in keywords:
                new_freq[kw] = new_freq.get(kw, 0) + 1

        if not self._last_keyword_freq:
            # First call — just store baseline
            self._last_keyword_freq = dict(new_freq)
            return False

        # Compute relative change
        all_keys = set(self._last_keyword_freq.keys()) | set(new_freq.keys())
        if not all_keys:
            return False

        changes = 0
        for key in all_keys:
            old_v = self._last_keyword_freq.get(key, 0)
            new_v = new_freq.get(key, 0)
            if old_v == 0 and new_v == 0:
                continue
            # Relative change per keyword
            max_v = max(old_v, new_v, 1)
            changes += abs(new_v - old_v) / max_v

        avg_change = changes / len(all_keys) if all_keys else 0.0

        # Update stored frequencies
        self._last_keyword_freq = dict(new_freq)

        if avg_change > _TOPIC_CHANGE_THRESHOLD:
            logger.info(
                "UserModelGenerator: emotion/topic trigger fired — "
                "keyword freq change = %.2f (> %.2f). v4.5.0 §3.4.2",
                avg_change,
                _TOPIC_CHANGE_THRESHOLD,
            )
            return True

        return False

    async def on_key_event(
        self,
        user_text: str,
        prior_model: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Handle key-event triggered micro-adjustment — v4.5.0 §3.4.2.

        When user explicitly reflects (e.g. "我最近一直在想..."), do a
        lightweight model update focused on the expressed topic.

        Args:
            user_text: The user's reflective statement.
            prior_model: Existing user model.

        Returns:
            Updated UserModel dict.
        """
        if prior_model is None:
            prior_model = await self._load_existing_model()
        if prior_model is None:
            prior_model = _make_new_user_model()

        prompt = self._build_key_event_prompt(user_text, prior_model)

        try:
            raw_output = await self._invoke_model(prompt)
        except Exception as exc:
            logger.warning(
                "UserModelGenerator: key-event model invocation failed: %s", exc
            )
            return prior_model

        new_model = self._parse_output(raw_output, prior_model)
        self._check_diff(prior_model, new_model)

        return new_model

    # ------------------------------------------------------------------
    # Internal — query cold memory
    # ------------------------------------------------------------------

    async def _query_recent_scenes(self) -> list[dict[str, Any]]:
        """Query cold memory for recent, important scenes — v4.5.0 §3.4.3.

        Filters: created_at within last 30 days, importance_score > 0.2.
        Returns up to _MAX_SCENE_SUMMARIES results sorted by importance desc.
        """
        if self._memory is None:
            return []

        cutoff = (datetime.now(timezone.utc) - timedelta(days=_COLD_MEMORY_LOOKBACK_DAYS)).isoformat()

        try:
            if self._memory._cold is not None:
                cold = self._memory._cold
                # ColdMemoryStore.search supports text search; try vector search for
                # high-importance scenes, then filter by date in-memory.
                if hasattr(cold, "_query_scenes_by_importance"):
                    results: list[dict[str, Any]] = await cold._query_scenes_by_importance(  # type: ignore[union-attr]
                        min_importance=_COLD_MEMORY_MIN_IMPORTANCE,
                        since=cutoff,
                        limit=_MAX_SCENE_SUMMARIES,
                    )
                    return results
                # fallback: generic search
                if hasattr(cold, "search"):
                    raw = await cold.search("", top_k=_MAX_SCENE_SUMMARIES)  # type: ignore[union-attr]
                    filtered: list[dict[str, Any]] = []
                    for rec in raw:
                        if not isinstance(rec, dict):
                            continue
                        if float(rec.get("importance_score", 0)) <= _COLD_MEMORY_MIN_IMPORTANCE:
                            continue
                        created = rec.get("created_at", "")
                        if created and created < cutoff:
                            continue
                        filtered.append(rec)
                    # Sort by importance desc
                    filtered.sort(
                        key=lambda r: float(r.get("importance_score", 0)),
                        reverse=True,
                    )
                    return filtered[: _MAX_SCENE_SUMMARIES]
        except Exception as exc:
            # Catches: LanceDB connection errors, query failures.
            # Safe: return empty list; caller handles via new-user template.
            logger.warning(
                "UserModelGenerator: cold memory query failed: %s", exc
            )

        return []

    # ------------------------------------------------------------------
    # Internal — prompt building
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        scenes: list[dict[str, Any]],
        prior_model: Optional[dict[str, Any]],
    ) -> str:
        """Build the LLM prompt for full user model generation.

        v4.5.0 §3.4.3: includes recent cold memory scene summaries and the
        existing user model as prior.
        """
        parts: list[str] = [_USER_MODEL_SYSTEM_PROMPT]

        # Scene summaries
        parts.append("\n## 用户近期交互记录\n")
        for i, scene in enumerate(scenes, 1):
            summary = scene.get("scene_summary", scene.get("summary", ""))
            importance = scene.get("importance_score", 0.5)
            created = scene.get("created_at", scene.get("timestamp", ""))
            parts.append(
                f"{i}. [重要性:{float(importance):.2f}, 时间:{created}] {summary}"
            )

        # Prior model
        if prior_model and prior_model.get("version", 0) > 0:
            parts.append("\n## 已有用户模型（作为先验）\n")
            prior_clean = _clean_for_prompt(prior_model)
            parts.append(json.dumps(prior_clean, ensure_ascii=False, indent=2))

        parts.append("\n## 输出\n请输出更新后的完整 UserModel JSON：")
        return "\n".join(parts)

    def _build_key_event_prompt(
        self,
        user_text: str,
        prior_model: dict[str, Any],
    ) -> str:
        """Build prompt for key-event micro-adjustment."""
        parts: list[str] = [
            "你是一个用户认知总结助手。用户表达了以下重要信息，"
            "请据此微调用户模型的相关字段。\n\n",
            "## 用户表达\n",
            user_text,
            "\n## 已有用户模型（先验）\n",
            json.dumps(_clean_for_prompt(prior_model), ensure_ascii=False, indent=2),
            "\n## 指令\n",
            "根据用户表达，更新用户模型中的相关字段（如 knowledge_profile.topics_of_interest、"
            "inferred_traits.personality 等）。对于用户未提及的字段，保持原值不变。"
            "输出完整的更新后 UserModel JSON。",
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal — model invocation (reuse 3B)
    # ------------------------------------------------------------------

    async def _invoke_model(self, prompt: str) -> str:
        """Invoke the 3B model via MainDecisionEngine's generate.

        v4.5.0 §3.4.3: reuse Qwen2.5-3B from MainDecisionEngine.
        Uses the decision engine's tokenizer and model.generate().
        """
        # Build a chat-template formatted prompt
        from src.decision.context_assembler import _IM_START, _IM_END, _ROLE_SYSTEM, _ROLE_USER

        chat_prompt = (
            f"{_IM_START}{_ROLE_SYSTEM}\n"
            f"{prompt}\n"
            f"{_IM_END}\n"
            f"{_IM_START}{_ROLE_USER}\n"
            f"请输出用户模型 JSON。\n"
            f"{_IM_END}\n"
            f"{_IM_START}assistant\n"
        )

        # Use MainDecisionEngine's internal generate path
        if hasattr(self._decision, "_generate"):
            gen_params: dict[str, float] = {
                "temperature": 0.5,  # Lower temp for structured output
                "top_p": 0.9,
                "repetition_penalty": 1.0,
            }
            return await self._decision._generate(
                chat_prompt, gen_params, max_new_tokens=1024
            )

        # Fallback: use a simpler generate interface
        if hasattr(self._decision, "_model") and self._decision._model is not None:
            import torch  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]

            tokenizer = self._decision._tokenizer
            model = self._decision._model
            inputs = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}  # type: ignore[assignment]
            with torch.no_grad():  # pyright: ignore[reportUnknownMemberType]
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.5,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]  # pyright: ignore[reportUnknownMemberType]
            generated_ids = outputs[0][input_len:]
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        raise RuntimeError(
            "UserModelGenerator: Decision engine model is not loaded. "
            "Cannot generate user model. v4.5.0 §3.4.3"
        )

    # ------------------------------------------------------------------
    # Internal — output parsing
    # ------------------------------------------------------------------

    def _parse_output(
        self,
        raw_output: str,
        prior_model: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Parse LLM output into a valid UserModel dict.

        v4.5.0 §3.4.1, §3.4.3:
          - Extracts JSON from model output.
          - Applies the prior model as fallback for missing fields.
          - Bumps version number.
          - Sets generated_at timestamp.
        """
        # Extract JSON block from possibly-wrapped output
        json_str = _extract_json(raw_output)

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(
                "UserModelGenerator: failed to parse model output as JSON. "
                "Raw output (first 200 chars): %s",
                raw_output[:200],
            )
            # Fall back to prior or template
            if prior_model:
                return self._merge_with_prior({}, prior_model)
            return _make_new_user_model()

        if not isinstance(parsed, dict):
            logger.warning(
                "UserModelGenerator: parsed output is not a dict. "
                "Falling back to prior."
            )
            if prior_model:
                return self._merge_with_prior({}, prior_model)
            return _make_new_user_model()

        # Merge with prior for safety
        if prior_model and prior_model.get("version", 0) > 0:
            model = self._merge_with_prior(parsed, prior_model)
        else:
            model = parsed

        # Ensure required fields exist
        model.setdefault("user_model_id", str(uuid.uuid4()))
        model.setdefault("inferred_traits", {})
        model.setdefault("knowledge_profile", {})
        model.setdefault("behavioral_insights", {})
        model.setdefault("key_memories", [])
        model.setdefault("relationship_meta", {})

        # Bump version
        prev_version = int(prior_model.get("version", 0)) if prior_model else 0
        model["version"] = prev_version + 1
        model["generated_at"] = datetime.now(timezone.utc).isoformat()

        # Ensure model_confidence default
        rel = model.setdefault("relationship_meta", {})
        if "model_confidence" not in rel:
            # Compute from per-field confidences
            conf = self._compute_overall_confidence(model)
            rel["model_confidence"] = conf
        rel.setdefault("user_verified_fields", [])

        # Strip internal fields
        model.get("behavioral_insights", {}).pop("_session_count", None)

        return model

    def _merge_with_prior(
        self,
        parsed: dict[str, Any],
        prior: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge parsed output with prior model, preferring parsed for non-empty values.

        This ensures fields the model didn't update are preserved from the prior.
        """
        result = deepcopy(prior)

        # Merge top-level sections
        for section in (
            "inferred_traits",
            "knowledge_profile",
            "behavioral_insights",
            "key_memories",
            "relationship_meta",
        ):
            if section in parsed and parsed[section]:
                if isinstance(parsed[section], dict) and isinstance(result.get(section), dict):
                    # For dict sections, update in place
                    result.setdefault(section, {})
                    result[section] = {**result[section], **parsed[section]}  # type: ignore[operator]
                else:
                    result[section] = parsed[section]

        return result

    def _compute_overall_confidence(self, model: dict[str, Any]) -> float:
        """Compute overall model confidence from per-field confidences.

        v4.5.0 §3.4.1: each inferred field has an optional _confidence sibling.
        """
        scores: list[float] = []
        traits = model.get("inferred_traits", {})
        for field in ("personality", "communication_style", "emotional_pattern"):
            conf = traits.get(f"{field}_confidence")
            if conf is not None:
                scores.append(min(1.0, max(0.0, float(conf))))
        if not scores:
            return 0.5
        return round(sum(scores) / len(scores), 3)

    # ------------------------------------------------------------------
    # Internal — quality control
    # ------------------------------------------------------------------

    def _check_diff(
        self,
        old_model: dict[str, Any],
        new_model: dict[str, Any],
    ) -> bool:
        """Compare old vs new user model for excessive field changes.

        v4.5.0 §3.4.3: if > 30 % fields changed, mark as "needs review"
        and log a WARNING.

        Returns:
            True if diff exceeds threshold (needs review), False otherwise.
        """
        if not old_model or old_model.get("version", 0) < 1:
            return False

        changed = self._count_field_changes(old_model, new_model)
        total = self._count_leaves(old_model)

        if total == 0:
            return False

        ratio = changed / total
        if ratio > _DIFF_NEEDS_REVIEW_THRESHOLD:
            logger.warning(
                "UserModelGenerator: Model diff ratio %.2f exceeds threshold %.2f. "
                "Marked as 'needs review'. v4.5.0 §3.4.3",
                ratio,
                _DIFF_NEEDS_REVIEW_THRESHOLD,
            )
            new_model.setdefault("relationship_meta", {})["_needs_review"] = True
            return True

        return False

    @staticmethod
    def _count_field_changes(
        old: dict[str, Any],
        new: dict[str, Any],
    ) -> int:
        """Count number of leaf-level fields that differ between two models."""
        changed = 0
        for key in set(old.keys()) | set(new.keys()):
            if key.startswith("_"):
                continue
            if key in ("generated_at", "version", "user_model_id", "total_interaction_hours"):
                continue
            old_v = old.get(key)
            new_v = new.get(key)
            if isinstance(old_v, dict) and isinstance(new_v, dict):
                changed += UserModelGenerator._count_field_changes(old_v, new_v)
            elif isinstance(old_v, list) and isinstance(new_v, list):
                if len(old_v) != len(new_v):
                    changed += 1
                else:
                    for a, b in zip(old_v, new_v):
                        if isinstance(a, dict) and isinstance(b, dict):
                            changed += UserModelGenerator._count_field_changes(a, b)
                        elif a != b:
                            changed += 1
            elif str(old_v) != str(new_v):
                changed += 1
        return changed

    @staticmethod
    def _count_leaves(model: dict[str, Any]) -> int:
        """Count total leaf-level fields in a model dict."""
        count = 0
        for key, value in model.items():
            if key.startswith("_"):
                continue
            if key in ("generated_at", "version", "user_model_id", "total_interaction_hours"):
                continue
            if isinstance(value, dict):
                count += UserModelGenerator._count_leaves(value)
            elif isinstance(value, list):
                count += len(value)
            else:
                count += 1
        return count

    # ------------------------------------------------------------------
    # Internal — keyword frequency tracking
    # ------------------------------------------------------------------

    def _update_keyword_freq(self, scenes: list[dict[str, Any]]) -> None:
        """Update stored keyword-frequency baseline from current scenes."""
        freq: dict[str, int] = {}
        for scene in scenes:
            summary = scene.get("scene_summary", scene.get("summary", ""))
            text = str(summary)
            keywords = re.findall(r"[\u4e00-\u9fff\w]{2,}", text.lower())
            for kw in keywords:
                freq[kw] = freq.get(kw, 0) + 1
        self._last_keyword_freq = freq

    # ------------------------------------------------------------------
    # Internal — load existing model from memory
    # ------------------------------------------------------------------

    async def _load_existing_model(self) -> Optional[dict[str, Any]]:
        """Load the current user model from the memory service.

        v4.5.0 §3.5: MemoryService.get_user_model() returns the latest version.
        """
        if self._memory is not None and hasattr(self._memory, "get_user_model"):
            try:
                return await self._memory.get_user_model()
            except Exception as exc:
                # Catches: storage backend errors.
                # Safe: return None; caller handles via new-user template.
                logger.warning(
                    "UserModelGenerator: failed to load existing user model: %s", exc
                )
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Extract a JSON object from model output.

    Handles outputs wrapped in ```json``` fences or with leading/trailing text.
    """
    # Try fenced code block first
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        return m.group(1)
    # Try bare JSON object
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return m.group(0)
    return text.strip()


def _clean_for_prompt(model: dict[str, Any]) -> dict[str, Any]:
    """Strip internal fields for prompt inclusion."""
    clean = deepcopy(model)
    for key in list(clean.keys()):
        if key.startswith("_"):
            del clean[key]
    bhv = clean.get("behavioral_insights")
    if isinstance(bhv, dict):
        bhv.pop("_session_count", None)
    rel = clean.get("relationship_meta")
    if isinstance(rel, dict):
        rel.pop("_needs_review", None)
    return clean
