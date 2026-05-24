"""Emotion Adjustment - real-time微调 driven by subjective emotion. v4.5.0 §4.5"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.personality.baseline import BaselinePersonality

logger = logging.getLogger(__name__)

# v4.5.0 §4.5 — valid emotion vocabulary for 0.5B output.
# Only joy, sadness, neutral are reliable per 项目宪法 emotion rules.
# anger and surprise are placeholder enums.
_VALID_EMOTIONS = frozenset({"joy", "sadness", "anger", "surprise", "neutral"})

# v4.5.0 §4.5 — Minimal classification prompt template.
_CLASSIFY_PROMPT_TEMPLATE = (
    "根据以下最近对话，判断用户当前的主导情绪。\n"
    "必须从以下词表中仅输出一个词：joy, sadness, anger, surprise, neutral\n"
    "\n"
    "最近对话：\n"
    "{recent_dialogues}\n"
    "\n"
    "情绪：\n"
)


def _resolve_project_root() -> Path:
    """Resolve the project root directory relative to this source file."""
    return Path(__file__).resolve().parent.parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file, returning an empty dict on failure.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed dict, or {} if the file is missing / unparseable.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logger.warning("Config file not found: %s", path)
        return {}
    except Exception as exc:
        logger.warning("Failed to parse config %s: %s", path, exc)
        return {}


def _resolve_model_path(config_key: str = "qwen_0.5b") -> str:
    """Resolve model path from ``config/model_paths.yaml``.

    Args:
        config_key: Key in model_paths.yaml. Defaults to ``qwen_0.5b``.

    Returns:
        Absolute or relative path string for the model.
    """
    root = _resolve_project_root()
    cfg = _load_yaml(root / "config" / "model_paths.yaml")
    return str(cfg.get(config_key, f"models/{config_key}"))


class SubjectiveEmotionClassifier:
    """Real-time subjective emotion classifier powered by Qwen2.5-0.5B-Instruct.

    v4.5.0 §4.5:
      - Loaded **on demand** (lazy), never at startup.
      - Runs on CPU (FP16) by default; can be overridden via ``device``.
      - The output emotion is the *subjective* response emotion
        (how the avatar should feel/sound), **NOT** the objective user
        emotion produced by the perception layer.
      - Falls back to ``"neutral"`` whenever the model is unavailable
        or the generated text is outside the vocabulary.

    Usage::

        classifier = SubjectiveEmotionClassifier()
        emotion = classifier.classify("用户：我今天很开心！\n助手：那太好了！")
        # emotion -> "joy"  (subjective response emotion for avatar)
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model_path: Override path to the 0.5B model directory.
                If ``None``, read from ``config/model_paths.yaml`` (key ``qwen_0.5b``).
            device: torch device string. 项目宪法 §12.1: 0.5B runs on CPU.
        """
        self._model_path: str = model_path or _resolve_model_path("qwen_0.5b")
        self._device: str = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded: bool = False

    # ------------------------------------------------------------------ #
    #  Model lifecycle (lazy loading)
    # ------------------------------------------------------------------ #

    def load_model(self) -> bool:
        """Lazy-load Qwen2.5-0.5B-Instruct (FP16).

        v4.5.0 §4.5: model loaded on demand, not at startup.
        v4.5.0 §3.1: empty CUDA cache before loading when GPU is used.

        Returns:
            ``True`` if the model loaded successfully, ``False`` otherwise.
        """
        if self._loaded:
            return True

        # Clean GPU cache if we happen to run on GPU — v4.5.0 §3.1
        if self._device != "cpu":
            try:
                import torch  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
                if torch.cuda.is_available():  # pyright: ignore[reportUnknownMemberType]
                    torch.cuda.empty_cache()  # pyright: ignore[reportUnknownMemberType]
            except ImportError:
                pass

        try:
            from transformers import (  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            logger.info(
                "Lazy-loading Qwen2.5-0.5B for subjective emotion classification from %s ...",
                self._model_path,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                device_map=self._device if self._device != "cpu" else None,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            if self._device == "cpu":
                self._model = self._model.to("cpu")  # type: ignore[union-attr]

            self._loaded = True
            logger.info("Qwen2.5-0.5B loaded successfully for emotion classification.")
            return True

        except Exception as exc:
            # Catches ImportError (transformers missing), OSError (model files
            # missing), RuntimeError (OOM), etc.
            # Safe fallback: classifier remains unloaded; classify() returns "neutral".
            logger.warning(
                "Failed to load Qwen2.5-0.5B for emotion classification from %s: %s. "
                "Subjective emotion will fall back to 'neutral'. v4.5.0 §4.5",
                self._model_path, exc,
            )
            self._loaded = False
            return False

    def unload_model(self) -> None:
        """Release the 0.5B model from memory.

        v4.5.0 §3.1: del model + torch.cuda.empty_cache().
        """
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False

        try:
            import torch  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
            if torch.cuda.is_available():  # pyright: ignore[reportUnknownMemberType]
                torch.cuda.empty_cache()  # pyright: ignore[reportUnknownMemberType]
        except ImportError:
            pass

        logger.info("Qwen2.5-0.5B emotion classifier unloaded.")

    @property
    def is_loaded(self) -> bool:
        """Return whether the model has been successfully loaded."""
        return self._loaded

    # ------------------------------------------------------------------ #
    #  Classification
    # ------------------------------------------------------------------ #

    def classify(self, recent_dialogues: str, trace_id: str = "") -> str:
        """Classify the *subjective* response emotion from recent dialogues.

        This is the **subjective** emotion (how the avatar should feel),
        which is distinct from the **objective** user emotion produced by
        the perception layer (``metadata.emotion``).

        v4.5.0 §4.5: If the 0.5B model is not loaded, it is loaded on demand.
        If loading fails or the generated text is not in the vocabulary,
        returns ``"neutral"``.

        Args:
            recent_dialogues: Recent user-avatar interaction text.
            trace_id: Optional trace identifier for logging.

        Returns:
            A valid emotion label from ``_VALID_EMOTIONS``,
            or ``"neutral"`` on any failure.
        """
        if not self._loaded and not self.load_model():
            logger.warning(
                "[%s] SubjectiveEmotionClassifier model unavailable; "
                "falling back to 'neutral'. v4.5.0 §4.5",
                trace_id,
            )
            return "neutral"

        prompt = _CLASSIFY_PROMPT_TEMPLATE.format(recent_dialogues=recent_dialogues)

        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")  # type: ignore[union-attr]
            if self._device != "cpu":
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

            outputs = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,  # type: ignore[union-attr]
            )

            # Decode only the newly generated tokens
            generated = self._tokenizer.decode(  # type: ignore[union-attr]
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            raw_label = generated.strip().lower().split()[0]

            if raw_label not in _VALID_EMOTIONS:
                logger.warning(
                    "[%s] 0.5B generated invalid emotion label %r; "
                    "falling back to 'neutral'. v4.5.0 §4.5",
                    trace_id, raw_label,
                )
                return "neutral"

            logger.info(
                "[%s] Subjective emotion classified as %r (objective emotion from "
                "perception is handled separately). v4.5.0 §4.5",
                trace_id, raw_label,
            )
            return raw_label

        except Exception as exc:
            # Catches tokenization errors, generation errors, CUDA errors, etc.
            logger.warning(
                "[%s] Subjective emotion classification failed: %s; "
                "falling back to 'neutral'. v4.5.0 §4.5",
                trace_id, exc,
            )
            return "neutral"


class EmotionAdj:
    """Produces real-time emotion-driven adjustments for dynamic personality.

    The emotion category managed here is the *subjective* response emotion
    (how the avatar should feel/sound), **NOT** the objective user emotion
    from perception (``metadata.emotion``).  v4.5.0 §4.5

    This class can operate in two modes:
      1. **Manual mode**: call ``set_emotion()`` with a fixed label.
      2. **Model-driven mode**: supply a ``SubjectiveEmotionClassifier`` and
         call ``classify_and_set()`` to let the 0.5B model infer the subjective
         emotion from recent dialogues.
    """

    # v4.5.0 §4.5 — emotion-driven field mappings (multiplier style)
    _EMOTION_MAP: dict[str, dict[str, dict[str, float]]] = {
        "joy": {
            "voice_style": {"speed": 1.05, "formality": -0.05},
            "avatar_style": {"expression_intensity": 1.1, "gesture_frequency": 1.15},
            "mouse_style": {"movement_speed": 1.05},
        },
        "sadness": {
            "voice_style": {"speed": 0.92, "formality": 0.05},
            "avatar_style": {"expression_intensity": 0.85, "gesture_frequency": 0.75},
            "mouse_style": {"movement_speed": 0.90},
        },
        "neutral": {
            "voice_style": {"speed": 1.0, "formality": 0.0},
            "avatar_style": {"expression_intensity": 1.0, "gesture_frequency": 1.0},
            "mouse_style": {"movement_speed": 1.0},
        },
        "anger": {
            "voice_style": {"speed": 1.08, "formality": 0.0},
            "avatar_style": {"expression_intensity": 1.15, "gesture_frequency": 1.1},
            "mouse_style": {"movement_speed": 1.1},
        },
        "surprise": {
            "voice_style": {"speed": 1.1, "formality": -0.02},
            "avatar_style": {"expression_intensity": 1.2, "gesture_frequency": 1.2},
            "mouse_style": {"movement_speed": 1.15},
        },
    }

    def __init__(
        self,
        baseline: BaselinePersonality,
        classifier: SubjectiveEmotionClassifier | None = None,
    ) -> None:
        """
        Args:
            baseline: Immutable personality baseline for bounds clamping.
            classifier: Optional 0.5B emotion classifier. If provided,
                ``classify_and_set()`` can be used for model-driven emotion
                inference.  v4.5.0 §4.5
        """
        self._baseline = baseline
        self._classifier = classifier
        self._current_emotion = "neutral"

    # ------------------------------------------------------------------ #
    #  Emotion state
    # ------------------------------------------------------------------ #

    @property
    def current_emotion(self) -> str:
        """Return the current subjective response emotion."""
        return self._current_emotion

    @property
    def classifier(self) -> SubjectiveEmotionClassifier | None:
        """Return the attached classifier, if any."""
        return self._classifier

    def set_emotion(self, emotion: str) -> None:
        """Set the current subjective response emotion manually.

        Args:
            emotion: Emotion label.  If not in the valid vocabulary,
                silently falls back to ``"neutral"``.
        """
        emotion = emotion.lower().strip()
        if emotion not in _VALID_EMOTIONS:
            self._current_emotion = "neutral"
            return
        self._current_emotion = emotion

    def classify_and_set(self, recent_dialogues: str, trace_id: str = "") -> str:
        """Use the 0.5B classifier to infer subjective emotion and update state.

        This is the **subjective** emotion (avatar response style), which is
        separate from the **objective** user emotion coming from perception.

        Args:
            recent_dialogues: Recent user-avatar interaction text.
            trace_id: Optional trace identifier for logging.

        Returns:
            The emotion label that was set (may be ``"neutral"`` on failure).
        """
        if self._classifier is None:
            logger.warning(
                "[%s] No SubjectiveEmotionClassifier attached; "
                "falling back to 'neutral'. v4.5.0 §4.5",
                trace_id,
            )
            self._current_emotion = "neutral"
            return "neutral"

        label = self._classifier.classify(recent_dialogues, trace_id=trace_id)
        self._current_emotion = label
        return label

    # ------------------------------------------------------------------ #
    #  Adjustment lookups
    # ------------------------------------------------------------------ #

    def get_adjustment(self, section: str, field: str) -> float:
        """Return the emotion adjustment factor for a field.

        Returns a multiplier (e.g., 1.05 = +5%).
        For fields not in the emotion map, returns 1.0 (no change).
        """
        section_map = self._EMOTION_MAP.get(self._current_emotion, {})
        field_map = section_map.get(section, {})
        return field_map.get(field, 1.0)

    def get_all_adjustments(self) -> dict[str, dict[str, float]]:
        """Return the full adjustment map for the current emotion."""
        return self._EMOTION_MAP.get(self._current_emotion, {}).copy()

    def compute_adjusted_value(self, section: str, field: str, base_value: float) -> float:
        """Apply emotion multiplier to a base value, clamped to baseline bounds."""
        factor = self.get_adjustment(section, field)
        adjusted = base_value * factor
        min_val = self._baseline.get_min(section, field)
        max_val = self._baseline.get_max(section, field)
        return max(min_val, min(max_val, adjusted))

    def emotion_to_l2d_expression(self, emotion: str) -> str:
        """Map emotion label to L2D expression string (pure function).

        v5.x: Stateless alternative to ``set_emotion()`` + ``_current_emotion``.
        Used by ``PersonaContext.generate()`` for pure computation — does **not**
        mutate internal emotion state.

        Args:
            emotion: Emotion label (``joy``, ``sadness``, ``neutral``).

        Returns:
            L2D expression string (e.g. ``"joy"``), or empty string for
            unknown / placeholder emotions.
        """
        # v5.x: Direct mapping from emotion label to L2D expression name.
        # Only joy/sadness/neutral are reliable per 项目宪法 emotion rules.
        _EXPRESSION_MAP: dict[str, str] = {
            "joy": "星星眼",
            "sadness": "晕晕眼",
            "neutral": "neutral",
        }
        return _EXPRESSION_MAP.get(emotion, "")
