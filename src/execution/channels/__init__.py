"""Execution channels — avatar_channel, mouse_channel, voice_channel."""

from src.execution.channels.avatar_channel import (
    AvatarChannel,
    Live2DRenderer,
    compute_rms_volume,
)
from src.execution.channels.companion_animation import (
    CompanionAnimationManager,
    AnimationState,
)
from src.execution.channels.mouse_channel import (
    MouseChannel,
    MouseAction,
    MouseController,
    PyAutoGUIController,
    SafetyLevel,
    Win32Controller,
    generate_bezier_path,
    get_foreground_window_info,
)
from src.execution.channels.voice_channel import VoiceChannel
from src.execution.transcript_overlay import TranscriptOverlay
from src.execution.tts_service.cosyvoice_adapter import (
    CosyVoiceAdapter,
    CosyVoiceEmotion,
    TTSAudioChunk,
    map_emotion_to_cosyvoice,
)
from src.execution.tts_service.stream_handler import TTSBackend

# v4.5.0 §7.5.3: backward-compat aliases for older names
TTSEmotion = CosyVoiceEmotion
TTSRoute = TTSBackend
get_tts_params = map_emotion_to_cosyvoice

__all__ = [
    "AvatarChannel",
    "Live2DRenderer",
    "compute_rms_volume",
    "CompanionAnimationManager",
    "AnimationState",
    "MouseChannel",
    "MouseAction",
    "MouseController",
    "PyAutoGUIController",
    "SafetyLevel",
    "Win32Controller",
    "generate_bezier_path",
    "get_foreground_window_info",
    "CosyVoiceAdapter",
    "CosyVoiceEmotion",
    "TTSAudioChunk",
    "TTSBackend",
    "TTSEmotion",  # backward-compat alias
    "TTSRoute",  # backward-compat alias
    "TranscriptOverlay",
    "VoiceChannel",
    "get_tts_params",  # backward-compat alias
    "map_emotion_to_cosyvoice",
]
