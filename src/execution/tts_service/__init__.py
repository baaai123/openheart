"""TTS service — cosyvoice_adapter, char_duration_predictor, stream_handler.

v4.5.0 §7.5: Voice coordination with CosyVoice gRPC streaming synthesis.
v4.5.0 §7.2.1: CharDurationPredictor for word-level speech timing estimation.
"""

from src.execution.tts_service.cosyvoice_adapter import (
    CosyVoiceAdapter,
    CosyVoiceEmotion,
    TTSAudioChunk,
    map_emotion_to_cosyvoice,
)
from src.execution.tts_service.char_duration_predictor import CharDurationPredictor
from src.execution.tts_service.stream_handler import (
    StreamState,
    TTSBackend,
    TTSStreamHandler,
)

__all__ = [
    "CharDurationPredictor",
    "CosyVoiceAdapter",
    "CosyVoiceEmotion",
    "TTSAudioChunk",
    "TTSBackend",
    "TTSStreamHandler",
    "StreamState",
    "map_emotion_to_cosyvoice",
]
