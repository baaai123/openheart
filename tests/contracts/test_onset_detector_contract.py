"""Contract tests for ChineseOnsetDetector (spec v4.5.0 §1.4.2)."""
from __future__ import annotations

import numpy as np
import pytest

from tests.contracts.conftest import require_module

require_module("src.perception.audio.onset_detector", "ChineseOnsetDetector")

from src.perception.audio.onset_detector import ChineseOnsetDetector  # noqa: E402


class TestChineseOnsetDetectorExists:
    def test_class_is_importable(self):
        assert ChineseOnsetDetector is not None

    def test_constructor_accepts_expected_kwargs(self):
        det = ChineseOnsetDetector(
            sample_rate=16000,
            highpass_cutoff=4000,
            energy_rise_threshold_db=15.0,
            rise_window_ms=8.0,
            cooldown_ms=200.0,
        )
        assert det.sample_rate == 16000
        assert det.highpass_cutoff == 4000


class TestChineseOnsetDetectorProcessFrame:
    def test_silence_returns_false(self):
        det = ChineseOnsetDetector(sample_rate=16000)
        silence = np.zeros(1600, dtype=np.float32)
        assert det.process_frame(silence, frame_start_sample=0) is False

    def test_empty_frame_returns_false(self):
        det = ChineseOnsetDetector(sample_rate=16000)
        assert det.process_frame(np.array([]), frame_start_sample=0) is False

    def test_high_freq_burst_triggers_onset(self):
        det = ChineseOnsetDetector(
            sample_rate=16000,
            energy_rise_threshold_db=10.0,
            rise_window_ms=5.0,
            cooldown_ms=50.0,
        )
        sr = 16000
        # 0.1 s silence
        silence = np.zeros(sr // 10, dtype=np.float32)
        det.process_frame(silence, frame_start_sample=0)

        # 0.05 s of 6 kHz sine wave (inside the 4–8 kHz band)
        t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
        burst = np.sin(2 * np.pi * 6000 * t).astype(np.float32)
        detected = det.process_frame(burst, frame_start_sample=len(silence))
        assert detected is True
        assert det.last_onset_sample >= len(silence)

    def test_low_freq_burst_does_not_trigger(self):
        det = ChineseOnsetDetector(
            sample_rate=16000,
            energy_rise_threshold_db=10.0,
            rise_window_ms=5.0,
            cooldown_ms=50.0,
        )
        sr = 16000
        # Warm up the filter with a low-frequency signal so transients don't leak
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        low_freq = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
        det.process_frame(low_freq[: len(low_freq) // 2], frame_start_sample=0)

        # Continue with the same signal — no high-frequency onset should appear
        detected = det.process_frame(
            low_freq[len(low_freq) // 2 :],
            frame_start_sample=len(low_freq) // 2,
        )
        assert detected is False

    def test_cooldown_blocks_second_onset(self):
        det = ChineseOnsetDetector(
            sample_rate=16000,
            energy_rise_threshold_db=10.0,
            rise_window_ms=5.0,
            cooldown_ms=500.0,
        )
        sr = 16000
        silence = np.zeros(sr // 10, dtype=np.float32)
        det.process_frame(silence, frame_start_sample=0)

        t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
        burst = np.sin(2 * np.pi * 6000 * t).astype(np.float32)
        first = det.process_frame(burst, frame_start_sample=len(silence))
        assert first is True

        # Immediate second burst inside cooldown window
        second = det.process_frame(burst, frame_start_sample=len(silence) + len(burst))
        assert second is False

    def test_last_onset_sample_is_set(self):
        det = ChineseOnsetDetector(
            sample_rate=16000,
            energy_rise_threshold_db=10.0,
            rise_window_ms=5.0,
            cooldown_ms=50.0,
        )
        sr = 16000
        silence = np.zeros(sr // 10, dtype=np.float32)
        det.process_frame(silence, frame_start_sample=0)

        t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
        burst = np.sin(2 * np.pi * 6000 * t).astype(np.float32)
        det.process_frame(burst, frame_start_sample=len(silence))
        assert det.last_onset_sample != -1


class TestChineseOnsetDetectorReset:
    def test_reset_clears_state(self):
        det = ChineseOnsetDetector(
            sample_rate=16000,
            energy_rise_threshold_db=10.0,
            rise_window_ms=5.0,
            cooldown_ms=50.0,
        )
        sr = 16000
        silence = np.zeros(sr // 10, dtype=np.float32)
        det.process_frame(silence, frame_start_sample=0)

        t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
        burst = np.sin(2 * np.pi * 6000 * t).astype(np.float32)
        det.process_frame(burst, frame_start_sample=len(silence))
        assert det.last_onset_sample != -1

        det.reset()
        assert det.last_onset_sample == -1
