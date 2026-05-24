"""Contract tests for AudioRingBuffer (spec v4.5.0 §1.4.1)."""
from __future__ import annotations

import numpy as np
import pytest

from tests.contracts.conftest import require_module

require_module("src.perception.audio.ring_buffer", "AudioRingBuffer")

from src.perception.audio.ring_buffer import AudioRingBuffer  # noqa: E402


class TestAudioRingBufferExists:
    def test_class_is_importable(self):
        assert AudioRingBuffer is not None

    def test_constructor_accepts_expected_kwargs(self):
        buf = AudioRingBuffer(sample_rate=16000, buffer_duration_sec=1.5, pre_roll_ms=200)
        assert buf.sample_rate == 16000
        assert buf.duration_sec == 1.5


class TestAudioRingBufferWrite:
    def test_write_increases_total_written(self):
        buf = AudioRingBuffer(sample_rate=16000, buffer_duration_sec=1.0)
        buf.write(np.zeros(1600, dtype=np.float32))
        assert buf.newest_sample_index == 1599

    def test_write_accepts_int16_and_converts(self):
        buf = AudioRingBuffer(sample_rate=16000, buffer_duration_sec=1.0)
        buf.write(np.zeros(100, dtype=np.int16))
        assert buf.newest_sample_index == 99

    def test_write_wraps_around(self):
        buf = AudioRingBuffer(sample_rate=100, buffer_duration_sec=1.0)
        first = np.ones(100, dtype=np.float32)
        second = np.full(100, 2.0, dtype=np.float32)
        buf.write(first)
        buf.write(second)
        # After wrap, the oldest 100 samples should be from the second batch
        seg = buf.get_segment(0, 100)
        assert seg.shape[0] == 100
        # The oldest samples (0-99) are now overwritten by second batch
        assert np.allclose(seg, 2.0)


class TestAudioRingBufferRead:
    def test_get_segment_returns_correct_shape(self):
        buf = AudioRingBuffer(sample_rate=100, buffer_duration_sec=1.0)
        buf.write(np.arange(100, dtype=np.float32))
        seg = buf.get_segment(10, 30)
        assert seg.shape == (20,)
        assert np.allclose(seg, np.arange(10, 30))

    def test_get_segment_zero_pads_when_buffer_not_yet_full(self):
        buf = AudioRingBuffer(sample_rate=100, buffer_duration_sec=10.0)
        buf.write(np.ones(50, dtype=np.float32))
        seg = buf.get_segment(0, 50)
        assert seg.shape == (50,)
        # First 50 samples were zero before any writes
        assert np.allclose(seg[:50], np.ones(50))

    def test_get_pre_roll_segment_uses_pre_roll_ms(self):
        buf = AudioRingBuffer(sample_rate=100, buffer_duration_sec=1.0, pre_roll_ms=50)
        buf.write(np.arange(100, dtype=np.float32))
        # pre_roll_ms=50 -> 5 samples
        seg = buf.get_pre_roll_segment(trigger_sample=50)
        assert seg.shape == (5,)
        assert np.allclose(seg, np.arange(45, 50))

    def test_get_pre_roll_segment_with_duration(self):
        buf = AudioRingBuffer(sample_rate=100, buffer_duration_sec=1.0, pre_roll_ms=50)
        buf.write(np.arange(100, dtype=np.float32))
        seg = buf.get_pre_roll_segment(trigger_sample=50, duration_samples=20)
        assert seg.shape == (20,)
        assert np.allclose(seg, np.arange(45, 65))

    def test_get_segment_clamps_to_newest_sample(self):
        buf = AudioRingBuffer(sample_rate=100, buffer_duration_sec=1.0)
        buf.write(np.ones(50, dtype=np.float32))
        seg = buf.get_segment(40, 200)
        # Should clamp to newest sample index (49)
        assert seg.shape == (10,)

    def test_get_segment_returns_empty_for_equal_indices(self):
        buf = AudioRingBuffer(sample_rate=100, buffer_duration_sec=1.0)
        buf.write(np.ones(10, dtype=np.float32))
        seg = buf.get_segment(5, 5)
        assert seg.shape == (0,)


class TestAudioRingBufferPreRoll:
    def test_pre_roll_never_negative(self):
        buf = AudioRingBuffer(sample_rate=16000, buffer_duration_sec=1.5, pre_roll_ms=200)
        buf.write(np.ones(100, dtype=np.float32))
        seg = buf.get_pre_roll_segment(trigger_sample=10)
        # trigger_sample - pre_roll_samples would be negative; clamped to 0
        assert seg.shape == (10,)

    def test_pre_roll_200ms_at_16khz_is_3200_samples(self):
        buf = AudioRingBuffer(sample_rate=16000, buffer_duration_sec=1.5, pre_roll_ms=200)
        assert buf._pre_roll_samples == 3200

    def test_capacity_matches_duration(self):
        buf = AudioRingBuffer(sample_rate=16000, buffer_duration_sec=1.5)
        assert buf.capacity_samples == 24000
