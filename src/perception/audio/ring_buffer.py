"""Ring buffer for audio perception — v4.5.0 §1.4.1

AudioRingBuffer keeps the most recent ``buffer_duration_sec`` of 16 kHz
mono audio in memory and allows retrieval from any trigger point with an
optional pre-roll window.
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


class AudioRingBuffer:
    """Circular audio buffer with pre-roll support.

    Parameters
    ----------
    sample_rate: int
        Sampling rate in Hz (default 16000).
    buffer_duration_sec: float
        Total ring-buffer duration in seconds (default 1.5).
    pre_roll_ms: int
        Milliseconds of audio to include *before* the trigger sample
        when using :meth:`get_pre_roll_segment` (default 200).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_duration_sec: float = 1.5,
        pre_roll_ms: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self.buffer_duration_sec = buffer_duration_sec
        self.pre_roll_ms = pre_roll_ms

        self._buffer_samples = int(sample_rate * buffer_duration_sec)
        self._pre_roll_samples = int(sample_rate * pre_roll_ms / 1000.0)

        # Use float32 to match typical audio pipeline dtype
        self._buffer = np.zeros(self._buffer_samples, dtype=np.float32)
        self._write_index = 0  # Next sample index to write
        self._total_written = 0  # Cumulative samples written (for bounds checking)

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    def write(self, audio_frame: np.ndarray) -> None:
        """Write a 1-D mono audio frame into the ring buffer.

        Parameters
        ----------
        audio_frame:
            1-D array of float32 (or castable) samples.
        """
        frame = np.asarray(audio_frame, dtype=np.float32).ravel()
        n = frame.shape[0]
        if n == 0:
            return

        # Fast path — frame fits contiguously in the remaining tail
        tail = self._buffer_samples - self._write_index
        if n <= tail:
            self._buffer[self._write_index : self._write_index + n] = frame
            self._write_index = (self._write_index + n) % self._buffer_samples
        else:
            # Wrap-around write
            self._buffer[self._write_index :] = frame[:tail]
            self._buffer[: n - tail] = frame[tail:]
            self._write_index = n - tail

        self._total_written += n

    # ------------------------------------------------------------------ #
    # Read helpers
    # ------------------------------------------------------------------ #

    def get_segment(self, start_sample: int, end_sample: int) -> np.ndarray:
        """Return audio samples in the half-open interval [start, end).

        If the requested range is not fully available (cold-start) the
        missing prefix is zero-padded.  The suffix is clamped to the
        newest sample currently stored.
        """
        if start_sample < 0:
            start_sample = 0
        if end_sample < start_sample:
            end_sample = start_sample

        newest_sample = self._total_written - 1
        if end_sample > newest_sample:
            end_sample = newest_sample + 1

        if start_sample >= end_sample:
            return np.empty(0, dtype=np.float32)

        length = end_sample - start_sample
        out = np.zeros(length, dtype=np.float32)

        # Map logical sample index -> physical buffer index
        def _phys(idx: int) -> int:
            # idx is a logical sample number counting from 0
            return idx % self._buffer_samples

        for i in range(length):
            out[i] = self._buffer[_phys(start_sample + i)]

        return out

    def get_pre_roll_segment(
        self,
        trigger_sample: int,
        duration_samples: int | None = None,
    ) -> np.ndarray:
        """Retrieve audio centred on *trigger_sample* with pre-roll.

        The returned segment starts ``pre_roll_ms`` before
        *trigger_sample* and extends for *duration_samples* (or just the
        pre-roll window if ``duration_samples`` is ``None``).

        Parameters
        ----------
        trigger_sample:
            Logical sample index that marks the event of interest.
        duration_samples:
            If given, total length of the returned segment.  The segment
            still begins ``pre_roll_samples`` before *trigger_sample* and
            extends forward for the remaining samples.
        """
        start_sample = max(0, trigger_sample - self._pre_roll_samples)

        if duration_samples is None:
            end_sample = trigger_sample
        else:
            end_sample = start_sample + duration_samples

        return self.get_segment(start_sample, end_sample)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def duration_sec(self) -> float:
        """Configured maximum duration of the buffer."""
        return self.buffer_duration_sec

    @property
    def capacity_samples(self) -> int:
        """Total sample capacity of the buffer."""
        return self._buffer_samples

    @property
    def newest_sample_index(self) -> int:
        """Logical index of the most recently written sample (or -1)."""
        return self._total_written - 1
