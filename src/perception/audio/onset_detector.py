"""Chinese onset detector for audio perception — v4.5.0 §1.4.2

Detects voiceless consonants and soft onsets in Mandarin Chinese by
tracking abrupt energy rises in the high-frequency band (4–8 kHz).
This prevents VAD from clipping the silent fricative/burst portions of
sounds like *s*, *sh*, *x*, *h*, *f*, and unaspirated *b*, *d*, *g*,
*j*, *zh*, *z*.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class ChineseOnsetDetector:
    """High-frequency energy-rise onset detector for Chinese phonetics.

    Parameters
    ----------
    sample_rate: int
        Sampling rate in Hz (default 16000).
    highpass_cutoff: int
        Lower bound of the high-frequency band in Hz (default 4000).
    energy_rise_threshold_db: float
        Minimum energy jump (in dB) to register as an onset (default 15.0).
    rise_window_ms: float
        Length of the analysis window for the energy rise calculation
        in milliseconds (default 8.0).
    cooldown_ms: float
        Minimum silence between two accepted onsets in milliseconds
        (default 200.0).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        highpass_cutoff: int = 4000,
        energy_rise_threshold_db: float = 15.0,
        rise_window_ms: float = 8.0,
        cooldown_ms: float = 200.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.highpass_cutoff = highpass_cutoff
        self.energy_rise_threshold_db = energy_rise_threshold_db
        self.rise_window_ms = rise_window_ms
        self.cooldown_ms = cooldown_ms

        self.last_onset_sample: int = -1

        # Design a 2nd-order Butterworth high-pass filter for the band of interest
        nyq = sample_rate / 2.0
        if highpass_cutoff >= nyq:
            highpass_cutoff = int(nyq) - 1
        self._filter_b, self._filter_a = signal.butter(
            N=2,
            Wn=highpass_cutoff / nyq,
            btype="high",
        )

        # Convert time parameters to sample counts
        self._rise_samples = max(1, int(sample_rate * rise_window_ms / 1000.0))
        self._cooldown_samples = max(1, int(sample_rate * cooldown_ms / 1000.0))

        # State for the causal filter
        self._zi = signal.lfilter_zi(self._filter_b, self._filter_a)

        # History of filtered energy (dB) for rise calculation
        self._energy_db_history: list[float] = []
        self._history_start_sample = 0  # Logical sample index of history[0]

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def process_frame(self, audio_frame: np.ndarray, frame_start_sample: int) -> bool:
        """Process one audio frame and return ``True`` if an onset is detected.

        Parameters
        ----------
        audio_frame:
            1-D array of float32 mono samples.
        frame_start_sample:
            Logical sample index of the first element in *audio_frame*.
        """
        frame = np.asarray(audio_frame, dtype=np.float32).ravel()
        if frame.size == 0:
            return False

        # High-pass filter to isolate the 4–8 kHz band
        filtered, self._zi = signal.lfilter(
            self._filter_b, self._filter_a, frame, zi=self._zi
        )

        # Energy per sample in dB (clamped to avoid log(0))
        energy = filtered ** 2
        energy_db = 10.0 * np.log10(energy + 1e-12)

        self._energy_db_history.extend(energy_db.tolist())

        # If we have no previous onset, the cooldown window is ignored
        cooldown_end = (
            self.last_onset_sample + self._cooldown_samples
            if self.last_onset_sample >= 0
            else -1
        )

        onset_detected = False
        for i, db in enumerate(energy_db):
            global_idx = frame_start_sample + i

            # Respect cooldown
            if global_idx <= cooldown_end:
                continue

            # Need enough history to compare with the window before this sample
            hist_idx = global_idx - self._history_start_sample
            if hist_idx < self._rise_samples:
                continue

            # Compute mean energy in the window just before this sample
            prev_window = self._energy_db_history[
                hist_idx - self._rise_samples : hist_idx
            ]
            prev_mean = float(np.mean(prev_window))

            rise = db - prev_mean
            if rise >= self.energy_rise_threshold_db:
                self.last_onset_sample = global_idx
                onset_detected = True
                cooldown_end = global_idx + self._cooldown_samples
                logger.debug(
                    "Onset detected at sample %d (rise=%.2f dB)",
                    global_idx,
                    rise,
                )
                break

        # Trim history to keep memory bounded — keep enough for the next frame's rise window
        keep = self._rise_samples + self._cooldown_samples + self.sample_rate // 10
        if len(self._energy_db_history) > keep:
            trim = len(self._energy_db_history) - keep
            self._energy_db_history = self._energy_db_history[trim:]
            self._history_start_sample += trim

        return onset_detected

    def reset(self) -> None:
        """Reset detector state (e.g. at the start of a new recording session)."""
        self.last_onset_sample = -1
        self._zi = signal.lfilter_zi(self._filter_b, self._filter_a)
        self._energy_db_history.clear()
        self._history_start_sample = 0
