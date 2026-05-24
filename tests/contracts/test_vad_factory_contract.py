"""Contract tests for VAD factory degradation chain.

Spec: v4.5.0 §1.4.3, degradation matrix §4.2.
"""

import numpy as np
import pytest

from tests.contracts import require_module

VAD_FACTORY = "src.perception.audio.vad_factory"
TEN_VAD = "src.perception.audio.ten_vad"
SILERO_VAD = "src.perception.audio.silero_vad"


class TestModuleExists:
    def test_vad_factory_module_available(self):
        require_module(VAD_FACTORY, "VADFactory")

    def test_ten_vad_module_available(self):
        require_module(TEN_VAD, "TENVAD")

    def test_silero_vad_module_available(self):
        require_module(SILERO_VAD, "SileroVAD")


class TestVADFactoryDegradationChain:
    @pytest.fixture
    def factory(self):
        from src.perception.audio.vad_factory import VADFactory

        return VADFactory

    def test_factory_creates_ten_vad_when_available(self, factory, monkeypatch):
        from src.perception.audio.ten_vad import TENVAD

        original_init = TENVAD.__init__

        def mock_init(self, *args, **kwargs):
            self.threshold = 0.5
            self.hop_size = 256
            self._model = None
            self._triggered = False
            self._speech_start = 0
            self._current_sample = 0

        monkeypatch.setattr(TENVAD, "__init__", mock_init)

        try:
            vad = factory.create("ten_vad")
            assert isinstance(vad, TENVAD)
            assert not vad.degraded
        finally:
            monkeypatch.setattr(TENVAD, "__init__", original_init)

    def test_factory_falls_back_to_silero_when_ten_unavailable(self, factory, monkeypatch):
        from src.perception.audio.ten_vad import TENVAD
        from src.perception.audio.silero_vad import SileroVAD

        monkeypatch.setattr(
            TENVAD, "__init__", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no libc++"))
        )

        vad = factory.create("ten_vad")
        assert isinstance(vad, SileroVAD)
        assert not vad.degraded

    def test_factory_degrades_to_continuous_asr_when_both_unavailable(self, factory, monkeypatch):
        from src.perception.audio.ten_vad import TENVAD
        from src.perception.audio.silero_vad import SileroVAD
        from src.perception.audio.vad_factory import ContinuousASRVAD

        # Force both VADs to fail on init.
        monkeypatch.setattr(
            TENVAD, "__init__", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no libc++"))
        )
        monkeypatch.setattr(
            SileroVAD, "__init__", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no torch"))
        )

        vad = factory.create("ten_vad")
        assert isinstance(vad, ContinuousASRVAD)
        assert vad.degraded

    def test_factory_creates_silero_directly(self, factory, monkeypatch):
        from src.perception.audio.silero_vad import SileroVAD
        from src.perception.audio.vad_factory import ContinuousASRVAD

        # Force SileroVAD to succeed by mocking its __init__.
        original_init = SileroVAD.__init__

        def mock_init(self, *args, **kwargs):
            self.threshold = 0.5
            self.min_speech_duration_ms = 250
            self.min_silence_duration_ms = 100
            self.speech_pad_ms = 30
            self._model = None
            self._iterator = None
            self._pending_start = None

        monkeypatch.setattr(SileroVAD, "__init__", mock_init)

        try:
            vad = factory.create("silero")
            assert isinstance(vad, SileroVAD)
            assert not vad.degraded
        finally:
            monkeypatch.setattr(SileroVAD, "__init__", original_init)

    def test_factory_degrades_to_continuous_asr_when_silero_directly_unavailable(self, factory, monkeypatch):
        from src.perception.audio.silero_vad import SileroVAD
        from src.perception.audio.vad_factory import ContinuousASRVAD

        monkeypatch.setattr(
            SileroVAD, "__init__", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no torch"))
        )

        vad = factory.create("silero")
        assert isinstance(vad, ContinuousASRVAD)
        assert vad.degraded


class TestBaseVADInterface:
    def test_all_vads_implement_basevad(self, monkeypatch):
        from src.perception.audio.vad_factory import BaseVAD, ContinuousASRVAD
        from src.perception.audio.ten_vad import TENVAD
        from src.perception.audio.silero_vad import SileroVAD

        assert issubclass(TENVAD, BaseVAD)
        assert issubclass(SileroVAD, BaseVAD)
        assert issubclass(ContinuousASRVAD, BaseVAD)

    def test_continuous_asr_always_returns_full_chunk(self):
        from src.perception.audio.vad_factory import ContinuousASRVAD, SpeechSegment

        vad = ContinuousASRVAD()
        chunk = np.zeros(16000, dtype=np.float32)
        segments = vad.process(chunk)

        assert len(segments) == 1
        assert segments[0] == SpeechSegment(start_sample=0, end_sample=16000, is_speech_end=True)

    def test_vad_set_threshold_exists_on_all_implementations(self, monkeypatch):
        from src.perception.audio.ten_vad import TENVAD
        from src.perception.audio.silero_vad import SileroVAD
        from src.perception.audio.vad_factory import ContinuousASRVAD

        # Mock inits so we can instantiate without heavy deps.
        monkeypatch.setattr(TENVAD, "__init__", lambda self, *a, **k: setattr(self, "threshold", 0.5) or setattr(self, "_current_sample", 0) or setattr(self, "_triggered", False) or setattr(self, "_speech_start", 0) or setattr(self, "_model", None))
        class FakeIterator:
            threshold = 0.5

        monkeypatch.setattr(
            SileroVAD,
            "__init__",
            lambda self, *a, **k: (
                setattr(self, "threshold", 0.5)
                or setattr(self, "_pending_start", None)
                or setattr(self, "_iterator", FakeIterator())
                or setattr(self, "_model", None)
            ),
        )

        ten = TENVAD()
        silero = SileroVAD()
        continuous = ContinuousASRVAD()

        ten.set_threshold(0.3)
        assert ten.threshold == 0.3

        silero.set_threshold(0.4)
        assert silero.threshold == 0.4

        continuous.set_threshold(0.6)

    def test_silero_vad_process_structure(self, monkeypatch):
        from src.perception.audio.silero_vad import SileroVAD
        from src.perception.audio.vad_factory import SpeechSegment

        # Mock init and iterator so we can test process() in isolation.
        monkeypatch.setattr(
            SileroVAD,
            "__init__",
            lambda self, *a, **k: (
                setattr(self, "threshold", 0.5)
                or setattr(self, "_pending_start", None)
                or setattr(self, "_iterator", None)
                or setattr(self, "_model", None)
            ),
        )

        silero = SileroVAD()

        # Simulate an end event from the iterator.
        class FakeIterator:
            def __call__(self, x, return_seconds=False):
                return {"end": 8000}

        silero._iterator = FakeIterator()
        silero._pending_start = 2000

        chunk = np.zeros(512, dtype=np.float32)
        segments = silero.process(chunk)

        assert len(segments) == 1
        assert segments[0] == SpeechSegment(start_sample=2000, end_sample=8000, is_speech_end=True)

    def test_ten_vad_process_structure(self, monkeypatch):
        from src.perception.audio.ten_vad import TENVAD
        from src.perception.audio.vad_factory import SpeechSegment

        # Mock TenVad model so we can control probability outputs.
        class FakeModel:
            def __init__(self):
                self._probs = [0.1, 0.1, 0.6, 0.6, 0.6, 0.1, 0.1]
                self._idx = 0

            def process(self, frame):
                prob = self._probs[self._idx]
                self._idx += 1
                return prob, 0

        monkeypatch.setattr(
            TENVAD,
            "__init__",
            lambda self, *a, **k: (
                setattr(self, "threshold", 0.5)
                or setattr(self, "hop_size", 256)
                or setattr(self, "_model", FakeModel())
                or setattr(self, "_triggered", False)
                or setattr(self, "_speech_start", 0)
                or setattr(self, "_current_sample", 0)
            ),
        )

        ten = TENVAD()
        # 7 frames * 256 samples = 1792 samples.
        chunk = np.zeros(1792, dtype=np.int16)
        segments = ten.process(chunk)

        assert len(segments) == 1
        # Speech starts at frame 2 (offset 512), ends after frame 5 (offset 1536).
        assert segments[0].start_sample == 512
        assert segments[0].end_sample == 1536
        assert segments[0].is_speech_end
