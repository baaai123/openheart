"""GPU Inference Thread — v4.5.0 §12.2

Singleton worker thread that owns ALL torch.cuda operations. Provides:
- CUDA stream isolation via torch.cuda.Stream
- model_load_semaphore (semaphore=1) for exclusive model hot-loading
- Queue-based task submission with Future return
- VRAM monitoring per spec §12.1 (threshold-based degradation triggering)

Usage:
    from src.infra.gpu_manager import gpu_inference_thread
    future = gpu_inference_thread.submit(some_inference_fn, *args)
    result = future.result(timeout=30)
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GPU_OOM_THRESHOLD_GB: float = 1.0
VRAM_MONITOR_INTERVAL_SEC: float = 5.0

# Sentinel object for shutdown signalling
_SENTINEL_FN: Any = object()


# ---------------------------------------------------------------------------
# Task wrapper
# ---------------------------------------------------------------------------

class _GPUTask:
    """A callable + Future pair queued for execution on the GPU thread."""

    __slots__ = ("fn", "args", "kwargs", "future", "trace_id")

    def __init__(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        future: Future[Any],
    ) -> None:
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.future = future
        self.trace_id = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# GPUInferenceThread
# ---------------------------------------------------------------------------


class GPUInferenceThread:
    """Singleton GPU worker thread — spec §12.2 model_load_semaphore.

    All torch.cuda operations MUST be submitted through submit() to
    ensure single-owner access to the GPU context and prevent CUDA
    synchronization bugs.

    Parameters
    ----------
    device_id: int
        CUDA device index (default 0).
    """

    _instance: GPUInferenceThread | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, device_id: int = 0) -> GPUInferenceThread:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self, device_id: int = 0) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._device_id = device_id
        self._task_queue: queue.Queue[_GPUTask] = queue.Queue()
        self._model_load_semaphore = threading.Semaphore(value=1)
        self._stream: Any = None  # torch.cuda.Stream — set when thread starts
        self._shutdown_flag = threading.Event()
        self._started = threading.Event()

        # VRAM monitoring
        self._vram_free_gb: float = 0.0
        self._vram_total_gb: float = 0.0
        self._vram_lock = threading.Lock()
        self._oom_triggered = threading.Event()

        # Start the worker thread
        self._worker: threading.Thread = threading.Thread(
            target=self._run_loop,
            name="gpu-inference",
            daemon=True,
        )
        self._worker.start()
        self._started.wait(timeout=5.0)

        logger.info(
            "GPUInferenceThread started on device %d, stream initialized",
            self._device_id,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        """Submit a callable for execution on the GPU thread.

        Returns a concurrent.futures.Future that resolves with the
        callable's return value or raises its exception.

        Raises RuntimeError if the GPU thread has been shut down.
        """
        if self._shutdown_flag.is_set():
            raise RuntimeError("GPUInferenceThread has been shut down")

        future: Future[Any] = Future()
        task = _GPUTask(fn=fn, args=args, kwargs=kwargs, future=future)

        logger.debug(
            "GPU: queued task trace_id=%s",
            task.trace_id,
        )
        self._task_queue.put(task)
        return future

    def acquire_model_load_semaphore(
        self, timeout: float | None = None
    ) -> bool:
        """Acquire the model_load_semaphore for exclusive model hot-loading.

        Returns True if acquired, False on timeout.
        spec §12.2: semaphore initial value 1.
        """
        acquired = self._model_load_semaphore.acquire(
            blocking=True,
            timeout=timeout,
        )
        if acquired:
            logger.info("GPU: model_load_semaphore acquired")
        return acquired

    def release_model_load_semaphore(self) -> None:
        """Release the model_load_semaphore after model load completes.

        Must be called after acquire_model_load_semaphore() succeeds.
        """
        self._model_load_semaphore.release()
        logger.info("GPU: model_load_semaphore released")

    def get_vram_info(self) -> tuple[float, float]:
        """Return (free_gb, total_gb) of current VRAM state.

        Updated every VRAM_MONITOR_INTERVAL_SEC by the monitoring loop.
        """
        with self._vram_lock:
            return (self._vram_free_gb, self._vram_total_gb)

    def is_oom_imminent(self) -> bool:
        """Check if VRAM is below the OOM threshold (1.0 GB).

        spec §5.4: when VRAM < 1.0 GB, context should be truncated to 50%.
        """
        return self._oom_triggered.is_set()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Gracefully shut down the GPU worker thread.

        Drains remaining tasks, then stops the loop.
        """
        if self._shutdown_flag.is_set():
            return

        logger.info("GPU: initiating shutdown")
        self._shutdown_flag.set()
        self._task_queue.put(
            _GPUTask(
                fn=_SENTINEL_FN,  # type: ignore[arg-type]
                args=(),
                kwargs={},
                future=Future(),
            )
        )
        self._worker.join(timeout=timeout)

        if self._worker.is_alive():
            logger.warning(
                "GPU: worker thread did not stop within %.1fs", timeout
            )

    # ------------------------------------------------------------------ #
    # Worker thread loop
    # ------------------------------------------------------------------ #

    def _run_loop(self) -> None:
        """Main GPU worker thread loop — runs ALL torch.cuda operations."""
        stream = self._init_stream()
        self._stream = stream
        self._started.set()

        # Initial VRAM reading
        self._update_vram_info()
        last_vram_check = time.monotonic()

        while not self._shutdown_flag.is_set():
            try:
                task = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                # Periodic VRAM check
                now = time.monotonic()
                if now - last_vram_check >= VRAM_MONITOR_INTERVAL_SEC:
                    self._update_vram_info()
                    self._check_oom_threshold()
                    last_vram_check = now
                continue

            if task.fn is _SENTINEL_FN:
                continue

            self._execute_task(task, stream)

        # Final sync before thread exit
        self._sync_stream(stream)
        logger.info("GPU: worker thread stopped")

    def _init_stream(self) -> Any:
        """Initialize the CUDA stream for this thread.

        Returns None if CUDA is unavailable (graceful degradation).
        """
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning(
                    "GPU: CUDA not available — GPU thread running in "
                    "CPU-only mode (stream=None)"
                )
                return None
            torch.cuda.set_device(self._device_id)
            stream = torch.cuda.Stream()
            logger.info("GPU: CUDA stream initialized on device %d", self._device_id)
            return stream
        except ImportError:
            logger.warning(
                "GPU: torch not available — GPU thread running in CPU-only mode"
            )
            return None

    def _execute_task(self, task: _GPUTask, stream: Any) -> None:
        """Execute one task on the designated CUDA stream."""
        if task.future.done():
            return

        try:
            with self._stream_context(stream):
                result = task.fn(*task.args, **task.kwargs)
            task.future.set_result(result)
        except Exception as exc:
            logger.warning(
                "GPU: task trace_id=%s failed: %s",
                task.trace_id,
                exc,
            )
            task.future.set_exception(exc)

    def _stream_context(self, stream: Any):
        """Context manager: use the thread's CUDA stream if available."""
        if stream is None:
            from contextlib import nullcontext
            return nullcontext()

        try:
            import torch
            return torch.cuda.stream(stream)
        except ImportError:
            from contextlib import nullcontext
            return nullcontext()

    def _sync_stream(self, stream: Any) -> None:
        """Synchronize the CUDA stream (blocking)."""
        if stream is None:
            return
        try:
            stream.synchronize()
        except Exception as exc:
            logger.warning("GPU: stream sync failed: %s", exc)

    def _update_vram_info(self) -> None:
        """Read current VRAM state from CUDA."""
        try:
            import torch
            if not torch.cuda.is_available():
                return
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            with self._vram_lock:
                self._vram_free_gb = free_gb
                self._vram_total_gb = total_gb
        except Exception as exc:
            logger.warning("GPU: VRAM info read failed: %s", exc)

    def _check_oom_threshold(self) -> None:
        """Check VRAM and set OOM flag if below threshold."""
        with self._vram_lock:
            free_gb = self._vram_free_gb

        was_triggered = self._oom_triggered.is_set()

        if free_gb < GPU_OOM_THRESHOLD_GB:
            self._oom_triggered.set()
            if not was_triggered:
                logger.warning(
                    "OPENMATE_OOM_PREVENTION: VRAM free %.2f GB < %.1f GB "
                    "threshold — context truncation recommended",
                    free_gb,
                    GPU_OOM_THRESHOLD_GB,
                )
        else:
            self._oom_triggered.clear()


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

gpu_inference_thread: GPUInferenceThread | None = None


def get_gpu_thread(device_id: int = 0) -> GPUInferenceThread:
    """Return the global GPUInferenceThread singleton, creating it if needed."""
    global gpu_inference_thread
    if gpu_inference_thread is None:
        gpu_inference_thread = GPUInferenceThread(device_id=device_id)
    return gpu_inference_thread
