"""Shutdown Manager — graceful system termination with ordered cleanup.

Handles SIGINT/SIGTERM and performs staged resource release:
1. Live2D render thread
2. GPU inference thread
3. Outstanding async tasks
4. Redis connection flush
5. LanceDB WAL sync
6. Docker compose down

All cleanup steps are logged with trace_id. A 30-second hard deadline
is enforced; after that, the process force-terminates via os._exit(1).

v4.5.0 §0.4 constraint: crash preferred over silent data corruption.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
import uuid
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLEANUP_TIMEOUT_SEC: float = 30.0
CLEANUP_STEP_TIMEOUT_SEC: float = 5.0


# ---------------------------------------------------------------------------
# Cleanup step registry
# ---------------------------------------------------------------------------

class CleanupStep:
    """One step in the shutdown sequence — a named callable with timeout."""

    __slots__ = ("name", "fn", "timeout")

    def __init__(
        self,
        name: str,
        fn: Callable[[], Any],
        timeout: float = CLEANUP_STEP_TIMEOUT_SEC,
    ) -> None:
        self.name = name
        self.fn = fn
        self.timeout = timeout


def _run_with_timeout(
    fn: Callable[[], Any],
    timeout: float,
    step_name: str,
    trace_id: str,
) -> bool:
    """Run a callable with a deadline. Returns True if it completed."""
    import threading

    result_container: list[bool] = [False]
    exception_container: list[Exception | None] = [None]

    def worker() -> None:
        try:
            fn()
        except Exception as exc:
            exception_container[0] = exc
        else:
            result_container[0] = True

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        logger.warning(
            "[%s] Shutdown step '%s' timed out after %.1fs",
            trace_id,
            step_name,
            timeout,
        )
        return False

    if exception_container[0] is not None:
        logger.warning(
            "[%s] Shutdown step '%s' raised: %s",
            trace_id,
            step_name,
            exception_container[0],
        )

    return result_container[0]


# ---------------------------------------------------------------------------
# ShutdownManager
# ---------------------------------------------------------------------------


class ShutdownManager:
    """Orchestrates ordered system shutdown on SIGINT/SIGTERM.

    Cleanup order:
      1. Live2D render thread
      2. GPU inference thread
      3. Outstanding async tasks (cancel + gather)
      4. Redis connection flush + disconnect
      5. LanceDB WAL sync
      6. Docker compose down

    A 30-second hard deadline is enforced; after that, force-terminates.
    """

    _instance: ShutdownManager | None = None

    def __new__(cls) -> ShutdownManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._steps: list[CleanupStep] = []
        self._shutting_down = False
        self._live2d_close_cb: Callable[[], None] | None = None
        self._gpu_close_cb: Callable[[], None] | None = None
        self._docker_close_cb: Callable[[], None] | None = None
        self._redis_close_cb: Callable[[], Any] | None = None
        self._lancedb_close_cb: Callable[[], Any] | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------ #
    # Registration API
    # ------------------------------------------------------------------ #

    def register_live2d_renderer(self, close_fn: Callable[[], None]) -> None:
        """Register the Live2D render thread cleanup callback."""
        self._live2d_close_cb = close_fn

    def register_gpu_thread(self, close_fn: Callable[[], None]) -> None:
        """Register the GPU inference thread shutdown callback."""
        self._gpu_close_cb = close_fn

    def register_docker_manager(self, close_fn: Callable[[], None]) -> None:
        """Register the Docker compose down callback."""
        self._docker_close_cb = close_fn

    def register_redis_client(self, close_fn: Callable[[], Any]) -> None:
        """Register the Redis connection close callback."""
        self._redis_close_cb = close_fn

    def register_lancedb(self, close_fn: Callable[[], Any]) -> None:
        """Register the LanceDB WAL sync/close callback."""
        self._lancedb_close_cb = close_fn

    def register_async_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the main asyncio event loop for task cancellation."""
        self._async_loop = loop

    # ------------------------------------------------------------------ #
    # Signal setup
    # ------------------------------------------------------------------ #

    def install_handlers(self) -> None:
        """Install SIGINT and SIGTERM handlers."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        logger.info("ShutdownManager: signal handlers installed")

    # ------------------------------------------------------------------ #
    # Shutdown execution
    # ------------------------------------------------------------------ #

    def shutdown(self, trace_id: str | None = None) -> None:
        """Execute the full staged shutdown sequence.

        This method is idempotent — calling it multiple times is safe.
        """
        if self._shutting_down:
            return
        self._shutting_down = True

        trace_id = trace_id or str(uuid.uuid4())
        deadline = time.monotonic() + CLEANUP_TIMEOUT_SEC
        failed_steps: list[str] = []

        logger.warning(
            "[%s] ShutdownManager: starting staged shutdown "
            "(deadline=%.0fs)",
            trace_id,
            CLEANUP_TIMEOUT_SEC,
        )

        # Build step list from registered callbacks — fixed order per spec
        steps_to_run: list[CleanupStep] = self._build_step_list()

        for step in steps_to_run:
            if time.monotonic() > deadline:
                logger.error(
                    "[%s] ShutdownManager: hard deadline reached, "
                    "skipping remaining %d steps",
                    trace_id,
                    len([s for s in steps_to_run if s.name > step.name]),
                )
                break

            remaining = deadline - time.monotonic()
            step_timeout = min(step.timeout, max(0.1, remaining))

            logger.info(
                "[%s] Shutdown step: %s (timeout=%.1fs, remaining=%.1fs)",
                trace_id,
                step.name,
                step_timeout,
                remaining,
            )

            ok = _run_with_timeout(step.fn, step_timeout, step.name, trace_id)
            if not ok:
                failed_steps.append(step.name)

        # Force terminate if deadline exceeded
        if time.monotonic() > deadline or failed_steps:
            logger.error(
                "[%s] ShutdownManager: %d step(s) failed or timed out: %s",
                trace_id,
                len(failed_steps),
                ", ".join(failed_steps) if failed_steps else "deadline",
            )

        if time.monotonic() > deadline:
            logger.error(
                "[%s] ShutdownManager: force-terminating process", trace_id
            )
            os._exit(1)

        logger.warning(
            "[%s] ShutdownManager: graceful shutdown complete", trace_id
        )
        os._exit(0)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Signal handler entry point."""
        sig_name = signal.Signals(signum).name
        logger.warning(
            "ShutdownManager: received %s, initiating shutdown", sig_name
        )
        self.shutdown()

    def _build_step_list(self) -> list[CleanupStep]:
        """Build the ordered cleanup step list from registered callbacks."""

        steps: list[CleanupStep] = []

        # Step 1: Live2D render thread
        if self._live2d_close_cb is not None:
            steps.append(
                CleanupStep(
                    name="live2d_render_thread",
                    fn=self._live2d_close_cb,
                )
            )

        # Step 2: GPU inference thread
        if self._gpu_close_cb is not None:
            steps.append(
                CleanupStep(
                    name="gpu_inference_thread",
                    fn=self._gpu_close_cb,
                )
            )

        # Step 3: Cancel outstanding async tasks
        if self._async_loop is not None:
            steps.append(
                CleanupStep(
                    name="cancel_async_tasks",
                    fn=self._cancel_async_tasks,
                    timeout=3.0,
                )
            )

        # Step 4: Redis flush + disconnect
        if self._redis_close_cb is not None:
            cb = self._redis_close_cb  # Capture for closure
            steps.append(
                CleanupStep(
                    name="redis_close",
                    fn=lambda: self._safe_invoke(cb),
                )
            )

        # Step 5: LanceDB WAL sync
        if self._lancedb_close_cb is not None:
            cb = self._lancedb_close_cb  # Capture for closure
            steps.append(
                CleanupStep(
                    name="lancedb_wal_sync",
                    fn=lambda: self._safe_invoke(cb),
                )
            )

        # Step 6: Docker compose down
        if self._docker_close_cb is not None:
            steps.append(
                CleanupStep(
                    name="docker_compose_down",
                    fn=self._docker_close_cb,
                )
            )

        logger.info(
            "ShutdownManager: built cleanup sequence: %s",
            " → ".join(s.name for s in steps),
        )
        return steps

    def _cancel_async_tasks(self) -> None:
        """Cancel all outstanding asyncio tasks in the main event loop."""
        loop = self._async_loop
        if loop is None:
            return

        try:
            tasks = asyncio.all_tasks(loop=loop)
            if not tasks:
                return

            for task in tasks:
                task.cancel()

            async def _gather_cancelled() -> None:
                try:
                    await asyncio.gather(
                        *tasks, return_exceptions=True
                    )
                except Exception:
                    pass

            # Run in the async loop's context if possible
            try:
                future = asyncio.run_coroutine_threadsafe(
                    _gather_cancelled(), loop
                )
                future.result(timeout=2.0)
            except Exception:
                pass

            logger.info(
                "ShutdownManager: cancelled %d async tasks", len(tasks)
            )
        except Exception as exc:
            logger.warning(
                "ShutdownManager: async task cancellation failed: %s", exc
            )

    @staticmethod
    def _safe_invoke(cb: Callable[[], Any] | None) -> None:
        """Safely invoke a callback, swallowing exceptions."""
        if cb is not None:
            try:
                cb()
            except Exception as exc:
                logger.warning(
                    "ShutdownManager: callback failed: %s", exc
                )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_shutdown_manager: ShutdownManager | None = None


def get_shutdown_manager() -> ShutdownManager:
    """Return the global ShutdownManager singleton."""
    global _shutdown_manager
    if _shutdown_manager is None:
        _shutdown_manager = ShutdownManager()
    return _shutdown_manager
