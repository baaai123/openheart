"""Centralised logging infrastructure — v4.5.0 §10.1

Features:
- JSON format logging with required fields: timestamp, trace_id, layer,
  component, level, operation, span_id, parent_span_id, duration_ms,
  status, message, extra.
- ``trace_id`` via ``contextvars.ContextVar`` — propagates across asyncio
  tasks automatically and sub-threads via ``copy_context``.
- File rotation: ``RotatingFileHandler``, 10 MB × 5 backup files.
- Console output: INFO level, ERROR-level messages in red (ANSI).
- ``setup_logging(level, log_dir)`` called once at startup.
- ``set_trace_id(trace_id)`` context manager for scoped trace assignment.

Usage::

    from src.infra.logging_setup import setup_logging, set_trace_id, log_context

    setup_logging(level="INFO", log_dir=Path("./logs"))

    with set_trace_id("abc-123"):
        logger.info("Processing request")  # includes trace_id=abc-123
"""
from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Context variables — §10.1 / §10.2
# ---------------------------------------------------------------------------

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default=""
)
span_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "span_id", default=""
)
parent_span_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "parent_span_id", default=""
)
layer_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "layer", default=""
)
component_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "component", default=""
)
operation_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "operation", default=""
)

# ---------------------------------------------------------------------------
# ANSI colour codes
# ---------------------------------------------------------------------------

_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"

# ---------------------------------------------------------------------------
# JSON formatter — §10.1 output shape
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """JSON log formatter matching the v4.5.0 §10.1 schema.

    Reads ``trace_id``, ``span_id``, ``parent_span_id``, ``layer``,
    ``component``, and ``operation`` from ContextVars.
    ``duration_ms`` and ``status`` are read from the LogRecord if set;
    otherwise they are omitted.
    """

    def format(self, record: logging.LogRecord) -> str:
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        log_entry: dict[str, Any] = {
            "timestamp": now,
            "trace_id": trace_id_var.get(),
            "layer": layer_var.get() or getattr(record, "layer", ""),
            "component": component_var.get() or getattr(record, "component", ""),
            "level": record.levelname,
            "operation": operation_var.get() or getattr(record, "operation", ""),
            "message": record.getMessage(),
        }

        # Optional span fields (§10.2)
        span = span_id_var.get()
        if span:
            log_entry["span_id"] = span
        parent = parent_span_id_var.get()
        if parent:
            log_entry["parent_span_id"] = parent

        # Optional performance fields
        duration = getattr(record, "duration_ms", None)
        if duration is not None:
            log_entry["duration_ms"] = duration
        status = getattr(record, "status", None)
        if status is not None:
            log_entry["status"] = status

        # Extra context
        extra = getattr(record, "extra", None)
        if extra is not None:
            log_entry["extra"] = extra

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with red ERROR lines."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"[{record.levelname:<5}] "
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        )
        tid = trace_id_var.get()
        if tid:
            base += f"[{tid[:8]}] "
        base += record.getMessage()

        if record.levelno >= logging.ERROR:
            return f"{_ANSI_RED}{base}{_ANSI_RESET}"

        return base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path = "logs/",
    *,
    file_level: str | None = None,
    console_level: str | None = None,
) -> None:
    """Configure the root logger with JSON file + coloured console handlers.

    Must be called once at startup.  Subsequent calls are idempotent.

    Parameters
    ----------
    level:
        Default log level for both handlers (e.g. ``"INFO"``).
    log_dir:
        Directory for rotated JSON log files.  Created if absent.
    file_level:
        Optional per-handler override for file output.
    console_level:
        Optional per-handler override for console output.
    """
    root = logging.getLogger()

    # Idempotency guard
    if any(getattr(h, "_openheart_setup", False) for h in root.handlers):
        return

    parsed_level = _resolve_level(level)
    fl = _resolve_level(file_level) if file_level else parsed_level
    cl = _resolve_level(console_level) if console_level else parsed_level

    # -- File handler with rotation (10 MB × 5) --------------------------- #
    log_path = Path(log_dir)
    # try/except: OSError on permission denied or disk full.
    # Safe: log directory creation failure falls back to console-only.
    try:
        log_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        logging.warning(
            "Cannot create log directory %s — file logging disabled.", log_path
        )

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_path / "openheart.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(fl)
    file_handler.setFormatter(JSONFormatter())

    # -- Console handler --------------------------------------------------- #
    console_handler = logging.StreamHandler()
    console_handler.setLevel(cl)
    console_handler.setFormatter(ConsoleFormatter())

    # -- Wire up ----------------------------------------------------------- #
    root.setLevel(min(fl, cl))
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Mark handlers to prevent double setup (attribute, not class replacement)
    setattr(file_handler, "_openheart_setup", True)
    setattr(console_handler, "_openheart_setup", True)

    root.info(
        "Logging configured: file=%s (level=%s, rotation=10MB×5), console=%s",
        log_path,
        logging.getLevelName(fl),
        logging.getLevelName(cl),
    )


@contextmanager
def set_trace_id(trace_id: str | None = None) -> Generator[None, None, None]:
    """Context manager that sets ``trace_id`` for the current scope.

    The trace_id is stored in a ``ContextVar`` so it propagates correctly
    across asyncio tasks.  For sub-threads, use :func:`run_with_trace_id`.

    Parameters
    ----------
    trace_id:
        UUID v4 string.  If ``None``, a fresh one is generated.
    """
    tid = trace_id or str(uuid.uuid4())
    token = trace_id_var.set(tid)
    try:
        yield
    finally:
        trace_id_var.reset(token)


@contextmanager
def log_context(
    *,
    trace_id: str | None = None,
    layer: str = "",
    component: str = "",
    operation: str = "",
) -> Generator[None, None, None]:
    """Set multiple context variables for the duration of a logging scope.

    This is the preferred way to annotate log output with per-component
    metadata.  All four context vars are restored on exit.

    Parameters
    ----------
    trace_id:
        Override trace_id.  If ``None``, a new one is generated only when
        the current value is empty.
    layer:
        Layer name (e.g. ``"perception"``, ``"decision"``).
    component:
        Component name (e.g. ``"AudioPipeline"``, ``"EasterEggSystem"``).
    operation:
        Operation name (e.g. ``"process_chunk"``, ``"check_all"``).
    """
    current_tid = trace_id_var.get()
    tid = trace_id or (current_tid if current_tid else str(uuid.uuid4()))

    tid_token = trace_id_var.set(tid)
    layer_token = layer_var.set(layer)
    comp_token = component_var.set(component)
    op_token = operation_var.set(operation)
    span = _make_span_id()
    span_token = span_id_var.set(span)

    try:
        yield
    finally:
        trace_id_var.reset(tid_token)
        layer_var.reset(layer_token)
        component_var.reset(comp_token)
        operation_var.reset(op_token)
        span_id_var.reset(span_token)


def run_with_trace_id(
    target: Any,
    *args: Any,
    trace_id: str | None = None,
    **kwargs: Any,
) -> threading.Thread:
    """Spawn a sub-thread that inherits the current trace_id context.

    Standard ``threading.Thread`` does NOT copy ContextVars.  This helper
    uses ``contextvars.copy_context().run()`` so the child thread logs
    with the same trace_id as the parent.

    Returns the started ``Thread`` object.
    """
    ctx = contextvars.copy_context()
    if trace_id is not None:
        ctx.run(trace_id_var.set, trace_id)

    thread = threading.Thread(
        target=ctx.run,
        args=(lambda: target(*args, **kwargs),),
    )
    thread.start()
    return thread


def get_logger(name: str) -> logging.Logger:
    """Return a logger ready to emit JSON-structured logs.

    Callers should include extra fields ``layer``, ``component``, and
    ``operation`` on every log call to comply with spec §10.1.
    """
    return logging.getLogger(name)


def spawn_traced_thread(
    target: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    daemon: bool = True,
) -> threading.Thread:
    """Start a ``Thread`` that inherits the current trace_id context.

    Wraps ``contextvars.copy_context().run()`` so the child thread sees
    the same ``trace_id_var`` value as the parent.
    """
    if kwargs is None:
        kwargs = {}

    ctx = contextvars.copy_context()

    def _runner() -> None:
        ctx.run(target, *args, **kwargs)

    t = threading.Thread(target=_runner, daemon=daemon)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_level(level: str) -> int:
    """Convert a level name to its numeric constant."""
    parsed = getattr(logging, level.upper(), None)
    if isinstance(parsed, int):
        return parsed
    logging.warning("Unknown log level %r — defaulting to INFO.", level)
    return logging.INFO


def _make_span_id() -> str:
    """Generate a short hex span identifier (§10.2)."""
    return uuid.uuid4().hex[:16]
