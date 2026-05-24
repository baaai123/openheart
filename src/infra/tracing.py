"""Distributed tracing infrastructure — TraceManager + @trace_span.

Reuses ContextVars from logging_setup.py (§10.1).
Provides:
  - ``TraceManager`` — context manager that creates/nests spans,
    sets ContextVars, records timing, logs on close.
  - ``trace_span(layer, component, operation)`` — async decorator
    wrapping any function in a TraceManager span.
  - ``generate_trace_id()`` — convenience for entry-point usage.

Usage::

    from src.infra.tracing import trace_span, generate_trace_id, TraceManager

    # Option A: decorator (most common)
    @trace_span(layer="perception", component="visual_orchestrator", operation="capture_screenshot")
    async def my_func():
        ...

    # Option B: manual context manager (for fine-grained sub-spans)
    async with TraceManager(layer="perception", component="ocr", operation="scan_full"):
        ...

    # Option C: at entry point, generate and set trace_id
    trace_id = generate_trace_id()
    async with TraceManager(layer="perception", component="perception_bus",
                            operation="cycle", trace_id=trace_id):
        ...

# v5.x §10.2 — Distributed tracing
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Optional

from src.infra.logging_setup import (
    component_var,
    layer_var,
    operation_var,
    parent_span_id_var,
    span_id_var,
    trace_id_var,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TraceRecord schema helpers (§10.2)
# ---------------------------------------------------------------------------

def _new_id() -> str:
    """Generate a uuid4 hex string (compact, no dashes)."""
    return uuid.uuid4().hex


def generate_trace_id() -> str:
    """Generate a new trace_id for entry-point use.

    Returns a compact 32-char hex string.
    """
    return _new_id()


# ---------------------------------------------------------------------------
# Span status constants (§10.1)
# ---------------------------------------------------------------------------

SPAN_SUCCESS = "SUCCESS"
SPAN_FAILURE = "FAILURE"
SPAN_DEGRADED = "DEGRADED"
SPAN_TIMEOUT = "TIMEOUT"


# ---------------------------------------------------------------------------
# TraceManager — per-span context manager
# ---------------------------------------------------------------------------

class TraceManager:
    """Context manager that creates a distributed-trace span.

    On enter:
      - Generates a new ``span_id``.
      - Saves the current ``span_id`` as ``parent_span_id`` for nesting.
      - Sets all ContextVars (trace_id, span_id, parent_span_id, layer,
        component, operation).
      - Records ``start_time``.

    On exit:
      - Computes ``duration_ms``.
      - Logs a span-close record with status, duration, and fields.
      - Restores the previous span state (so parent spans resume correctly).

    Parameters
    ----------
    layer : str
        System layer (e.g. ``"perception"``, ``"memory"``, ``"decision"``).
    component : str
        Component name (e.g. ``"visual_orchestrator"``, ``"retrieval_gate"``).
    operation : str
        Specific operation (e.g. ``"capture_screenshot"``, ``"query"``).
    trace_id : str, optional
        Explicit trace_id. If omitted, uses the current ContextVar value.
        Generate one via ``generate_trace_id()`` for entry points.
    """

    def __init__(
        self,
        layer: str,
        component: str,
        operation: str,
        trace_id: Optional[str] = None,
    ) -> None:
        self._layer = layer
        self._component = component
        self._operation = operation
        self._explicit_trace_id = trace_id

        # Saved state for restoration
        self._saved: dict[str, str] = {}
        self._start_time: float = 0.0
        self._span_id: str = ""

    async def __aenter__(self) -> TraceManager:
        self._start_time = time.monotonic()

        # Save current ContextVar values for restoration on exit
        self._saved = {
            "trace_id": trace_id_var.get(),
            "span_id": span_id_var.get(),
            "parent_span_id": parent_span_id_var.get(),
            "layer": layer_var.get(),
            "component": component_var.get(),
            "operation": operation_var.get(),
        }

        # Determine trace_id: explicit > incoming
        current_trace_id = (
            self._explicit_trace_id or self._saved["trace_id"] or _new_id()
        )

        # The current span_id becomes the parent for the new span
        current_parent = self._saved["span_id"] or ""

        # Generate new span_id
        self._span_id = _new_id()

        # Set all ContextVars
        trace_id_var.set(current_trace_id)
        span_id_var.set(self._span_id)
        parent_span_id_var.set(current_parent)
        layer_var.set(self._layer)
        component_var.set(self._component)
        operation_var.set(self._operation)

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        duration_ms = (time.monotonic() - self._start_time) * 1000.0

        # Determine status
        if exc_type is asyncio.CancelledError:
            status = SPAN_TIMEOUT
        elif exc_type is not None:
            status = SPAN_FAILURE
        else:
            status = SPAN_SUCCESS

        # Log span close — matches §10.1 JSON schema
        logger.info(
            "Span closed",
            extra={
                "trace_id": trace_id_var.get(),
                "span_id": self._span_id,
                "parent_span_id": parent_span_id_var.get(),
                "layer": self._layer,
                "component": self._component,
                "operation": self._operation,
                "duration_ms": round(duration_ms, 2),
                "status": status,
            },
        )

        # Restore previous ContextVar state
        trace_id_var.set(self._saved["trace_id"])
        span_id_var.set(self._saved["span_id"])
        parent_span_id_var.set(self._saved["parent_span_id"])
        layer_var.set(self._saved["layer"])
        component_var.set(self._saved["component"])
        operation_var.set(self._saved["operation"])


# ---------------------------------------------------------------------------
# @trace_span decorator — wraps an async callable in TraceManager
# ---------------------------------------------------------------------------

def trace_span(
    layer: str,
    component: str,
    operation: str,
    trace_id: Optional[str] = None,
) -> Callable[..., Any]:
    """Decorator that wraps an async function in a ``TraceManager`` span."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with TraceManager(
                layer=layer,
                component=component,
                operation=operation,
                trace_id=trace_id,
            ):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Synchronous convenience
# ---------------------------------------------------------------------------

@contextmanager
def sync_trace_span(
    layer: str,
    component: str,
    operation: str,
    trace_id: Optional[str] = None,
) -> Any:
    """Synchronous context manager version of TraceManager.

    Use inside blocking GPU calls or thread-pool tasks where ``async with``
    is unavailable::

        with sync_trace_span(layer="perception", component="region_proposer",
                             operation="propose"):
            ...
    """
    start = time.monotonic()
    saved = {
        "trace_id": trace_id_var.get(),
        "span_id": span_id_var.get(),
        "parent_span_id": parent_span_id_var.get(),
        "layer": layer_var.get(),
        "component": component_var.get(),
        "operation": operation_var.get(),
    }

    current_trace_id = trace_id or saved["trace_id"] or _new_id()
    current_parent = saved["span_id"] or ""
    span_id = _new_id()

    trace_id_var.set(current_trace_id)
    span_id_var.set(span_id)
    parent_span_id_var.set(current_parent)
    layer_var.set(layer)
    component_var.set(component)
    operation_var.set(operation)

    status = SPAN_SUCCESS
    try:
        yield
    except asyncio.CancelledError:
        status = SPAN_TIMEOUT
        raise
    except Exception:
        status = SPAN_FAILURE
        raise
    finally:
        duration_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Sync span closed",
            extra={
                "trace_id": trace_id_var.get(),
                "span_id": span_id,
                "parent_span_id": saved["span_id"],
                "layer": layer,
                "component": component,
                "operation": operation,
                "duration_ms": round(duration_ms, 2),
                "status": status,
            },
        )
        # Restore
        trace_id_var.set(saved["trace_id"])
        span_id_var.set(saved["span_id"])
        parent_span_id_var.set(saved["parent_span_id"])
        layer_var.set(saved["layer"])
        component_var.set(saved["component"])
        operation_var.set(saved["operation"])
