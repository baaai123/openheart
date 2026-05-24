"""Integration tests for logging infrastructure — v4.5.0 §10.1.

Validates JSON format, trace_id propagation across threads, file rotation,
console output, and the set_trace_id context manager.
"""
from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import threading
import time
from pathlib import Path

import pytest

from src.infra.logging_setup import (
    JSONFormatter,
    ConsoleFormatter,
    setup_logging,
    set_trace_id,
    log_context,
    run_with_trace_id,
    trace_id_var,
    layer_var,
    component_var,
    operation_var,
)


class TestSetupLogging:
    @pytest.fixture(autouse=True)
    def _reset_logging(self):
        """Reset logging state so each test gets a fresh setup."""
        # Clear _openheart_setup flags to defeat idempotency guard
        for handler in list(logging.root.handlers):
            if hasattr(handler, "_openheart_setup"):
                delattr(handler, "_openheart_setup")
        yield
        # Clean up after test
        for handler in list(logging.root.handlers):
            if getattr(handler, "_openheart_setup", False):
                logging.root.removeHandler(handler)

    def test_setup_creates_handlers(self, tmp_path: Path):
        """setup_logging should add file + console handlers to root logger."""
        root = logging.getLogger()
        initial_count = len(root.handlers)

        setup_logging(level="DEBUG", log_dir=tmp_path)

        new_count = len(root.handlers)
        assert new_count > initial_count, "Expected handlers to be added"

    def test_setup_is_idempotent(self, tmp_path: Path):
        """Calling setup_logging twice should not duplicate handlers."""
        setup_logging(level="INFO", log_dir=tmp_path)
        count_after_first = len(logging.root.handlers)
        setup_logging(level="INFO", log_dir=tmp_path)
        assert len(logging.root.handlers) == count_after_first

    def test_log_file_created(self, tmp_path: Path):
        """File handler should write a log file on first message."""
        setup_logging(level="INFO", log_dir=tmp_path)
        logger = logging.getLogger("test_file_created")
        logger.info("hello file")
        log_files = list(tmp_path.glob("openheart*"))
        assert len(log_files) >= 1, f"Expected log file not found in {tmp_path}"

    def test_log_file_is_json_lines(self, tmp_path: Path):
        """Each line in the log file should be valid JSON."""
        setup_logging(level="INFO", log_dir=tmp_path)
        logger = logging.getLogger("test_json_lines")
        logger.info("line one")
        logger.info("line two")
        log_file = tmp_path / "openheart.log"
        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) >= 2
        for line in lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON line: {line[:80]}... error: {e}")
            for key in ("timestamp", "level", "message"):
                assert key in obj, f"Missing key '{key}' in log entry"


class TestJSONFormatter:
    def test_formatter_includes_trace_id(self):
        """trace_id from ContextVar should appear in output."""
        token = trace_id_var.set("test-tid-123")
        try:
            formatter = JSONFormatter()
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=1,
                msg="hello", args=(), exc_info=None,
            )
            entry = json.loads(formatter.format(record))
            assert entry["trace_id"] == "test-tid-123"
        finally:
            trace_id_var.reset(token)

    def test_formatter_includes_layer_component_operation(self):
        """layer, component, operation should appear when set."""
        tid_tok = trace_id_var.set("tid")
        lay_tok = layer_var.set("perception")
        comp_tok = component_var.set("audio")
        op_tok = operation_var.set("process_chunk")
        try:
            formatter = JSONFormatter()
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=1,
                msg="test", args=(), exc_info=None,
            )
            entry = json.loads(formatter.format(record))
            assert entry["layer"] == "perception"
            assert entry["component"] == "audio"
            assert entry["operation"] == "process_chunk"
        finally:
            trace_id_var.reset(tid_tok)
            layer_var.reset(lay_tok)
            component_var.reset(comp_tok)
            operation_var.reset(op_tok)

    def test_formatter_empty_context_vars_produce_empty_strings(self):
        """When context vars are unset, trace_id etc. should be ''."""
        # Ensure clean state — previous tests may have set these vars
        for var in (trace_id_var, layer_var, component_var, operation_var):
            var.set("")
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=1,
            msg="test", args=(), exc_info=None,
        )
        entry = json.loads(formatter.format(record))
        assert entry["trace_id"] == ""
        assert entry["layer"] == ""

    def test_formatter_optional_fields_absent_when_not_set(self):
        """span_id, duration_ms, status should be absent when not provided."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=1,
            msg="test", args=(), exc_info=None,
        )
        entry = json.loads(formatter.format(record))
        assert "span_id" not in entry
        assert "duration_ms" not in entry
        assert "status" not in entry

    def test_formatter_optional_fields_present_when_set(self):
        """span_id, duration_ms, status should appear when set on record."""
        from src.infra.logging_setup import span_id_var, parent_span_id_var
        span_tok = span_id_var.set("span-001")
        parent_tok = parent_span_id_var.set("parent-000")
        try:
            formatter = JSONFormatter()
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=1,
                msg="test", args=(), exc_info=None,
            )
            record.duration_ms = 42.5
            record.status = "SUCCESS"
            entry = json.loads(formatter.format(record))
            assert entry["span_id"] == "span-001"
            assert entry["parent_span_id"] == "parent-000"
            assert entry["duration_ms"] == 42.5
            assert entry["status"] == "SUCCESS"
        finally:
            span_id_var.reset(span_tok)
            parent_span_id_var.reset(parent_tok)


class TestConsoleFormatter:
    def test_red_for_error(self):
        """ERROR-level messages should contain ANSI red codes."""
        formatter = ConsoleFormatter()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=1,
            msg="something broke", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "\033[31m" in output, "Expected red ANSI code in ERROR output"

    def test_no_red_for_info(self):
        """INFO-level messages should NOT contain ANSI red codes."""
        formatter = ConsoleFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=1,
            msg="all good", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "\033[31m" not in output


class TestSetTraceId:
    def test_context_manager_sets_and_restores(self):
        """set_trace_id should set trace_id in scope and restore after."""
        original = trace_id_var.get()
        with set_trace_id("ctx-tid-456"):
            assert trace_id_var.get() == "ctx-tid-456"
        assert trace_id_var.get() == original

    def test_context_manager_generates_when_none(self):
        """set_trace_id(None) should auto-generate a UUID."""
        with set_trace_id(None) as ctx:
            tid = trace_id_var.get()
            assert len(tid) == 36
            assert tid.count("-") == 4

    def test_nested_context_managers(self):
        """Nested set_trace_id should restore correctly."""
        with set_trace_id("outer"):
            assert trace_id_var.get() == "outer"
            with set_trace_id("inner"):
                assert trace_id_var.get() == "inner"
            assert trace_id_var.get() == "outer"


class TestLogContext:
    def test_sets_all_context_vars(self):
        """log_context should set trace_id, layer, component, operation."""
        with log_context(
            trace_id="tid-ctx", layer="fusion", component="sc",
            operation="merge",
        ):
            assert trace_id_var.get() == "tid-ctx"
            assert layer_var.get() == "fusion"
            assert component_var.get() == "sc"
            assert operation_var.get() == "merge"

    def test_restores_all_on_exit(self):
        """All context vars should be restored after log_context exits."""
        orig_tid = trace_id_var.get()
        orig_layer = layer_var.get()
        with log_context(layer="test-layer"):
            assert layer_var.get() == "test-layer"
        assert layer_var.get() == orig_layer


class TestTraceIdAcrossThreads:
    def test_trace_id_propagates_to_sub_thread(self):
        """run_with_trace_id should copy trace_id to the sub-thread."""
        captured_tid: list[str] = []

        def child_work():
            captured_tid.append(trace_id_var.get())

        with set_trace_id("parent-tid-thread"):
            thread = run_with_trace_id(child_work)
            thread.join(timeout=5)

        assert len(captured_tid) == 1
        assert captured_tid[0] == "parent-tid-thread"

    def test_trace_id_in_thread_is_independent_after_exit(self):
        """ContextVar changes in sub-thread should not affect parent."""
        with set_trace_id("parent-main"):
            with set_trace_id("parent-inner"):
                captured: list[str] = []

                def child():
                    captured.append(trace_id_var.get())

                thread = run_with_trace_id(child)
                thread.join(timeout=5)
                assert captured[0] == "parent-inner"

    def test_sub_thread_can_override_its_own_trace_id(self):
        """Sub-thread can set its own trace_id without affecting parent."""
        child_tids: list[str] = []

        def child():
            child_tids.append(trace_id_var.get())
            with set_trace_id("child-override"):
                child_tids.append(trace_id_var.get())
            child_tids.append(trace_id_var.get())

        with set_trace_id("parent-core"):
            thread = run_with_trace_id(child)
            thread.join(timeout=5)
            assert trace_id_var.get() == "parent-core"

        assert child_tids == ["parent-core", "child-override", "parent-core"]


class TestTraceIdAcrossAsync:
    @pytest.mark.asyncio
    async def test_trace_id_propagates_to_async_task(self):
        """ContextVar should automatically propagate to asyncio tasks."""
        captured: list[str] = []

        async def child_coro():
            captured.append(trace_id_var.get())

        with set_trace_id("async-parent-tid"):
            await child_coro()

        assert captured[0] == "async-parent-tid"

    @pytest.mark.asyncio
    async def test_trace_id_in_async_task_isolated_from_parent(self):
        """Changing trace_id in an async task should not affect parent."""
        with set_trace_id("parent-async"):
            async def child():
                token = trace_id_var.set("child-async")
                try:
                    await asyncio.sleep(0)
                finally:
                    trace_id_var.reset(token)

            await child()
            assert trace_id_var.get() == "parent-async"


class TestFileRotation:
    def test_rotating_file_handler_configured(self, tmp_path: Path):
        """setup_logging should set maxBytes=10MB and backupCount=5."""
        import logging.handlers
        setup_logging(level="INFO", log_dir=tmp_path)
        for handler in logging.root.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                assert handler.maxBytes == 10 * 1024 * 1024
                assert handler.backupCount == 5
                return
        pytest.fail("RotatingFileHandler not found on root logger")
