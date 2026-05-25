"""
Unit tests for the GET /api/ping health check endpoint.

v5.x: Validates HTTP 200, status field, ISO 8601 timestamp, and hex trace_id.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import pytest
from aiohttp.test_utils import make_mocked_request

from src.config.api_server import get_ping


class TestApiPing:
    """Tests for the GET /api/ping health check endpoint."""

    async def _call_ping(self) -> dict:
        """Helper: call the ping handler and return parsed JSON body."""
        req = make_mocked_request("GET", "/api/ping")
        resp = await get_ping(req)
        assert resp.status == 200  # sanity: always 200 on success
        return json.loads(resp.text)

    # ── Individual tests ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_ping_returns_200(self) -> None:
        """GET /api/ping must return HTTP 200."""
        req = make_mocked_request("GET", "/api/ping")
        resp = await get_ping(req)
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_ping_has_status_ok(self) -> None:
        """Response JSON must contain ``status`` equal to ``"ok"``."""
        data = await self._call_ping()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_ping_timestamp_is_iso8601(self) -> None:
        """``timestamp`` field must be a valid ISO 8601 UTC datetime string."""
        data = await self._call_ping()
        ts = data["timestamp"]

        # Parse ISO 8601 — must succeed
        dt = datetime.fromisoformat(ts)

        # Must be timezone-aware (UTC)
        assert dt.tzinfo is not None, "timestamp must be timezone-aware"
        assert dt.utcoffset() is not None, "timestamp must have a UTC offset"

    @pytest.mark.asyncio
    async def test_ping_trace_id_is_hex(self) -> None:
        """``trace_id`` must be a 12-character lowercase hex string."""
        data = await self._call_ping()
        tid = data["trace_id"]

        assert isinstance(tid, str), "trace_id must be a string"
        assert len(tid) == 12, f"trace_id must be exactly 12 chars, got {len(tid)}"
        assert re.fullmatch(r"[0-9a-f]{12}", tid), (
            f"trace_id '{tid}' is not a valid 12-char hex string"
        )
