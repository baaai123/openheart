"""
Unit tests for the REST API server endpoints.

v5.x: Validates /api/ping, /api/visual/status, /api/visual/toggle,
/api/config visualEnabled sync, and /api/status visual_enabled field.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp.test_utils import make_mocked_request

import src.config.api_server as srv
from src.config.api_server import (
    get_ping,
    get_status,
    get_visual_status,
    post_config,
    post_visual_toggle,
    set_visual_enabled_callback,
)


def _reset_globals() -> None:
    """Reset module-level globals between tests."""
    import src.config.api_server as srv

    srv._on_visual_enabled_change = None
    srv._voice_pipeline_ref = None


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_config_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect config file paths to temp dirs so tests don't touch real files."""
    _reset_globals()
    ui_file = tmp_path / "ui_settings.json"
    srv_file = tmp_path / "server_config.json"
    monkeypatch.setattr("src.config.api_server.UI_CONFIG_PATH", ui_file)
    monkeypatch.setattr("src.config.api_server.CONFIG_PATH", srv_file)
    # Write defaults to both files
    ui_file.write_text(json.dumps({"visual_enabled": True, "voice_mode": "asr"}))
    srv_file.write_text(json.dumps({}))
    yield


# ── Helpers ──────────────────────────────────────────────────


async def _call(handler: Any, method: str = "GET", path: str = "/", body: dict | None = None) -> Any:
    """Call a handler with an optional JSON body."""
    req = make_mocked_request(method, path)
    if body is not None:
        mock_json = AsyncMock(return_value=body)
        req.json = mock_json
    return await handler(req)


def _ui_config() -> dict:
    """Read current ui_settings.json from temp path."""
    return json.loads(Path(str(srv.UI_CONFIG_PATH)).read_text())


def _srv_config() -> dict:
    """Read current server_config.json from temp path."""
    return json.loads(Path(str(srv.CONFIG_PATH)).read_text())


# ── /api/ping tests ─────────────────────────────────────────


class TestApiPing:
    """Tests for the GET /api/ping health check endpoint."""

    async def _call_ping(self) -> dict:
        """Helper: call the ping handler and return parsed JSON body."""
        req = make_mocked_request("GET", "/api/ping")
        resp = await get_ping(req)
        assert resp.status == 200
        return json.loads(resp.text)

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
        """timestamp field must be a valid ISO 8601 UTC datetime string."""
        data = await self._call_ping()
        ts = data["timestamp"]
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None
        assert dt.utcoffset() is not None

    @pytest.mark.asyncio
    async def test_ping_trace_id_is_hex(self) -> None:
        """trace_id must be a 12-character lowercase hex string."""
        data = await self._call_ping()
        tid = data["trace_id"]
        assert isinstance(tid, str)
        assert len(tid) == 12
        assert re.fullmatch(r"[0-9a-f]{12}", tid)


# ── /api/visual/status tests ────────────────────────────────


class TestApiVisualStatus:
    """Tests for GET /api/visual/status."""

    @pytest.mark.asyncio
    async def test_returns_visual_enabled_true_by_default(self) -> None:
        """Should return visual_enabled=True when ui_settings has it True."""
        resp = await _call(get_visual_status)
        assert resp.status == 200
        data = json.loads(resp.text)
        assert data["visual_enabled"] is True

    @pytest.mark.asyncio
    async def test_returns_visual_enabled_false_when_disabled(self) -> None:
        """Should return visual_enabled=False when ui_settings has it False."""
        ui_file = Path(str(srv.UI_CONFIG_PATH))
        ui_file.write_text(json.dumps({"visual_enabled": False, "voice_mode": "asr"}))
        resp = await _call(get_visual_status)
        assert resp.status == 200
        data = json.loads(resp.text)
        assert data["visual_enabled"] is False

    @pytest.mark.asyncio
    async def test_returns_true_on_missing_file(self) -> None:
        """Should default to True when ui_settings.json is missing."""
        Path(str(srv.UI_CONFIG_PATH)).unlink(missing_ok=True)
        resp = await _call(get_visual_status)
        assert resp.status == 200
        data = json.loads(resp.text)
        assert data["visual_enabled"] is True


# ── /api/visual/toggle tests ────────────────────────────────


class TestApiVisualToggle:
    """Tests for POST /api/visual/toggle."""

    @pytest.mark.asyncio
    async def test_toggle_enable(self) -> None:
        """POST {"enabled": true} should persist to both config files."""
        # Start with disabled
        Path(str(srv.UI_CONFIG_PATH)).write_text(json.dumps({"visual_enabled": False}))
        resp = await _call(post_visual_toggle, "POST", "/", {"enabled": True})
        assert resp.status == 200
        data = json.loads(resp.text)
        assert data["visual_enabled"] is True
        # Check both config files
        assert _ui_config()["visual_enabled"] is True
        assert _srv_config()["visualEnabled"] is True

    @pytest.mark.asyncio
    async def test_toggle_disable(self) -> None:
        """POST {"enabled": false} should persist to both config files."""
        resp = await _call(post_visual_toggle, "POST", "/", {"enabled": False})
        assert resp.status == 200
        data = json.loads(resp.text)
        assert data["visual_enabled"] is False
        assert _ui_config()["visual_enabled"] is False
        assert _srv_config()["visualEnabled"] is False

    @pytest.mark.asyncio
    async def test_toggle_flips_current_state(self) -> None:
        """POST {} with no 'enabled' key should toggle current state."""
        # Start with True
        Path(str(srv.UI_CONFIG_PATH)).write_text(json.dumps({"visual_enabled": True}))
        resp = await _call(post_visual_toggle, "POST", "/", {})
        assert resp.status == 200
        assert json.loads(resp.text)["visual_enabled"] is False
        # Toggle again
        resp = await _call(post_visual_toggle, "POST", "/", {})
        assert json.loads(resp.text)["visual_enabled"] is True

    @pytest.mark.asyncio
    async def test_toggle_invalid_json_returns_400(self) -> None:
        """Non-JSON body should return 400."""
        req = make_mocked_request("POST", "/")
        mock_json = AsyncMock(side_effect=json.JSONDecodeError("Expecting value", "", 0))
        req.json = mock_json
        resp = await post_visual_toggle(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_toggle_fires_callback(self) -> None:
        """set_visual_enabled_callback should fire on toggle."""
        callback = MagicMock()
        set_visual_enabled_callback(callback)
        await _call(post_visual_toggle, "POST", "/", {"enabled": False})
        callback.assert_called_once_with(False)

    @pytest.mark.asyncio
    async def test_toggle_callback_not_fired_when_not_set(self) -> None:
        """No error when callback is None."""
        resp = await _call(post_visual_toggle, "POST", "/", {"enabled": False})
        assert resp.status == 200


# ── /api/config visualEnabled sync tests ────────────────────


class TestApiConfigVisualSync:
    """Tests that POST /api/config with visualEnabled syncs to ui_settings.json."""

    @pytest.mark.asyncio
    async def test_visual_enabled_syncs_to_ui_settings(self) -> None:
        """visualEnabled in POST /api/config should write visual_enabled to ui_settings.json."""
        resp = await _call(post_config, "POST", "/", {"visualEnabled": False})
        assert resp.status == 200
        assert _ui_config()["visual_enabled"] is False

    @pytest.mark.asyncio
    async def test_visual_enabled_does_not_affect_voice_mode(self) -> None:
        """Changing visualEnabled should not clobber voice_mode in ui_settings."""
        Path(str(srv.UI_CONFIG_PATH)).write_text(json.dumps({"voice_mode": "text", "visual_enabled": True}))
        await _call(post_config, "POST", "/", {"visualEnabled": False})
        ui = _ui_config()
        assert ui["visual_enabled"] is False
        assert ui["voice_mode"] == "text"

    @pytest.mark.asyncio
    async def test_visual_enabled_not_in_body_does_not_change_ui_settings(self) -> None:
        """POST /api/config without visualEnabled should not touch ui_settings."""
        Path(str(srv.UI_CONFIG_PATH)).write_text(json.dumps({"visual_enabled": False, "voice_mode": "asr"}))
        await _call(post_config, "POST", "/", {"model": "gpt-4"})
        assert _ui_config()["visual_enabled"] is False

    @pytest.mark.asyncio
    async def test_visual_enabled_fires_callback(self) -> None:
        """Sync via post_config should fire the visual enabled callback."""
        callback = MagicMock()
        set_visual_enabled_callback(callback)
        await _call(post_config, "POST", "/", {"visualEnabled": False})
        callback.assert_called_once_with(False)


# ── /api/status visual_enabled tests ────────────────────────


class TestApiStatusVisualField:
    """Tests that GET /api/status includes visual_enabled."""

    @pytest.mark.asyncio
    async def test_status_includes_visual_enabled_true(self) -> None:
        """GET /api/status should contain visual_enabled: true by default."""
        resp = await _call(get_status)
        assert resp.status == 200
        data = json.loads(resp.text)
        assert "visual_enabled" in data
        assert data["visual_enabled"] is True

    @pytest.mark.asyncio
    async def test_status_includes_visual_enabled_false(self) -> None:
        """GET /api/status should reflect disabled state from ui_settings."""
        Path(str(srv.UI_CONFIG_PATH)).write_text(json.dumps({"visual_enabled": False}))
        resp = await _call(get_status)
        data = json.loads(resp.text)
        assert data["visual_enabled"] is False
