"""OpenHeart REST API server for UI config and status. v5.x

Endpoints:
  POST /api/config  — save config to config/server_config.json
  GET  /api/config  — read current config
  GET  /api/status  — backend + L2D status

Runs on aiohttp, port 8081 (env: OPENMATE_API_PORT).
CORS open for Electron dev.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from src.voice_pipeline import VoicePipeline

_log = logging.getLogger("api_server")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "server_config.json"
UI_CONFIG_PATH = PROJECT_ROOT / "config" / "ui_settings.json"
DEFAULT_PORT = int(os.environ.get("OPENMATE_API_PORT", "8081"))
_VALID_VOICE_MODES = frozenset({"asr", "text"})

# Global reference to the active VoicePipeline (set at startup via set_voice_pipeline)
_voice_pipeline_ref: VoicePipeline | None = None

DEFAULT_CONFIG = {
    "baseUrl": "",
    "model": "",
    "apiKey": "",
    "systemPrompt": "",
    "voiceEnabled": True,
    "visualEnabled": True,
    "l2dEnabled": True,
    "voiceMode": "asr",
}

ALLOWED_FIELDS = frozenset(DEFAULT_CONFIG.keys())
BOOL_FIELDS = frozenset({"voiceEnabled", "visualEnabled", "l2dEnabled"})


def _module_available(name: str) -> bool:
    try:
        import importlib.util
        spec = importlib.util.find_spec(name)
        return spec is not None
    except (ImportError, ModuleNotFoundError):
        return False


def _load_config() -> dict:
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("Failed to load config from %s: %s", CONFIG_PATH, exc)
    return dict(DEFAULT_CONFIG)


def _save_config(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        tmp.replace(CONFIG_PATH)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def set_voice_pipeline(pipeline: VoicePipeline | None) -> None:
    """Set the active VoicePipeline reference for voice toggle endpoints.

    Called once at startup from the runtime loop after creating the
    VoicePipeline instance.
    """
    global _voice_pipeline_ref  # noqa: PLW0603
    _voice_pipeline_ref = pipeline


def _read_ui_config() -> dict[str, object]:
    """Read the current ui_settings.json, returning empty dict on failure."""
    try:
        return json.loads(UI_CONFIG_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_ui_config(data: dict[str, object]) -> None:
    """Atomically write ui_settings.json with the given data."""
    tmp = UI_CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(UI_CONFIG_PATH)


# ── Handlers ────────────────────────────────────────────

async def get_config(request: web.Request) -> web.Response:
    cfg = _load_config()
    return web.json_response(cfg)


async def post_config(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception) as exc:
        _log.warning("POST /api/config — invalid JSON: %s", exc)
        return web.json_response(
            {"error": "Invalid JSON body", "trace_id": _new_trace_id()}, status=400
        )

    if not isinstance(body, dict):
        return web.json_response(
            {"error": "Body must be a JSON object", "trace_id": _new_trace_id()}, status=400
        )

    unknown = set(body.keys()) - ALLOWED_FIELDS
    if unknown:
        return web.json_response(
            {"error": f"Unknown fields: {', '.join(sorted(unknown))}", "trace_id": _new_trace_id()},
            status=400,
        )

    if "voiceMode" in body and body["voiceMode"] not in _VALID_VOICE_MODES:
        return web.json_response(
            {"error": f"voiceMode must be one of: {', '.join(sorted(_VALID_VOICE_MODES))}",
             "trace_id": _new_trace_id()},
            status=400,
        )

    for field in BOOL_FIELDS:
        if field in body and not isinstance(body[field], bool):
            return web.json_response(
                {"error": f"{field} must be a boolean", "trace_id": _new_trace_id()},
                status=400,
            )

    existing = _load_config()
    existing.update(body)

    try:
        _save_config(existing)
    except OSError as exc:
        tid = _new_trace_id()
        _log.warning("[%s] Config persist failed: %s", tid, exc)
        return web.json_response(
            {"error": "Failed to save configuration", "trace_id": tid}, status=500
        )

    # v4.5.0 §0.6 — apply voiceEnabled change live to VoicePipeline
    if "voiceEnabled" in body and _voice_pipeline_ref is not None:
        if body["voiceEnabled"]:
            _voice_pipeline_ref.resume()
        else:
            _voice_pipeline_ref.pause()

    # v4.5.0 §0.6 — persist voiceMode to ui_settings.json for runtime loop
    if "voiceMode" in body:
        ui_config = _read_ui_config()
        ui_config["voice_mode"] = body["voiceMode"]
        _write_ui_config(ui_config)

    safe = {k: v for k, v in existing.items() if k != "apiKey"}
    _log.info("Config saved: %s", safe)
    return web.json_response(existing)


# ── Voice control handlers ───────────────────────────────

async def get_voice_status(request: web.Request) -> web.Response:
    # v4.5.0 §0.6 — return current voice pipeline state
    if _voice_pipeline_ref is None:
        return web.json_response({
            "voice_enabled": False,
            "voice_mode": _load_config().get("voiceMode", "asr"),
            "voice_pipeline": "uninitialized",
        })

    ui_config = _read_ui_config()
    return web.json_response({
        "voice_enabled": _voice_pipeline_ref.voice_enabled,
        "voice_mode": ui_config.get("voice_mode", _load_config().get("voiceMode", "asr")),
        "voice_pipeline": "ready",
    })


async def post_voice_toggle(request: web.Request) -> web.Response:
    # v4.5.0 §0.6 — POST body: {"enabled": true/false}
    if _voice_pipeline_ref is None:
        return web.json_response(
            {"error": "Voice pipeline not initialized", "trace_id": _new_trace_id()},
            status=503,
        )

    try:
        data: dict[str, object] = await request.json()
    except Exception:
        return web.json_response(
            {"error": "Invalid JSON body", "trace_id": _new_trace_id()},
            status=400,
        )

    enabled = data.get("enabled")
    if enabled is None:
        # No explicit value → toggle current state
        enabled = not _voice_pipeline_ref.voice_enabled

    if enabled:
        _voice_pipeline_ref.resume()
    else:
        _voice_pipeline_ref.pause()

    # Persist to both config files for consistency
    srv_config = _load_config()
    srv_config["voiceEnabled"] = enabled
    _save_config(srv_config)

    ui_config = _read_ui_config()
    ui_config["voice_enabled"] = enabled
    _write_ui_config(ui_config)

    _log.info("Voice toggle via API: enabled=%s", enabled)
    return web.json_response({"voice_enabled": enabled})


async def get_status(request: web.Request) -> web.Response:
    tts_ok = _module_available("cosyvoice")
    l2d_ok = _module_available("live2d")
    backend_running = True

    # Include voice state in the status response
    voice_state = {
        "voice_enabled": _voice_pipeline_ref.voice_enabled if _voice_pipeline_ref is not None else False,
    }

    return web.json_response({
        "backend": "running" if backend_running else "stopped",
        "status": "running" if backend_running else "stopped",
        "l2d": "running" if l2d_ok else "stopped",
        "l2d_status": "running" if l2d_ok else "stopped",
        "tts": "running" if tts_ok else "stopped",
        **voice_state,
    })


# ── Health check ───────────────────────────────────────────

async def get_ping(request: web.Request) -> web.Response:
    """Health check endpoint for monitoring and Docker health checks.

    Returns:
        JSON with status, ISO 8601 UTC timestamp, and trace_id.

    v5.x: Standard /api/ping health check.
    """
    return web.json_response({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": _new_trace_id(),
    })


# ── CORS middleware ─────────────────────────────────────

@web.middleware
async def cors_middleware(request: web.Request, handler) -> web.StreamResponse:
    if request.method == "OPTIONS":
        resp = web.Response(status=204)
    else:
        try:
            resp = await handler(request)
        except web.HTTPException:
            raise
        except Exception:
            _log.exception("Unhandled error in handler")
            resp = web.json_response(
                {"error": "Internal server error", "trace_id": _new_trace_id()}, status=500
            )

    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp


# ── 404 catch-all ───────────────────────────────────────

async def handle_404(request: web.Request) -> web.Response:
    return web.json_response(
        {"error": "Not found", "path": request.path, "trace_id": _new_trace_id()},
        status=404,
    )


# ── App factory ─────────────────────────────────────────

def create_app() -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get("/api/config", get_config)
    app.router.add_post("/api/config", post_config)
    app.router.add_get("/api/status", get_status)
    app.router.add_get("/api/ping", get_ping)  # v5.x: health check endpoint
    app.router.add_get("/api/voice/status", get_voice_status)
    app.router.add_post("/api/voice/toggle", post_voice_toggle)
    # Catch-all for graceful 404 on unknown routes
    app.router.add_route("*", "/{tail:.*}", handle_404)
    _log.info(
        "Routes registered: GET|POST /api/config, GET /api/status, "
        "GET /api/ping, GET /api/voice/status, POST /api/voice/toggle"
    )
    return app


# ── Entry point ─────────────────────────────────────────

def main(port: int = DEFAULT_PORT) -> None:
    app = create_app()
    _log.info("API server starting on 0.0.0.0:%d", port)
    web.run_app(app, host="0.0.0.0", port=port, print=None)


if __name__ == "__main__":
    main()
