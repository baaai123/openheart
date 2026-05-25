#!/usr/bin/env python3
"""OpenHeart Frontend Server — pre-upload security UI. v5.x
Serves index.html, Live2D assets, REST API for file scanning + L2D WS proxy.

Usage: python frontend/server.py [--port 8000] [--l2d-port 9876]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
_log = logging.getLogger("frontend")

ROOT = Path(__file__).resolve().parent
LIVE2D_ROOT = ROOT.parent / "live2d"
SRC_ROOT = ROOT.parent / "src"

# Try to import privacy_filter; degrade gracefully if unavailable
_has_filter = False
_filter_sensitive = None
_generate_local_summary = None
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
try:
    from src.memory.privacy_filter import filter_sensitive, generate_local_summary
    _filter_sensitive = filter_sensitive
    _generate_local_summary = generate_local_summary
    _has_filter = True
    _log.info("privacy_filter loaded — real scanning enabled")
except ImportError:
    _log.warning("privacy_filter unavailable — using regex-only fallback")
    _has_filter = False


def _mask_fallback(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"1[3-9]\d{9}", lambda m: m.group()[:3] + "****" + m.group()[-4:], text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.\w{2,}", "***@***.***", text)
    return text


# ── Scan endpoints ────────────────────────────────────────────────

def _scan_text(text: str) -> dict:
    """Scan text for sensitive info, return results with trace_id."""
    trace_id = uuid.uuid4().hex[:12]
    source = "privacy_filter" if _has_filter else "fallback_regex"
    mask_fn = _filter_sensitive if _filter_sensitive else _mask_fallback

    cleaned = mask_fn(text)
    flagged = []
    for original_line in text.split("\n"):
        masked_line = mask_fn(original_line)
        if masked_line != original_line:
            # Extract what was masked
            for i, (c1, c2) in enumerate(zip(original_line, masked_line)):
                if c1 != c2 and c2 == "*":
                    start = max(0, i - 4)
                    end = min(len(original_line), i + 8)
                    context = original_line[start:end]
                    flagged.append({
                        "type": "sensitive_pattern",
                        "original_context": context,
                        "masked": masked_line[start:end] if end <= len(masked_line) else "***",
                        "position": i,
                    })
                    break

    return {
        "trace_id": trace_id,
        "source": source,
        "total_chars": len(text),
        "flagged_count": len(flagged),
        "flagged_items": flagged[:20],
        "cleaned_text": cleaned,
        "is_safe": len(flagged) == 0,
        "metadata": {
            "degraded": not _has_filter,
            "source_layer": "execution",
        },
    }


async def handle_scan_text(request: web.Request) -> web.Response:
    """POST /api/scan — scan submitted text for sensitive content."""
    body = await request.json()
    text = body.get("text", "")
    result = _scan_text(text)
    return web.json_response(result)


async def handle_scan_file(request: web.Request) -> web.Response:
    """POST /api/scan-file — upload + scan file for sensitive content."""
    reader = await request.multipart()
    field = await reader.next()
    if not field or field.name != "file":
        return web.json_response({"error": "no file field"}, status=400)

    filename = field.filename or "unknown"
    content = await field.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("utf-8", errors="replace")

    result = _scan_text(text)
    result["filename"] = filename
    result["file_size"] = len(content)
    return web.json_response(result)


# ── Chat queue endpoint ─────────────────────────────────────────────

CHAT_QUEUE = Path("/tmp/openheart_chat_queue.jsonl")


async def handle_chat(request: web.Request) -> web.Response:
    """POST /api/chat — queue user text for the backend to process.

    Writes one JSON line per message to CHAT_QUEUE (JSONL format);
    the backend (e.g. runtime_loop.py) tails this file and processes
    each line as a chat request.
    Returns {status: 'queued'} immediately.
    """
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return web.json_response({"status": "error", "message": "text is required"}, status=400)

    entry = {"text": text, "source": "frontend_ui"}
    try:
        with open(CHAT_QUEUE, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        _log.warning("Failed to write to chat queue %s: %s", CHAT_QUEUE, exc)
        return web.json_response({"status": "error", "message": str(exc)}, status=500)

    _log.info("Chat queued (%d chars): %s …", len(text), text[:80])
    return web.json_response({"status": "queued"})


# ── Live2D WebSocket proxy ─────────────────────────────────────────

class L2DProxy:
    """Proxy frontend WS <-> backend Live2DServer (port 9876)."""
    def __init__(self, l2d_port: int = 9876):
        self._l2d_port = l2d_port

    async def handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)

        try:
            session = aiohttp.ClientSession()
            backend = await session.ws_connect(
                f"ws://127.0.0.1:{self._l2d_port}", timeout=5)
        except Exception as exc:
            _log.warning("L2D backend offline (port %d): %s", self._l2d_port, exc)
            # Standalone echo mode — respond with ack + state
            await ws.send_json({"type": "l2d_status", "connected": False})
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = msg.json()
                    await ws.send_json({"type": "ack", "command": data.get("type")})
            return ws

        async def _relay(fr, to):
            async for msg in fr:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await to.send_str(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await to.send_bytes(msg.data)
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break

        await asyncio.gather(_relay(ws, backend), _relay(backend, ws))


# ── App ────────────────────────────────────────────────────────────

def make_app(l2d_port: int = 9876) -> web.Application:
    app = web.Application()

    @web.middleware
    async def cors(request: web.Request, handler):
        if request.method == "OPTIONS":
            return web.Response(headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            })
        resp = await handler(request)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp
    app.middlewares.append(cors)

    app.router.add_get("/", lambda r: web.FileResponse(ROOT / "index.html"))
    app.router.add_static("/frontend", ROOT, name="frontend")
    app.router.add_static("/live2d", LIVE2D_ROOT, name="live2d")
    app.router.add_post("/api/scan", handle_scan_text)
    app.router.add_post("/api/scan-file", handle_scan_file)
    app.router.add_post("/api/chat", handle_chat)

    app.router.add_get("/ws/l2d", L2DProxy(l2d_port).handler)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--l2d-port", type=int, default=9876)
    args = parser.parse_args()

    print(f"\n  OpenHeart Pre-Upload UI → http://localhost:{args.port}")
    print(f"  Privacy filter: {'LIVE' if _has_filter else 'FALLBACK (regex)'}")
    print(f"  L2D WS proxy      → ws://localhost:{args.l2d_port}\n")

    app = make_app(args.l2d_port)
    web.run_app(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
