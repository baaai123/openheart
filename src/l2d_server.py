"""WebSocket bridge: Python → Electron L2D control.  v5.x"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)


class Live2DServer:
    """
    WebSocket server that controls Electron Live2D renderer.

    Usage in runtime_loop:
        _l2d = Live2DServer(port=9876)
        await _l2d.start()
        ...
        _l2d.set_expression("smile")
        _l2d.send_audio(pcm16_bytes)
    """

    def __init__(self, port: int = 9876):
        self._port = port
        self._clients: list[Any] = []
        self._server: Any = None
        self._ready = asyncio.Event()

    async def start(self):
        import logging as _ws_log
        _ws_log.getLogger("websockets").setLevel(logging.WARNING)
        import websockets
        self._server = await websockets.serve(
            self._handler, "0.0.0.0", self._port)
        logger.info(f"L2D WS server on port {self._port}")

    async def _handler(self, websocket):
        self._clients.append(websocket)
        logger.info("L2D client connected (total=%d)", len(self._clients))
        try:
            async for msg in websocket:
                # v5.x debug — log EVERY received message to diagnose ready-signal issue
                logger.info("L2D _handler received raw msg (len=%d, type=%s): %.200s",
                            len(msg), type(msg).__name__, msg)
                try:
                    data = json.loads(msg)
                    logger.info("L2D _handler parsed JSON: type=%s, keys=%s",
                                data.get("type", "?"), list(data.keys()))
                    if data.get("type") == "ready":
                        logger.info("L2D _handler: setting _ready event")
                        self._ready.set()
                    else:
                        logger.info("L2D _handler: unhandled message type=%s", data.get("type", "?"))
                except json.JSONDecodeError as e:
                    # JSON parse failure — log raw bytes for diagnosis
                    logger.warning("L2D _handler JSON parse error: %s, raw=%.200s", e, msg)
        except Exception as e:
            # Catch any unexpected exception in the handler loop
            logger.warning("L2D _handler exception: %s (type=%s)", e, type(e).__name__,
                            exc_info=True)
        finally:
            # Client disconnected or handler terminated — remove from broadcast list
            if websocket in self._clients:
                self._clients.remove(websocket)
            logger.info("L2D client disconnected (total=%d)", len(self._clients))

    async def wait_ready(self, timeout: float = 10.0) -> bool:
        if self._ready.is_set():
            return True
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning("L2D wait_ready timed out after %.1fs", timeout)
            return False

    async def _broadcast(self, msg: dict[str, Any]):
        _msg_type = msg.get("type", "?")
        logger.info("L2D _broadcast: type=%s, clients=%d", _msg_type, len(self._clients))
        if not self._clients:
            return
        payload = json.dumps(msg)
        for ws in self._clients[:]:
            try:
                await ws.send(payload)
            except Exception:
                # Client disconnected — _handler.finally handles removal
                pass

    def set_expression(self, name: str):
        """Send expression command to L2D (non-blocking)."""
        asyncio.ensure_future(
            self._broadcast({"type": "expression", "name": name}))

    def start_motion(self, name: str):
        """Send motion command to L2D."""
        asyncio.ensure_future(
            self._broadcast({"type": "motion", "name": name}))

    def send_audio(self, pcm16_bytes: bytes):
        """Deprecated: use start_speaking() for proper playback-timed lip-sync."""
        pass  # v5.x: timing fixed via start_speaking/stop_speaking below

    def start_speaking(self, duration_seconds: float):
        """Tell L2D to animate mouth for the full TTS playback duration."""
        asyncio.ensure_future(
            self._start_speaking_async(duration_seconds))

    async def _start_speaking_async(self, duration_seconds: float):
        logger.info("L2D start_speaking: duration=%.1fs, clients=%d",
                     duration_seconds, len(self._clients))
        await self._broadcast({
            "type": "speak",
            "duration": duration_seconds
        })
        logger.info("L2D start_speaking: broadcast done")

    def stop_speaking(self):
        """Reset mouth to closed after TTS playback ends."""
        asyncio.ensure_future(
            self._stop_speaking_async())

    async def _stop_speaking_async(self):
        logger.info("L2D stop_speaking: clients=%d", len(self._clients))
        await self._broadcast({"type": "speak_stop"})
        logger.info("L2D stop_speaking: broadcast done")

    def send_subtitle(self, role: str, text: str):
        """Send subtitle text to Electron L2D window for on-canvas display."""
        if not self._clients:
            return
        asyncio.ensure_future(
            self._broadcast({
                "type": "subtitle",
                "role": role,
                "text": text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }))

    def send_start_signal(self, sentence: str):
        """Send start signal to Electron — sentence text + mouth start."""
        asyncio.ensure_future(self._broadcast({
            "type": "start",
            "sentence": sentence,
        }))

    def send_finish_signal(self):
        """Send finish signal to Electron — mouth stop."""
        asyncio.ensure_future(self._broadcast({
            "type": "finish",
        }))

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()


# Quick self-test when run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Live2DServer module loaded successfully (no clients connected)")
