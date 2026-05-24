"""Minimal test: verify ws 8.x async send behavior."""
import asyncio
import json
import logging
import sys
sys.path.insert(0, '/home/baaai/projects/openheart')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

from src.l2d_server import Live2DServer

async def main():
    server = Live2DServer(port=9877)
    await server.start()
    print("\n>>> Server on 9877. Waiting 5s for messages...", flush=True)
    try:
        await asyncio.wait_for(server._ready.wait(), timeout=5.0)
        print(">>> READY received!", flush=True)
    except asyncio.TimeoutError:
        print(">>> TIMEOUT", flush=True)
    print(">>> Client count:", len(server._clients), flush=True)
    await server.stop()

asyncio.run(main())
