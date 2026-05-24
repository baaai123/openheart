"""Quick test: start Live2DServer, wait for ready signal."""
import asyncio
import logging
import sys
sys.path.insert(0, '/home/baaai/projects/openheart')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

from src.l2d_server import Live2DServer

async def main():
    server = Live2DServer(port=9876)
    await server.start()
    print("\n>>> Server started. Waiting for ready signal (timeout 10s)...", flush=True)
    try:
        await asyncio.wait_for(server._ready.wait(), timeout=10.0)
        print(">>> READY signal received!", flush=True)
    except asyncio.TimeoutError:
        print(">>> TIMEOUT: No ready signal received in 10s", flush=True)
    await server.stop()

asyncio.run(main())
