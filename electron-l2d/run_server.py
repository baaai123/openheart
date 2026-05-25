import asyncio
import logging
import os
import sys

# Resolve project root: env var overrides, otherwise derive from script location
PROJECT_ROOT = os.environ.get('PROJECT_ROOT') or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)
sys.path.insert(0, PROJECT_ROOT)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
from src.l2d_server import Live2DServer

async def main():
    server = Live2DServer(port=9876)
    await server.start()
    print("SERVER_READY_9876", flush=True)
    # Keep running until signaled; use Event for clean shutdown
    stop_event = asyncio.Event()
    await stop_event.wait()
    await server.stop()

asyncio.run(main())
