import asyncio
import logging
import sys
sys.path.insert(0, '/home/baaai/projects/openheart')
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
