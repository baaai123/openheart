"""Test client — sends commands to Electron L2D."""
import asyncio
import json
import time

import numpy as np
import websockets


async def main():
    async with websockets.connect("ws://127.0.0.1:9876") as ws:
        await asyncio.sleep(1)
        for expr in ["smile", "sad", "neutral", "surprised"]:
            await ws.send(json.dumps({"type": "expression", "name": expr}))
            await asyncio.sleep(1.5)
        for _ in range(10):
            pcm = (np.sin(np.linspace(0, 6.28, 512)) * 0.3).astype(np.float32)
            await ws.send(json.dumps({
                "type": "audio",
                "samples": pcm.tolist(),
            }))
            await asyncio.sleep(0.05)


if __name__ == "__main__":
    asyncio.run(main())
