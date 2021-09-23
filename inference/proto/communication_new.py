import asyncio, threading, time
import websockets

from configuration import *

connected = False

async def connect():
    async with websockets.connect(WS_URI) as ws:
        connected = True
        while True:
            await send_msg(ws, "hello")
            await get_msg(ws)
            time.sleep(0.5)

async def send_msg(ws, msg):
    await ws.send(msg)

async def get_msg(ws):
    data = await ws.recv()
    print("received:", data)

while True:
    try:
        asyncio.get_event_loop().run_until_complete(connect())
    except:
        continue