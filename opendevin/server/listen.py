from fastapi import FastAPI, WebSocket

import agenthub  # noqa F401 (we import this to get the agents registered)
from opendevin.server.session import Session

app = FastAPI()


# This endpoint recieves events from the client (i.e. the browser)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = Session(websocket)
    # TODO: should this use asyncio instead of await?
    await session.start_listening()
