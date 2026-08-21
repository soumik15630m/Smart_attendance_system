import asyncio
import ipaddress
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.config import settings

router = APIRouter(prefix="/ws", tags=["streaming"])


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_video(self, frame_bytes: bytes):
        """Send binary video frame to all browsers, in parallel.

        A slow or dead viewer no longer blocks delivery to everyone else
        behind it, and connections that fail to send are dropped instead
        of silently lingering until their own receive loop notices.
        """
        connections = list(self.active_connections)
        if not connections:
            return

        results = await asyncio.gather(
            *(connection.send_bytes(frame_bytes) for connection in connections),
            return_exceptions=True,
        )
        for connection, result in zip(connections, results):
            if isinstance(result, Exception):
                self.disconnect(connection)

    async def broadcast_notification(self, data: dict):
        """Send JSON check-in data (Name, Time, Status) for Toasts, in parallel."""
        connections = list(self.active_connections)
        if not connections:
            return

        results = await asyncio.gather(
            *(connection.send_json(data) for connection in connections),
            return_exceptions=True,
        )
        for connection, result in zip(connections, results):
            if isinstance(result, Exception):
                self.disconnect(connection)


manager = ConnectionManager()


def _is_loopback_client(websocket: WebSocket) -> bool:
    client = websocket.client
    if not client:
        return False
    host = client.host
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    if settings.LOCAL_ONLY and not _is_loopback_client(websocket):
        await websocket.close(code=1008)
        return

    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/video-input")
async def video_input_endpoint(websocket: WebSocket):
    """
    to process gpu frames by camera_client.py
    """
    if settings.LOCAL_ONLY and not _is_loopback_client(websocket):
        await websocket.close(code=1008)
        return

    """Receive frames from camera client and fan out to viewers."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            await manager.broadcast_video(data)
    except WebSocketDisconnect:
        print("Camera Client Disconnected")
