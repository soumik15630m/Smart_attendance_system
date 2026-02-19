from typing import List
import ipaddress

from src.config import settings

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/ws", tags=["streaming"])


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast_video(self, frame_bytes: bytes):
        """Send binary video frame to all browsers"""
        for connection in self.active_connections:
            try:
                await connection.send_bytes(frame_bytes)
            except Exception:
                continue

    async def broadcast_notification(self, data: dict):
        """Send JSON check-in data (Name, Time, Status) for Toasts"""
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                continue


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
