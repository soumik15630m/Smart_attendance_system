from typing import List

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


@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/video-input")
async def video_input_endpoint(websocket: WebSocket):
    """Receive frames from camera client and fan out to viewers."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            await manager.broadcast_video(data)
    except WebSocketDisconnect:
        print("Camera Client Disconnected")
