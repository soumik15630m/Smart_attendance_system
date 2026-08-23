"""Tests for src/routers/web_stream.py.

Covers the ConnectionManager fan-out logic (a failing/slow connection
shouldn't block or crash delivery to the others) at the unit level with
fake websockets, plus the two websocket endpoints end-to-end using
Starlette's synchronous TestClient, which supports real websocket
handshakes without needing a live server.
"""

import pytest
from starlette.testclient import TestClient

from src.config import settings
from src.routers.web_stream import ConnectionManager
from src.routers.web_stream import router as web_stream_router


class _FakeWebSocket:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.sent_bytes: list[bytes] = []
        self.sent_json: list[dict] = []

    async def send_bytes(self, data: bytes) -> None:
        if self.fail:
            raise RuntimeError("connection reset by peer")
        self.sent_bytes.append(data)

    async def send_json(self, data: dict) -> None:
        if self.fail:
            raise RuntimeError("connection reset by peer")
        self.sent_json.append(data)


# --- ConnectionManager.broadcast_video --------------------------------


@pytest.mark.asyncio
async def test_broadcast_video_delivers_to_all_connections():
    manager = ConnectionManager()
    good_a, good_b = _FakeWebSocket(), _FakeWebSocket()
    manager.active_connections = [good_a, good_b]  # type: ignore[list-item]

    await manager.broadcast_video(b"frame-1")

    assert good_a.sent_bytes == [b"frame-1"]
    assert good_b.sent_bytes == [b"frame-1"]


@pytest.mark.asyncio
async def test_broadcast_video_drops_failing_connection_without_blocking_others():
    manager = ConnectionManager()
    failing = _FakeWebSocket(fail=True)
    healthy = _FakeWebSocket()
    manager.active_connections = [failing, healthy]  # type: ignore[list-item]

    await manager.broadcast_video(b"frame-1")

    assert healthy.sent_bytes == [b"frame-1"]
    assert failing not in manager.active_connections
    assert healthy in manager.active_connections


@pytest.mark.asyncio
async def test_broadcast_video_noop_with_no_connections():
    manager = ConnectionManager()
    manager.active_connections = []
    await manager.broadcast_video(b"frame-1")  # should not raise


# --- ConnectionManager.broadcast_notification --------------------------


@pytest.mark.asyncio
async def test_broadcast_notification_delivers_to_all_connections():
    manager = ConnectionManager()
    good_a, good_b = _FakeWebSocket(), _FakeWebSocket()
    manager.active_connections = [good_a, good_b]  # type: ignore[list-item]

    payload = {"name": "Alice", "status": "success"}
    await manager.broadcast_notification(payload)

    assert good_a.sent_json == [payload]
    assert good_b.sent_json == [payload]


@pytest.mark.asyncio
async def test_broadcast_notification_drops_failing_connection():
    manager = ConnectionManager()
    failing = _FakeWebSocket(fail=True)
    healthy = _FakeWebSocket()
    manager.active_connections = [failing, healthy]  # type: ignore[list-item]

    await manager.broadcast_notification({"name": "Bob"})

    assert healthy.sent_json == [{"name": "Bob"}]
    assert failing not in manager.active_connections


@pytest.mark.asyncio
async def test_broadcast_notification_noop_with_no_connections():
    manager = ConnectionManager()
    manager.active_connections = []
    await manager.broadcast_notification({"name": "Nobody"})  # should not raise


# --- disconnect ----------------------------------------------------------


def test_disconnect_removes_known_connection():
    manager = ConnectionManager()
    ws = _FakeWebSocket()
    manager.active_connections = [ws]  # type: ignore[list-item]
    manager.disconnect(ws)  # type: ignore[arg-type]
    assert ws not in manager.active_connections


def test_disconnect_is_a_noop_for_unknown_connection():
    manager = ConnectionManager()
    ws = _FakeWebSocket()
    manager.active_connections = []
    manager.disconnect(ws)  # type: ignore[arg-type]  # should not raise


# --- websocket endpoints (isolated app, no DB/Redis/lifespan needed) -----


def _stream_app():
    """A bare FastAPI app mounting only the websocket router.

    Deliberately not `src.main.app`: that app's lifespan makes a real
    Redis connection on startup, which these websocket-only tests don't
    need and which CI doesn't provision a Redis service for.
    """
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(web_stream_router)
    return app


def test_stream_endpoint_accepts_loopback_client(monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_ONLY", True)
    with (
        TestClient(_stream_app(), client=("127.0.0.1", 12345)) as tc,
        tc.websocket_connect("/ws/stream") as ws,
    ):
        ws.close()


def test_stream_endpoint_rejects_non_loopback_client_when_local_only(monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setattr(settings, "LOCAL_ONLY", True)
    with TestClient(_stream_app(), client=("8.8.8.8", 12345)) as tc:
        with (
            pytest.raises(WebSocketDisconnect) as exc_info,
            tc.websocket_connect("/ws/stream"),
        ):
            pass
        assert exc_info.value.code == 1008


def test_stream_endpoint_allows_any_client_when_not_local_only(monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_ONLY", False)
    with (
        TestClient(_stream_app(), client=("8.8.8.8", 12345)) as tc,
        tc.websocket_connect("/ws/stream") as ws,
    ):
        ws.close()


def test_video_input_broadcasts_frames_to_connected_viewers(monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_ONLY", False)
    with (
        TestClient(_stream_app()) as tc,
        tc.websocket_connect("/ws/stream") as viewer,
        tc.websocket_connect("/ws/video-input") as camera,
    ):
        camera.send_bytes(b"\x00\x01\x02frame")
        received = viewer.receive_bytes()
        assert received == b"\x00\x01\x02frame"
