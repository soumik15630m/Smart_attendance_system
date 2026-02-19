from __future__ import annotations

import ipaddress
import platform
import socket
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.config import settings
from src.database import get_db
from src.models.attendance import Attendance
from src.models.person import Person
from src.services.script_runner import LocalScriptRunner, default_script_specs

router = APIRouter(tags=["local-ui"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEBUI_DIR = PROJECT_ROOT / "src" / "webui"
SCRIPT_RUNNER = LocalScriptRunner(PROJECT_ROOT, default_script_specs())


class ScriptStartRequest(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    employee_id: str | None = Field(default=None, max_length=60)


def _is_loopback_client(host: str | None) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def ensure_local_access(request: Request) -> None:
    if not settings.LOCAL_ONLY:
        return
    client_host = request.client.host if request.client else None
    if not _is_loopback_client(client_host):
        raise HTTPException(
            status_code=403,
            detail="Local-only mode enabled: dashboard access allowed from this machine only.",
        )


def _status_to_step(script_status: str) -> str:
    if script_status in {"running", "stopping"}:
        return "in_progress"
    if script_status == "completed":
        return "complete"
    if script_status == "failed":
        return "blocked"
    return "action"


async def _db_summary(db: AsyncSession) -> dict[str, int | str | bool | None]:
    try:
        people_count_result = await db.execute(select(func.count(Person.id)))
        attendance_today_result = await db.execute(
            select(func.count(Attendance.id)).where(Attendance.date == date.today())
        )
        return {
            "db_status": "up",
            "people_count": int(people_count_result.scalar_one() or 0),
            "attendance_today": int(attendance_today_result.scalar_one() or 0),
            "db_error": None,
        }
    except Exception as exc:
        return {
            "db_status": "down",
            "people_count": None,
            "attendance_today": None,
            "db_error": str(exc),
        }


@router.get("/ui", include_in_schema=False)
async def serve_dashboard(request: Request):
    ensure_local_access(request)
    index_path = WEBUI_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Dashboard files are missing.")
    return FileResponse(index_path)


@router.get("/ui/api/overview")
async def dashboard_overview(request: Request, db: AsyncSession = Depends(get_db)):
    ensure_local_access(request)
    db_data = await _db_summary(db)
    running = [s for s in SCRIPT_RUNNER.list_scripts() if s["status"] == "running"]

    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "local_only": settings.LOCAL_ONLY,
        "running_scripts": len(running),
        **db_data,
    }


@router.get("/ui/api/scripts")
async def list_scripts(request: Request):
    ensure_local_access(request)
    return {"scripts": SCRIPT_RUNNER.list_scripts()}


@router.post("/ui/api/scripts/{script_id}/start")
async def start_script(
    script_id: str,
    request: Request,
    payload: ScriptStartRequest | None = None,
):
    ensure_local_access(request)
    try:
        state = SCRIPT_RUNNER.start_script(
            script_id,
            payload.model_dump(exclude_none=True) if payload else {},
        )
        return {"script": state}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown script '{script_id}'.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/ui/api/scripts/{script_id}/stop")
async def stop_script(script_id: str, request: Request):
    ensure_local_access(request)
    try:
        state = SCRIPT_RUNNER.stop_script(script_id)
        return {"script": state}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown script '{script_id}'.")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/ui/api/scripts/{script_id}/logs")
async def read_logs(
    script_id: str,
    request: Request,
    tail: int = Query(default=160, ge=20, le=600),
):
    ensure_local_access(request)
    try:
        script = SCRIPT_RUNNER.get_script(script_id)
        logs = SCRIPT_RUNNER.get_logs(script_id, tail=tail)
        return {"script": script, "logs": logs}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown script '{script_id}'.")


@router.get("/ui/api/onboarding")
async def onboarding_status(request: Request, db: AsyncSession = Depends(get_db)):
    ensure_local_access(request)
    db_data = await _db_summary(db)
    scripts = {item["id"]: item for item in SCRIPT_RUNNER.list_scripts()}

    people_count = db_data.get("people_count")
    attendance_today = db_data.get("attendance_today")
    register_complete = bool(people_count and people_count > 0)
    attendance_seen = bool(attendance_today and attendance_today > 0)

    steps = [
        {
            "id": "gpu",
            "title": "Validate GPU Runtime",
            "description": "Run diagnostics and confirm CUDA provider availability.",
            "status": _status_to_step(scripts["test_gpu"]["status"]),
        },
        {
            "id": "seed",
            "title": "Initialize Sample Data",
            "description": "Create starter records for end-to-end verification.",
            "status": _status_to_step(scripts["seed_db"]["status"]),
        },
        {
            "id": "enroll",
            "title": "Enroll Your First Person",
            "description": "Open camera-based registration and capture one face.",
            "status": (
                "complete"
                if register_complete
                else _status_to_step(scripts["register_face"]["status"])
            ),
        },
        {
            "id": "camera",
            "title": "Run Edge Camera Client",
            "description": "Start local recognition and websocket relay streaming.",
            "status": _status_to_step(scripts["camera_client"]["status"]),
        },
        {
            "id": "attendance",
            "title": "Verify Attendance Logs",
            "description": "Confirm attendance events are being written to the database.",
            "status": "complete" if attendance_seen else "action",
        },
    ]

    return {
        "steps": steps,
        "db_status": db_data["db_status"],
        "db_error": db_data["db_error"],
    }


@router.get("/ui/api/attendance/recent")
async def recent_attendance(
    request: Request,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=8, ge=1, le=30),
):
    ensure_local_access(request)
    try:
        query = (
            select(Attendance)
            .options(selectinload(Attendance.person))
            .order_by(Attendance.created_at.desc())
            .limit(limit)
        )
        result = await db.execute(query)
        rows = []
        for record in result.scalars().all():
            rows.append(
                {
                    "id": record.id,
                    "person_name": record.person.name if record.person else "Unknown",
                    "employee_id": (
                        record.person.employee_id if record.person else None
                    ),
                    "method": record.method,
                    "date": str(record.date),
                    "time": record.created_at.strftime("%H:%M:%S"),
                }
            )
        return {"records": rows}
    except Exception as exc:
        return {"records": [], "error": str(exc)}
