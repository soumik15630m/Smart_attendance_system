import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database import get_db
from src.models.attendance import Attendance
from src.redis_config import get_redis
from src.schemas.attendance import AttendanceRead
from src.services.attendance import AttendanceService
from src.services.recognition import RecognitionService

router = APIRouter(prefix="/attendance", tags=["attendance"])


class IdentifyRequest(BaseModel):
    embedding: List[float]
    camera_id: str


@router.post("/identify")
async def identify_and_mark(
    request: IdentifyRequest,
    db: AsyncSession = Depends(get_db),
    cache=Depends(get_redis),
):
    rec_service = RecognitionService(db)
    att_service = AttendanceService(db, cache)

    person = await rec_service.find_nearest_match(request.embedding)

    if not person:
        return {"status": "unknown", "message": "No matching person found"}

    record, created = await att_service.mark_attendance(person.id)

    if created:
        return {
            "status": "success",
            "person_name": person.name,
            "employee_id": person.employee_id,
        }
    else:
        return {
            "status": "ignored",
            "message": "Attendance already marked recently",
            "person_name": person.name,
        }


@router.get("/history", response_model=List[AttendanceRead])
async def get_attendance_history(
    skip: int = 0,
    limit: int = 100,
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch attendance history with optional date filtering.
    """
    query = select(Attendance).options(selectinload(Attendance.person))

    if date:
        try:
            filter_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
            query = query.where(Attendance.date == filter_date)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
            )

    query = query.order_by(Attendance.created_at.desc()).offset(skip).limit(limit)

    result = await db.execute(query)
    return result.scalars().all()
