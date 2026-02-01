from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from src.database import get_db
from src.redis_config import get_redis
from src.services.attendance import AttendanceService
from src.services.person import PersonService
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/attendance", tags=["attendance"])

class IdentifyRequest(BaseModel):
    embedding: List[float]
    camera_id: str

@router.post("/identify")
async def identify_and_mark(
        request: IdentifyRequest,
        db: AsyncSession = Depends(get_db),
        redis = Depends(get_redis)
):
    person_service = PersonService(db)
    att_service = AttendanceService(db, redis)

    person = await person_service.identify_person(request.embedding)

    if not person:
        return {"status": "unknown", "message": "No matching person found"}

    record, created = await att_service.mark_attendance(person.id)

    if created:
        return {
            "status": "success",
            "person_name": person.name,
            "employee_id": person.employee_id
        }
    else:
        return {
            "status": "ignored",
            "message": "Attendance already marked recently",
            "person_name": person.name
        }