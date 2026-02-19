import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .person import PersonRead


class AttendanceBase(BaseModel):
    method: str = Field("face_bio", examples=["face_bio", "manual", "qr"])
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class AttendanceCreate(AttendanceBase):
    person_id: int
    date: datetime.date = Field(default_factory=datetime.date.today)


class AttendanceRead(AttendanceBase):
    id: int
    person_id: int
    date: datetime.date
    timestamp: datetime.datetime = Field(alias="created_at")

    person: Optional[PersonRead] = None

    model_config = ConfigDict(from_attributes=True)
