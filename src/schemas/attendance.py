from typing import Optional
import datetime  # Import module to avoid name collision with the field 'date'
from pydantic import BaseModel, Field, ConfigDict
from .person import PersonRead

# --- Base Schema ---
class AttendanceBase(BaseModel):
    method: str = Field("face_bio", examples=["face_bio", "manual", "qr"])
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

# --- Create Schema (Input) ---
class AttendanceCreate(AttendanceBase):
    person_id: int
    # FIX: Use datetime.date explicitly for the type, avoiding the clash
    date: datetime.date = Field(default_factory=datetime.date.today)

# --- Read Schema (Output) ---
class AttendanceRead(AttendanceBase):
    id: int
    person_id: int
    date: datetime.date
    timestamp: datetime.datetime = Field(alias="created_at")

    # Optional: Include full person details in the response
    person: Optional[PersonRead] = None

    model_config = ConfigDict(from_attributes=True)