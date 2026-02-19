from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PersonBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, examples=["John Doe"])
    employee_id: Optional[str] = Field(None, max_length=50, examples=["EMP-001"])
    role: str = Field("employee", max_length=20)
    is_active: bool = True


class PersonCreate(PersonBase):
    embedding: List[float] = Field(
        ..., min_length=512, max_length=512, description="512-dim face vector"
    )


class PersonRead(PersonBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
