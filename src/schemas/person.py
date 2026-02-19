from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# --- Base Schema (Shared properties) ---
class PersonBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, examples=["John Doe"])
    employee_id: Optional[str] = Field(None, max_length=50, examples=["EMP-001"])
    role: str = Field("employee", max_length=20)
    is_active: bool = True


# --- Create Schema (Input) ---
class PersonCreate(PersonBase):
    # The API might receive the embedding directly from the recognition service
    embedding: List[float] = Field(
        ..., min_length=512, max_length=512, description="512-dim face vector"
    )


# --- Read Schema (Output) ---
class PersonRead(PersonBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
