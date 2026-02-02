from typing import List, Optional
from sqlalchemy import String, Boolean, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from .base import Base, TimestampMixin


class Person(Base, TimestampMixin):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    employee_id: Mapped[Optional[str]] = mapped_column(
        String(50), unique=True, index=True
    )
    role: Mapped[str] = mapped_column(String(20), server_default="employee")

    embedding: Mapped[List[float]] = mapped_column(Vector(512), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    attendance_logs = relationship(
        "Attendance", back_populates="person", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index(
            "ix_persons_embedding_cosine",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self):
        return f"<Person(id={self.id}, name='{self.name}')>"
