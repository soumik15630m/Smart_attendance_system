from datetime import date
from sqlalchemy import ForeignKey, Date, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


class Attendance(Base, TimestampMixin):
    __tablename__ = "attendance"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Foreign Key to Person
    person_id: Mapped[int] = mapped_column(
        ForeignKey("persons.id"), nullable=False, index=True
    )

    # The actual date of attendance (stored separately for easy unique indexing)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # Method of verification (e.g., "face_bio", "manual_override")
    method: Mapped[str] = mapped_column(server_default="face_bio")

    # Confidence score of the match at that moment
    confidence_score: Mapped[float] = mapped_column(nullable=True)

    # Relationships
    person = relationship("Person", back_populates="attendance_logs")

    # one record per person per day.
    __table_args__ = (
        UniqueConstraint("person_id", "date", name="uq_person_attendance_daily"),
    )

    def __repr__(self):
        return f"<Attendance(person_id={self.person_id}, date='{self.date}')>"
