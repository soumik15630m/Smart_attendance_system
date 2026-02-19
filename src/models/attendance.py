from datetime import date

from sqlalchemy import Date, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


class Attendance(Base, TimestampMixin):
    __tablename__ = "attendance"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    person_id: Mapped[int] = mapped_column(
        ForeignKey("persons.id"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    method: Mapped[str] = mapped_column(server_default="face_bio")
    confidence_score: Mapped[float] = mapped_column(nullable=True)
    person = relationship("Person", back_populates="attendance_logs")
    __table_args__ = (
        UniqueConstraint("person_id", "date", name="uq_person_attendance_daily"),
    )

    def __repr__(self):
        return f"<Attendance(person_id={self.person_id}, date='{self.date}')>"
