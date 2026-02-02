import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError  # Add this import
from src.models.attendance import Attendance
from redis.asyncio import Redis


class AttendanceService:
    def __init__(self, db: AsyncSession, redis: Redis):
        self.db = db
        self.redis = redis

    async def mark_attendance(self, person_id: int):
        today = datetime.date.today()
        redis_key = f"attendance:{person_id}:{today}"

        # Quick check in Redis (The fast path)
        if await self.redis.get(redis_key):
            return None, False

        try:
            # Attempt to save to Postgres
            new_record = Attendance(
                person_id=person_id,
                date=today,
                method="face_bio",
                confidence_score=0.99,
            )
            self.db.add(new_record)
            await self.db.commit()
            await self.db.refresh(new_record)

            # Success! Set Redis cache for 12 hours
            await self.redis.setex(redis_key, 43200, "marked")
            return new_record, True

        except IntegrityError:
            # Handle the case where the DB already has a record (Safety fallback)
            await self.db.rollback()
            # Backfill Redis if it was somehow cleared
            await self.redis.setex(redis_key, 43200, "marked")
            return None, False
