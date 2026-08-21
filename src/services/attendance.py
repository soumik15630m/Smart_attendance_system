import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.attendance import Attendance
from src.redis_config import CacheClient
from src.utils.logging import logger


class AttendanceService:
    def __init__(self, db: AsyncSession, cache: CacheClient):
        self.db = db
        self.cache = cache

    async def _is_recently_marked(self, key: str) -> bool:
        try:
            return bool(await self.cache.get(key))
        except Exception:
            logger.warning(
                "Cache unavailable on cooldown check, falling back to DB "
                "unique constraint for key=%s",
                key,
                exc_info=True,
            )
            return False

    async def _mark_recently_marked(self, key: str) -> None:
        try:
            await self.cache.setex(key, 43200, "marked")
        except Exception:
            logger.warning(
                "Cache unavailable, could not set cooldown for key=%s",
                key,
                exc_info=True,
            )
            return

    async def mark_attendance(self, person_id: int, confidence_score: float):
        today = datetime.date.today()
        cache_key = f"attendance:{person_id}:{today}"

        if await self._is_recently_marked(cache_key):
            return None, False

        try:
            new_record = Attendance(
                person_id=person_id,
                date=today,
                method="face_bio",
                confidence_score=confidence_score,
            )
            self.db.add(new_record)
            await self.db.commit()
            await self.db.refresh(new_record)

            await self._mark_recently_marked(cache_key)
            return new_record, True

        except IntegrityError:
            await self.db.rollback()
            await self._mark_recently_marked(cache_key)
            return None, False
