from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models.person import Person


class RecognitionService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def find_nearest_match(self, embedding: List[float]) -> Optional[Person]:
        """Return nearest match if distance is below threshold."""

        query_with_dist = (
            select(
                Person, Person.embedding.cosine_distance(embedding).label("distance")
            )
            .order_by("distance")
            .limit(1)
        )

        result = await self.db.execute(query_with_dist)
        match = result.first()

        if not match:
            return None

        person_obj, distance = match

        if distance < settings.SIMILARITY_THRESHOLD:
            return person_obj

        return None
