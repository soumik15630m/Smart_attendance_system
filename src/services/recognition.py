from typing import List, Optional, Tuple

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models.person import Person


class RecognitionService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def find_nearest_match(
        self, embedding: List[float]
    ) -> Optional[Tuple[Person, float]]:
        """Return (person, distance) for the nearest active match below threshold."""

        # SET LOCAL scopes this to the current transaction only, so it can't
        # leak into other requests sharing a pooled connection.
        await self.db.execute(
            text("SET LOCAL hnsw.ef_search = :ef_search"),
            {"ef_search": settings.HNSW_EF_SEARCH},
        )

        query_with_dist = (
            select(
                Person, Person.embedding.cosine_distance(embedding).label("distance")
            )
            .where(Person.is_active.is_(True))
            .order_by("distance")
            .limit(1)
        )

        result = await self.db.execute(query_with_dist)
        match = result.first()

        if not match:
            return None

        person_obj, distance = match

        if distance < settings.SIMILARITY_THRESHOLD:
            return person_obj, distance

        return None
