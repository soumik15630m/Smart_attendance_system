from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models.person import Person


class RecognitionService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def find_nearest_match(
        self, embedding: list[float]
    ) -> tuple[Person, float] | None:
        """Return (person, distance) for the nearest active match below threshold."""

        # SET LOCAL scopes this to the current transaction only, so it can't
        # leak into other requests sharing a pooled connection.
        #
        # asyncpg sends parameterized statements through the extended query
        # protocol, and Postgres's SET command does not accept a bind
        # parameter ($1) as its value under that protocol -- it raises a
        # syntax error. hnsw.ef_search is an internal, admin-controlled
        # setting (not user input), so it's safe to inline as a literal
        # once validated as an int.
        ef_search = int(settings.HNSW_EF_SEARCH)
        await self.db.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))

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
