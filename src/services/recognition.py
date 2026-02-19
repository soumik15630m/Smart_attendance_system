from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models.person import Person


class RecognitionService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def find_nearest_match(self, embedding: List[float]) -> Optional[Person]:
        """
        Performs a vector similarity search in the database.

        Args:
            embedding: The 512-dimensional vector from the camera.

        Returns:
            The matched Person object if the distance is below the threshold,
            otherwise None.
        """

        query = (
            select(Person)
            .order_by(Person.embedding.cosine_distance(embedding))
            .limit(1)
        )

        result = await self.db.execute(query)
        person = result.scalars().first()

        if not person:
            return None

        # Validation: Check the actual distance
        # We need to calculate the distance again or trust the visual match?
        # A robust way is to re-fetch the distance or simply check if the DB returned *any* valid match
        # assuming we might want to put a WHERE clause for threshold in the query itself later.

        # However, purely relying on ORDER BY limit 1 always returns *someone*.
        # We must verify if that "someone" is actually close enough.

        # Optimized approach: Calculate distance in Python for the single result
        # OR update query to return distance. Let's do the query update for precision.

        query_with_dist = (
            select(
                Person, Person.embedding.cosine_distance(embedding).label("distance")
            )
            .order_by("distance")
            .limit(1)
        )

        result = await self.db.execute(query_with_dist)
        match = result.first()  # Returns (Person, distance) tuple

        if not match:
            return None

        person_obj, distance = match

        # Threshold Check
        # distance = 0.0 (Identical) -> 1.0 (Orthogonal) -> 2.0 (Opposite)
        # settings.SIMILARITY_THRESHOLD is around 0.4 or 0.5 for InsightFace
        if distance < settings.SIMILARITY_THRESHOLD:
            return person_obj

        return None
