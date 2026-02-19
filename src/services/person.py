from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.models.person import Person


class PersonService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def identify_person(self, embedding: list):
        """
        Uses the pgvector operator (<=>) to find the closest match
        in the database based on cosine distance.
        """
        # We query the 'persons' table and sort by the closest embedding
        # LIMIT 1 gives us the best match
        query = (
            select(Person)
            .order_by(Person.embedding.cosine_distance(embedding))
            .limit(1)
        )
        result = await self.db.execute(query)
        person = result.scalar_one_or_none()

        # we can add a distance threshold check here
        # to prevent "false matches" if the face is too different
        return person
