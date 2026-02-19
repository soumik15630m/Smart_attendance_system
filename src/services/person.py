from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.models.person import Person


class PersonService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def identify_person(self, embedding: list):
        """Return the nearest person match by cosine distance."""
        query = (
            select(Person)
            .order_by(Person.embedding.cosine_distance(embedding))
            .limit(1)
        )
        result = await self.db.execute(query)
        person = result.scalar_one_or_none()
        return person
