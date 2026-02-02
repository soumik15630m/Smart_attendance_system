import asyncio
import sys
import os
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import AsyncSessionLocal
from src.models.person import Person
from sqlalchemy import select


async def seed():
    async with AsyncSessionLocal() as session:
        # Check if DB is already seeded
        result = await session.execute(select(Person).limit(1))
        if result.scalars().first():
            print("Database already contains data. Skipping seed.")
            return

        print("Seeding database with test users...")

        # Create a dummy user with a random embedding
        fake_embedding = [random.random() for _ in range(512)]

        test_user = Person(
            name="Elon Musk",
            role="admin",
            employee_id="TESLA-001",
            embedding=fake_embedding,
        )

        session.add(test_user)
        await session.commit()
        print(f"Added user: {test_user.name} with ID: {test_user.employee_id}")


if __name__ == "__main__":
    asyncio.run(seed())
