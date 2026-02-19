from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.database import get_db
from src.models.person import Person
from src.schemas.person import PersonCreate, PersonRead
from src.services.recognition import RecognitionService

router = APIRouter(prefix="/persons", tags=["persons"])


@router.post("/register", response_model=PersonRead)
async def register_person(person_in: PersonCreate, db: AsyncSession = Depends(get_db)):
    """Register person and block duplicate IDs or embeddings."""

    if person_in.employee_id:
        query = select(Person).where(Person.employee_id == person_in.employee_id)
        result = await db.execute(query)
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Employee ID '{person_in.employee_id}' already registered.",
            )

    rec_service = RecognitionService(db)
    existing_person = await rec_service.find_nearest_match(person_in.embedding)

    if existing_person:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Face already registered! Matched with: {existing_person.name} ({existing_person.employee_id})",
        )

    new_person = Person(
        name=person_in.name,
        employee_id=person_in.employee_id,
        role=person_in.role,
        embedding=person_in.embedding,
        is_active=person_in.is_active,
    )

    db.add(new_person)
    try:
        await db.commit()
        await db.refresh(new_person)
        return new_person
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Integrity Error: Duplicate data or invalid format.",
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
