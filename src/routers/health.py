from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from src.database import get_db

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Simple probe to verify the API server is up.
    """
    return {"status": "ok", "service": "Face Attendance API"}


@router.get("/db", status_code=status.HTTP_200_OK)
async def db_health_check(db: AsyncSession = Depends(get_db)):
    """
    Deep probe to verify the Database connection is active.
    """
    try:
        result = await db.execute(text("SELECT 1"))
        if result.scalar_one() == 1:
            return {"status": "up", "database": "connected"}
        else:
            raise HTTPException(
                status_code=500, detail="Database returned unexpected result"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection failed: {str(e)}",
        )
