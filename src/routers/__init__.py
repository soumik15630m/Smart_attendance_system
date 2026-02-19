from .attendance import router as attendance_router
from .health import router as health_router
from .persons import router as persons_router

__all__ = ["attendance_router", "health_router", "persons_router"]
