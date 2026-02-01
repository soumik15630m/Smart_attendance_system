from .attendance import router as attendance_router
from .health import router as health_router

# Huh!! for  wildcard imports
__all__ = ["attendance_router", "health_router"]