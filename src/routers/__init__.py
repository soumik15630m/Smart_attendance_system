from .attendance import router as attendance_router
from .health import router as health_router
from .local_ui import router as local_ui_router
from .persons import router as persons_router

# Huh!! for  wildcard imports
__all__ = ["attendance_router", "health_router", "persons_router", "local_ui_router"]
