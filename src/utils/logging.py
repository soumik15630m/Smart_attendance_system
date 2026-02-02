import logging
import sys

# Configure standard logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # Log to console
        logging.StreamHandler(sys.stdout),
        # Log to file
        logging.FileHandler("app.log", mode="a", encoding="utf-8"),
    ],
)


def get_logger(name: str):
    return logging.getLogger(name)


logger = get_logger("face_attendance_app")
