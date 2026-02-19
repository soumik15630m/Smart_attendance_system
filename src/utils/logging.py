import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", mode="a", encoding="utf-8"),
    ],
)


def get_logger(name: str):
    return logging.getLogger(name)


logger = get_logger("face_attendance_app")
