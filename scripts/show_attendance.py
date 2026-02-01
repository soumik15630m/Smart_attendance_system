import asyncio
import sys
import os
import logging
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.orm import selectinload

script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == 'scripts':
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir

sys.path.append(project_root)
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

# Disable the root logger and specific SQLAlchemy loggers
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.dialects").setLevel(logging.WARNING)

try:
    import src.config
    src.config.settings.DEBUG = False
except ImportError:
    pass

from src.database import AsyncSessionLocal
from src.models.attendance import Attendance

async def show_attendance():
    print("\n" + "="*95)
    print(f" {'ID':<5} | {'Date':<12} | {'Time':<10} | {'Name':<20} | {'Employee ID':<15} | {'Method':<10}")
    print("="*95)

    try:
        async with AsyncSessionLocal() as session:
            query = (
                select(Attendance)
                .options(selectinload(Attendance.person))
                .order_by(Attendance.date.desc(), Attendance.created_at.desc())
            )

            result = await session.execute(query)
            records = result.scalars().all()

            if not records:
                print(f" {'No records found.':<90}")
            else:
                for record in records:
                    name = record.person.name if record.person else "Unknown"
                    emp_id = record.person.employee_id if record.person else "---"
                    time_str = record.created_at.strftime("%H:%M:%S")

                    print(f" {record.id:<5} | {str(record.date):<12} | {time_str:<10} | {name:<20} | {emp_id:<15} | {record.method:<10}")

    except Exception as e:
        print(f"\n[!] Error fetching data: {e}")
        if "DATABASE_URL" in str(e):
            print("    Hint: Check your .env file location.")

    print("="*95 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(show_attendance())
    except KeyboardInterrupt:
        pass