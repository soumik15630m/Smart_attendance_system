FROM python:3.10-slim

WORKDIR /app

# Native deps for OpenCV + PostgreSQL wheels.
# apt-get upgrade pulls in patched util-linux (CVE-2026-53612/53613/53614/53615).
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# setuptools/wheel pinned to patched versions (CVE-2025-47273, CVE-2026-24049).
RUN pip install --upgrade pip "setuptools>=78.1.1" "wheel>=0.46.2" && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
