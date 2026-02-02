# src/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# FIX: Added 'build-essential' (for C++ compilation) and 'python3-dev' (for headers)
# FIX: Kept 'libgl1' for OpenCV support
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Upgrading pip first is often safer for binary wheels
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run the app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]