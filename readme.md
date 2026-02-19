# Smart Attendance System

A **Hybrid Edge–Cloud Face Recognition System** using FastAPI, InsightFace, and pgvector.

---

## What it does

The **Smart Attendance System** automates personnel check-ins by performing real-time face recognition. It uses a **hybrid architecture** where face detection and vector embedding happen at the **Edge (client device)**, while vector similarity search and record-keeping happen in the **Cloud (server)**.

---

## Why it exists

To eliminate manual attendance logs and reduce network bandwidth usage by transmitting **lightweight vector embeddings** instead of heavy image or video streams.

---

## 1. Problem Statement

Traditional biometric attendance systems often suffer from:

- **High Latency**  
  Uploading full images to the cloud for processing is slow.

- **Bandwidth Saturation**  
  Streaming video from multiple cameras cripples local networks.

- **Scalability Issues**  
  Linear search in databases degrades as user counts grow.

This system shifts **AI inference to the edge** and uses **HNSW (Hierarchical Navigable Small World)** indexing via `pgvector` for **O(log n)** similarity search, enabling instant matching even with thousands of users.

---

## 2. Key Features

- **Edge AI Processing**  
  Runs InsightFace locally to extract 512-dimensional embeddings; only JSON data is sent to the server.

- **Vector Similarity Search**  
  PostgreSQL + pgvector for high-performance cosine similarity matching.

- **Anti-Spam Cooldowns**  
  Cache-backed cooldown prevents duplicate attendance entries (default: 12 hours) using Upstash REST or Redis with automatic local startup.

- **Real-time Monitoring**  
  WebSocket relay broadcasts live video feeds to a web dashboard without server-side processing.

- **GPU Accelerated**  
  Native NVIDIA CUDA support via ONNX Runtime.

- **Automatic Migrations**  
  Self-healing database schema using Alembic.

**Non-Goals**

This system is **not** a payroll calculator or HR management suite.  
It is strictly an **identity verification and attendance logging engine**.

---

## 3. Architecture Overview

The system follows a **Split-Compute** pattern.

### Client (Edge)

- Captures video frames
- Performs image quality checks (brightness, face size)
- Generates face embeddings (vectors)
- Sends `{ vector + camera_id }` to the API

### Server (Core)

- FastAPI receives the embedding
- Cache checks cooldown (Upstash REST / Redis with local auto-start)
- PostgreSQL performs vector similarity search
- Attendance is logged if confidence exceeds threshold

---

## 4. Repository Structure

```text
smart-attendance-system/
├── src/                  # Core server implementation
│   ├── models/           # SQLAlchemy database models
│   ├── routers/          # API endpoints (attendance, persons, stream)
│   ├── services/         # Business logic (recognition, attendance cooldown)
│   └── main.py           # Application entry point
├── scripts/              # Client-side tools
│   ├── camera_client.py  # Edge AI camera runner
│   ├── register_face.py  # CLI tool to enroll new users
│   └── test_gpu.py       # CUDA diagnostics
├── alembic/              # Database migrations
├── docker-compose.yml    # Container orchestration
└── .env.example          # Configuration template
````

---

## 5. Installation

### Requirements

* **OS**

    * Server: Linux
    * Client: Windows / Linux
* **Runtime**

    * Python 3.10+ or Docker Desktop
* **Hardware**

    * NVIDIA GPU recommended for client (CUDA 12.x)

---

### Method 1: Docker (Server — Recommended)

Clone the repository:

```bash
git clone https://github.com/soumik15630/smart_attendance_system.git
cd smart_attendance_system
```

Configure environment:

```bash
cp .env.example .env
# Edit .env with database credentials
```

Run with Docker Compose:

```bash
docker-compose up --build
```

---

### Method 2: Manual (Client)

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify GPU support (optional):

```bash
python scripts/test_gpu.py
```

---

## 6. Quick Start

1. **Start the Server**

Ensure Docker is running and the database is healthy. If Upstash is not configured, the app will auto-start local Redis in `CACHE_BACKEND=auto` mode.
API endpoint:

```
http://localhost:8000
```

Web dashboard (local machine only):

```
http://127.0.0.1:8000/ui
```

2. **Register a User**

Run on a machine with a webcam:

```bash
python scripts/register_face.py
```

Follow prompts to capture face and enter Name/ID.

3. **Start the Camera Client**

```bash
python scripts/camera_client.py
```

Faces will be detected and attendance results rendered on the video feed.

---

## 7. Usage Guide

### Common Workflows

#### A. Enrolling New Employees

* Run `scripts/register_face.py`
* Ensures no duplicate faces exist
* Press `s` to save when prompted

#### B. Daily Operation

* Run `scripts/camera_client.py` on entrance kiosk

Visual feedback:

* **Green box** → Known user, attendance marked
* **Red box** → Unknown user or low confidence

#### C. Viewing Logs

Via API or CLI helper:

```bash
python scripts/show_attendance.py
```

---

## 8. Configuration

Configuration is managed via the `.env` file.

| Variable             | Description                       | Example / Default                      |
| -------------------- | --------------------------------- | -------------------------------------- |
| DATABASE_URL         | PostgreSQL connection string      | postgresql+asyncpg://user:pass@host/db |
| CACHE_BACKEND        | `auto`, `upstash_rest`, `redis`     | auto                                   |
| REDIS_URL            | Redis connection URL              | redis://localhost:6379/0               |
| AUTO_START_LOCAL_REDIS | Auto-start local Redis if needed | true                                  |
| LOCAL_REDIS_START_CMD | Optional custom startup command   | redis-server --port 6379 ...           |
| PREFER_DOCKER_REDIS | Try Docker Redis command before OS-native command | true                     |
| UPSTASH_REDIS_REST_URL | Upstash REST endpoint           | https://<id>.upstash.io                |
| UPSTASH_REDIS_REST_TOKEN | Upstash REST bearer token     | <token>                                |
| SIMILARITY_THRESHOLD | Match strictness (0.0–1.0)        | 0.5                                    |
| API_KEY              | Client authentication key         | my-secret-admin-key                    |
| SERVER_IP            | Server address for client scripts | 127.0.0.1                              |

---

## 9. Performance & Benchmarks

* **Search Complexity**
  O(log n) using HNSW indexes on pgvector
  Tested with 1M+ vectors in millisecond range.

* **Network Load**
  < 5 KB per recognition event (JSON only)
  vs ~2 MB/s continuous video streaming.

* **Latency**
  < 100 ms average round-trip on local LAN.

---

## 10. Error Handling & Diagnostics

* **Logs**
  Output to stdout and `app.log`

* **Database Resilience**
  Uses `pool_pre_ping=True` for auto-recovery

* **GPU Issues**
  If the client crashes on startup, run:

```bash
python scripts/test_gpu.py
```

---

## 11. Security Considerations

* **API Security**
  Uses static `X-API-Key` header
  *Production recommendation:* OAuth2 or mTLS

* **Data Privacy**
  No images are stored—only vector embeddings

* **Replay Attacks**
  V1 does not fully prevent video replay attacks

---

## 12. Roadmap

* [ ] Next.js dashboard for analytics
* [ ] Liveness detection (anti-spoofing)
* [ ] Multi-tenant support
* [ ] Slack / Email notifications

---

## 13. License

Licensed under the **GNU General Public License v3.0**.
See the `LICENSE` file for details.

