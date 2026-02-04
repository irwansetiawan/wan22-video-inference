# WAN 2.2 EC2 API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a FastAPI server that wraps WAN 2.2 TI2V-5B model for video generation, deployable on AWS EC2 g5.xlarge.

**Architecture:** Single Python process with FastAPI serving HTTP requests, SQLite for job persistence, background thread processing one job at a time via CLI wrapper, S3 for video storage.

**Tech Stack:** Python 3.10+, FastAPI, SQLite, boto3, WAN 2.2 (via subprocess)

---

## Task 1: Project Structure & Config

**Files:**
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `requirements.txt`

**Step 1: Create project structure**

```bash
mkdir -p src scripts
touch src/__init__.py
```

**Step 2: Create requirements.txt**

```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-dotenv==1.0.0
boto3==1.34.0
httpx==0.26.0
pydantic==2.5.3
```

**Step 3: Create src/config.py**

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API
API_SECRET_KEY = os.environ["API_SECRET_KEY"]
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# AWS
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
S3_BUCKET = os.environ["S3_BUCKET"]
S3_REGION = os.getenv("S3_REGION", "ap-southeast-1")

# Paths
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
MODEL_DIR = DATA_DIR / "Wan2.2-TI2V-5B"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"
DB_PATH = DATA_DIR / "jobs.db"
WAN_REPO_DIR = DATA_DIR / "Wan2.2"

# Ensure directories exist
INPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Video settings
VIDEO_SIZE = "1280*720"
PRESIGNED_URL_EXPIRY = 3600  # 1 hour
```

**Step 4: Commit**

```bash
git add src/ requirements.txt
git commit -m "feat: add project structure and config"
```

---

## Task 2: Database Module

**Files:**
- Create: `src/database.py`

**Step 1: Create src/database.py**

```python
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from src.config import DB_PATH


def init_db():
    """Initialize the database schema."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'queued',
                prompt TEXT NOT NULL,
                image_path TEXT,
                video_path TEXT,
                s3_key TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        conn.commit()


@contextmanager
def get_connection():
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def create_job(prompt: str, image_path: Optional[str] = None) -> dict:
    """Create a new job and return it."""
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO jobs (id, prompt, image_path, created_at)
               VALUES (?, ?, ?, ?)""",
            (job_id, prompt, image_path, now)
        )
        conn.commit()

    return get_job(job_id)


def get_job(job_id: str) -> Optional[dict]:
    """Get a job by ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return dict(row) if row else None


def get_next_queued_job() -> Optional[dict]:
    """Get the oldest queued job."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT * FROM jobs WHERE status = 'queued'
               ORDER BY created_at ASC LIMIT 1"""
        ).fetchone()
        return dict(row) if row else None


def get_queue_position(job_id: str) -> int:
    """Get position in queue (0 = next, -1 = not queued)."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id FROM jobs WHERE status = 'queued'
               ORDER BY created_at ASC"""
        ).fetchall()
        for i, row in enumerate(rows):
            if row["id"] == job_id:
                return i
        return -1


def update_job_status(
    job_id: str,
    status: str,
    video_path: Optional[str] = None,
    s3_key: Optional[str] = None,
    error: Optional[str] = None
):
    """Update job status and related fields."""
    now = datetime.utcnow().isoformat()

    with get_connection() as conn:
        if status == "processing":
            conn.execute(
                "UPDATE jobs SET status = ?, started_at = ? WHERE id = ?",
                (status, now, job_id)
            )
        elif status == "completed":
            conn.execute(
                """UPDATE jobs SET status = ?, video_path = ?, s3_key = ?,
                   completed_at = ? WHERE id = ?""",
                (status, video_path, s3_key, now, job_id)
            )
        elif status == "failed":
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, completed_at = ? WHERE id = ?",
                (status, error, now, job_id)
            )
        else:
            conn.execute(
                "UPDATE jobs SET status = ? WHERE id = ?", (status, job_id)
            )
        conn.commit()
```

**Step 2: Commit**

```bash
git add src/database.py
git commit -m "feat: add SQLite database module for job tracking"
```

---

## Task 3: S3 Storage Module

**Files:**
- Create: `src/storage.py`

**Step 1: Create src/storage.py**

```python
import boto3
from pathlib import Path

from src.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    S3_BUCKET,
    S3_REGION,
    PRESIGNED_URL_EXPIRY,
)


def get_s3_client():
    """Get an S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=S3_REGION,
    )


def upload_video(local_path: Path, s3_key: str) -> str:
    """Upload a video file to S3 and return the S3 key."""
    client = get_s3_client()
    client.upload_file(
        str(local_path),
        S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": "video/mp4"}
    )
    return s3_key


def get_presigned_url(s3_key: str) -> str:
    """Generate a presigned URL for downloading a video."""
    client = get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=PRESIGNED_URL_EXPIRY,
    )


def download_image(url: str, local_path: Path) -> Path:
    """Download an image from URL to local path."""
    import httpx

    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()

    local_path.write_bytes(response.content)
    return local_path
```

**Step 2: Commit**

```bash
git add src/storage.py
git commit -m "feat: add S3 storage module for video uploads"
```

---

## Task 4: Inference Module (WAN 2.2 CLI Wrapper)

**Files:**
- Create: `src/inference.py`

**Step 1: Create src/inference.py**

```python
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from src.config import MODEL_DIR, WAN_REPO_DIR, OUTPUTS_DIR, VIDEO_SIZE


class InferenceError(Exception):
    """Raised when video generation fails."""
    pass


def generate_video(
    job_id: str,
    prompt: str,
    image_path: Optional[Path] = None,
) -> Path:
    """
    Generate a video using WAN 2.2 TI2V-5B model.

    Returns the path to the generated video file.
    Raises InferenceError if generation fails.
    """
    output_dir = OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "python", str(WAN_REPO_DIR / "generate.py"),
        "--task", "ti2v-5B",
        "--size", VIDEO_SIZE,
        "--ckpt_dir", str(MODEL_DIR),
        "--output_dir", str(output_dir),
        "--offload_model", "True",
        "--convert_model_dtype",
        "--t5_cpu",
        "--prompt", prompt,
    ]

    if image_path:
        cmd.extend(["--image", str(image_path)])

    # Run inference
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            cwd=str(WAN_REPO_DIR),
        )
    except subprocess.TimeoutExpired:
        raise InferenceError("Video generation timed out after 30 minutes")

    if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown error"
        raise InferenceError(f"Generation failed: {error_msg[:500]}")

    # Find the generated video file
    video_files = list(output_dir.glob("*.mp4"))
    if not video_files:
        raise InferenceError("No video file generated")

    return video_files[0]


def cleanup_job_files(job_id: str, input_path: Optional[Path] = None):
    """Clean up temporary files after job completion."""
    output_dir = OUTPUTS_DIR / job_id
    if output_dir.exists():
        shutil.rmtree(output_dir)

    if input_path and input_path.exists():
        input_path.unlink()
```

**Step 2: Commit**

```bash
git add src/inference.py
git commit -m "feat: add inference module wrapping WAN 2.2 CLI"
```

---

## Task 5: Background Worker

**Files:**
- Create: `src/worker.py`

**Step 1: Create src/worker.py**

```python
import threading
import time
import logging
from pathlib import Path
from typing import Optional

from src.database import get_next_queued_job, update_job_status
from src.inference import generate_video, cleanup_job_files, InferenceError
from src.storage import upload_video, get_presigned_url

logger = logging.getLogger(__name__)


class VideoWorker:
    """Background worker that processes video generation jobs."""

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_job_id: Optional[str] = None

    def start(self):
        """Start the worker thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Worker started")

    def stop(self):
        """Stop the worker thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Worker stopped")

    @property
    def current_job_id(self) -> Optional[str]:
        """Get the currently processing job ID."""
        return self._current_job_id

    def _run(self):
        """Main worker loop."""
        while self._running:
            job = get_next_queued_job()

            if job is None:
                time.sleep(2)  # Poll every 2 seconds
                continue

            self._process_job(job)

    def _process_job(self, job: dict):
        """Process a single job."""
        job_id = job["id"]
        self._current_job_id = job_id

        logger.info(f"Processing job {job_id}")
        update_job_status(job_id, "processing")

        input_path = Path(job["image_path"]) if job["image_path"] else None

        try:
            # Generate video
            video_path = generate_video(
                job_id=job_id,
                prompt=job["prompt"],
                image_path=input_path,
            )

            # Upload to S3
            s3_key = f"videos/{job_id}.mp4"
            upload_video(video_path, s3_key)

            # Mark complete
            update_job_status(
                job_id,
                "completed",
                video_path=str(video_path),
                s3_key=s3_key,
            )
            logger.info(f"Job {job_id} completed")

        except InferenceError as e:
            logger.error(f"Job {job_id} failed: {e}")
            update_job_status(job_id, "failed", error=str(e))

        except Exception as e:
            logger.exception(f"Job {job_id} unexpected error")
            update_job_status(job_id, "failed", error=f"Unexpected error: {e}")

        finally:
            self._current_job_id = None
            cleanup_job_files(job_id, input_path)


# Global worker instance
worker = VideoWorker()
```

**Step 2: Commit**

```bash
git add src/worker.py
git commit -m "feat: add background worker for job processing"
```

---

## Task 6: FastAPI Server

**Files:**
- Create: `src/server.py`

**Step 1: Create src/server.py**

```python
import base64
import logging
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

from src.config import API_SECRET_KEY, INPUTS_DIR
from src.database import init_db, create_job, get_job, get_queue_position
from src.storage import download_image, get_presigned_url
from src.worker import worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    init_db()
    worker.start()
    logger.info("Server started")
    yield
    worker.stop()
    logger.info("Server stopped")


app = FastAPI(
    title="WAN 2.2 Video Generation API",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Auth ---

def verify_api_key(x_api_key: str = Header(...)):
    """Verify the API key header."""
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# --- Models ---

class GenerateRequest(BaseModel):
    prompt: str
    image_base64: Optional[str] = None
    image_url: Optional[str] = None


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    position: int


class StatusResponse(BaseModel):
    job_id: str
    status: str
    video_url: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    current_job: Optional[str] = None


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=True,  # Model loads on first inference
        current_job=worker.current_job_id,
    )


@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(verify_api_key)])
async def generate(request: GenerateRequest):
    """Submit a video generation job."""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    image_path = None

    # Handle image input
    if request.image_base64:
        try:
            image_data = base64.b64decode(request.image_base64)
            image_path = INPUTS_DIR / f"{uuid.uuid4()}.jpg"
            image_path.write_bytes(image_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    elif request.image_url:
        try:
            image_path = INPUTS_DIR / f"{uuid.uuid4()}.jpg"
            download_image(request.image_url, image_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")

    # Create job
    job = create_job(
        prompt=request.prompt,
        image_path=str(image_path) if image_path else None,
    )

    position = get_queue_position(job["id"])

    logger.info(f"Created job {job['id']} at position {position}")

    return GenerateResponse(
        job_id=job["id"],
        status=job["status"],
        position=position,
    )


@app.get("/status/{job_id}", response_model=StatusResponse, dependencies=[Depends(verify_api_key)])
async def status(job_id: str):
    """Get the status of a job."""
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    video_url = None
    if job["status"] == "completed" and job["s3_key"]:
        video_url = get_presigned_url(job["s3_key"])

    return StatusResponse(
        job_id=job["id"],
        status=job["status"],
        video_url=video_url,
        error=job["error"],
    )
```

**Step 2: Commit**

```bash
git add src/server.py
git commit -m "feat: add FastAPI server with endpoints"
```

---

## Task 7: Setup Script

**Files:**
- Create: `scripts/setup.sh`

**Step 1: Create scripts/setup.sh**

```bash
#!/bin/bash
set -e

echo "=== WAN 2.2 EC2 Setup ==="

# Configuration
DATA_DIR="${DATA_DIR:-/data}"
MODEL_NAME="Wan-AI/Wan2.2-TI2V-5B"

# Create data directories
echo "Creating directories..."
sudo mkdir -p "$DATA_DIR"
sudo chown -R ubuntu:ubuntu "$DATA_DIR"
mkdir -p "$DATA_DIR/inputs" "$DATA_DIR/outputs"

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Create virtual environment
echo "Creating virtual environment..."
cd "$DATA_DIR"
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "huggingface_hub[cli]"

# Clone WAN 2.2 repository
echo "Cloning WAN 2.2 repository..."
if [ ! -d "$DATA_DIR/Wan2.2" ]; then
    git clone https://github.com/Wan-Video/Wan2.2.git "$DATA_DIR/Wan2.2"
fi

# Install WAN 2.2 dependencies
echo "Installing WAN 2.2 dependencies..."
cd "$DATA_DIR/Wan2.2"
pip install -r requirements.txt

# Download model weights (this takes a while)
echo "Downloading model weights (~20GB)..."
huggingface-cli download "$MODEL_NAME" --local-dir "$DATA_DIR/Wan2.2-TI2V-5B"

# Install API dependencies
echo "Installing API dependencies..."
pip install fastapi uvicorn python-dotenv boto3 httpx pydantic

# Setup systemd service
echo "Setting up systemd service..."
sudo tee /etc/systemd/system/wan-api.service > /dev/null <<EOF
[Unit]
Description=WAN 2.2 Video Generation API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/wan22
Environment=PATH=$DATA_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=DATA_DIR=$DATA_DIR
EnvironmentFile=/home/ubuntu/wan22/.env
ExecStart=$DATA_DIR/venv/bin/uvicorn src.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable wan-api

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Copy your project to /home/ubuntu/wan22"
echo "2. Create .env file with required variables"
echo "3. Run: sudo systemctl start wan-api"
echo "4. Check logs: sudo journalctl -u wan-api -f"
```

**Step 2: Make executable and commit**

```bash
chmod +x scripts/setup.sh
git add scripts/setup.sh
git commit -m "feat: add EC2 setup script"
```

---

## Task 8: Start/Stop Scripts

**Files:**
- Create: `scripts/start.sh`
- Create: `scripts/stop.sh`

**Step 1: Create scripts/start.sh**

```bash
#!/bin/bash
set -e

DATA_DIR="${DATA_DIR:-/data}"

# Activate virtual environment
source "$DATA_DIR/venv/bin/activate"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

export DATA_DIR

echo "Starting WAN 2.2 API server..."
exec uvicorn src.server:app --host 0.0.0.0 --port 8000
```

**Step 2: Create scripts/stop.sh**

```bash
#!/bin/bash

echo "Stopping WAN 2.2 API server..."

# Find and kill uvicorn process
pkill -f "uvicorn src.server:app" || true

echo "Server stopped"
```

**Step 3: Make executable and commit**

```bash
chmod +x scripts/start.sh scripts/stop.sh
git add scripts/start.sh scripts/stop.sh
git commit -m "feat: add start/stop scripts"
```

---

## Task 9: README

**Files:**
- Create: `README.md`

**Step 1: Create README.md**

```markdown
# WAN 2.2 Video Generation API

Self-hosted WAN 2.2 TI2V-5B video generation API for AWS EC2.

## Prerequisites

- AWS EC2 g5.xlarge instance (or any instance with 24GB+ VRAM)
- Ubuntu 22.04 Deep Learning AMI
- S3 bucket for video storage

## Quick Start

### 1. Launch EC2 Instance

Launch a g5.xlarge instance with:
- AMI: Deep Learning AMI GPU PyTorch 2.0+ (Ubuntu 22.04)
- Storage: 100GB EBS
- Security Group: Allow ports 22 (SSH) and 8000 (API)

### 2. Clone and Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<ec2-ip>

# Clone this repo
git clone <your-repo-url> ~/wan22
cd ~/wan22

# Run setup (downloads model, ~30 min)
./scripts/setup.sh
```

### 3. Configure Environment

Create `.env` file:

```bash
cat > .env << 'EOF'
API_SECRET_KEY=your-secret-key-here
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
S3_BUCKET=your-bucket-name
S3_REGION=ap-southeast-1
EOF
```

Generate a secure API key:
```bash
openssl rand -hex 32
```

### 4. Start the Server

```bash
# Using systemd (recommended)
sudo systemctl start wan-api
sudo journalctl -u wan-api -f

# Or manually
./scripts/start.sh
```

## API Usage

### Health Check

```bash
curl http://<ec2-ip>:8000/health
```

### Generate Video (Text-to-Video)

```bash
curl -X POST http://<ec2-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"prompt": "A cat dancing on the moon"}'
```

Response:
```json
{"job_id": "abc-123", "status": "queued", "position": 0}
```

### Generate Video (Image-to-Video)

```bash
# With image URL
curl -X POST http://<ec2-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "prompt": "Make this character dance",
    "image_url": "https://example.com/image.jpg"
  }'

# With base64 image
curl -X POST http://<ec2-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "prompt": "Make this character dance",
    "image_base64": "'$(base64 -i image.jpg)'"
  }'
```

### Check Status

```bash
curl http://<ec2-ip>:8000/status/abc-123 \
  -H "X-API-Key: your-secret-key"
```

Response (when complete):
```json
{
  "job_id": "abc-123",
  "status": "completed",
  "video_url": "https://s3.../presigned-url...",
  "error": null
}
```

## Node.js Client Example

```typescript
const API_URL = 'http://<ec2-ip>:8000';
const API_KEY = 'your-secret-key';

async function generateVideo(prompt: string, imageUrl?: string) {
  // Submit job
  const response = await fetch(`${API_URL}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
    },
    body: JSON.stringify({ prompt, image_url: imageUrl }),
  });

  const { job_id } = await response.json();

  // Poll for completion
  while (true) {
    const statusRes = await fetch(`${API_URL}/status/${job_id}`, {
      headers: { 'X-API-Key': API_KEY },
    });
    const status = await statusRes.json();

    if (status.status === 'completed') {
      return status.video_url;
    }
    if (status.status === 'failed') {
      throw new Error(status.error);
    }

    await new Promise(r => setTimeout(r, 10000)); // Wait 10s
  }
}
```

## Cost Management

- g5.xlarge costs ~$1.00/hour
- Stop instance when not in use: `aws ec2 stop-instances --instance-ids <id>`
- Start when needed: `aws ec2 start-instances --instance-ids <id>`

## Troubleshooting

### Check logs
```bash
sudo journalctl -u wan-api -f
```

### Manual test inference
```bash
cd /data/Wan2.2
source /data/venv/bin/activate
python generate.py --task ti2v-5B --size 1280*720 \
  --ckpt_dir /data/Wan2.2-TI2V-5B \
  --offload_model True --convert_model_dtype --t5_cpu \
  --prompt "A cat dancing"
```

### Out of memory
The A10G has 24GB VRAM. If OOM occurs, the flags `--offload_model True --t5_cpu` should help. Reduce `--size` to `832*480` if still failing.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and usage instructions"
```

---

## Task 10: Final Verification

**Step 1: Verify all files exist**

```bash
ls -la src/
ls -la scripts/
cat requirements.txt
```

Expected structure:
```
src/
├── __init__.py
├── config.py
├── database.py
├── storage.py
├── inference.py
├── worker.py
└── server.py

scripts/
├── setup.sh
├── start.sh
└── stop.sh
```

**Step 2: Final commit with all changes**

```bash
git status
git log --oneline
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Project structure & config |
| 2 | SQLite database module |
| 3 | S3 storage module |
| 4 | WAN 2.2 inference wrapper |
| 5 | Background worker |
| 6 | FastAPI server |
| 7 | EC2 setup script |
| 8 | Start/stop scripts |
| 9 | README documentation |
| 10 | Final verification |
