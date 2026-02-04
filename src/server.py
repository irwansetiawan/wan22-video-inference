import base64
import logging
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

from src.config import API_SECRET_KEY, INPUTS_DIR
from src.database import init_db, create_job, get_job, get_queue_position, list_jobs
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


class JobItem(BaseModel):
    job_id: str
    status: str
    prompt: str
    video_url: Optional[str] = None
    error: Optional[str] = None
    created_at: str


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


@app.get("/jobs", response_model=list[JobItem], dependencies=[Depends(verify_api_key)])
async def list_all_jobs(status: Optional[str] = None, limit: int = 100):
    """List all jobs with optional status filter."""
    jobs = list_jobs(status=status, limit=limit)

    result = []
    for job in jobs:
        video_url = None
        if job["status"] == "completed" and job["s3_key"]:
            video_url = get_presigned_url(job["s3_key"])

        result.append(JobItem(
            job_id=job["id"],
            status=job["status"],
            prompt=job["prompt"],
            video_url=video_url,
            error=job["error"],
            created_at=job["created_at"],
        ))

    return result
