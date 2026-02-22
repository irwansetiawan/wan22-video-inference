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
                audio_prompt=job.get("audio_prompt"),
            )

            # Upload to S3
            ext = video_path.suffix  # .webp from ComfyUI
            s3_key = f"videos/{job_id}{ext}"
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
