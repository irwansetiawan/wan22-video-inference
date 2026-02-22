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

        # Migrate: add audio_prompt column if it doesn't exist
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN audio_prompt TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists


@contextmanager
def get_connection():
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def create_job(prompt: str, image_path: Optional[str] = None, audio_prompt: Optional[str] = None) -> dict:
    """Create a new job and return it."""
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO jobs (id, prompt, image_path, audio_prompt, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (job_id, prompt, image_path, audio_prompt, now)
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


def list_jobs(status: Optional[str] = None, limit: int = 100) -> list[dict]:
    """List jobs with optional status filter."""
    with get_connection() as conn:
        if status:
            rows = conn.execute(
                """SELECT * FROM jobs WHERE status = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (status, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM jobs
                   ORDER BY created_at DESC LIMIT ?""",
                (limit,)
            ).fetchall()
        return [dict(row) for row in rows]


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
