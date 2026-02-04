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
