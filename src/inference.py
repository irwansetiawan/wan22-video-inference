import json
import shutil
import subprocess
import time
import logging
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional

from src.config import (
    COMFYUI_DIR,
    COMFYUI_URL,
    OUTPUTS_DIR,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_FRAME_NUM,
    VIDEO_STEPS,
    VIDEO_CFG,
    VIDEO_SPLIT_STEP,
    VIDEO_FPS,
    VIDEO_SIGMAS,
    I2V_HIGH_NOISE_MODEL,
    I2V_LOW_NOISE_MODEL,
    TEXT_ENCODER_MODEL,
    VAE_MODEL,
    MMAUDIO_MODEL,
    MMAUDIO_VAE,
    MMAUDIO_SYNCHFORMER,
    MMAUDIO_CLIP,
    MMAUDIO_STEPS,
    MMAUDIO_CFG,
)

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Raised when video generation fails."""
    pass


def _build_i2v_workflow(prompt: str, image_path: Path, seed: int) -> dict:
    """Build a WAN 2.2 14B FP8 distilled image-to-video workflow for ComfyUI.

    Uses dual high-noise / low-noise models with 4-step denoising via
    WanVideoWrapper (Kijai) custom nodes.
    """
    return {
        # --- Text encoding ---
        "11": {
            "class_type": "LoadWanVideoT5TextEncoder",
            "inputs": {
                "t5_model": TEXT_ENCODER_MODEL,
                "precision": "fp8_e4m3fn",
            },
        },
        "16": {
            "class_type": "WanVideoTextEncode",
            "inputs": {
                "prompt": prompt,
                "t5_encoder": ["11", 0],
                "force_offload": True,
            },
        },
        # --- Model loading (dual high/low noise) ---
        "22": {
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "model": I2V_HIGH_NOISE_MODEL,
                "base_precision": "fp8_e4m3fn",
            },
        },
        "71": {
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "model": I2V_LOW_NOISE_MODEL,
                "base_precision": "fp8_e4m3fn",
            },
        },
        # --- VAE ---
        "109": {
            "class_type": "WanVideoVAELoader",
            "inputs": {
                "vae_name": VAE_MODEL,
            },
        },
        # --- Block swap (memory management) ---
        "39": {
            "class_type": "WanVideoBlockSwap",
            "inputs": {
                "double_blocks_to_swap": 0,
                "single_blocks_to_swap": 0,
            },
        },
        "92": {
            "class_type": "WanVideoSetBlockSwap",
            "inputs": {
                "wan_model": ["22", 0],
                "block_swap": ["39", 0],
            },
        },
        "93": {
            "class_type": "WanVideoSetBlockSwap",
            "inputs": {
                "wan_model": ["71", 0],
                "block_swap": ["39", 0],
            },
        },
        # --- Image input ---
        "106": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_path.name,
            },
        },
        "108": {
            "class_type": "ImageResizeKJv2",
            "inputs": {
                "width": VIDEO_WIDTH,
                "height": VIDEO_HEIGHT,
                "interpolation": "lanczos",
                "condition": "always",
                "multiple_of": 16,
                "round_method": "round",
                "image": ["106", 0],
            },
        },
        # --- Image-to-video encode ---
        "110": {
            "class_type": "WanVideoImageToVideoEncode",
            "inputs": {
                "vae": ["109", 0],
                "image": ["108", 0],
                "width": VIDEO_WIDTH,
                "height": VIDEO_HEIGHT,
                "length": VIDEO_FRAME_NUM,
            },
        },
        # --- Sigma schedule ---
        "100": {
            "class_type": "StringToFloatList",
            "inputs": {
                "float_string": VIDEO_SIGMAS,
            },
        },
        "99": {
            "class_type": "FloatToSigmas",
            "inputs": {
                "float_list": ["100", 0],
            },
        },
        # --- Sampling (high-noise, steps 0→split) ---
        "27": {
            "class_type": "WanVideoSampler",
            "inputs": {
                "seed": seed,
                "steps": VIDEO_STEPS,
                "cfg": VIDEO_CFG,
                "start_step": 0,
                "end_step": VIDEO_SPLIT_STEP,
                "model": ["92", 0],
                "positive": ["16", 0],
                "negative": ["16", 0],
                "latent_image": ["110", 0],
                "image_embeds": ["110", 1],
                "sigmas": ["99", 0],
                "force_offload": True,
            },
        },
        # --- Sampling (low-noise, steps split→end) ---
        "90": {
            "class_type": "WanVideoSampler",
            "inputs": {
                "seed": seed,
                "steps": VIDEO_STEPS,
                "cfg": VIDEO_CFG,
                "start_step": VIDEO_SPLIT_STEP,
                "end_step": VIDEO_STEPS,
                "model": ["93", 0],
                "positive": ["16", 0],
                "negative": ["16", 0],
                "latent_image": ["27", 0],
                "image_embeds": ["110", 1],
                "sigmas": ["99", 0],
                "force_offload": True,
            },
        },
        # --- Decode ---
        "28": {
            "class_type": "WanVideoDecode",
            "inputs": {
                "samples": ["90", 0],
                "vae": ["109", 0],
            },
        },
        # --- Output ---
        "60": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["28", 0],
                "frame_rate": float(VIDEO_FPS),
                "loop_count": 0,
                "filename_prefix": "wan22_video",
                "format": "video/h264-mp4",
                "save_output": True,
            },
        },
    }


def _build_audio_nodes(audio_prompt: str, seed: int) -> dict:
    """Build MMAudio nodes to append to a video workflow.

    Expects node "28" (WanVideoDecode) to exist in the parent workflow,
    providing decoded video frames as input to MMAudioSampler.
    """
    return {
        "80": {
            "class_type": "MMAudioModelLoader",
            "inputs": {
                "mmaudio_model": MMAUDIO_MODEL,
                "base_precision": "fp16",
            },
        },
        "81": {
            "class_type": "MMAudioFeatureUtilsLoader",
            "inputs": {
                "vae_model": MMAUDIO_VAE,
                "synchformer_model": MMAUDIO_SYNCHFORMER,
                "clip_model": MMAUDIO_CLIP,
                "mode": "44k",
                "precision": "fp16",
            },
        },
        "82": {
            "class_type": "MMAudioSampler",
            "inputs": {
                "mmaudio_model": ["80", 0],
                "feature_utils": ["81", 0],
                "images": ["28", 0],
                "prompt": audio_prompt,
                "negative_prompt": "",
                "seed": seed,
                "steps": MMAUDIO_STEPS,
                "cfg": MMAUDIO_CFG,
                "duration": round(VIDEO_FRAME_NUM / VIDEO_FPS, 2),
                "mask_away_clip": False,
                "force_offload": True,
            },
        },
        "83": {
            "class_type": "SaveAudio",
            "inputs": {
                "filename_prefix": "wan22_audio",
                "audio": ["82", 0],
            },
        },
    }


def _submit_prompt(workflow: dict) -> str:
    """Submit a workflow to ComfyUI and return the prompt_id."""
    payload = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:2000]
        raise InferenceError(f"ComfyUI rejected workflow ({e.code}): {body}")
    except Exception as e:
        raise InferenceError(f"Failed to submit to ComfyUI: {e}")

    if "error" in result:
        raise InferenceError(f"ComfyUI rejected workflow: {result['error']}")

    return result["prompt_id"]


def _poll_completion(prompt_id: str, timeout: int = 2700) -> dict:
    """Poll ComfyUI /history until the job completes or times out."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(
                f"{COMFYUI_URL}/history/{prompt_id}", timeout=10
            ) as resp:
                history = json.loads(resp.read())
        except Exception:
            time.sleep(5)
            continue

        if prompt_id in history:
            entry = history[prompt_id]
            if "status" in entry and entry["status"].get("status_str") == "error":
                messages = entry["status"].get("messages", [])
                error_msg = str(messages)[:2000] if messages else "Unknown ComfyUI error"
                raise InferenceError(f"ComfyUI execution failed: {error_msg}")
            return entry

        time.sleep(5)

    raise InferenceError(f"Video generation timed out after {timeout // 60} minutes")


def _download_comfyui_file(item: dict, output_dir: Path) -> Path:
    """Download a single file from ComfyUI output."""
    filename = item["filename"]
    params = urllib.parse.urlencode({
        "filename": filename,
        "subfolder": item.get("subfolder", ""),
        "type": item.get("type", "output"),
    })
    url = f"{COMFYUI_URL}/view?{params}"
    out_path = output_dir / filename
    urllib.request.urlretrieve(url, str(out_path))
    return out_path


def _download_output(
    history_entry: dict, output_dir: Path, has_audio: bool = False
) -> tuple[Path, "Path | None"]:
    """Download generated video (and optionally audio) from ComfyUI output."""
    outputs = history_entry.get("outputs", {})
    video_path = None
    audio_path = None

    for node_id, node_output in outputs.items():
        # Look for video files
        for key in ["images", "gifs", "videos"]:
            if key not in node_output:
                continue
            for item in node_output[key]:
                if item["filename"].endswith((".mp4", ".webp", ".gif")):
                    try:
                        video_path = _download_comfyui_file(item, output_dir)
                    except Exception as e:
                        raise InferenceError(f"Failed to download video: {e}")

        # Look for audio files
        if has_audio and "audio" in node_output:
            for item in node_output["audio"]:
                if item["filename"].endswith((".flac", ".wav", ".mp3", ".ogg", ".opus")):
                    try:
                        audio_path = _download_comfyui_file(item, output_dir)
                    except Exception as e:
                        raise InferenceError(f"Failed to download audio: {e}")

    if video_path is None:
        raise InferenceError("No video file found in ComfyUI output")

    if has_audio and audio_path is None:
        logger.warning("Audio was requested but no audio file found in ComfyUI output")

    return video_path, audio_path


def _combine_audio_video(video_path: Path, audio_path: Path, output_dir: Path) -> Path:
    """Mux audio into video using ffmpeg."""
    combined_path = output_dir / f"wan22_combined_{video_path.stem}.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(combined_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise InferenceError(f"ffmpeg muxing failed: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        raise InferenceError("ffmpeg muxing timed out after 60 seconds")
    except FileNotFoundError:
        raise InferenceError("ffmpeg not found on system")

    return combined_path


def _copy_image_to_comfyui(image_path: Path) -> None:
    """Copy input image to ComfyUI's input directory for LoadImage node."""
    comfyui_input = COMFYUI_DIR / "input"
    comfyui_input.mkdir(parents=True, exist_ok=True)
    dest = comfyui_input / image_path.name
    shutil.copy2(str(image_path), str(dest))


def generate_video(
    job_id: str,
    prompt: str,
    image_path: Path,
    audio_prompt: Optional[str] = None,
) -> Path:
    """
    Generate a video using ComfyUI with WAN 2.2 14B FP8 distilled model (I2V).
    Optionally generates synchronized audio via MMAudio.

    Returns the path to the generated video file (with audio if requested).
    Raises InferenceError if generation fails.
    """
    output_dir = OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = hash(job_id) % (2**32)

    _copy_image_to_comfyui(image_path)
    workflow = _build_i2v_workflow(prompt, image_path, seed)

    has_audio = bool(audio_prompt)
    if has_audio:
        workflow.update(_build_audio_nodes(audio_prompt, seed))

    logger.info(f"Submitting job {job_id} to ComfyUI (audio={'yes' if has_audio else 'no'})")
    prompt_id = _submit_prompt(workflow)
    logger.info(f"Job {job_id} submitted as ComfyUI prompt {prompt_id}")

    history = _poll_completion(prompt_id)
    logger.info(f"Job {job_id} ComfyUI execution complete, downloading output")

    video_path, audio_path = _download_output(history, output_dir, has_audio=has_audio)
    logger.info(f"Job {job_id} video saved to {video_path}")

    if audio_path:
        logger.info(f"Job {job_id} combining audio with video")
        combined_path = _combine_audio_video(video_path, audio_path, output_dir)
        logger.info(f"Job {job_id} combined video saved to {combined_path}")
        return combined_path

    return video_path


def cleanup_job_files(job_id: str, input_path: Optional[Path] = None):
    """Clean up temporary files after job completion."""
    output_dir = OUTPUTS_DIR / job_id
    if output_dir.exists():
        shutil.rmtree(output_dir)

    if input_path and input_path.exists():
        comfyui_copy = COMFYUI_DIR / "input" / input_path.name
        if comfyui_copy.exists():
            comfyui_copy.unlink()
        input_path.unlink()
