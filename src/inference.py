import json
import shutil
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
    VIDEO_SHIFT,
    T2V_HIGH_NOISE_MODEL,
    T2V_LOW_NOISE_MODEL,
    I2V_HIGH_NOISE_MODEL,
    I2V_LOW_NOISE_MODEL,
    TEXT_ENCODER_MODEL,
    VAE_MODEL,
    CLIP_VISION_MODEL,
)

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Raised when video generation fails."""
    pass


def _build_t2v_workflow(prompt: str, seed: int) -> dict:
    """Build a WAN 2.2 14B FP8 text-to-video workflow for ComfyUI."""
    return {
        "10": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": T2V_HIGH_NOISE_MODEL,
                "weight_dtype": "default",
            },
        },
        "11": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": T2V_LOW_NOISE_MODEL,
                "weight_dtype": "default",
            },
        },
        "38": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER_MODEL,
                "type": "wan",
            },
        },
        "39": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": VAE_MODEL,
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["38", 0],
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "worst quality, blurry, distorted, low quality, static, watermark",
                "clip": ["38", 0],
            },
        },
        "40": {
            "class_type": "EmptyHunyuanLatentVideo",
            "inputs": {
                "width": VIDEO_WIDTH,
                "height": VIDEO_HEIGHT,
                "length": VIDEO_FRAME_NUM,
                "batch_size": 1,
            },
        },
        "48": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["10", 0],
                "shift": VIDEO_SHIFT,
            },
        },
        "49": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["11", 0],
                "shift": VIDEO_SHIFT,
            },
        },
        "50": {
            "class_type": "WanDualModelMerge",
            "inputs": {
                "high_noise_model": ["48", 0],
                "low_noise_model": ["49", 0],
                "threshold": 0.5,
            },
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": VIDEO_STEPS,
                "cfg": VIDEO_CFG,
                "sampler_name": "uni_pc",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["50", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["40", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["39", 0],
            },
        },
        "60": {
            "class_type": "SaveAnimatedWEBP",
            "inputs": {
                "filename_prefix": "wan22_video",
                "fps": 16,
                "lossless": False,
                "quality": 90,
                "method": "default",
                "images": ["8", 0],
            },
        },
    }


def _build_i2v_workflow(prompt: str, image_path: Path, seed: int) -> dict:
    """Build a WAN 2.2 14B FP8 image-to-video workflow for ComfyUI."""
    return {
        "10": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": I2V_HIGH_NOISE_MODEL,
                "weight_dtype": "default",
            },
        },
        "11": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": I2V_LOW_NOISE_MODEL,
                "weight_dtype": "default",
            },
        },
        "38": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER_MODEL,
                "type": "wan",
            },
        },
        "39": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": VAE_MODEL,
            },
        },
        "42": {
            "class_type": "CLIPVisionLoader",
            "inputs": {
                "clip_name": CLIP_VISION_MODEL,
            },
        },
        "20": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_path.name,
            },
        },
        "43": {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["42", 0],
                "image": ["20", 0],
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["38", 0],
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "worst quality, blurry, distorted, low quality, static, watermark",
                "clip": ["38", 0],
            },
        },
        "44": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "width": VIDEO_WIDTH,
                "height": VIDEO_HEIGHT,
                "length": VIDEO_FRAME_NUM,
                "batch_size": 1,
                "clip_vision_output": ["43", 0],
                "start_image": ["20", 0],
                "vae": ["39", 0],
            },
        },
        "48": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["10", 0],
                "shift": VIDEO_SHIFT,
            },
        },
        "49": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["11", 0],
                "shift": VIDEO_SHIFT,
            },
        },
        "50": {
            "class_type": "WanDualModelMerge",
            "inputs": {
                "high_noise_model": ["48", 0],
                "low_noise_model": ["49", 0],
                "threshold": 0.5,
            },
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": VIDEO_STEPS,
                "cfg": VIDEO_CFG,
                "sampler_name": "uni_pc",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["50", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["44", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["39", 0],
            },
        },
        "60": {
            "class_type": "SaveAnimatedWEBP",
            "inputs": {
                "filename_prefix": "wan22_video",
                "fps": 16,
                "lossless": False,
                "quality": 90,
                "method": "default",
                "images": ["8", 0],
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


def _download_output(history_entry: dict, output_dir: Path) -> Path:
    """Download the generated video from ComfyUI output."""
    outputs = history_entry.get("outputs", {})
    for node_id, node_output in outputs.items():
        for key in ["images", "gifs", "videos"]:
            if key not in node_output:
                continue
            for item in node_output[key]:
                filename = item["filename"]
                if filename.endswith((".mp4", ".webp", ".gif")):
                    params = urllib.parse.urlencode({
                        "filename": filename,
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                    })
                    url = f"{COMFYUI_URL}/view?{params}"
                    out_path = output_dir / filename
                    try:
                        urllib.request.urlretrieve(url, str(out_path))
                    except Exception as e:
                        raise InferenceError(f"Failed to download output: {e}")
                    return out_path

    raise InferenceError("No video file found in ComfyUI output")


def _copy_image_to_comfyui(image_path: Path) -> None:
    """Copy input image to ComfyUI's input directory for LoadImage node."""
    comfyui_input = COMFYUI_DIR / "input"
    comfyui_input.mkdir(parents=True, exist_ok=True)
    dest = comfyui_input / image_path.name
    shutil.copy2(str(image_path), str(dest))


def generate_video(
    job_id: str,
    prompt: str,
    image_path: Optional[Path] = None,
) -> Path:
    """
    Generate a video using ComfyUI with WAN 2.2 14B FP8 models.

    Returns the path to the generated video file.
    Raises InferenceError if generation fails.
    """
    output_dir = OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = hash(job_id) % (2**32)

    if image_path:
        _copy_image_to_comfyui(image_path)
        workflow = _build_i2v_workflow(prompt, image_path, seed)
    else:
        workflow = _build_t2v_workflow(prompt, seed)

    logger.info(f"Submitting job {job_id} to ComfyUI")
    prompt_id = _submit_prompt(workflow)
    logger.info(f"Job {job_id} submitted as ComfyUI prompt {prompt_id}")

    history = _poll_completion(prompt_id)
    logger.info(f"Job {job_id} ComfyUI execution complete, downloading output")

    video_path = _download_output(history, output_dir)
    logger.info(f"Job {job_id} video saved to {video_path}")

    return video_path


def cleanup_job_files(job_id: str, input_path: Optional[Path] = None):
    """Clean up temporary files after job completion."""
    output_dir = OUTPUTS_DIR / job_id
    if output_dir.exists():
        shutil.rmtree(output_dir)

    if input_path and input_path.exists():
        # Also remove the copy in ComfyUI's input directory
        comfyui_copy = COMFYUI_DIR / "input" / input_path.name
        if comfyui_copy.exists():
            comfyui_copy.unlink()
        input_path.unlink()
