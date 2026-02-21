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
COMFYUI_DIR = DATA_DIR / "ComfyUI"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"
DB_PATH = DATA_DIR / "jobs.db"

# Ensure directories exist
INPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ComfyUI backend
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")

# Video settings
VIDEO_WIDTH = 832
VIDEO_HEIGHT = 480
VIDEO_FRAME_NUM = 81  # Must be 4n+1. 81 frames = ~5s at 16fps
VIDEO_STEPS = 30
VIDEO_CFG = 6.0
VIDEO_SHIFT = 8.0
PRESIGNED_URL_EXPIRY = 3600  # 1 hour

# WAN 2.2 14B FP8 model filenames (Comfy-Org repackaged)
T2V_HIGH_NOISE_MODEL = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
T2V_LOW_NOISE_MODEL = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
I2V_HIGH_NOISE_MODEL = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
I2V_LOW_NOISE_MODEL = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
TEXT_ENCODER_MODEL = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_MODEL = "wan_2.1_vae.safetensors"
CLIP_VISION_MODEL = "clip_vision_h.safetensors"
