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
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 704
VIDEO_FRAME_NUM = 121  # Must be 4n+1. 121 frames at 24fps → ~5.04s
VIDEO_STEPS = 30
VIDEO_CFG = 5.0
VIDEO_SHIFT = 8.0
VIDEO_FPS = 24
PRESIGNED_URL_EXPIRY = 3600  # 1 hour

# MMAudio settings
MMAUDIO_MODEL = "mmaudio_large_44k_v2_fp16.safetensors"
MMAUDIO_VAE = "mmaudio_vae_44k_fp16.safetensors"
MMAUDIO_SYNCHFORMER = "mmaudio_synchformer_fp16.safetensors"
MMAUDIO_CLIP = "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors"
MMAUDIO_STEPS = 25
MMAUDIO_CFG = 4.5

# WAN 2.2 5B FP16 model filenames (Comfy-Org repackaged, unified TI2V)
TI2V_MODEL = "wan2.2_ti2v_5B_fp16.safetensors"
TEXT_ENCODER_MODEL = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_MODEL = "wan2.2_vae.safetensors"
