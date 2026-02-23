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

# I2V 14B FP8 Distilled model filenames
I2V_HIGH_NOISE_MODEL = "wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors"
I2V_LOW_NOISE_MODEL = "wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors"
TEXT_ENCODER_MODEL = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_MODEL = "wan2.2_vae.safetensors"

# Video settings (distilled defaults)
VIDEO_WIDTH = 832
VIDEO_HEIGHT = 480
VIDEO_FRAME_NUM = 81  # Must be 4n+1. 81 frames at 16fps → ~5.06s
VIDEO_STEPS = 4
VIDEO_CFG = 1.0
VIDEO_SPLIT_STEP = 2
VIDEO_FPS = 16
VIDEO_SIGMAS = "1.0, 0.9375001, 0.8333333, 0.625, 0.0000"
PRESIGNED_URL_EXPIRY = 3600  # 1 hour

# MMAudio settings
MMAUDIO_MODEL = "mmaudio_large_44k_v2_fp16.safetensors"
MMAUDIO_VAE = "mmaudio_vae_44k_fp16.safetensors"
MMAUDIO_SYNCHFORMER = "mmaudio_synchformer_fp16.safetensors"
MMAUDIO_CLIP = "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors"
MMAUDIO_STEPS = 25
MMAUDIO_CFG = 4.5
