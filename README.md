# WAN 2.2 Video Generation API

Self-hosted WAN 2.2 14B FP8 distilled video generation API powered by ComfyUI on AWS EC2. Image-to-video only.

## Architecture

- **Model**: WAN 2.2 14B FP8 Distilled (dual high/low-noise models, 4-step denoising, ~15GB each)
- **Audio Model**: MMAudio Large 44k v2 (optional, ~3.9GB)
- **Backend**: ComfyUI (localhost:8188) — handles model loading, inference, and GPU management
- **API**: FastAPI (port 8000) — public REST API that submits workflows to ComfyUI
- **EC2 (Compute)**: g5.2xlarge — NVIDIA A10G 24GB VRAM
- **S3 (Storage)**: S3 bucket for video storage

## Prerequisites

- AWS EC2 g5.2xlarge instance (24GB VRAM A10G GPU)
- Ubuntu 22.04 Deep Learning AMI
- Storage: 150GB EBS (models are ~37GB total)
- S3 bucket for video storage

## Quick Start

### 1. Launch EC2 Instance

Launch a g5.2xlarge instance with:
- AMI: Deep Learning AMI GPU PyTorch 2.0+ (Ubuntu 22.04)
- Storage: 150GB EBS
- Security Group: Allow ports 22 (SSH) and 8000 (API)

### 2. Clone and Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<ec2-ip>

# Clone this repo
git clone <your-repo-url> ~/wan22
cd ~/wan22

# Run setup (downloads ~37GB of models, takes ~15-20 min)
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
S3_REGION=your-s3-region
EOF
```

Generate a secure API key:
```bash
openssl rand -hex 32
```

### 4. Start the Services

```bash
# Start ComfyUI backend first, then the API
sudo systemctl start comfyui
sudo systemctl start wan-api

# Check logs
sudo journalctl -u comfyui -f
sudo journalctl -u wan-api -f
```

## API Usage

See [docs/API.md](docs/API.md) for full API reference.

### Quick Examples

```bash
# Health check
curl http://<ec2-ip>:8000/health

# Image-to-Video
curl -X POST http://<ec2-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "prompt": "Make this character wave hello",
    "image_url": "https://example.com/character.jpg"
  }'

# Image-to-Video with Audio
curl -X POST http://<ec2-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "prompt": "Ocean waves crashing on a rocky shoreline",
    "image_url": "https://example.com/shoreline.jpg",
    "audio_prompt": "Sound of ocean waves crashing, seagulls calling"
  }'

# Check status
curl http://<ec2-ip>:8000/status/<job_id> \
  -H "X-API-Key: your-secret-key"
```

## Video Output

- **Resolution**: 832x480
- **Frame rate**: 16fps
- **Duration**: ~5 seconds (81 frames)
- **Format**: MP4 (H.264 video, AAC audio when `audio_prompt` is provided)
- **Audio**: AI-generated via MMAudio (optional, synced to video content)
- **Estimated generation time**: ~2-3 min (I2V), ~3-4 min (with audio)

## Cost Management

- Stop instance when not in use: `aws ec2 stop-instances --instance-ids <id>`
- Start when needed: `aws ec2 start-instances --instance-ids <id>`

## Troubleshooting

### Check logs
```bash
# ComfyUI backend
sudo journalctl -u comfyui -f

# API server
sudo journalctl -u wan-api -f
```

### ComfyUI not responding
```bash
# Restart ComfyUI
sudo systemctl restart comfyui

# Wait for it to load, then restart API
sleep 10
sudo systemctl restart wan-api
```

### Out of memory
The A10G has 24GB VRAM. The 14B FP8 distilled model at 832x480 uses block swapping to manage VRAM. If OOM occurs:
- Increase block swap values in the workflow (see `src/inference.py`)
- Reduce `VIDEO_FRAME_NUM` in `src/config.py` (e.g., 49 frames instead of 81)
- Ensure no other GPU processes are running: `nvidia-smi`
