# WAN 2.2 Video Generation API

Self-hosted WAN 2.2 14B FP8 video generation API powered by ComfyUI on AWS EC2.

## Architecture

- **Model**: WAN 2.2 14B FP8 (dual expert: high-noise + low-noise)
- **Backend**: ComfyUI (localhost:8188) — handles model loading, inference, and GPU management
- **API**: FastAPI (port 8000) — public REST API that submits workflows to ComfyUI
- **EC2 (Compute)**: g5.xlarge in **Tokyo (ap-northeast-1)** — NVIDIA A10G 24GB VRAM, ~$1.46/hr
- **S3 (Storage)**: Bucket in **Singapore (ap-southeast-1)** — close to end users for fast video downloads

## Prerequisites

- AWS EC2 g5.xlarge instance (24GB VRAM A10G GPU)
- Ubuntu 22.04 Deep Learning AMI
- Storage: 150GB EBS (models are ~60GB total)
- S3 bucket in ap-southeast-1 for video storage

## Quick Start

### 1. Launch EC2 Instance

Launch a g5.xlarge instance in **ap-northeast-1 (Tokyo)** with:
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

# Run setup (downloads ~60GB of models, takes ~20-30 min)
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
S3_REGION=ap-southeast-1  # S3 in Singapore, close to end users
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

# Text-to-Video
curl -X POST http://<ec2-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"prompt": "A cat dancing on the moon"}'

# Check status
curl http://<ec2-ip>:8000/status/<job_id> \
  -H "X-API-Key: your-secret-key"
```

## Video Output

- **Resolution**: 832x480 (480p)
- **Frames**: 81 (~5 seconds at 16fps)
- **Format**: WEBP (animated)
- **Estimated generation time**: ~6-10 minutes per video

## Cost Management

- g5.xlarge in Tokyo costs ~$1.46/hour
- Stop instance when not in use: `aws ec2 stop-instances --instance-ids <id> --region ap-northeast-1`
- Start when needed: `aws ec2 start-instances --instance-ids <id> --region ap-northeast-1`
- Running 4 hours/day ≈ $175/month

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
The A10G has 24GB VRAM. WAN 2.2 14B FP8 at 480p typically uses ~18-22GB VRAM. If OOM occurs:
- Reduce `VIDEO_FRAME_NUM` in `src/config.py` (e.g., 41 frames instead of 81)
- Ensure no other GPU processes are running: `nvidia-smi`
