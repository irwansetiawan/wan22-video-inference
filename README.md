# WAN 2.2 Video Generation API

Self-hosted WAN 2.2 TI2V-5B video generation API for AWS EC2.

## Prerequisites

- AWS EC2 g5.xlarge instance (or any instance with 24GB+ VRAM)
- Ubuntu 22.04 Deep Learning AMI
- S3 bucket for video storage

## Quick Start

### 1. Launch EC2 Instance

Launch a g5.xlarge instance with:
- AMI: Deep Learning AMI GPU PyTorch 2.0+ (Ubuntu 22.04)
- Storage: 100GB EBS
- Security Group: Allow ports 22 (SSH) and 8000 (API)

### 2. Clone and Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<ec2-ip>

# Clone this repo
git clone <your-repo-url> ~/wan22
cd ~/wan22

# Run setup (downloads model, ~30 min)
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
S3_REGION=ap-southeast-1
EOF
```

Generate a secure API key:
```bash
openssl rand -hex 32
```

### 4. Start the Server

```bash
# Using systemd (recommended)
sudo systemctl start wan-api
sudo journalctl -u wan-api -f

# Or manually
./scripts/start.sh
```

## API Usage

### Health Check

```bash
curl http://<ec2-ip>:8000/health
```

### Generate Video (Text-to-Video)

```bash
curl -X POST http://<ec2-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"prompt": "A cat dancing on the moon"}'
```

Response:
```json
{"job_id": "abc-123", "status": "queued", "position": 0}
```

### Generate Video (Image-to-Video)

```bash
# With image URL
curl -X POST http://<ec2-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "prompt": "Make this character dance",
    "image_url": "https://example.com/image.jpg"
  }'

# With base64 image
curl -X POST http://<ec2-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "prompt": "Make this character dance",
    "image_base64": "'$(base64 -i image.jpg)'"
  }'
```

### Check Status

```bash
curl http://<ec2-ip>:8000/status/abc-123 \
  -H "X-API-Key: your-secret-key"
```

Response (when complete):
```json
{
  "job_id": "abc-123",
  "status": "completed",
  "video_url": "https://s3.../presigned-url...",
  "error": null
}
```

## Node.js Client Example

```typescript
const API_URL = 'http://<ec2-ip>:8000';
const API_KEY = 'your-secret-key';

async function generateVideo(prompt: string, imageUrl?: string) {
  // Submit job
  const response = await fetch(`${API_URL}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
    },
    body: JSON.stringify({ prompt, image_url: imageUrl }),
  });

  const { job_id } = await response.json();

  // Poll for completion
  while (true) {
    const statusRes = await fetch(`${API_URL}/status/${job_id}`, {
      headers: { 'X-API-Key': API_KEY },
    });
    const status = await statusRes.json();

    if (status.status === 'completed') {
      return status.video_url;
    }
    if (status.status === 'failed') {
      throw new Error(status.error);
    }

    await new Promise(r => setTimeout(r, 10000)); // Wait 10s
  }
}
```

## Cost Management

- g5.xlarge costs ~$1.00/hour
- Stop instance when not in use: `aws ec2 stop-instances --instance-ids <id>`
- Start when needed: `aws ec2 start-instances --instance-ids <id>`

## Troubleshooting

### Check logs
```bash
sudo journalctl -u wan-api -f
```

### Manual test inference
```bash
cd /data/Wan2.2
source /data/venv/bin/activate
python generate.py --task ti2v-5B --size 1280*720 \
  --ckpt_dir /data/Wan2.2-TI2V-5B \
  --offload_model True --convert_model_dtype --t5_cpu \
  --prompt "A cat dancing"
```

### Out of memory
The A10G has 24GB VRAM. If OOM occurs, the flags `--offload_model True --t5_cpu` should help. Reduce `--size` to `832*480` if still failing.
