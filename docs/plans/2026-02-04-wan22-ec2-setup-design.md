# WAN 2.2 EC2 Video Generation API - Design Document

## Overview

Self-hosted WAN 2.2 TI2V-5B (Text+Image to Video) on AWS EC2 g5.xlarge, exposed via FastAPI for local development use.

## Architecture

```
┌─────────────────┐         ┌─────────────────────────────────┐
│  localhost      │         │      EC2 g5.xlarge              │
│  (Node.js)      │  HTTPS  │                                 │
│                 │────────▶│  FastAPI + WAN 2.2              │
│                 │         │                                 │
└─────────────────┘         └─────────────────────────────────┘
                                          │
                                          ▼
                                   ┌─────────────┐
                                   │  S3 bucket  │
                                   └─────────────┘
```

## Components

### EC2 Instance
- **Type**: g5.xlarge (1x NVIDIA A10G, 24GB VRAM)
- **AMI**: Ubuntu 22.04 Deep Learning AMI (CUDA pre-installed)
- **Storage**: 100GB EBS (model weights ~20GB + working space)
- **Security group**: port 22 (SSH), port 8000 (API)
- **Region**: ap-southeast-1 (Singapore)

### Application
- Single Python process running FastAPI on port 8000
- WAN 2.2 model loaded in GPU memory
- SQLite database for job tracking
- Background thread processing job queue (one job at a time due to VRAM constraints)

### S3 Bucket
- Stores generated videos
- Returns presigned URLs (valid 1 hour)

## API Design

### Authentication
All requests require `X-API-Key` header matching `API_SECRET_KEY` env var.

### Endpoints

```
GET  /health
  → { "status": "ok", "gpu": "A10G", "model_loaded": true }

POST /generate
  Headers: X-API-Key: xxx
  Body: {
    "prompt": "A cat dancing on the moon",
    "image_base64": "...",      // optional, for image-to-video
    "image_url": "https://..."  // optional, alternative to base64
  }
  → { "job_id": "abc123", "status": "queued", "position": 2 }

GET  /status/{job_id}
  Headers: X-API-Key: xxx
  → {
      "job_id": "abc123",
      "status": "completed",  // queued | processing | completed | failed
      "video_url": "https://s3...presigned...",
      "error": null
    }
```

### Job Lifecycle
1. `queued` - Waiting in line
2. `processing` - GPU working on it
3. `completed` - Video ready, URL provided
4. `failed` - Error occurred, message in `error` field

## File Structure

```
wan22/
├── scripts/
│   ├── setup.sh          # One-time EC2 setup (install deps, download model)
│   ├── start.sh          # Start the API server
│   └── stop.sh           # Graceful shutdown
├── src/
│   ├── server.py         # FastAPI app, endpoints, auth
│   ├── worker.py         # Background job processor
│   ├── inference.py      # WAN 2.2 wrapper (runs the model)
│   ├── storage.py        # S3 upload + presigned URL generation
│   ├── database.py       # SQLite job CRUD
│   └── config.py         # Environment variables, constants
├── requirements.txt      # Python dependencies
└── README.md             # Setup + usage instructions
```

### Data Directory (on EC2)
```
/data/
├── jobs.db               # SQLite database
├── Wan2.2-TI2V-5B/       # Model weights (downloaded during setup)
├── inputs/               # Temp storage for input images
└── outputs/              # Temp storage before S3 upload
```

## Environment Variables

```
API_SECRET_KEY=<random-string-you-generate>
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
S3_BUCKET=your-video-bucket
S3_REGION=ap-southeast-1
```

## Operations

### Initial Setup (~30 min)
1. Launch g5.xlarge via AWS Console with Deep Learning AMI
2. SSH in, clone this repo
3. Run `./scripts/setup.sh` - installs deps, downloads 20GB model weights
4. Set environment variables in `/data/.env`
5. Run `./scripts/start.sh`

### Daily Usage
- Start instance via AWS Console or CLI
- API auto-starts via systemd service
- Stop instance when done to save credits

## Cost Estimate

- g5.xlarge: ~$1.00/hour
- Running 4 hours/day: ~$120/month
- $500 credits: ~500 hours total runtime or ~4 months at 4hr/day

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Concurrency | Single job at a time | A10G has 24GB VRAM, model needs ~20-24GB |
| Job persistence | SQLite | Simple, no extra services, good enough for single instance |
| Authentication | API key header | Simple, localhost IP may change |
| Infrastructure | Manual EC2 | No Terraform/Docker needed for single instance |
| Queue | In-process with SQLite | No Redis needed, jobs survive restarts via DB |

## Future Improvements (not in scope)

- Terraform for reproducible infrastructure
- Docker for easier deployment
- Multi-instance scaling with shared SQS queue
- Webhook callbacks instead of polling
