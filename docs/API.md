# WAN 2.2 Video Generation API

REST API for generating videos using the WAN 2.2 14B model (FP8 quantized, powered by ComfyUI). Supports both text-to-video and image-to-video generation.

## Base URL

```
http://<your-server-ip>:8000
```

## Authentication

All endpoints (except `/health`) require an API key passed via the `X-API-Key` header.

```
X-API-Key: <your-api-key>
```

Requests with a missing or invalid key receive a `401 Unauthorized` response.

## Endpoints

### Health Check

Check if the server is running and the model is loaded.

```
GET /health
```

**Response**

```json
{
  "status": "ok",
  "model_loaded": true,
  "current_job": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"ok"` if the server is running |
| `model_loaded` | boolean | Whether the model is ready for inference |
| `current_job` | string \| null | Job ID currently being processed, or `null` if idle |

---

### Generate Video

Submit a video generation job. Jobs are queued and processed one at a time.

```
POST /generate
```

**Headers**

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | `application/json` |
| `X-API-Key` | Yes | Your API key |

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | Text description of the video to generate |
| `image_url` | string | No | URL of a reference image for image-to-video generation |
| `image_base64` | string | No | Base64-encoded reference image for image-to-video generation |

> **Note:** If both `image_url` and `image_base64` are provided, `image_base64` takes precedence. If neither is provided, the API performs text-to-video generation.

**Example — Text-to-Video**

```bash
curl -X POST http://<your-server-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{"prompt": "A cat dancing on the moon"}'
```

**Example — Image-to-Video (URL)**

```bash
curl -X POST http://<your-server-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{
    "prompt": "Make this character wave hello",
    "image_url": "https://example.com/character.jpg"
  }'
```

**Example — Image-to-Video (Base64)**

```bash
curl -X POST http://<your-server-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{
    "prompt": "Make this character wave hello",
    "image_base64": "'$(base64 -i image.jpg)'"
  }'
```

**Response**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "position": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Unique job identifier (UUID) |
| `status` | string | Always `"queued"` on successful submission |
| `position` | integer | Position in the queue (0 = next to be processed) |

**Errors**

| Status | Description |
|--------|-------------|
| 400 | Missing prompt, invalid base64, or failed image download |
| 401 | Invalid or missing API key |

---

### Check Job Status

Get the current status of a job.

```
GET /status/{job_id}
```

**Example**

```bash
curl http://<your-server-ip>:8000/status/550e8400-e29b-41d4-a716-446655440000 \
  -H "X-API-Key: <your-api-key>"
```

**Response**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "video_url": "https://s3.amazonaws.com/...",
  "error": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | The job identifier |
| `status` | string | One of: `queued`, `processing`, `completed`, `failed` |
| `video_url` | string \| null | Presigned S3 download URL (only when `completed`) |
| `error` | string \| null | Error message (only when `failed`) |

> **Note:** The `video_url` is a presigned S3 URL that expires after **1 hour**. Request a fresh URL by calling this endpoint again.

**Errors**

| Status | Description |
|--------|-------------|
| 401 | Invalid or missing API key |
| 404 | Job not found |

---

### List Jobs

List all jobs with optional filtering by status.

```
GET /jobs
```

**Query Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `status` | string | No | — | Filter by status: `queued`, `processing`, `completed`, `failed` |
| `limit` | integer | No | 100 | Maximum number of jobs to return |

**Example — All Jobs**

```bash
curl http://<your-server-ip>:8000/jobs \
  -H "X-API-Key: <your-api-key>"
```

**Example — Completed Jobs Only**

```bash
curl "http://<your-server-ip>:8000/jobs?status=completed&limit=10" \
  -H "X-API-Key: <your-api-key>"
```

**Response**

```json
[
  {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "prompt": "A cat dancing on the moon",
    "video_url": "https://s3.amazonaws.com/...",
    "error": null,
    "created_at": "2026-02-21T15:01:21.000000"
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | The job identifier |
| `status` | string | Job status |
| `prompt` | string | The original prompt |
| `video_url` | string \| null | Presigned S3 download URL (only when `completed`) |
| `error` | string \| null | Error message (only when `failed`) |
| `created_at` | string | ISO 8601 timestamp of when the job was created |

Jobs are returned in reverse chronological order (newest first).

---

## Job Lifecycle

```
queued → processing → completed
                    → failed
```

1. **queued** — Job is waiting in the queue
2. **processing** — Video is being generated (typically 6–10 minutes at 480p)
3. **completed** — Video is ready, `video_url` is available
4. **failed** — Generation failed, see `error` for details

Jobs are processed sequentially, one at a time.

## Video Output

- **Model:** WAN 2.2 14B (FP8 quantized, dual expert architecture)
- **Format:** MP4 (H.264)
- **Resolution:** 832x480 (480p)
- **Frame rate:** 16 fps
- **Duration:** ~5 seconds (81 frames)
- **Storage:** Videos are uploaded to S3 and served via presigned URLs
- **URL Expiry:** Presigned URLs expire after 1 hour. Call `/status/{job_id}` again to get a fresh URL.

## Rate Limits

There are no explicit rate limits, but since jobs are processed sequentially, submitting many jobs will queue them. Use the `position` field from `/generate` to estimate wait time.

## Client Example (Node.js / TypeScript)

```typescript
const API_URL = "http://<your-server-ip>:8000";
const API_KEY = "<your-api-key>";

const headers = {
  "Content-Type": "application/json",
  "X-API-Key": API_KEY,
};

// Submit a job
const res = await fetch(`${API_URL}/generate`, {
  method: "POST",
  headers,
  body: JSON.stringify({ prompt: "A cat dancing on the moon" }),
});
const { job_id } = await res.json();

// Poll until complete
let video_url: string | null = null;
while (!video_url) {
  await new Promise((r) => setTimeout(r, 10_000));
  const status = await fetch(`${API_URL}/status/${job_id}`, { headers }).then(
    (r) => r.json()
  );
  if (status.status === "completed") video_url = status.video_url;
  if (status.status === "failed") throw new Error(status.error);
}

console.log("Video ready:", video_url);
```

## Client Example (Python)

```python
import time
import requests

API_URL = "http://<your-server-ip>:8000"
HEADERS = {"X-API-Key": "<your-api-key>"}

# Submit a job
res = requests.post(
    f"{API_URL}/generate",
    headers=HEADERS,
    json={"prompt": "A cat dancing on the moon"},
)
job_id = res.json()["job_id"]

# Poll until complete
while True:
    time.sleep(10)
    status = requests.get(f"{API_URL}/status/{job_id}", headers=HEADERS).json()
    if status["status"] == "completed":
        print("Video ready:", status["video_url"])
        break
    if status["status"] == "failed":
        raise Exception(status["error"])
```
