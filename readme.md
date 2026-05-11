# ViMax API Server

FastAPI wrapper for the [ViMax](https://github.com/HKUDS/ViMax) agentic video generation framework. Exposes `idea2video` and `script2video` pipelines over HTTP so any Hermes agent (or web app) can submit jobs and poll for results.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| POST | `/generate/idea` | Submit idea → video job |
| POST | `/generate/script` | Submit script → video job |
| GET | `/status/{job_id}` | Poll job status |
| GET | `/download/{job_id}` | Download completed MP4 |
| GET | `/jobs` | List all jobs |

## Quick Start (Docker)

```bash
git clone https://github.com/YOURUSER/vimax-api.git
cd vimax-api

# Copy a config with your API keys filled in
cp configs/idea2video.yaml configs/idea2video.production.yaml
# Edit configs/idea2video.production.yaml — add your OpenRouter / Google API keys

docker build -t vimax-api .
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/configs/idea2video.production.yaml:/app/configs/idea2video.yaml \
  -e VIMAX_CONFIG=/app/configs/idea2video.yaml \
  -e OPENROUTER_API_KEY=sk-... \
  vimax-api
```

## Submit a Job

```bash
curl -X POST https://vimax.yourdomain.com/generate/idea \
  -H "Content-Type: application/json" \
  -d '{
    "idea": "A woman walks through a modern Berlin apartment, sunlight streaming through floor-to-ceiling windows",
    "user_requirement": "Max 2 scenes, 3 shots each. Slow, elegant pacing.",
    "style": "Cinematic, warm tones"
  }'
```

Response:
```json
{"job_id": "a1b2c3d4", "status": "pending", "poll": "/status/a1b2c3d4"}
```

## Poll for Result

```bash
curl https://vimax.yourdomain.com/status/a1b2c3d4
```

When `status` is `completed`, download:
```bash
curl -O https://vimax.yourdomain.com/download/a1b2c3d4
```

## Coolify Deployment

1. Create a **Private** or **Public** GitHub repo with this code
2. In Coolify: **New Service → GitHub → Select repo**
3. **Set `PORT=8000`** explicitly in Coolify service settings
4. Point a domain (e.g. `vimax.loomyloo.app`)
5. Add environment variables for your API keys
6. Deploy

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | HTTP server port |
| `VIMAX_CONFIG` | `configs/idea2video.yaml` | YAML config path inside container |
| `VIMAX_OUTPUT_DIR` | `./output` | Where final videos are stored |
| `VIMAX_JOBS_FILE` | `./jobs.json` | Persistent job state |

API keys are read from the YAML config (see ViMax docs), but you can also pass them as env vars if your config references `$ENV_VAR`.

## Architecture

```
Browser / Hermes Agent
    │ POST /generate/idea
    ▼
FastAPI Server (Coolify/Hetzner)
    │ background task
    ▼
ViMax Pipeline (Idea2Video / Script2Video)
    │ calls external APIs
    ▼
OpenRouter (LLM) → Google Imagen (images) → Google Veo (video)
    │
    ▼
MP4 saved → /download/{job_id}
```

## Notes

- **ViMax is a coordinator, not a model.** It orchestrates paid APIs (OpenRouter, Google Imagen, Google Veo). Budget accordingly.
- **Jobs are async.** Submit → poll `/status` until `completed` or `failed`. Typical runtime: 5–30 minutes depending on shot count.
- **No GPU required on server.** All heavy inference happens via cloud APIs.
- **Resume support.** ViMax caches intermediate frames in the working directory. If a job fails partway, re-running with the same working dir picks up where it left off.
