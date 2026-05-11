"""
ViMax FastAPI Server
Wraps the ViMax agentic video generation pipelines behind HTTP endpoints.
Supports both idea-to-video and script-to-video workflows.
"""

import os
import uuid
import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# ViMax imports
from pipelines.idea2video_pipeline import Idea2VideoPipeline
from pipelines.script2video_pipeline import Script2VideoPipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(os.environ.get("VIMAX_OUTPUT_DIR", "./output"))
JOBS_FILE = Path(os.environ.get("VIMAX_JOBS_FILE", "./jobs.json"))
CONFIG_PATH = os.environ.get("VIMAX_CONFIG", "configs/idea2video.yaml")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Job State
# ---------------------------------------------------------------------------

jobs: Dict[str, Dict[str, Any]] = {}


def _load_jobs():
    global jobs
    if JOBS_FILE.exists():
        with open(JOBS_FILE, "r") as f:
            jobs = json.load(f)


def _save_jobs():
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, default=str, indent=2)


def _new_job(job_type: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "type": job_type,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "output_path": None,
        "error": None,
        "progress": [],
    }
    _save_jobs()
    return job_id


def _update_job(job_id: str, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)
        jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
        _save_jobs()


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class IdeaRequest(BaseModel):
    idea: str = Field(..., description="Raw idea / concept for the video")
    user_requirement: str = Field("", description="Constraints, e.g. 'max 3 scenes, 5 shots each'")
    style: str = Field("Realistic", description="Visual style, e.g. 'Anime', 'Realistic', 'Cinematic'")
    config_path: Optional[str] = Field(CONFIG_PATH, description="Path to YAML config")


class ScriptRequest(BaseModel):
    script: str = Field(..., description="Full screenplay text")
    user_requirement: str = Field("", description="Constraints")
    style: str = Field("Realistic", description="Visual style")
    config_path: Optional[str] = Field("configs/script2video.yaml", description="Path to YAML config")


class JobStatusResponse(BaseModel):
    id: str
    type: str
    status: str  # pending | running | completed | failed
    created_at: str
    updated_at: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    progress: list = []
    download_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Background Worker
# ---------------------------------------------------------------------------

async def _run_idea2video(job_id: str, req: IdeaRequest):
    try:
        _update_job(job_id, status="running")
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Override working dir via env for this run
        pipeline = Idea2VideoPipeline.init_from_config(config_path=req.config_path)
        pipeline.working_dir = str(job_dir)

        final_path = await pipeline(
            idea=req.idea,
            user_requirement=req.user_requirement,
            style=req.style,
        )

        _update_job(
            job_id,
            status="completed",
            output_path=final_path,
            download_url=f"/download/{job_id}",
        )
    except Exception as e:
        _update_job(job_id, status="failed", error=str(e))


async def _run_script2video(job_id: str, req: ScriptRequest):
    try:
        _update_job(job_id, status="running")
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        pipeline = Script2VideoPipeline.init_from_config(config_path=req.config_path)
        pipeline.working_dir = str(job_dir)

        final_path = await pipeline(
            script=req.script,
            user_requirement=req.user_requirement,
            style=req.style,
        )

        _update_job(
            job_id,
            status="completed",
            output_path=final_path,
            download_url=f"/download/{job_id}",
        )
    except Exception as e:
        _update_job(job_id, status="failed", error=str(e))


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_jobs()
    yield


app = FastAPI(
    title="ViMax API",
    description="Agentic video generation via HTTP. Wraps ViMax pipelines.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "ViMax API",
        "version": "1.0.0",
        "endpoints": [
            "POST /generate/idea",
            "POST /generate/script",
            "GET  /status/{job_id}",
            "GET  /download/{job_id}",
        ],
    }


@app.post("/generate/idea")
async def generate_idea(req: IdeaRequest, background_tasks: BackgroundTasks):
    job_id = _new_job("idea2video")
    background_tasks.add_task(_run_idea2video, job_id, req)
    return {"job_id": job_id, "status": "pending", "poll": f"/status/{job_id}"}


@app.post("/generate/script")
async def generate_script(req: ScriptRequest, background_tasks: BackgroundTasks):
    job_id = _new_job("script2video")
    background_tasks.add_task(_run_script2video, job_id, req)
    return {"job_id": job_id, "status": "pending", "poll": f"/status/{job_id}"}


@app.get("/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**jobs[job_id])


@app.get("/download/{job_id}")
def download(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    path = job.get("output_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(path, media_type="video/mp4", filename=f"vimax_{job_id}.mp4")


@app.get("/jobs")
def list_jobs():
    return {"jobs": list(jobs.values())}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
