"""OpenRouter video generation adapter with async polling.

Works with ANY video model on OpenRouter (Veo 3.1, Kling, etc.)
Uses OpenRouter's async job API: submit → poll → download.

Config example:
  video_generator:
    class_path: tools.video_generator_openrouter_api.VideoGeneratorOpenRouterAPI
    init_args:
      api_key: ${OPENROUTER_API_KEY}
      model: google/veo-3.1
"""

import logging
import asyncio
from typing import List, Optional
import httpx
from tenacity import retry, stop_after_attempt

from interfaces.video_output import VideoOutput
from interfaces.image_output import ImageOutput
from utils.retry import after_func
from utils.rate_limiter import RateLimiter


class VideoGeneratorOpenRouterAPI:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "google/veo-3.1",
        rate_limiter: Optional[RateLimiter] = None,
        poll_interval: int = 5,
        max_poll_time: int = 600,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @retry(stop=stop_after_attempt(3), after=after_func)
    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: List[str] = [],
        resolution: str = "1080p",
        aspect_ratio: str = "16:9",
        duration: int = 8,
        **kwargs,
    ) -> VideoOutput:
        logging.info(f"Calling OpenRouter video endpoint for {self.model} ...")

        if self.rate_limiter:
            await self.rate_limiter.acquire()

        # 1. Submit job
        payload = {"model": self.model, "prompt": prompt}

        # Add aspect_ratio hint if provider supports it
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/videos",
                headers=self.headers,
                json=payload,
            )
            resp.raise_for_status()
            submit_data = resp.json()

        job_id = submit_data.get("id")
        polling_url = submit_data.get("polling_url")
        if not job_id or not polling_url:
            raise RuntimeError(f"OpenRouter did not return job_id + polling_url. Got: {submit_data}")

        logging.info(f"Video job submitted: {job_id}. Polling...")

        # 2. Poll until done
        start = asyncio.get_event_loop().time()
        while True:
            async with httpx.AsyncClient() as client:
                poll_resp = await client.get(polling_url, headers=self.headers, timeout=30)
                poll_resp.raise_for_status()
                status_data = poll_resp.json()

            status = status_data.get("status")
            logging.info(f"Video job {job_id} status: {status}")

            if status == "completed":
                urls = status_data.get("unsigned_urls", [])
                if not urls:
                    raise RuntimeError("Job completed but no unsigned_urls returned")
                return VideoOutput(fmt="url", ext="mp4", data=urls[0])

            if status == "failed":
                error = status_data.get("error", "Unknown error")
                raise RuntimeError(f"Video generation failed: {error}")

            if (asyncio.get_event_loop().time() - start) > self.max_poll_time:
                raise RuntimeError(f"Video generation timed out after {self.max_poll_time}s")

            await asyncio.sleep(self.poll_interval)
