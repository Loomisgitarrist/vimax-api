"""Generic OpenAI-compatible video generator.

WARNING: Most video providers do NOT expose an OpenAI-compatible endpoint.
This adapter works only with providers that do (e.g. some FAL endpoints).
For Google Veo, use VideoGeneratorVeoGoogleAPI directly.
For FAL native API, a separate adapter is needed.

Config example:
  video_generator:
    class_path: tools.video_generator_openai_api.VideoGeneratorOpenAIAPI
    init_args:
      api_key: ${VIDEO_API_KEY}
      base_url: https://api.openai.com/v1
      model: gpt-4o-mini-video
"""

import logging
from typing import List, Optional
import httpx
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt

from interfaces.video_output import VideoOutput
from utils.retry import after_func
from utils.rate_limiter import RateLimiter


class VideoGeneratorOpenAIAPI:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini-video",
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.model = model
        self.rate_limiter = rate_limiter
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @retry(stop=stop_after_attempt(3), after=after_func)
    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: List[str],
        resolution: str = "1080p",
        aspect_ratio: str = "16:9",
        duration: int = 8,
        **kwargs,
    ) -> VideoOutput:
        logging.warning(
            "VideoGeneratorOpenAIAPI is a placeholder. Most video providers "
            "require custom adapters. Falling back to HTTP POST."
        )

        if self.rate_limiter:
            await self.rate_limiter.acquire()

        # Generic fallback — POST to /v1/videos/generations (non-standard)
        # Providers that support this will work; others will fail.
        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": self._resolution_to_size(resolution, aspect_ratio),
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.client.base_url}/videos/generations",
                headers={"Authorization": f"Bearer {self.client.api_key}"},
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()

        video_url = data.get("data", [{}])[0].get("url")
        if not video_url:
            raise ValueError("No video URL returned from API")

        return VideoOutput(fmt="url", ext="mp4", data=video_url)

    def _resolution_to_size(self, resolution: str, aspect_ratio: str) -> str:
        mapping = {
            ("1080p", "16:9"): "1920x1080",
            ("720p", "16:9"): "1280x720",
            ("1080p", "9:16"): "1080x1920",
        }
        return mapping.get((resolution, aspect_ratio), "1920x1080")
