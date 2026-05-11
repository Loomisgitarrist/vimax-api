"""Generic OpenAI-compatible image generator.

Works with any provider that exposes an OpenAI-style images/generations
endpoint: OpenRouter, FAL (OpenAI-compatible mode), direct OpenAI, etc.

Config example:
  image_generator:
    class_path: tools.image_generator_openai_api.ImageGeneratorOpenAIAPI
    init_args:
      api_key: ${IMAGE_API_KEY}
      base_url: https://openrouter.ai/api/v1
      model: recraft-v3
"""

import logging
import base64
from io import BytesIO
from typing import List, Optional
from PIL import Image
import httpx
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt

from interfaces.image_output import ImageOutput
from utils.retry import after_func
from utils.rate_limiter import RateLimiter


class ImageGeneratorOpenAIAPI:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "dall-e-3",
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.model = model
        self.rate_limiter = rate_limiter
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @retry(stop=stop_after_attempt(3), after=after_func)
    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] = [],
        aspect_ratio: Optional[str] = "16:9",
        **kwargs,
    ) -> ImageOutput:
        logging.info(f"Calling {self.model} via OpenAI-compatible API to generate image...")

        if self.rate_limiter:
            await self.rate_limiter.acquire()

        # Convert aspect_ratio to size if needed (OpenAI uses size, not aspect_ratio)
        size = kwargs.get("size", self._aspect_to_size(aspect_ratio))

        response = await self.client.images.generate(
            model=self.model,
            prompt=prompt,
            n=1,
            size=size,
            response_format="b64_json",
        )

        b64_data = response.data[0].b64_json
        if not b64_data:
            raise ValueError("No image returned from API")

        image = Image.open(BytesIO(base64.b64decode(b64_data)))
        return ImageOutput(fmt="pil", ext="png", data=image)

    def _aspect_to_size(self, aspect_ratio: str) -> str:
        mapping = {
            "16:9": "1792x1024",
            "9:16": "1024x1792",
            "1:1": "1024x1024",
            "4:3": "1024x768",
            "3:4": "768x1024",
        }
        return mapping.get(aspect_ratio, "1792x1024")
