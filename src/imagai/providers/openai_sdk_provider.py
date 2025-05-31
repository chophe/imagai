import httpx
from openai import OpenAI, AsyncOpenAI
from openai.types.images_response import Image
from imagai.config import EngineConfig
from imagai.models import ImageGenerationRequest, ImageGenerationResponse
from imagai.providers.base_provider import BaseImageProvider
from typing import List
import logging

logger = logging.getLogger(__name__)


class OpenAISDKProvider(BaseImageProvider):
    def __init__(self, engine_config: EngineConfig):
        self.config = engine_config
        self.client_params = {
            "api_key": self.config.api_key,
        }
        if self.config.base_url:
            self.client_params["base_url"] = str(self.config.base_url)
        self.async_client = AsyncOpenAI(**self.client_params)

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> List[ImageGenerationResponse]:
        responses: List[ImageGenerationResponse] = []
        try:
            api_response = await self.async_client.images.generate(
                model=self.config.model or "dall-e-2",
                prompt=request.prompt,
                n=request.n or 1,
                size=request.size or "1024x1024",
                quality=request.quality
                if self.config.model and "dall-e-3" in self.config.model
                else None,
                style=request.style
                if self.config.model and "dall-e-3" in self.config.model
                else None,
                response_format=request.response_format or "url",
            )
            for image_data in api_response.data:
                img_response = ImageGenerationResponse()
                if image_data.url:
                    img_response.image_url = image_data.url
                elif image_data.b64_json:
                    img_response.image_b64_json = image_data.b64_json
                else:
                    img_response.error = "No image data found in API response."
                responses.append(img_response)
            return responses
        except Exception as e:
            logger.error(f"Error generating image with engine {request.engine}: {e}")
            return [ImageGenerationResponse(error=str(e))]

    async def close(self):
        await self.async_client.close()
