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
            kwargs = {
                "model": self.config.model or "dall-e-2",
                "prompt": request.prompt,
                "n": request.n or 1,
                "size": request.size or "1024x1024",
                "response_format": request.response_format or "url",
            }

            if self.config.model and "dall-e-3" in self.config.model:
                kwargs["quality"] = request.quality
                kwargs["style"] = request.style

            if self.config.model and "stability" in self.config.model.lower():
                # For Stability AI models, 'n' is typically not used or handled differently.
                # The AvalAI examples, which this provider might be used with via base_url, don't show 'n'.
                if "n" in kwargs:
                    del kwargs["n"]

                stability_specific_params = [
                    "negative_prompt",
                    "seed",
                    "strength",
                    "output_format",
                    "aspect_ratio",
                    "mode",
                ]
                extra_body = {}

                # Populate extra_body from request.extra_params if they are provided
                # This assumes 'request' has an 'extra_params' attribute (e.g., a dictionary)
                # containing additional parameters passed from the CLI or other sources.
                if hasattr(request, "extra_params") and request.extra_params:
                    for key in stability_specific_params:
                        if (
                            key in request.extra_params
                            and request.extra_params[key] is not None
                        ):
                            extra_body[key] = request.extra_params[key]

                # Handle 'mode' parameter logic for Stability AI
                # If user explicitly provided 'mode' via extra_params, it's already in extra_body.
                # Otherwise, apply specific defaults based on model type.
                if "mode" not in extra_body:
                    # For SD3 models, do not add 'mode' by default.
                    # It should be explicitly set by the user if needed (e.g., for 'image-to-image').
                    # For other (non-SD3) stability models, default to "text-to-image".
                    if "sd3" not in self.config.model.lower():
                        extra_body["mode"] = "text-to-image"

                # Handle potential conflict between 'size' and 'aspect_ratio'
                # If 'aspect_ratio' is provided (now in extra_body), 'size' might be redundant or conflicting.
                if "aspect_ratio" in extra_body and "size" in kwargs:
                    del kwargs["size"]  # Prefer aspect_ratio if explicitly provided

                # Add the collected stability-specific parameters to the API call via extra_body
                if extra_body:
                    kwargs["extra_body"] = extra_body

            # The existing placeholder comments regarding Avaloq documentation and specific
            # parameter handling (like removing 'size' if 'aspect_ratio' is used)
            # are addressed by the logic above.

            api_response = await self.async_client.images.generate(**kwargs)
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
