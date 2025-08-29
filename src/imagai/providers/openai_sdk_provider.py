import httpx
import os
from openai import OpenAI, AsyncOpenAI
from openai.types.images_response import Image
from imagai.config import EngineConfig
from imagai.models import ImageGenerationRequest, ImageGenerationResponse
from imagai.providers.base_provider import BaseImageProvider
from typing import List
import logging
import json
from rich.console import Console
from rich.table import Table

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
            # Detect OpenRouter endpoint
            is_openrouter = False
            if getattr(self.config, "base_url", None):
                try:
                    is_openrouter = "openrouter.ai" in str(self.config.base_url)
                except Exception:
                    is_openrouter = False

            model_name = self.config.model or "dall-e-3"

            # If using OpenRouter with a Gemini model or chat-only model, use chat.completions
            if is_openrouter and ("gemini" in model_name.lower()):
                client = OpenAI(**self.client_params)
                # Optional OpenRouter ranking headers from env
                extra_headers = {}
                ref = os.environ.get("OPENROUTER_HTTP_REFERER")
                ttl = os.environ.get("OPENROUTER_X_TITLE")
                if ref:
                    extra_headers["HTTP-Referer"] = ref
                if ttl:
                    extra_headers["X-Title"] = ttl

                # Build messages; allow vision via extra_params.image_url
                content_items = []
                if request.prompt:
                    content_items.append({"type": "text", "text": request.prompt})
                img_url = None
                if getattr(request, "extra_params", None):
                    img_url = request.extra_params.get("image_url")
                if img_url:
                    content_items.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": img_url},
                        }
                    )
                messages = [
                    {
                        "role": "user",
                        "content": content_items
                        if content_items
                        else [{"type": "text", "text": request.prompt or ""}],
                    }
                ]
                
                # Check if this is an image generation model
                is_image_model = "image" in model_name.lower()
                extra_body = {}
                if is_image_model:
                    extra_body["modalities"] = ["image", "text"]

                if request.verbose:
                    print("--- OpenRouter Chat.Completions Request ---")
                    try:
                        print(
                            json.dumps(
                                {
                                    "model": model_name,
                                    "messages": messages,
                                    "extra_headers": extra_headers or None,
                                },
                                indent=2,
                            )
                        )
                    except Exception:
                        print({"model": model_name, "messages": messages})
                    print("------------------------------------------")

                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    extra_headers=extra_headers or None,
                    extra_body=extra_body,
                )

                # Extract content and images from response
                try:
                    message = completion.choices[0].message
                    content = message.content
                    images = getattr(message, "images", None)
                except Exception:
                    content = None
                    images = None

                resp = ImageGenerationResponse()
                
                # Handle image generation models
                if is_image_model and images:
                    # Extract base64 image data from the first image
                    try:
                        image_data = images[0]["image_url"]["url"]
                        if image_data.startswith("data:image/"):
                            # Extract base64 data from data URL
                            base64_data = image_data.split(",", 1)[1]
                            resp.image_b64_json = base64_data
                        else:
                            resp.image_url = image_data
                    except (KeyError, IndexError, TypeError) as e:
                        resp.error = f"Failed to extract image data: {e}"
                
                # Always include text content if available
                if content:
                    resp.text_content = content
                
                # Set error if no content was found
                if not content and not (is_image_model and images):
                    resp.error = (
                        "No content returned from OpenRouter chat completion."
                    )
                
                # Include usage if available
                usage = getattr(completion, "usage", None)
                if usage is not None:
                    try:
                        resp.usage = (
                            usage.model_dump()
                            if hasattr(usage, "model_dump")
                            else dict(usage)
                        )
                    except Exception:
                        resp.usage = None
                responses.append(resp)
                return responses

            kwargs = {
                "model": model_name,
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
                if "response_format" in kwargs:
                    del kwargs["response_format"]

                stability_specific_params = [
                    "negative_prompt",
                    "seed",
                    "strength",
                    "output_format",
                    "aspect_ratio",
                    "mode",
                ]
                extra_body = {}

                # Populate stability-specific params directly into kwargs from request.extra_params if provided
                if hasattr(request, "extra_params") and request.extra_params:
                    for key in stability_specific_params:
                        if key in request.extra_params and request.extra_params[key] is not None:
                            extra_body[key] = request.extra_params[key]

                # Handle 'mode' parameter logic for Stability AI
                if "mode" not in extra_body:
                    if "sd3" not in self.config.model.lower():
                        extra_body["mode"] = "text-to-image"

                # Handle potential conflict between 'size' and 'aspect_ratio'
                if "aspect_ratio" in extra_body and "size" in kwargs:
                    del kwargs["size"]  # Prefer aspect_ratio if explicitly provided

                if extra_body:
                    kwargs["extra_body"] = extra_body

            if request.verbose:
                print("--- API Request Body ---")
                try:
                    print(json.dumps(kwargs, indent=2, default=str))
                except TypeError:
                    print(kwargs)
                print("------------------------")

            api_response = await self.async_client.images.generate(**kwargs)
            usage = getattr(api_response, "usage", None)
            estimated_cost = getattr(api_response, "estimated_cost", None)
            for image_data in api_response.data:
                img_response = ImageGenerationResponse()
                if image_data.url:
                    img_response.image_url = image_data.url
                elif image_data.b64_json:
                    img_response.image_b64_json = image_data.b64_json
                else:
                    img_response.error = "No image data found in API response."
                img_response.usage = usage
                img_response.estimated_cost = estimated_cost
                responses.append(img_response)

            console = Console()
            table = Table(
                title="API Usage & Cost Info",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Category", style="cyan", width=15)
            table.add_column("Attribute", style="green", width=20)
            table.add_column("Value", style="yellow")

            has_data_to_display = False

            if usage:
                attributes_to_check = [
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "input_tokens_details",
                ]
                usage_details_added = False
                for attr in attributes_to_check:
                    if hasattr(usage, attr):
                        value = getattr(usage, attr)
                        table.add_row(
                            "Usage",
                            attr,
                            str(value) if value is not None else "N/A",
                        )
                        usage_details_added = True
                        has_data_to_display = True

                if not usage_details_added:
                    # Fallback if no known attributes found, print str(usage)
                    table.add_row("Usage", "Raw Data", str(usage))
                    has_data_to_display = True

            if estimated_cost:  # This is a dict
                for key, value in estimated_cost.items():
                    table.add_row("Estimated Cost", key, str(value))
                    has_data_to_display = True

            if has_data_to_display:
                console.print(table)
            else:
                console.print("No usage or cost information available.")

            return responses
        except Exception as e:
            logger.error(f"Error generating image with engine {request.engine}: {e}")
            return [ImageGenerationResponse(error=str(e))]

    async def close(self):
        await self.async_client.close()
