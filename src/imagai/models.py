from pydantic import BaseModel, Field
from typing import Optional, Literal


class ImageGenerationRequest(BaseModel):
    prompt: str
    engine: str
    output_filename: Optional[str] = None
    size: Optional[
        Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
    ] = "1024x1024"
    quality: Optional[Literal["standard", "hd"]] = "standard"
    n: Optional[int] = Field(
        1, ge=1, le=10, description="Number of images to generate."
    )
    style: Optional[Literal["vivid", "natural"]] = "vivid"
    response_format: Optional[Literal["url", "b64_json"]] = "url"
    extra_params: Optional[dict] = None
    verbose: bool = False
    auto_filename: bool = False
    random_filename: bool = False


class ImageGenerationResponse(BaseModel):
    image_url: Optional[str] = None
    image_b64_json: Optional[str] = None
    saved_path: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[dict] = None
    estimated_cost: Optional[dict] = None
    # For OpenRouter chat-based responses (e.g., Gemini vision)
    text_content: Optional[str] = None
