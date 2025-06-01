import httpx
import base64
from pathlib import Path
from datetime import datetime
from PIL import Image
import io
import logging
from typing import Optional
import uuid
import re
from imagai.config import settings
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a valid filename."""
    name = re.sub(r'[<>:"/\\\\|?*\\x00-\\x1F]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name[:100]
    return name


async def generate_filename_from_prompt_llm(prompt: str, extension: str = "png") -> str:
    """Generates a filename from a prompt using an LLM."""
    # Determine which engine configuration to use for filename generation
    # Priority: Dedicated "filename_generation" engine, then default_engine, then first available OpenAI engine
    engine_to_use_for_filename_gen = None
    filename_engine_config = None

    if "filename_generation" in settings.engines:
        engine_to_use_for_filename_gen = "filename_generation"
        filename_engine_config = settings.engines[engine_to_use_for_filename_gen]
    elif (
        settings.default_engine and settings.default_engine in settings.engines
    ):  # Check if default_engine is configured
        engine_to_use_for_filename_gen = settings.default_engine
        filename_engine_config = settings.engines[engine_to_use_for_filename_gen]
    else:  # Fallback to the first openai engine if no specific or default engine for filenames
        for name, config in settings.engines.items():
            if "openai" in name:  # A simple check, could be made more robust
                engine_to_use_for_filename_gen = name
                filename_engine_config = config
                break

    if (
        not filename_engine_config
        or not filename_engine_config.api_key
        or filename_engine_config.api_key == "YOUR_OPENAI_API_KEY"
    ):
        logger.warning(
            "OpenAI API key for filename generation not configured. Using default filename."
        )
        return generate_filename(prompt, extension)

    client = AsyncOpenAI(
        api_key=filename_engine_config.api_key,
        base_url=str(filename_engine_config.base_url)
        if filename_engine_config.base_url
        else None,
    )
    try:
        response = await client.chat.completions.create(
            model=filename_engine_config.model
            or "gpt-3.5-turbo",  # Default to gpt-3.5-turbo if not specified
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates concise, descriptive, and filesystem-safe filenames based on user prompts. The filename should not include the file extension. Max 5 words.",
                },
                {
                    "role": "user",
                    "content": f"Generate a filename for this prompt: {prompt}",
                },
            ],
            max_tokens=20,
            temperature=0.2,
        )
        raw_filename = response.choices[0].message.content.strip()
        sanitized = sanitize_filename(raw_filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{sanitized}_{timestamp}.{extension}"
    except Exception as e:
        logger.error(f"Error generating filename with LLM: {e}. Using default method.")
        return generate_filename(prompt, extension)  # Fallback to default
    finally:
        if "client" in locals():
            await client.close()


def generate_random_filename(extension: str = "png") -> str:
    """Generates a random filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = str(uuid.uuid4())[:8]
    return f"image_{timestamp}_{random_str}.{extension}"


def generate_filename(prompt: Optional[str] = None, extension: str = "png") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prompt:
        sane_prompt = "".join(
            c if c.isalnum() or c in (" ", "-") else "_" for c in prompt[:30]
        ).rstrip()
        sane_prompt = sane_prompt.replace(" ", "_")
        return f"{sane_prompt}_{timestamp}.{extension}"
    return f"image_{timestamp}.{extension}"


async def save_image_from_url(image_url: str, output_path: Path) -> Optional[Path]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(io.BytesIO(response.content))
            img.save(output_path)
            logger.info(f"Image saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(
                f"Failed to process and save image from {image_url} to {output_path}: {e}"
            )
            return None
    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTP error downloading image {image_url}: {e.response.status_code} - {e.response.text}"
        )
        return None
    except Exception as e:
        logger.error(f"Error downloading or saving image {image_url}: {e}")
        return None


async def save_image_from_b64(b64_json: str, output_path: Path) -> Optional[Path]:
    try:
        image_bytes = base64.b64decode(b64_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.save(output_path)
            logger.info(f"Image saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to process and save b64 image to {output_path}: {e}")
            return None
    except Exception as e:
        logger.error(f"Error decoding or saving base64 image: {e}")
        return None


def get_image_extension(filename: str) -> str:
    ext = Path(filename).suffix[1:].lower()
    if ext in ["jpg", "jpeg", "png", "gif", "webp"]:
        return ext
    return "png"
