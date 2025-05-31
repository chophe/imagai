import httpx
import base64
from pathlib import Path
from datetime import datetime
from PIL import Image
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


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
