from imagai.config import settings
from imagai.models import ImageGenerationRequest, ImageGenerationResponse
from imagai.providers.openai_sdk_provider import OpenAISDKProvider
from imagai.utils import (
    save_image_from_url,
    save_image_from_b64,
    generate_filename,
    generate_random_filename,
    generate_filename_from_prompt_llm,
    get_image_extension,
)
from pathlib import Path
import logging
from typing import List

logger = logging.getLogger(__name__)


async def generate_image_core(
    request: ImageGenerationRequest,
) -> List[ImageGenerationResponse]:
    engine_name = request.engine
    if engine_name not in settings.engines:
        error_msg = f"Engine '{engine_name}' not configured. Available engines: {list(settings.engines.keys())}"
        logger.error(error_msg)
        return [ImageGenerationResponse(error=error_msg)]
    engine_config = settings.engines[engine_name]
    provider = OpenAISDKProvider(engine_config)
    final_responses: List[ImageGenerationResponse] = []
    try:
        api_responses = await provider.generate_image(request)
        for i, api_response in enumerate(api_responses):
            if api_response.error:
                final_responses.append(api_response)
                continue
            # If we received a text-only response (e.g., OpenRouter chat-based vision),
            # skip image-saving logic and return the text content without error.
            if (
                getattr(api_response, "text_content", None)
                and not getattr(api_response, "image_url", None)
                and not getattr(api_response, "image_b64_json", None)
            ):
                final_responses.append(api_response)
                continue
            base_filename = request.output_filename
            output_ext = "png"
            if base_filename:
                output_ext = get_image_extension(base_filename)
                if request.n > 1:
                    name_part, _ = Path(base_filename).stem, Path(base_filename).suffix
                    current_filename = f"{name_part}_{i + 1}.{output_ext}"
                else:
                    current_filename = base_filename
            elif request.auto_filename:
                current_filename = await generate_filename_from_prompt_llm(
                    prompt=request.prompt, extension=output_ext, verbose=request.verbose
                )
                if request.n > 1:
                    name_part, ext_part = (
                        Path(current_filename).stem,
                        Path(current_filename).suffix,
                    )
                    current_filename = f"{name_part}_{i + 1}{ext_part}"
            elif request.random_filename:
                current_filename = generate_random_filename(extension=output_ext)
                if request.n > 1:
                    name_part, ext_part = (
                        Path(current_filename).stem,
                        Path(current_filename).suffix,
                    )
                    current_filename = f"{name_part}_{i + 1}{ext_part}"
            else:
                current_filename = generate_filename(
                    prompt=request.prompt, extension=output_ext
                )
            output_file_path = Path(settings.output_dir) / current_filename
            saved_path = None
            model_name = (
                engine_config.model
                if hasattr(engine_config, "model")
                else str(engine_config)
            )
            if api_response.image_url:
                saved_path = await save_image_from_url(
                    api_response.image_url, output_file_path, request.prompt, model_name
                )
            elif api_response.image_b64_json:
                saved_path = await save_image_from_b64(
                    api_response.image_b64_json,
                    output_file_path,
                    request.prompt,
                    model_name,
                )
            if saved_path:
                api_response.saved_path = str(saved_path)
            else:
                api_response.error = (
                    api_response.error or f"Failed to save image to {output_file_path}"
                )
            final_responses.append(api_response)
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred in generate_image_core for engine {engine_name}: {e}"
        )
        return [ImageGenerationResponse(error=f"Core generation error: {e}")]
    finally:
        if "provider" in locals() and hasattr(provider, "close"):
            await provider.close()
    return final_responses
