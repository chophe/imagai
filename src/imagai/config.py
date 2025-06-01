from pydantic import BaseModel, HttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Optional
import os


class EngineConfig(BaseModel):
    api_key: str = Field(..., description="API key for the image generation engine.")
    base_url: Optional[HttpUrl] = Field(
        None, description="Base URL for the API (for OpenAI-compatible engines)."
    )
    model: Optional[str] = Field(
        None, description="Default model to use for this engine."
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="IMAGAI__",
        env_nested_delimiter="__",
        extra="ignore",
    )

    output_dir: str = Field(
        "generated_images", description="Default directory to save generated images."
    )
    default_engine: Optional[str] = Field(
        None, description="Default engine to use if not specified in command."
    )
    engines: Dict[str, EngineConfig] = {
        "openai_dalle3": EngineConfig(api_key="YOUR_OPENAI_API_KEY", model="dall-e-3")
    }


settings = Settings()

for key, value in os.environ.items():
    if key.startswith("IMAGAI__ENGINES__"):
        parts = key.split("__")
        if len(parts) >= 4:
            engine_name = parts[2].lower()
            config_key = parts[3].lower()
            if engine_name not in settings.engines:
                settings.engines[engine_name] = EngineConfig(api_key="dummy")
            if hasattr(settings.engines[engine_name], config_key):
                if config_key == "base_url":
                    try:
                        value = HttpUrl(value)
                    except Exception:
                        pass
                setattr(settings.engines[engine_name], config_key, value)
            elif config_key == "api_key" and not settings.engines[engine_name].api_key:
                settings.engines[engine_name].api_key = value

from pathlib import Path

Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
