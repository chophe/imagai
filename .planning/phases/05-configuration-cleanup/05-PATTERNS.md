# Phase 5: Configuration Cleanup - Pattern Map

**Mapped:** 2026-10-04
**Files analyzed:** 3 (2 modified, 1 new)
**Analogs found:** 3 / 3

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/imagai/config.py` | config | N/A (declaration) | `src/imagai/config.py` (self — current state) | exact |
| `src/imagai/web_server.py` | route | request-response | `src/imagai/utils.py` (write-time mkdir pattern) | role-match |
| `tests/test_config.py` | test | N/A | `tests/test_cli.py` | exact |

## Pattern Assignments

### `src/imagai/config.py` (config, declaration)

**Analog:** `src/imagai/config.py` (current state — the file being modified)

**Current imports pattern** (lines 1-4):
```python
from pydantic import BaseModel, HttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Optional
import os
```
After cleanup: `import os` is no longer needed (the os.environ loop is removed). The `from pathlib import Path` at line 56 is also removed.

**EngineConfig — where `api_key="dummy"` default goes** (lines 7-14):
```python
class EngineConfig(BaseModel):
    api_key: str = Field(..., description="API key for the image generation engine.")
    base_url: Optional[HttpUrl] = Field(
        None, description="Base URL for the API (for OpenAI-compatible engines)."
    )
    model: Optional[str] = Field(
        None, description="Default model to use for this engine."
    )
```
Change: `Field(...)` → `Field("dummy", ...)` on `api_key` to preserve the leniency the old loop provided (D-01).

**Settings — pydantic-settings config** (lines 17-33):
```python
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
```
This stays unchanged — native pydantic-settings reproduces the env parsing (D-01, D-03).

**Module-level singleton** (line 36):
```python
settings = Settings()
```
Stays unchanged — all consumers (`core.py`, `cli.py`, `utils.py`, `web_server.py`) import this singleton.

**REMOVED — os.environ loop** (lines 38-54):
```python
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
```
Entire block deleted. The `api_key="dummy"` leniency is preserved by the `EngineConfig.api_key` default instead.

**REMOVED — import-time mkdir** (lines 56-58):
```python
from pathlib import Path

Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
```
Entire block deleted (D-04). Write-time creation in `utils.py` is the sole creation point.

**Pydantic DTO pattern** (from `src/imagai/models.py` lines 1-32):
```python
from pydantic import BaseModel, Field
from typing import Optional, Literal

class ImageGenerationRequest(BaseModel):
    prompt: str
    engine: str
    output_filename: Optional[str] = None
    # ... fields with Field() for defaults and descriptions
```
The project convention: Pydantic `BaseModel` for all DTOs, `Field()` for defaults + descriptions, `Optional` for nullable fields.

---

### `src/imagai/web_server.py` (route, request-response)

**Analog:** `src/imagai/utils.py` (write-time mkdir pattern)

**REMOVED — import-time mkdir** (line 34):
```python
UPLOAD_FOLDER = Path("generated_images")
UPLOAD_FOLDER.mkdir(exist_ok=True)  # <-- DELETE THIS LINE
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
```
Delete line 34 only. The `UPLOAD_FOLDER` Path assignment and `app.config` assignment stay.

**Read-time guard pattern that stays** (lines 269, 322):
```python
if UPLOAD_FOLDER.exists():
    for img_file in UPLOAD_FOLDER.glob("*.png"):
```
This is the correct guard — no directory creation, just existence check before reading.

**404-when-absent pattern that stays** (lines 312-314):
```python
return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
```
`send_from_directory` returns 404 when the directory is absent — correct behavior when no images exist yet (D-05).

**Write-time mkdir pattern from `utils.py`** (lines 162, 196):
```python
output_path.parent.mkdir(parents=True, exist_ok=True)
```
This is the sole directory creation point after the phase. Both `save_image_from_url` and `save_image_from_b64` use this pattern — create parent directories at write time, not import time.

---

### `tests/test_config.py` (test, N/A)

**Analog:** `tests/test_cli.py`

**Test file structure** (lines 1-6):
```python
def test_app_version():
    assert True


def test_generate_help():
    assert True
```
The project uses bare pytest functions (no classes, no fixtures, no conftest.py). Tests are simple `def test_*():` functions with `assert` statements.

**Test conventions observed:**
- No test classes — flat functions
- No fixtures or setup/teardown
- No conftest.py exists
- pytest is the runner (`uv run pytest` per pyproject.toml dev dependency)
- Tests live in `tests/` directory with `__init__.py` present

**Expected test patterns for this phase** (from CONTEXT.md specifics):
- Fresh-process import test (subprocess) to verify no import side effects (CFG-02)
- Parametrized test over `.env.example` engine names to verify parsing (CFG-01)
- Test pinning the `__`-in-engine-name unsupported decision (D-02)

---

## Shared Patterns

### pydantic-settings v2 configuration
**Source:** `src/imagai/config.py` lines 17-23
**Apply to:** `src/imagai/config.py` (the Settings class)
```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="IMAGAI__",
        env_nested_delimiter="__",
        extra="ignore",
    )
```

### Pydantic DTO pattern
**Source:** `src/imagai/models.py` lines 1-21
**Apply to:** `src/imagai/config.py` (EngineConfig)
```python
from pydantic import BaseModel, Field
from typing import Optional

class ImageGenerationRequest(BaseModel):
    prompt: str
    engine: str
    output_filename: Optional[str] = None
    n: Optional[int] = Field(1, ge=1, le=10, description="Number of images to generate.")
```

### Write-time directory creation
**Source:** `src/imagai/utils.py` lines 162, 196
**Apply to:** All file-writing code (replaces import-time mkdir)
```python
output_path.parent.mkdir(parents=True, exist_ok=True)
```

### Settings singleton consumption
**Source:** `src/imagai/core.py` line 1, `src/imagai/cli.py` line 161, `src/imagai/utils.py` line 11, `src/imagai/web_server.py` line 25
**Apply to:** All modules that read engine configuration
```python
from imagai.config import settings
# ...
engine_config = settings.engines[engine_name]
```

### Engine lookup + error message pattern
**Source:** `src/imagai/core.py` lines 23-27
**Apply to:** Any code that looks up engines from settings
```python
if engine_name not in settings.engines:
    error_msg = f"Engine '{engine_name}' not configured. Available engines: {list(settings.engines.keys())}"
    logger.error(error_msg)
    return [ImageGenerationResponse(error=error_msg)]
```

## No Analog Found

All files have analogs. The `tests/test_config.py` file is new but follows the exact pattern of `tests/test_cli.py`.

## Metadata

**Analog search scope:** `src/imagai/`, `tests/`, `.env.example`, `pyproject.toml`
**Files scanned:** 8
**Pattern extraction date:** 2026-10-04
