# Phase 2: Path Containment - Pattern Map

**Mapped:** 2026-10-04
**Files analyzed:** 4
**Analogs found:** 4 / 4

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/imagai/utils.py` | utility | file-I/O | Same file — `save_image_from_url` / `save_image_from_b64` functions | exact (self) |
| `src/imagai/core.py` | service | request-response | Same file — filename-strategy block + error handling | exact (self) |
| `src/imagai/web_server.py` | controller | request-response | Same file — `UPLOAD_FOLDER` usage + `secure_filename` | exact (self) |
| `tests/test_cli.py` | test | N/A | Same file — existing pytest structure | exact (self) |

## Pattern Assignments

### `src/imagai/utils.py` (utility, file-I/O) — ENFORCEMENT BOUNDARY

**Analog:** Self — the save functions are the enforcement boundary per D-01.

**Imports pattern** (lines 1-13):
```python
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
from PIL import PngImagePlugin
```

**Existing filename sanitization pattern** (lines 18-23):
```python
def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a valid filename."""
    name = re.sub(r'[<>:"/\\\\|?*\\x00-\\x1F]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name[:100]
    return name
```

**Core save pattern — `save_image_from_url`** (lines 155-188):
```python
async def save_image_from_url(
    image_url: str, output_path: Path, prompt: str = None, model: str = None
) -> Optional[Path]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(io.BytesIO(response.content))
            ext = output_path.suffix[1:].lower()
            if prompt and model:
                img = _inject_metadata(img, prompt, model, ext)
            if ext == "png" and "pnginfo" in img.info:
                img.save(output_path, pnginfo=img.info["pnginfo"])
            elif ext in ("jpg", "jpeg") and "exif" in img.info:
                img.save(output_path, exif=img.info["exif"])
            else:
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
```

**Core save pattern — `save_image_from_b64`** (lines 191-215):
```python
async def save_image_from_b64(
    b64_json: str, output_path: Path, prompt: str = None, model: str = None
) -> Optional[Path]:
    try:
        image_bytes = base64.b64decode(b64_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(io.BytesIO(image_bytes))
            ext = output_path.suffix[1:].lower()
            if prompt and model:
                img = _inject_metadata(img, prompt, model, ext)
            if ext == "png" and "pnginfo" in img.info:
                img.save(output_path, pnginfo=img.info["pnginfo"])
            elif ext in ("jpg", "jpeg") and "exif" in img.info:
                img.save(output_path, exif=img.info["exif"])
            else:
                img.save(output_path)
            logger.info(f"Image saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to process and save b64 image to {output_path}: {e}")
            return None
    except Exception as e:
        logger.error(f"Error decoding or saving base64 image: {e}")
        return None
```

**Error handling pattern:** Both save functions return `None` on any failure (never raise). The containment check must follow this contract — return `None` on rejection, do not raise.

**Where to insert containment check:** Inside each save function, after the `try:` block opens but before `output_path.parent.mkdir(parents=True, exist_ok=True)`. The check validates `output_path` against `settings.output_dir` and returns `None` if the path escapes.

---

### `src/imagai/core.py` (service, request-response) — FILENAME STRATEGY + ERROR CONTRACT

**Analog:** Self — the filename-strategy block and error handling are the patterns to preserve.

**Imports pattern** (lines 1-14):
```python
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
```

**Filename strategy block** (lines 45-76):
```python
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
```

**Error handling pattern — rejection contract** (lines 94-99):
```python
            if saved_path:
                api_response.saved_path = str(saved_path)
            else:
                api_response.error = (
                    api_response.error or f"Failed to save image to {output_file_path}"
                )
```

**Pattern to preserve:** When `saved_path` is `None` (which happens when the containment check rejects), `api_response.error` is populated with a clear message. The containment rejection message should follow this same pattern — a descriptive string naming the rejection reason.

---

### `src/imagai/web_server.py` (controller, request-response) — UPLOAD_FOLDER UNIFICATION

**Analog:** Self — the `UPLOAD_FOLDER` usage and `secure_filename` pattern.

**Current UPLOAD_FOLDER definition** (lines 33-35):
```python
UPLOAD_FOLDER = Path("generated_images")
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
```

**Pattern to apply:** Replace `Path("generated_images")` with `Path(settings.output_dir)`. The `mkdir(exist_ok=True)` and `app.config` assignment remain unchanged. Import `settings` is already present at line 25.

**Existing security pattern — `secure_filename`** (lines 311-312):
```python
        filename = secure_filename(filename)
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
```

**UPLOAD_FOLDER usage locations to unify:**
- Line 33: `UPLOAD_FOLDER = Path("generated_images")` → `Path(settings.output_dir)`
- Line 34: `UPLOAD_FOLDER.mkdir(exist_ok=True)` — unchanged
- Line 35: `app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)` — unchanged
- Line 269: `if UPLOAD_FOLDER.exists():` — unchanged (now points to settings.output_dir)
- Line 270: `for img_file in UPLOAD_FOLDER.glob("*.png"):` — unchanged
- Line 312: `send_from_directory(app.config["UPLOAD_FOLDER"], filename)` — unchanged
- Line 322: `if UPLOAD_FOLDER.exists():` — unchanged
- Line 323: `for img_file in UPLOAD_FOLDER.glob("*"):` — unchanged
- Line 372: `print(f"📁 Generated images will be saved to: {UPLOAD_FOLDER.absolute()}")` — unchanged

**Pattern:** Only line 33 changes. All other references to `UPLOAD_FOLDER` automatically use the unified value.

---

### `tests/test_cli.py` (test, N/A) — CONTAINMENT TESTS

**Analog:** Self — the existing pytest function structure.

**Existing test pattern** (lines 1-6):
```python
def test_app_version():
    assert True


def test_generate_help():
    assert True
```

**Pattern to follow:** Simple pytest functions with descriptive names. New containment tests should:
- Use `pytest` (already the runner)
- Test the save functions directly with malicious paths
- Assert that rejected paths return `None` and no file is written outside the output directory
- Assert that valid paths still write correctly (happy path preserved)

**Suggested test structure:**
```python
import pytest
from pathlib import Path
from imagai.utils import save_image_from_url, save_image_from_b64
from imagai.config import settings

def test_save_image_from_url_rejects_absolute_path():
    # /tmp/evil.png → rejected, no file written
    pass

def test_save_image_from_url_rejects_traversal():
    # ../../escape.png → rejected, nothing written outside output_dir
    pass

def test_save_image_from_b64_rejects_absolute_path():
    # Same for b64 variant
    pass

def test_save_image_from_url_accepts_valid_filename():
    # my_image.png → written to settings.output_dir/my_image.png
    pass
```

---

## Shared Patterns

### Error Handling — Rejection Contract
**Source:** `src/imagai/utils.py` lines 155-215 (save functions) + `src/imagai/core.py` lines 94-99
**Apply to:** `src/imagai/utils.py` (containment check), `src/imagai/core.py` (error message)
```python
# utils.py: return None on rejection (never raise)
# core.py: set api_response.error with descriptive message
if saved_path:
    api_response.saved_path = str(saved_path)
else:
    api_response.error = (
        api_response.error or f"Failed to save image to {output_file_path}"
    )
```

### Path Construction
**Source:** `src/imagai/core.py` line 76
**Apply to:** `src/imagai/utils.py` (containment check must validate this path)
```python
output_file_path = Path(settings.output_dir) / current_filename
```

### Security — Filename Sanitization
**Source:** `src/imagai/utils.py` lines 18-23 (`sanitize_filename`) + `src/imagai/web_server.py` line 311 (`secure_filename`)
**Apply to:** `src/imagai/utils.py` (containment check should reject path separators, not just sanitize)
```python
# utils.py: sanitize replaces separators with underscore (existing)
# web_server.py: werkzeug secure_filename for serve path (existing)
# NEW: containment check must REJECT (not sanitize) any path with separators
```

### Configuration — Single Source of Truth
**Source:** `src/imagai/config.py` lines 25-27 (`settings.output_dir`)
**Apply to:** `src/imagai/web_server.py` (UPLOAD_FOLDER unification), `src/imagai/utils.py` (containment root)
```python
# config.py:
output_dir: str = Field(
    "generated_images", description="Default directory to save generated images."
)
```

## No Analog Found

All files have exact analogs (self-references). The containment check is new logic, but it follows established patterns in the same files.

## Metadata

**Analog search scope:** `src/imagai/`, `tests/`
**Files scanned:** 6 source files + 2 test files
**Pattern extraction date:** 2026-10-04
