"""SEC-01 rejection tests and SEC-02 containment proof tests."""

import asyncio
from pathlib import Path

from imagai.utils import (
    save_image_from_url,
    save_image_from_b64,
    _contained_path,
    generate_filename,
    generate_random_filename,
    generate_filename_from_prompt_llm,
)
from imagai.config import settings

B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def test_save_image_from_url_rejects_absolute_path():
    async def run():
        result = await save_image_from_url(
            "http://example.com/img.png", Path("/tmp/evil.png")
        )
        assert result is None
        assert not Path("/tmp/evil.png").exists()

    asyncio.run(run())


def test_save_image_from_url_rejects_traversal():
    async def run():
        result = await save_image_from_url(
            "http://example.com/img.png", Path("../../escape.png")
        )
        assert result is None
        resolved = Path("../../escape.png").resolve()
        assert not resolved.exists()

    asyncio.run(run())


def test_save_image_from_b64_rejects_absolute_path():
    async def run():
        result = await save_image_from_b64(B64, Path("/tmp/evil_b64.png"))
        assert result is None
        assert not Path("/tmp/evil_b64.png").exists()

    asyncio.run(run())


def test_save_image_from_b64_rejects_traversal():
    async def run():
        result = await save_image_from_b64(B64, Path("../../escape_b64.png"))
        assert result is None
        resolved = Path("../../escape_b64.png").resolve()
        assert not resolved.exists()

    asyncio.run(run())


def test_save_image_from_b64_rejects_subdirectory():
    async def run():
        result = await save_image_from_b64(B64, Path("sub/img.png"))
        assert result is None

    asyncio.run(run())


# ── SEC-02: Containment proof tests ──


def test_manual_filename_contained():
    assert _contained_path(Path(settings.output_dir) / "my_image.png")
    assert _contained_path(Path(settings.output_dir) / "my_image_2.png")


def test_prompt_derived_filename_contained():
    result = generate_filename(prompt="a cat sitting on a mat", extension="png")
    assert _contained_path(Path(settings.output_dir) / result)


def test_random_filename_contained():
    result = generate_random_filename(extension="png")
    assert _contained_path(Path(settings.output_dir) / result)


def test_llm_filename_contained():
    async def run():
        result = await generate_filename_from_prompt_llm(
            prompt="test prompt", extension="png"
        )
        assert _contained_path(Path(settings.output_dir) / result)

    asyncio.run(run())


def test_n_greater_than_1_variants_contained():
    assert _contained_path(Path(settings.output_dir) / "photo_2.png")
    assert _contained_path(Path(settings.output_dir) / "photo_3.png")


def test_happy_path_writes_to_output_dir():
    async def run():
        result = await save_image_from_b64(
            B64, Path(settings.output_dir) / "my_image.png"
        )
        assert result is not None
        assert (Path(settings.output_dir) / "my_image.png").exists()

    asyncio.run(run())


def test_web_server_upload_folder_unified():
    from imagai.web_server import UPLOAD_FOLDER

    assert UPLOAD_FOLDER == Path(settings.output_dir)


def test_absolute_output_dir_still_contains_basename():
    """CR-01 regression: an absolute output_dir must not cause every save to be rejected."""
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        abs_dir = str(Path(tmpdir) / "out")
        with patch.object(settings, "output_dir", abs_dir):
            candidate = Path(abs_dir) / "my_image.png"
            assert _contained_path(candidate) is True
            # Attack paths still rejected under an absolute output_dir
            assert _contained_path(Path("/tmp/evil.png")) is False
            assert _contained_path(Path(abs_dir) / "sub" / "img.png") is False
