#!/usr/bin/env python3
"""
Web server for Imagai - AI Image Generation Web Interface
Provides REST API endpoints to execute CLI commands from the web interface.
"""

import os
import sys
import json
import shlex
import subprocess
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import base64
import mimetypes

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from imagai.config import settings
from imagai.core import generate_image_core
from imagai.models import ImageGenerationRequest
from imagai.utils import _contained_path
from pydantic import ValidationError

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path(settings.output_dir)
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size


def _field_errors(ve: ValidationError) -> Dict[str, str]:
    """Render a pydantic ValidationError as a {field: message} map.

    Only the field name and message are echoed back. pydantic's `input`, `url`
    and `ctx` sub-keys carry the caller's own payload and internal schema URLs,
    and mirroring those at an untrusted caller is a disclosure (T-03-03).
    """
    errors: Dict[str, str] = {}
    for err in ve.errors():
        loc = err.get("loc") or ()
        key = str(loc[-1]) if loc else "_request"
        errors.setdefault(key, err.get("msg", "Invalid value"))
    return errors


@app.route("/")
def index():
    """Serve the main web interface"""
    # Resolve relative to this file so the server works from any CWD:
    # web_interface.html sits at the repo root, two levels above this package.
    html_path = Path(__file__).resolve().parents[2] / "web_interface.html"
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return (
            "Web interface file not found. Please ensure web_interface.html exists.",
            404,
        )


@app.route("/api/engines")
def get_engines():
    """Get available engines from configuration"""
    try:
        engines = []
        for engine_name, engine_config in settings.engines.items():
            engines.append(
                {
                    "name": engine_name,
                    "model": engine_config.model,
                    "base_url": engine_config.base_url,
                }
            )

        return jsonify(
            {
                "success": True,
                "engines": engines,
                "default_engine": settings.default_engine,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def generate_image():
    """Generate image using the core functionality"""
    try:
        data = request.get_json()

        if not data or "prompt" not in data:
            return jsonify({"success": False, "error": "Prompt is required"}), 400

        # Extract parameters from request
        prompt = data["prompt"]
        engine = data.get("engine") or settings.default_engine

        # Input image for image-to-image generation
        input_image = data.get("input_image")

        if not engine:
            return jsonify(
                {
                    "success": False,
                    "error": "No engine specified and no default engine configured",
                }
            ), 400

        if engine not in settings.engines:
            return jsonify(
                {"success": False, "error": f'Engine "{engine}" is not configured'}
            ), 400

        # Build extra parameters
        extra_params = {}
        param_mapping = {
            "negative_prompt": "negative_prompt",
            "seed": "seed",
            "strength": "strength",
            "output_format": "output_format",
            "aspect_ratio": "aspect_ratio",
            "mode": "mode",
        }

        for key, param_key in param_mapping.items():
            if key in data and data[key]:
                if key in ["seed"] and data[key]:
                    extra_params[param_key] = int(data[key])
                elif key in ["strength"] and data[key]:
                    extra_params[param_key] = float(data[key])
                else:
                    extra_params[param_key] = data[key]

        # Decode the input image once, before the request is built, so the
        # single build below is constructed with the decoded bytes.
        if input_image:
            if input_image.startswith("data:image/"):
                # Extract base64 data from the data URL
                encoded = input_image.split(",", 1)[1]
            else:
                # Assume it is already base64 encoded
                encoded = input_image
            # Add to extra_params for the core function
            extra_params["input_image"] = base64.b64decode(encoded)

        # Pass only the keys actually present. A `.get()` with a None fallback
        # would override the model's own defaults (e.g. n=1), so absent fields
        # keep pydantic's default while present ones arrive raw to be validated.
        validated = {
            key: data[key] for key in ("size", "n", "quality", "style") if key in data
        }

        # Create request object
        try:
            image_request = ImageGenerationRequest(
                prompt=prompt,
                engine=engine,
                output_filename=data.get("output"),
                response_format=data.get("response_format", "b64_json"),
                extra_params=extra_params,
                verbose=data.get("verbose", False),
                auto_filename=data.get("auto_filename", False),
                random_filename=data.get("random_filename", False),
                **validated,
            )
        except ValidationError as ve:
            # Malformed payload: a field-level 400, never the catch-all 500 and
            # never the provider.
            return jsonify({"success": False, "error": _field_errors(ve)}), 400

        # Reject an escaping output here, exactly where the save layer would.
        # Without this, a traversal attempt is silently absorbed downstream
        # (save returns None) and the caller never learns the path was refused.
        if image_request.output_filename and not _contained_path(
            Path(settings.output_dir) / image_request.output_filename
        ):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {
                            "output": "output must be a bare filename inside the configured output directory"
                        },
                    }
                ),
                400,
            )

        # Generate image using core functionality
        async def _generate():
            return await generate_image_core(image_request)

        results = asyncio.run(_generate())

        # Process results
        response_data = {
            "success": True,
            "results": [],
            "command": f"Generated using engine: {engine}",
        }

        for i, result in enumerate(results):
            result_data = {
                "index": i + 1,
                "success": not result.error,
                "error": result.error,
            }

            if not result.error:
                if result.saved_path:
                    result_data["saved_path"] = result.saved_path
                    # Try to encode image as base64 for preview
                    try:
                        if os.path.exists(result.saved_path):
                            with open(result.saved_path, "rb") as img_file:
                                img_data = img_file.read()
                                img_b64 = base64.b64encode(img_data).decode("utf-8")
                                mime_type = (
                                    mimetypes.guess_type(result.saved_path)[0]
                                    or "image/png"
                                )
                                result_data["image_data"] = (
                                    f"data:{mime_type};base64,{img_b64}"
                                )
                    except Exception as e:
                        result_data["preview_error"] = str(e)

                if result.image_url:
                    result_data["image_url"] = result.image_url

                if result.image_b64_json:
                    result_data["image_b64_json"] = result.image_b64_json

            response_data["results"].append(result_data)

        return jsonify(response_data)

    except Exception as e:
        return jsonify(
            {"success": False, "error": str(e), "type": type(e).__name__}
        ), 500


# Tokens that may precede the subcommand. Kept verbatim from the strings they
# replace at the old string-prefix guard.
ALLOWED_COMMAND_PREFIXES = (
    ("imagai",),
    ("uv", "run", "imagai"),
    ("python", "-m", "imagai"),
)
# The Typer app in cli.py exposes exactly these two subcommands.
ALLOWED_SUBCOMMANDS = ("generate", "list-engines")


def _allowed_argv(tokens: List[str]) -> bool:
    """True when tokens are an allowed prefix followed by an allowed subcommand.

    The compare is on argv tokens, not on the raw string, so `imagai; <cmd>` (whose
    first token matches an allowed prefix) is rejected exactly like any other
    non-allow-listed sequence — no shell ever sees the string (D-01, D-02).
    """
    for prefix in ALLOWED_COMMAND_PREFIXES:
        if len(tokens) > len(prefix) and tuple(tokens[: len(prefix)]) == prefix:
            if tokens[len(prefix)] in ALLOWED_SUBCOMMANDS:
                return True
    return False


def _png_snapshot() -> set:
    """The *.png files currently in UPLOAD_FOLDER, read at call time.

    Reading the module global here (rather than capturing it) lets a test patch
    `web_server.UPLOAD_FOLDER` and redirect the scan.
    """
    if UPLOAD_FOLDER.exists():
        return {p for p in UPLOAD_FOLDER.glob("*.png") if p.is_file()}
    return set()


# Serializes snapshot -> run -> diff so a concurrent request's file cannot land
# in the window this request is diffing (SEC-08). CONTEXT D-04 accepted that
# residual while the server was assumed single-tenant; the review demonstrated
# the race on a threaded server, so the accepted debt is paid here.
# Cost: concurrent /api/generate-cli calls now run sequentially, so one caller
# can wait up to the 300s subprocess timeout behind another.
# ponytail: global lock; a per-request output dir is the real fix, and needs
# CLI support the save layer does not have.
_GENERATE_CLI_LOCK = threading.Lock()


@app.route("/api/generate-cli", methods=["POST"])
def generate_image_cli():
    """Generate image using CLI command execution (alternative method)"""
    try:
        data = request.get_json()

        if not data or "command" not in data:
            return jsonify({"success": False, "error": "Command is required"}), 400

        command = data["command"]

        # JSON lets any type into this field. shlex.split on a non-string raises
        # AttributeError (not ValueError), which escapes the parse branch below
        # and reaches the outer handler as a 500 carrying the internal message.
        # A malformed field is the caller's mistake: 400, no subprocess.
        if not isinstance(command, str):
            return (
                jsonify({"success": False, "error": "Command must be a string"}),
                400,
            )

        # Tokenize without a shell, then gate on the resulting token sequence.
        try:
            tokens = shlex.split(command)
        except ValueError as e:
            return (
                jsonify({"success": False, "error": f"Could not parse command: {e}"}),
                400,
            )

        if not _allowed_argv(tokens):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Command is not in the allowed imagai command set",
                    }
                ),
                400,
            )

        # Execute command
        try:
            with _GENERATE_CLI_LOCK:
                # Snapshot before the run so the response can carry only the
                # files this request created.
                before = _png_snapshot()

                result = subprocess.run(
                    tokens,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    cwd=os.getcwd(),
                )

                response_data = {
                    "success": result.returncode == 0,
                    "command": command,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }

                # Try to find generated images
                if result.returncode == 0:
                    generated_images = []
                    for img_file in sorted(_png_snapshot() - before):
                        try:
                            with open(img_file, "rb") as f:
                                img_data = f.read()
                                img_b64 = base64.b64encode(img_data).decode("utf-8")
                                generated_images.append(
                                    {
                                        "filename": img_file.name,
                                        "path": str(img_file),
                                        "data": f"data:image/png;base64,{img_b64}",
                                    }
                                )
                        except Exception as e:
                            print(f"Error reading image {img_file}: {e}")

                    response_data["generated_images"] = generated_images

                return jsonify(response_data)

        except subprocess.TimeoutExpired:
            return jsonify(
                {"success": False, "error": "Command timed out after 5 minutes"}
            ), 408

        except Exception as e:
            return jsonify(
                {"success": False, "error": f"Command execution failed: {str(e)}"}
            ), 500

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/images/<filename>")
def serve_image(filename):
    """Serve generated images"""
    try:
        filename = secure_filename(filename)
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 404


@app.route("/api/images")
def list_images():
    """List all generated images"""
    try:
        images = []
        if UPLOAD_FOLDER.exists():
            for img_file in UPLOAD_FOLDER.glob("*"):
                if img_file.is_file() and img_file.suffix.lower() in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                ]:
                    stat = img_file.stat()
                    images.append(
                        {
                            "filename": img_file.name,
                            "size": stat.st_size,
                            "created": datetime.fromtimestamp(
                                stat.st_ctime
                            ).isoformat(),
                            "modified": datetime.fromtimestamp(
                                stat.st_mtime
                            ).isoformat(),
                            "url": f"/api/images/{img_file.name}",
                        }
                    )

        # Sort by creation time, newest first
        images.sort(key=lambda x: x["created"], reverse=True)

        return jsonify({"success": True, "images": images})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500


def main(
    host: str = "127.0.0.1", port: int = 5000, debug: bool = False, threaded: bool = True
):
    print("🎨 Starting Imagai Web Server...")
    print(f"📁 Generated images will be saved to: {UPLOAD_FOLDER.absolute()}")
    print(f"🔧 Available engines: {list(settings.engines.keys())}")
    print(f"🌐 Web interface will be available at: http://localhost:{port}")
    print("\n" + "=" * 50)

    app.run(host=host, port=port, debug=debug, threaded=threaded)


if __name__ == "__main__":
    main()
