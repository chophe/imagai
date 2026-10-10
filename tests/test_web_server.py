"""SEC-04 boundary rejection tests, plus the SEC-03/05/06/07/08 safety tests."""

import base64
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from imagai import web_server
from imagai.config import settings
from imagai.models import ImageGenerationRequest, ImageGenerationResponse

B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


@pytest.fixture
def client():
    """One shared Flask test client for the whole file."""
    return web_server.app.test_client()


@pytest.fixture
def engine_ready():
    """Make a configured engine resolve, so a payload reaches the schema gate.

    `generate_image()` checks engine presence/configuration *before* the
    pydantic gate. With `default_engine` unset every payload would 400 for the
    engine reason, so a field-rejection test would pass for the wrong reason
    and a config variance could masquerade as a pass. Patched once, here, and
    asserted once, here.
    """
    if not settings.engines:
        pytest.fail("no engines configured; every SEC-04 payload needs a valid one")
    name = settings.default_engine or next(iter(settings.engines))
    assert name in settings.engines, f"{name!r} is not a configured engine"
    with patch.object(settings, "default_engine", name):
        yield name


def _dispatch(seen):
    """A mocked provider that records the request it was handed and returns no images."""

    async def _dispatch_inner(request):
        seen.append(request)
        return []

    return _dispatch_inner


# ── SEC-04: server startup ordering and bind defaults ──


def test_main_defined_before_main_guard():
    """SEC-03: running the file as a script must not call main() before it exists."""
    source = (Path(__file__).resolve().parents[1] / "src" / "imagai" / "web_server.py").read_text()
    assert source.index("def main(") < source.index('if __name__ == "__main__":')


def test_main_defaults_loopback_no_debug():
    """SEC-06: a bare imagai-web binds loopback with the Werkzeug debugger off."""
    import inspect

    from imagai.web_server import main

    sig = inspect.signature(main)
    assert sig.parameters["host"].default == "127.0.0.1"
    assert sig.parameters["debug"].default is False


def test_script_mode_reaches_app_run(capsys):
    """SEC-03: running the file as __main__ reaches app.run and prints the banner.

    The pre-plan failure was NameError at the guard. The reach is asserted
    positively (a recorded call), so no bind failure can masquerade as success --
    and no real socket is opened, so the test is identical on a machine where
    port 5000 is taken.
    """
    import runpy
    import sys

    from flask.app import Flask

    calls = []
    original_run = Flask.run
    Flask.run = lambda self, host=None, port=None, **kwargs: calls.append(host)
    try:
        runpy.run_path("src/imagai/web_server.py", run_name="__main__")
    finally:
        Flask.run = original_run

    assert calls, "script raised before app.run was reached"
    assert calls[0] == "127.0.0.1"
    assert "Starting Imagai Web Server" in capsys.readouterr().out

# ── SEC-05: command execution runs argv-only ──


def test_generate_cli_rejects_shell_metacharacters():
    """A shell metacharacter must be rejected before any subprocess exists."""
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch

    from imagai import web_server
    from imagai.web_server import app

    client = app.test_client()
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(web_server, "UPLOAD_FOLDER", Path(tmpdir)):
            with patch.object(
                web_server.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            ) as m:
                resp = client.post(
                    "/api/generate-cli",
                    json={"command": "imagai; touch /tmp/pwned"},
                )

    assert resp.status_code == 400
    assert resp.get_json()["success"] is False
    assert m.called is False
    assert not Path("/tmp/pwned").exists()


def test_generate_cli_runs_argv_only():
    """An allow-listed command executes as a token list with shell=False."""
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch

    from imagai import web_server
    from imagai.web_server import app

    client = app.test_client()
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(web_server, "UPLOAD_FOLDER", Path(tmpdir)):
            with patch.object(
                web_server.subprocess,
                "run",
                side_effect=lambda argv, **kwargs: SimpleNamespace(
                    returncode=0, stdout="ok", stderr=""
                ),
            ) as m:
                resp = client.post(
                    "/api/generate-cli",
                    json={"command": "imagai generate -p a cat"},
                )

    assert resp.status_code == 200
    args, kwargs = m.call_args
    assert isinstance(args[0], list)
    assert args[0] == ["imagai", "generate", "-p", "a", "cat"]
    assert kwargs["shell"] is False

    body = resp.get_json()
    assert set(body) >= {
        "success",
        "command",
        "stdout",
        "stderr",
        "returncode",
    }


# ── SEC-05: allow-list edge cases on argv tokens ──


def test_generate_cli_rejects_metacharacters_after_allowed_prefix():
    """ROADMAP criterion 3: an allowed first token must not excuse a later `;`."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from imagai import web_server
    from imagai.web_server import app

    client = app.test_client()
    with patch.object(
        web_server.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
    ) as m:
        resp = client.post(
            "/api/generate-cli",
            json={"command": "imagai generate; touch /tmp/pwned"},
        )

    assert resp.status_code == 400
    assert resp.get_json()["success"] is False
    assert m.called is False
    assert not Path("/tmp/pwned").exists()


def test_generate_cli_rejects_disallowed_subcommand():
    """`imagai rm -rf /` matches an allowed prefix but not a subcommand."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from imagai import web_server
    from imagai.web_server import app

    client = app.test_client()
    with patch.object(
        web_server.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
    ) as m:
        resp = client.post(
            "/api/generate-cli", json={"command": "imagai rm -rf /"}
        )

    assert resp.status_code == 400
    assert resp.get_json()["success"] is False
    assert m.called is False


def test_generate_cli_rejects_non_allowlisted_prefix():
    """A command outside the three prefixes is rejected before any subprocess."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from imagai import web_server
    from imagai.web_server import app

    client = app.test_client()
    with patch.object(
        web_server.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
    ) as m:
        resp = client.post("/api/generate-cli", json={"command": "ls -la"})

    assert resp.status_code == 400
    assert resp.get_json()["success"] is False
    assert m.called is False


def test_generate_cli_rejects_unbalanced_quotes():
    """shlex.split failure must return a populated 400 body, never a raise."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from imagai import web_server
    from imagai.web_server import app

    client = app.test_client()
    with patch.object(
        web_server.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
    ) as m:
        resp = client.post(
            "/api/generate-cli", json={"command": 'imagai generate -p "unbalanced'}
        )

    assert resp.status_code == 400
    assert resp.get_json()["success"] is False
    assert m.called is False
    error = resp.get_json()["error"]
    assert isinstance(error, str) and error


def test_generate_cli_rejects_non_string_command():
    """A non-string `command` is a malformed field: 400, not a 500 leak.

    shlex.split raises AttributeError (not ValueError) on an int/list/dict, so
    before the isinstance guard the request escaped the inner `except ValueError`
    and the outer `except Exception` answered 500 with the raw internal message.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    from imagai import web_server
    from imagai.web_server import app

    client = app.test_client()
    for payload in (123, ["imagai", "generate"], {"a": 1}, None, True):
        with patch.object(
            web_server.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ) as m:
            resp = client.post("/api/generate-cli", json={"command": payload})

        assert resp.status_code == 400, payload
        body = resp.get_json()
        assert body["success"] is False
        assert body["error"] == "Command must be a string"
        assert m.called is False


def test_no_unused_flask_imports():
    """IN-03: every name web_server.py imports from flask must be referenced.

    `render_template_string` survived Phase 2.5's drop of flask_cors and was
    never called. An unused name in a security-reviewed module is lint noise
    that outlives its reason to exist, so the module is parsed and each imported
    flask name must still appear as a Name or attribute somewhere in the file.
    """
    import ast

    from imagai import web_server

    tree = ast.parse(Path(web_server.__file__).read_text())

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "flask":
            imported.update(alias.name for alias in node.names)

    used = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
    }

    assert imported == {
        "Flask",
        "request",
        "jsonify",
        "send_from_directory",
    }, f"flask import list changed unexpectedly: {sorted(imported)}"
    assert not (imported - used), f"unused flask imports: {sorted(imported - used)}"


# ── SEC-08: cross-request image leakage ──


def test_generate_cli_returns_only_new_files():
    """SEC-08: a pre-existing image in the output dir must not be returned."""
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch

    from imagai import web_server
    from imagai.config import settings
    from imagai.web_server import app

    png_bytes = base64.b64decode(B64)

    client = app.test_client()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "out"
        out.mkdir()
        (out / "old.png").write_bytes(png_bytes)

        def side_effect(argv, **kwargs):
            (out / "new_1.png").write_bytes(png_bytes)
            (out / "new_2.png").write_bytes(png_bytes)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch.object(web_server, "UPLOAD_FOLDER", out), patch.object(
            settings, "output_dir", str(out)
        ):
            with patch.object(web_server.subprocess, "run", side_effect=side_effect):
                resp = client.post(
                    "/api/generate-cli",
                    json={"command": "imagai generate -p a cat"},
                )

    assert resp.status_code == 200
    filenames = [img["filename"] for img in resp.get_json()["generated_images"]]
    assert "new_1.png" in filenames
    assert "new_2.png" in filenames
    assert "old.png" not in filenames


def test_generate_cli_concurrent_requests_return_only_their_own_images():
    """SEC-08: a concurrent request's file must not appear in this response.

    Request A is held inside its subprocess while request B writes its own
    image and completes -- the exact window the before/after set-diff races in.
    Before the _GENERATE_CLI_LOCK, A's diff observed B's file, so A's response
    carried B's image (the review's demonstrated defeat). With the lock, B
    cannot enter its own critical section until A leaves, so each response
    carries only its own file.

    The non-overlap assertion is the measured cost the lock pays: the two runs
    are strictly sequential, so concurrent callers queue rather than interleave.
    """
    import tempfile
    import threading
    import time
    from types import SimpleNamespace
    from unittest.mock import patch

    from imagai import web_server
    from imagai.config import settings
    from imagai.web_server import app

    png_bytes = base64.b64decode(B64)

    # A holds inside its subprocess until B's file is on disk, then proceeds so
    # A's diff runs after the foreign write. Without the lock that is the race:
    # A's diff observes B's file. With the lock B cannot enter its own critical
    # section until A leaves, so A's wait is released by the timeout instead.
    a_entered = threading.Event()
    b_wrote = threading.Event()

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "out"
        out.mkdir()
        responses = {}
        windows = []

        def side_effect(argv, **kwargs):
            tag = argv[3]  # `imagai generate -p <TAG>`
            started = time.monotonic()
            (out / f"from_request_{tag}.png").write_bytes(png_bytes)
            if tag == "A":
                a_entered.set()
                b_wrote.wait(timeout=2)
            else:
                b_wrote.set()
            windows.append((tag, started, time.monotonic()))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def run(tag):
            client = app.test_client()
            resp = client.post(
                "/api/generate-cli",
                json={"command": f"imagai generate -p {tag}"},
            )
            responses[tag] = resp

        with patch.object(web_server, "UPLOAD_FOLDER", out), patch.object(
            settings, "output_dir", str(out)
        ):
            with patch.object(web_server.subprocess, "run", side_effect=side_effect):
                a = threading.Thread(target=run, args=("A",))
                a.start()
                # Start B only once A is genuinely inside its subprocess, so
                # B's write always lands in A's run window.
                assert a_entered.wait(timeout=5), "A never entered its subprocess"
                b = threading.Thread(target=run, args=("B",))
                b.start()
                a.join(timeout=10)
                b.join(timeout=10)

        assert not a.is_alive() and not b.is_alive(), "a request thread hung"

        # Both images exist, so B's write really landed inside A's window.
        assert (out / "from_request_B.png").exists()
        assert (out / "from_request_A.png").exists()

        assert responses["A"].status_code == 200
        assert responses["B"].status_code == 200
        a_files = [
            img["filename"] for img in responses["A"].get_json()["generated_images"]
        ]
        b_files = [
            img["filename"] for img in responses["B"].get_json()["generated_images"]
        ]
        assert a_files == ["from_request_A.png"]
        assert b_files == ["from_request_B.png"]

        spans = {tag: (start, end) for tag, start, end in windows}
        (a_start, a_end), (b_start, b_end) = spans["A"], spans["B"]
        assert a_end <= b_start or b_end <= a_start, (
            "runs interleaved: A=[%r, %r] B=[%r, %r]" % (a_start, a_end, b_start, b_end)
        )


# ── SEC-07: no cross-origin grant ──


def test_no_cors_grant_on_cross_origin_post():
    """SEC-07: a POST carrying an Origin header must receive no CORS grant."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from imagai import web_server
    from imagai.web_server import app

    client = app.test_client()
    with patch.object(
        web_server.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
    ) as m:
        resp = client.post(
            "/api/generate-cli",
            json={"command": "ls -la"},
            headers={"Origin": "http://evil.example"},
        )

    # The command is rejected by the Plan 01 allow-list either way; the
    # assertion is on the header, which is a CORS fact independent of the
    # endpoint's status.
    assert resp.headers.get("Access-Control-Allow-Origin") is None


# ── ROADMAP criterion 5: the surface this phase did not touch ──


def test_generate_help_exits_0():
    """A pre-existing stub; now it actually proves `generate --help` exits 0."""
    from typer.testing import CliRunner

    from imagai.cli import app as cli_app

    result = CliRunner().invoke(cli_app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "prompt" in result.output.lower()


def test_api_generate_contract_unchanged():
    """`POST /api/generate` keeps its 400 contract -- this phase did not touch it."""
    from imagai.web_server import app

    client = app.test_client()
    resp = client.post("/api/generate", json={})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Prompt is required"


def test_cross_origin_preflight_receives_no_grant():
    """SEC-07: a preflight OPTIONS is the real drive-by probe -- no grant."""
    from imagai.web_server import app

    client = app.test_client()
    resp = client.options(
        "/api/generate-cli",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert resp.headers.get("Access-Control-Allow-Origin") is None


# ── SEC-04: one rejection per malformed field (T-03-05) ──
#
# Every payload below is otherwise valid (prompt + a configured engine via the
# `engine_ready` fixture), and dispatch is always mocked, so the only thing
# under test is the boundary's decision about that one field.


def test_api_generate_rejects_size_outside_enum(client, engine_ready):
    core = AsyncMock(return_value=[])
    with patch.object(web_server, "generate_image_core", core):
        resp = client.post("/api/generate", json={"prompt": "a cat", "size": "10x10"})

    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False
    assert "size" in body["error"]
    assert core.called is False


def test_api_generate_rejects_non_integer_n(client, engine_ready):
    """The ROADMAP-named ValueError case: a clean 400, not the old 500."""
    core = AsyncMock(return_value=[])
    with patch.object(web_server, "generate_image_core", core):
        resp = client.post("/api/generate", json={"prompt": "a cat", "n": "abc"})

    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False
    assert "n" in body["error"]
    assert core.called is False


def test_api_generate_rejects_n_below_minimum(client, engine_ready):
    core = AsyncMock(return_value=[])
    with patch.object(web_server, "generate_image_core", core):
        resp = client.post("/api/generate", json={"prompt": "a cat", "n": 0})

    assert resp.status_code == 400
    assert "n" in resp.get_json()["error"]
    assert core.called is False


def test_api_generate_rejects_n_above_maximum(client, engine_ready):
    core = AsyncMock(return_value=[])
    with patch.object(web_server, "generate_image_core", core):
        resp = client.post("/api/generate", json={"prompt": "a cat", "n": 11})

    assert resp.status_code == 400
    assert "n" in resp.get_json()["error"]
    assert core.called is False


def test_api_generate_accepts_n_at_both_bounds(client, engine_ready):
    """The bounds are pydantic's `ge=1, le=10` -- no new limit was introduced."""
    core = AsyncMock(return_value=[])
    with patch.object(web_server, "generate_image_core", core):
        for n in (1, 10):
            resp = client.post("/api/generate", json={"prompt": "a cat", "n": n})

            assert resp.status_code == 200, f"n={n} was schema-rejected"
            assert "n" not in (resp.get_json().get("error") or {})


def test_api_generate_rejects_quality_outside_enum(client, engine_ready):
    core = AsyncMock(return_value=[])
    with patch.object(web_server, "generate_image_core", core):
        resp = client.post("/api/generate", json={"prompt": "a cat", "quality": "ultra"})

    assert resp.status_code == 400
    assert "quality" in resp.get_json()["error"]
    assert core.called is False


def test_api_generate_rejects_style_outside_enum(client, engine_ready):
    core = AsyncMock(return_value=[])
    with patch.object(web_server, "generate_image_core", core):
        resp = client.post("/api/generate", json={"prompt": "a cat", "style": "neon"})

    assert resp.status_code == 400
    assert "style" in resp.get_json()["error"]
    assert core.called is False


def test_api_generate_rejects_response_format_outside_enum(client, engine_ready):
    core = AsyncMock(return_value=[])
    with patch.object(web_server, "generate_image_core", core):
        resp = client.post(
            "/api/generate", json={"prompt": "a cat", "response_format": "jpeg"}
        )

    assert resp.status_code == 400
    assert "response_format" in resp.get_json()["error"]
    assert core.called is False


def test_api_generate_rejects_non_string_prompt(client, engine_ready):
    """A present-but-malformed prompt is a field error, not the old 500."""
    core = AsyncMock(return_value=[])
    with patch.object(web_server, "generate_image_core", core):
        resp = client.post("/api/generate", json={"prompt": 12345})

    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False
    assert "prompt" in body["error"]
    assert core.called is False


def test_api_generate_rejects_absolute_output(client, engine_ready):
    core = AsyncMock(return_value=[])
    with patch.object(web_server, "generate_image_core", core):
        resp = client.post(
            "/api/generate", json={"prompt": "a cat", "output": "/tmp/evil.png"}
        )

    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False
    assert "output" in body["error"]
    assert core.called is False
    assert not Path("/tmp/evil.png").exists()


def test_api_generate_rejects_traversal_output(client, engine_ready, tmp_path):
    """`../../escape.png` is refused at the boundary and nothing is written."""
    out = tmp_path / "out"
    out.mkdir()
    core = AsyncMock(return_value=[])
    with patch.object(settings, "output_dir", str(out)):
        with patch.object(web_server, "generate_image_core", core):
            resp = client.post(
                "/api/generate", json={"prompt": "a cat", "output": "../../escape.png"}
            )

    assert resp.status_code == 400
    assert "output" in resp.get_json()["error"]
    assert core.called is False
    # The escape target sits inside the (fresh) temp root, so this is the
    # whole filesystem neighbourhood the traversal could have reached.
    assert not (tmp_path / "escape.png").exists()


def test_api_generate_rejects_subdirectory_output(client, engine_ready, tmp_path):
    """A subdirectory is outside the bare-basename contract, so it is refused."""
    out = tmp_path / "out"
    out.mkdir()
    core = AsyncMock(return_value=[])
    with patch.object(settings, "output_dir", str(out)):
        with patch.object(web_server, "generate_image_core", core):
            resp = client.post(
                "/api/generate", json={"prompt": "a cat", "output": "sub/img.png"}
            )

    assert resp.status_code == 400
    assert "output" in resp.get_json()["error"]
    assert core.called is False
    assert not (out / "sub").exists()


@pytest.mark.parametrize(
    "malformed",
    [
        {"size": "10x10"},
        {"n": "abc"},
        {"n": 0},
        {"n": 11},
        {"quality": "ultra"},
        {"style": "neon"},
        {"response_format": "jpeg"},
        {"output": "/tmp/evil.png"},
        {"output": "../x.png"},
    ],
)
def test_api_generate_malformed_payload_never_reaches_provider(
    client, engine_ready, malformed
):
    """The one CONTEXT sentence: no malformed payload crosses into the provider.

    The single proof is the mock: `generate_image_core` is replaced for the
    request, so a 400 here cannot have come from a live call -- and if the
    boundary ever stops rejecting, `called` turns True and this goes red.
    """
    core = AsyncMock(return_value=[])
    payload = {"prompt": "a cat"}
    payload.update(malformed)

    with patch.object(web_server, "generate_image_core", core):
        resp = client.post("/api/generate", json=payload)

    assert resp.status_code == 400
    assert resp.get_json()["success"] is False
    assert core.called is False


def test_api_generate_error_body_never_echoes_caller_input(client, engine_ready):
    """T-03-03: the field map carries messages only -- no payload, no schema URLs."""
    secret = "SEKRIT-PAYLOAD-9f3a"
    core = AsyncMock(return_value=[])
    with patch.object(web_server, "generate_image_core", core):
        resp = client.post(
            "/api/generate",
            json={"prompt": "a cat", "n": secret, "size": secret, "quality": secret},
        )

    assert resp.status_code == 400
    body = resp.get_data(as_text=True)
    error = resp.get_json()["error"]
    assert set(error) == {"n", "size", "quality"}
    for leak in (secret, "pydantic", "errors/", "input_value", "ctx", "http"):
        assert leak not in body, f"{leak!r} leaked into the 400 body: {body}"
    assert core.called is False


def test_api_generate_absent_fields_keep_model_defaults(client, engine_ready):
    """The only-present-keys splat: an absent `n` keeps pydantic's default of 1."""
    seen = []
    with patch.object(web_server, "generate_image_core", _dispatch(seen)):
        resp = client.post("/api/generate", json={"prompt": "a cat"})

    assert resp.status_code == 200
    assert seen[0].n == 1
    assert seen[0].size == "1024x1024"
    assert seen[0].response_format == "b64_json"


def test_api_generate_builds_the_request_once(client, engine_ready):
    """The duplicate build is really gone: one construction, fed decoded bytes."""
    built = []
    real = web_server.ImageGenerationRequest

    class Counting(real):
        def __init__(self, **kwargs):
            built.append(kwargs)
            super().__init__(**kwargs)

    with patch.object(web_server, "ImageGenerationRequest", Counting), patch.object(
        web_server, "generate_image_core", _dispatch([])
    ):
        resp = client.post(
            "/api/generate",
            json={"prompt": "a cat", "input_image": f"data:image/png;base64,{B64}"},
        )

    assert resp.status_code == 200
    assert len(built) == 1, "the request was built more than once"
    assert built[0]["extra_params"]["input_image"] == base64.b64decode(B64)


def test_api_generate_build_is_identical_before_and_after_input_image_decode(
    client, engine_ready
):
    """The collapsed build: with an image, the request is the plain one + bytes."""
    seen = []
    with patch.object(web_server, "generate_image_core", _dispatch(seen)):
        plain = client.post("/api/generate", json={"prompt": "a cat"})
        with_image = client.post(
            "/api/generate", json={"prompt": "a cat", "input_image": B64}
        )

    assert plain.status_code == 200
    assert with_image.status_code == 200
    assert len(seen) == 2

    without, with_img = seen
    assert (
        with_img.model_dump(exclude={"extra_params"})
        == without.model_dump(exclude={"extra_params"})
    )
    assert with_img.extra_params["input_image"] == base64.b64decode(B64)
    assert "input_image" not in (without.extra_params or {})


# ── SEC-04: the valid path is unchanged (T-03-06) ──


def test_api_generate_valid_request_returns_frozen_results_shape(
    client, engine_ready, tmp_path
):
    """A valid request still returns 200 with exactly the frozen result keys."""
    png = tmp_path / "happy.png"
    png.write_bytes(base64.b64decode(B64))

    async def dispatch(request):
        return [
            ImageGenerationResponse(
                saved_path=str(png),
                image_url="https://example.invalid/happy.png",
                image_b64_json=B64,
            )
        ]

    with patch.object(web_server, "UPLOAD_FOLDER", tmp_path), patch.object(
        settings, "output_dir", str(tmp_path)
    ), patch.object(web_server, "generate_image_core", dispatch):
        resp = client.post("/api/generate", json={"prompt": "a cat"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert set(body) == {"success", "results", "command"}

    assert len(body["results"]) == 1
    result = body["results"][0]
    assert set(result) == {
        "index",
        "success",
        "error",
        "saved_path",
        "image_data",
        "image_url",
        "image_b64_json",
    }
    assert result["index"] == 1
    assert result["success"] is True
    assert result["error"] is None
    assert result["saved_path"] == str(png)
    assert result["image_data"].startswith("data:image/png;base64,")
    assert result["image_url"] == "https://example.invalid/happy.png"
    assert result["image_b64_json"] == B64


# ── SEC-03 / Phase 4 boundary: the surface this phase did not touch ──


def test_no_422_for_malformed_input():
    """Every field rejection is a 400; pydantic's 422 is not the contract."""
    source = (Path(__file__).resolve().parents[1] / "src" / "imagai" / "web_server.py").read_text()
    assert "422" not in source


def test_generate_catch_all_500_body_is_untouched():
    """Phase 3 did not start Phase 4: the `str(e)` 500 body is still there."""
    source = (Path(__file__).resolve().parents[1] / "src" / "imagai" / "web_server.py").read_text()
    assert '"error": str(e), "type": type(e).__name__}' in source


# ── SEC-04: `seed`/`strength` are coerced outside the pydantic gate (CR-01) ──
#
# `seed` and `strength` are provider extra_params the request model does not
# carry, so the endpoint coerces them itself, ahead of the gate. `int()` and
# `float()` raise ValueError/TypeError whose message quotes the caller's own
# value -- that used to travel into the catch-all as a 500, which is the exact
# echo class T-03-03 forbids.


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", "abc"),
        ("seed", "3.5"),
        ("seed", [1]),
        ("seed", {"a": 1}),
        ("strength", "abc"),
        ("strength", [0.5]),
    ],
)
def test_api_generate_rejects_non_numeric_seed_or_strength(
    client, engine_ready, field, value
):
    """A non-numeric seed/strength is a field-level 400, never a 500."""
    core = AsyncMock(return_value=[])
    with patch.object(web_server, "generate_image_core", core):
        resp = client.post("/api/generate", json={"prompt": "a cat", field: value})

    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False
    assert field in body["error"]
    assert core.called is False


# ── SEC-04: `input_image` is decoded ahead of the gate (WR-01) ──
#
# The base64 decode runs before the request is built, so a malformed value has
# three distinct raise paths -- binascii.Error (a ValueError) on bad base64,
# IndexError on a data URL without a comma, and AttributeError on a non-string
# -- and all three quoted caller input on their way into the catch-all 500.


@pytest.mark.parametrize(
    "value",
    ["!!!x!!!", "data:image/png;base64", 12345, ["x"]],
)
def test_api_generate_rejects_malformed_input_image(client, engine_ready, value):
    """A malformed input_image is a field-level 400, never a 500."""
    core = AsyncMock(return_value=[])
    with patch.object(web_server, "generate_image_core", core):
        resp = client.post(
            "/api/generate", json={"prompt": "a cat", "input_image": value}
        )

    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False
    assert "input_image" in body["error"]
    assert core.called is False


def test_api_generate_still_decodes_a_valid_input_image(client, engine_ready):
    """The guard pins the malformed paths only; a valid decode is unchanged."""
    seen = []
    with patch.object(web_server, "generate_image_core", _dispatch(seen)):
        resp = client.post(
            "/api/generate",
            json={"prompt": "a cat", "input_image": f"data:image/png;base64,{B64}"},
        )

    assert resp.status_code == 200
    assert seen[0].extra_params["input_image"] == base64.b64decode(B64)


# ── SEC-04: `engine` is tested for membership before the gate (WR-02) ──
#
# `engine not in settings.engines` is one line above the new pydantic gate, so
# an unhashable engine (a JSON object or array) raised `unhashable type` and
# became a 500 that named the caller's own JSON type.


@pytest.mark.parametrize(
    "value,leaked",
    [({"a": 1}, "unhashable type"), (["a"], "unhashable type"), (1234567, "1234567")],
)
def test_api_generate_rejects_non_string_engine(client, engine_ready, value, leaked):
    """A non-string engine is a malformed field: 400 naming no internal detail.

    A hashable non-string (an int) is covered too: the membership test answers
    a 400 that quotes the caller's value, so the assertion is that neither the
    value nor the internal exception text reaches the body.
    """
    core = AsyncMock(return_value=[])
    with patch.object(web_server, "generate_image_core", core):
        resp = client.post("/api/generate", json={"prompt": "a cat", "engine": value})

    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False
    assert "engine" in body["error"]
    assert leaked not in resp.get_data(as_text=True)
    assert core.called is False


# ── SEC-04: an unreadable request body (WR-03) ──
#
# `request.get_json()` raises for a malformed body (Flask's own 400) and for a
# wrong Content-Type (415); inside the route's try both surfaced as the
# catch-all 500. A JSON *array* body is the same gap one layer down: the key
# test passes by membership and `data[key]` then raises TypeError.


@pytest.mark.parametrize("endpoint", ["/api/generate", "/api/generate-cli"])
@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            {"data": "{not json", "content_type": "application/json"},
            id="malformed-json",
        ),
        pytest.param(
            {
                "data": '{"prompt": "a cat", "command": "imagai list-engines"}',
                "content_type": "text/plain",
            },
            id="wrong-content-type",
        ),
        pytest.param({"json": ["prompt", "command"]}, id="json-array-body"),
    ],
)
def test_malformed_request_body_is_a_400_not_the_catch_all_500(
    client, endpoint, body
):
    """A body the boundary cannot read is the caller's 400, on either endpoint."""
    core = AsyncMock(return_value=[])
    with patch.object(web_server, "generate_image_core", core):
        resp = client.post(endpoint, **body)

    assert resp.status_code == 400
    payload = resp.get_json()
    assert payload["success"] is False
    assert "type" not in payload, "the catch-all 500 shape answered a malformed body"
    assert core.called is False
