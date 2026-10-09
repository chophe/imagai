"""SEC-03/05/06/07/08 web server safety tests."""

import base64
from pathlib import Path

B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

# ── SEC-03/SEC-06: server startup ordering and bind defaults ──


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
