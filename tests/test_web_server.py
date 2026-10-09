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
