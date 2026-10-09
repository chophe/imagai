"""SEC-05 command-execution tests and SEC-08 cross-request leak tests."""

import base64
from pathlib import Path

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
