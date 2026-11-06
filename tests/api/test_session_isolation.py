"""Anonymous session isolation for generated artifacts and evaluation history.

Every visitor of an internal deployment shares one API key, so authentication
alone cannot keep one visitor's artifacts away from another. These tests pin the
session boundary that closes that gap.
"""

import asyncio
import hashlib
from pathlib import Path

import pytest

from deploy.web import api
from deploy.web.exceptions import ArtifactNotFoundError


def _request(path: str = "/v1/files/viz-x/gradcam.png", token: str | None = None) -> api.Request:
    headers = []
    if token is not None:
        headers.append((b"cookie", f"{api.CLIENT_SESSION_COOKIE}={token}".encode()))
    return api.Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": headers,
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        }
    )


def _issued_token(response: api.Response) -> str:
    header = response.headers["set-cookie"]
    return header.split(";", 1)[0].split("=", 1)[1]


def _owner(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def test_session_owner_is_a_digest_and_is_stable_per_token() -> None:
    first = api._client_session_owner(_request(token="token-a"))
    assert first == _owner("token-a")
    assert first != "token-a"
    assert api._client_session_owner(_request(token="token-a")) == first
    assert api._client_session_owner(_request(token="token-b")) != first


def test_missing_session_has_no_owner() -> None:
    assert api._client_session_owner(_request()) is None


def test_session_is_issued_once_then_reused() -> None:
    response = api.Response()
    owner = api._ensure_client_session(_request(), response)
    token = _issued_token(response)
    assert _owner(token) == owner

    reuse = api.Response()
    assert api._ensure_client_session(_request(token=token), reuse) == owner
    # An existing session must not be replaced on every request.
    assert "set-cookie" not in reuse.headers


def test_artifact_owner_round_trip(tmp_path: Path) -> None:
    directory = tmp_path / "viz-x"
    api._write_artifact_owner(directory, "abc123")

    assert api._artifact_owner(directory) == "abc123"
    assert api._artifact_owner(tmp_path / "missing") is None


def _artifact_tree(tmp_path: Path, token: str) -> tuple[Path, Path]:
    root = tmp_path / "iqa-test"
    artifact = root / "viz-abc" / "gradcam.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"png")
    api._write_artifact_owner(artifact.parent, _owner(token))
    return root, artifact


def test_generated_file_is_visible_only_to_its_owning_session(monkeypatch, tmp_path: Path) -> None:
    root, artifact = _artifact_tree(tmp_path, "owner-token")
    monkeypatch.setattr(api, "_generated_file_root", lambda: root.resolve())
    monkeypatch.setattr(api, "AUTH_MODE", "none")
    monkeypatch.setattr(api, "EVALUATION_STORE_BACKEND", "sqlite")

    allowed = asyncio.run(api.get_generated_file("viz-abc/gradcam.png", _request(token="owner-token")))
    assert Path(allowed.path) == artifact.resolve()

    for intruder in ("other-token", None):
        with pytest.raises(ArtifactNotFoundError) as error:
            asyncio.run(api.get_generated_file("viz-abc/gradcam.png", _request(token=intruder)))
        # 404 rather than 403: do not confirm that another session's file exists.
        assert error.value.http_status_code == 404


def test_generated_file_rejects_the_owner_sidecar(monkeypatch, tmp_path: Path) -> None:
    root, _ = _artifact_tree(tmp_path, "owner-token")
    monkeypatch.setattr(api, "_generated_file_root", lambda: root.resolve())
    monkeypatch.setattr(api, "AUTH_MODE", "none")
    monkeypatch.setattr(api, "EVALUATION_STORE_BACKEND", "sqlite")

    with pytest.raises(ArtifactNotFoundError) as error:
        asyncio.run(api.get_generated_file("viz-abc/.owner", _request(token="owner-token")))
    assert error.value.http_status_code == 404


def test_history_is_scoped_to_the_server_issued_session(monkeypatch) -> None:
    monkeypatch.setattr(api, "AUTH_MODE", "none")
    monkeypatch.setattr(api, "EVALUATION_STORE_BACKEND", "sqlite")
    seen: list[str | None] = []

    def fake_read(limit: int, session_id: str | None = None) -> list[dict]:
        seen.append(session_id)
        return []

    monkeypatch.setattr(api, "_read_frontend_evaluation_log", fake_read)

    asyncio.run(api.list_frontend_evaluations(_request(token="token-a")))
    assert seen == [_owner("token-a")]

    # Without a session there is nothing to claim, so the store is not queried.
    seen.clear()
    asyncio.run(api.list_frontend_evaluations(_request()))
    assert seen == []


def test_expired_artifacts_are_purged_but_fresh_and_unrelated_dirs_survive(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "iqa-test"
    expired = root / "viz-expired"
    fresh = root / "viz-fresh"
    # The CLI writes into the same root; it must never be swept.
    cli_output = root / "visualizations"
    for directory in (expired, fresh, cli_output):
        directory.mkdir(parents=True)
        (directory / "gradcam.png").write_bytes(b"png")

    import os
    import time

    stale = time.time() - 8 * 86400
    os.utime(expired, (stale, stale))
    monkeypatch.setattr(api, "ARTIFACT_RETENTION_SECONDS", 7 * 86400)

    assert api._purge_expired_artifacts(root) == 1
    assert not expired.exists()
    assert fresh.exists()
    assert cli_output.exists()


def test_purge_is_bounded_and_ignores_symlinks(monkeypatch, tmp_path: Path) -> None:
    import os
    import time

    root = tmp_path / "iqa-test"
    root.mkdir()
    for index in range(3):
        directory = root / f"viz-old-{index}"
        directory.mkdir()
        stale = time.time() - 8 * 86400
        os.utime(directory, (stale, stale))

    target = tmp_path / "outside"
    target.mkdir()
    (root / "viz-link").symlink_to(target)

    monkeypatch.setattr(api, "ARTIFACT_RETENTION_SECONDS", 7 * 86400)
    monkeypatch.setattr(api, "ARTIFACT_SWEEP_LIMIT", 2)

    assert api._purge_expired_artifacts(root) == 2
    assert target.exists()
    assert (root / "viz-link").is_symlink()
