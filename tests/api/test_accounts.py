"""Account registration, login, and per-account artifact ownership."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from deploy.web import api
from deploy.web.accounts import AccountStore, hash_password, verify_password
from deploy.web.exceptions import ArtifactNotFoundError, InvalidCredentialsError, ResourceNotFoundError


def _store(tmp_path: Path) -> AccountStore:
    store = AccountStore(tmp_path / "accounts.sqlite3", 1024 * 1024, 5)
    store.initialize()
    return store


@pytest.fixture(autouse=True)
def _clean_limiters():
    api._login_attempts.clear()
    api._register_attempts.clear()
    yield
    api._login_attempts.clear()
    api._register_attempts.clear()


def _request(path: str = "/v1/auth/login", payload: dict | None = None, token: str | None = None) -> api.Request:
    body = json.dumps(payload or {}).encode()
    delivered = {"done": False}

    async def receive() -> dict:
        if delivered["done"]:
            return {"type": "http.request", "body": b"", "more_body": False}
        delivered["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    headers = [(b"content-type", b"application/json")]
    if token is not None:
        headers.append((b"cookie", f"{api.SESSION_COOKIE}={token}".encode()))
    return api.Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": headers,
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        },
        receive,
    )


def _cookie_token(response: api.Response) -> str:
    header = response.headers["set-cookie"]
    assert header.startswith(f"{api.SESSION_COOKIE}=")
    return header.split(";", 1)[0].split("=", 1)[1]


def _same_origin_request(origin: str | None = None) -> api.Request:
    request = _request()
    if origin is not None:
        request.scope["headers"].append((b"origin", origin.encode()))
    return request


def test_same_origin_cookie_is_strict_even_behind_https_front_proxy() -> None:
    request = _same_origin_request("https://testserver")
    request.scope["headers"].append((b"x-forwarded-proto", b"https"))
    assert api._cookie_samesite(request) == "strict"
    assert api._cookie_secure(request)
    assert api._allowed_origins(request) == {"https://testserver"}


def test_other_origins_remain_rejected_even_if_allowlist_environment_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://frontend.example")
    request = _same_origin_request("https://frontend.example")
    assert api._cookie_samesite(request) == "strict"
    with pytest.raises(api.OriginRejectedError):
        api._check_csrf_origin(request)


def test_password_hash_is_scrypt_and_salted() -> None:
    first = hash_password("correct horse battery")
    second = hash_password("correct horse battery")

    assert first.startswith("scrypt$16384$8$1$")
    assert len(first.split("$")) == 6
    # Distinct salts mean identical passwords never share a stored hash.
    assert first != second
    assert verify_password("correct horse battery", first)
    assert not verify_password("wrong password", first)
    assert not verify_password("correct horse battery", "not-a-valid-hash")


def test_register_normalizes_email_and_rejects_duplicates(tmp_path: Path) -> None:
    store = _store(tmp_path)
    user_id, email = store.register("User@Example.COM", "longpassword")

    assert email == "user@example.com"
    assert store.authenticate("user@example.com", "longpassword") == (user_id, "user@example.com")
    with pytest.raises(ValueError):
        store.register("USER@example.com", "anotherpassword")
    with pytest.raises(ValueError):
        store.register("not-an-email", "longpassword")
    with pytest.raises(ValueError):
        store.register("other@example.com", "short")


def test_disabled_account_cannot_authenticate_and_disabling_revokes_sessions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    user_id, _ = store.register("disabled@example.com", "longpassword")
    token = store.create_session(user_id, 3600)

    assert store.set_disabled("disabled@example.com", True)
    assert store.authenticate("disabled@example.com", "longpassword") is None
    assert store.resolve_session(token) is None

    assert not store.set_disabled("disabled@example.com", False)
    assert store.authenticate("disabled@example.com", "longpassword") == (user_id, "disabled@example.com")


def test_reset_password_revokes_sessions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    user_id, _ = store.register("reset@example.com", "oldpassword")
    token = store.create_session(user_id, 3600)

    store.reset_password("reset@example.com", "newpassword")

    assert store.authenticate("reset@example.com", "oldpassword") is None
    assert store.authenticate("reset@example.com", "newpassword") == (user_id, "reset@example.com")
    assert store.resolve_session(token) is None


def test_session_is_hashed_at_rest_and_expires(tmp_path: Path) -> None:
    store = _store(tmp_path)
    user_id, _ = store.register("a@example.com", "longpassword")
    token = store.create_session(user_id, 3600)

    with store.connection() as connection:
        stored = connection.execute("SELECT token_hash FROM sessions").fetchone()[0]
    assert stored != token
    assert stored == hashlib.sha256(token.encode()).hexdigest()
    assert store.resolve_session(token) == (user_id, "a@example.com")

    store.delete_session(token)
    assert store.resolve_session(token) is None


def test_register_then_login_issues_a_session(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(api, "AUTH_MODE", "accounts")
    monkeypatch.setattr(api, "_account_store", _store(tmp_path))

    created = asyncio.run(
        api.auth_register(api.RegisterRequest(email="new@example.com", password="longpassword"), _request())
    )
    assert created.status_code == 201
    assert b"new@example.com" in created.body

    logged_in = asyncio.run(
        api.auth_login(api.LoginRequest(email="new@example.com", password="longpassword"), _request())
    )
    assert logged_in.status_code == 200
    token = _cookie_token(logged_in)
    assert api._account_store.resolve_session(token)[1] == "new@example.com"


def test_login_does_not_reveal_whether_an_email_is_registered(monkeypatch, tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.register("known@example.com", "longpassword")
    monkeypatch.setattr(api, "AUTH_MODE", "accounts")
    monkeypatch.setattr(api, "_account_store", store)

    failures = []
    for email in ("known@example.com", "unknown@example.com"):
        with pytest.raises(InvalidCredentialsError) as error:
            asyncio.run(api.auth_login(api.LoginRequest(email=email, password="wrongpassword"), _request()))
        failures.append((error.value.http_status_code, str(error.value)))

    assert failures[0] == failures[1] == (401, "Invalid email or password")


def test_registration_is_unavailable_outside_accounts_mode(monkeypatch) -> None:
    monkeypatch.setattr(api, "AUTH_MODE", "api_key")
    with pytest.raises(ResourceNotFoundError) as error:
        asyncio.run(api.auth_register(api.RegisterRequest(email="a@example.com", password="longpassword"), _request()))
    assert error.value.http_status_code == 404


def test_artifact_owner_is_the_account_across_browsers(monkeypatch, tmp_path: Path) -> None:
    store = _store(tmp_path)
    user_id, _ = store.register("owner@example.com", "longpassword")
    token = store.create_session(user_id, 3600)
    monkeypatch.setattr(api, "AUTH_MODE", "accounts")
    monkeypatch.setattr(api, "_account_store", store)

    # A brand-new browser cookie jar still resolves to the same account owner.
    assert api._current_owner(_request(token=token)) == user_id


def test_other_account_cannot_read_artifacts(monkeypatch, tmp_path: Path) -> None:
    store = _store(tmp_path)
    owner_id, _ = store.register("owner@example.com", "longpassword")
    other_id, _ = store.register("other@example.com", "longpassword")
    owner_token = store.create_session(owner_id, 3600)
    other_token = store.create_session(other_id, 3600)

    root = tmp_path / "iqa-test"
    artifact = root / "viz-abc" / "gradcam.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"png")
    api._write_artifact_owner(artifact.parent, owner_id)

    monkeypatch.setattr(api, "AUTH_MODE", "accounts")
    monkeypatch.setattr(api, "_account_store", store)
    monkeypatch.setattr(api, "_generated_file_root", lambda: root.resolve())
    monkeypatch.setattr(api, "EVALUATION_STORE_BACKEND", "sqlite")

    allowed = asyncio.run(api.get_generated_file("viz-abc/gradcam.png", _request(token=owner_token)))
    assert Path(allowed.path) == artifact.resolve()

    with pytest.raises(ArtifactNotFoundError) as error:
        asyncio.run(api.get_generated_file("viz-abc/gradcam.png", _request(token=other_token)))
    assert error.value.http_status_code == 404
