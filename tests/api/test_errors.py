"""Error responses are stable across domain, validation and routing failures."""

import asyncio
import json
from pathlib import Path

from deploy.web import api

ROOT = Path(__file__).resolve().parents[2]


def request(method: str, path: str, body: bytes = b"", headers=(), *, server_error: bool = False):
    """Exercise the ASGI stack without starting stores or loading models."""
    messages = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "root_path": "",
        "query_string": b"",
        "headers": list(headers),
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }
    try:
        asyncio.run(api.app(scope, receive, send))
    except Exception:
        # Starlette sends the 500 response and then re-raises for server logging.
        if not server_error:
            raise
    start = next(message for message in messages if message["type"] == "http.response.start")
    content = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return start["status"], dict(start["headers"]), json.loads(content)


def test_unknown_model_has_stable_code():
    status, _, data = request("GET", "/v1/models/missing")
    assert status == 404
    assert data == {"code": 1100, "error": "Model 'missing' is not configured", "detail": None}


def test_api_trailing_slash_does_not_redirect_or_change_method():
    status, headers, data = request("GET", "/v1/health/")
    assert status == 404
    assert b"location" not in headers
    assert data["code"] == 1400


def test_unknown_route_uses_error_envelope():
    status, _, data = request("GET", "/v1/missing")
    assert status == 404
    assert data == {"code": 1400, "error": "Not Found", "detail": None}


def test_method_not_allowed_preserves_allow_header():
    status, headers, data = request("DELETE", "/v1/health")
    assert status == 405
    assert headers[b"allow"] == b"GET"
    assert data["code"] == 1402


def test_validation_response_does_not_echo_password(monkeypatch):
    monkeypatch.setattr(api, "AUTH_MODE", "accounts")
    secret = "sensitive-password"
    body = json.dumps({"email": "new@example.com", "password": {"secret": secret}}).encode()
    status, _, data = request("POST", "/v1/auth/register", body, [(b"content-type", b"application/json")])
    assert status == 422
    assert data["code"] == 1200
    assert data["error"] == "Request validation failed"
    assert "body.password" in data["detail"]
    assert secret not in json.dumps(data)


def test_auth_required_and_cross_origin_have_distinct_codes(monkeypatch):
    monkeypatch.setattr(api, "AUTH_MODE", "api_key")
    status, response_headers, data = request("GET", "/v1/frontend-evaluations")
    assert (status, data["code"]) == (401, 1300)
    assert response_headers[b"www-authenticate"] == b'Session realm="Deep-VQA"'
    status, _, data = request(
        "POST",
        "/v1/evaluations",
        headers=[(b"origin", b"http://untrusted.example"), (b"content-type", b"multipart/form-data; boundary=media")],
        body=b'--media\r\nContent-Disposition: form-data; name="file"; filename="test.png"\r\n\r\npng\r\n--media--\r\n',
    )
    assert (status, data["code"]) == (403, 1302)


def test_registration_conflict_and_invalid_login_keep_their_codes(monkeypatch, tmp_path):
    from deploy.web.accounts import AccountStore

    store = AccountStore(tmp_path / "accounts.sqlite3", 1024 * 1024, 5)
    store.initialize()
    monkeypatch.setattr(api, "AUTH_MODE", "accounts")
    monkeypatch.setattr(api, "_account_store", store)
    api._register_attempts.clear()
    api._login_attempts.clear()
    headers = [(b"content-type", b"application/json")]
    body = json.dumps({"email": "known@example.com", "password": "longpassword"}).encode()
    try:
        status, _, data = request("POST", "/v1/auth/register", body, headers)
        assert status == 201
        assert data["authenticated"] is True
        status, _, data = request("POST", "/v1/auth/register", body, headers)
        assert (status, data["code"]) == (422, 1202)
        failures = []
        for email in ("known@example.com", "unknown@example.com"):
            body = json.dumps({"email": email, "password": "wrongpassword"}).encode()
            status, _, data = request("POST", "/v1/auth/login", body, headers)
            failures.append((status, data))
        assert failures[0] == failures[1]
        assert failures[0][0] == 401
        assert failures[0][1]["code"] == 1301
    finally:
        api._register_attempts.clear()
        api._login_attempts.clear()


def test_rate_limit_response_preserves_retry_after(monkeypatch):
    monkeypatch.setattr(api, "AUTH_MODE", "api_key")
    monkeypatch.setattr(api, "LOGIN_ATTEMPTS", 1)
    api._login_attempts.clear()
    headers = [(b"content-type", b"application/json")]
    try:
        request("POST", "/v1/auth/login", b'{"api_key":"wrong"}', headers)
        status, headers, data = request("POST", "/v1/auth/login", b'{"api_key":"wrong"}', headers)
        assert (status, data["code"]) == (429, 1303)
        assert int(headers[b"retry-after"]) >= 1
    finally:
        api._login_attempts.clear()


def test_unexpected_failure_is_redacted(monkeypatch):
    def fail(_model_id):
        raise RuntimeError("private filesystem and credentials")

    monkeypatch.setattr(api, "_model_summary", fail)
    status, _, data = request("GET", "/v1/models", server_error=True)
    assert status == 500
    assert data == {"code": 1000, "error": "Internal server error", "detail": None}


def test_artifact_absence_and_other_owner_have_identical_responses(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "AUTH_MODE", "none")
    monkeypatch.setattr(api, "EVALUATION_STORE_BACKEND", "sqlite")
    monkeypatch.setattr(api, "_generated_file_root", lambda: tmp_path)
    artifact = tmp_path / "viz-other" / "gradcam.png"
    artifact.parent.mkdir()
    artifact.write_bytes(b"png")
    api._write_artifact_owner(artifact.parent, "different-owner")
    headers = [(b"cookie", f"{api.CLIENT_SESSION_COOKIE}=browser-session".encode())]
    first = request("GET", "/v1/files/viz-missing/gradcam.png", headers=headers)
    second = request("GET", "/v1/files/viz-other/gradcam.png", headers=headers)
    assert first[0] == second[0] == 404
    assert first[2] == second[2]
    assert first[2]["code"] == 1401


def test_business_error_codes_do_not_overlap_http_status_range():
    from deploy.web.exceptions import ErrorCode

    assert all(code.value < 100 or code.value > 599 for code in ErrorCode)
    assert len({code.value for code in ErrorCode}) == len(ErrorCode)


def test_generated_openapi_matches_runtime():
    schema = api.app.openapi()
    assert schema == json.loads((ROOT / "docs/openapi.json").read_text())
    assert "code" in schema["components"]["schemas"]["ErrorResponse"]["required"]
    response = schema["paths"]["/v1/auth/login"]["post"]["responses"]["422"]
    assert response["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorResponse")


def _upload_body(payload):
    return (
        b'--media\r\nContent-Disposition: form-data; name="file"; filename="test.png"\r\n'
        b"Content-Type: image/png\r\n\r\n" + payload + b"\r\n--media--\r\n"
    )


def test_upload_failure_codes_precede_model_execution(monkeypatch):
    monkeypatch.setattr(api, "AUTH_MODE", "none")
    headers = [(b"content-type", b"multipart/form-data; boundary=media")]
    status, _, data = request("POST", "/v1/evaluations", _upload_body(b"plain text"), headers)
    assert (status, data["code"]) == (422, 1504)
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)
    status, _, data = request("POST", "/v1/evaluations", _upload_body(b"\x89PNG\r\n\x1a\n"), headers)
    assert (status, data["code"]) == (413, 1502)


def test_inference_failure_hides_underlying_exception(monkeypatch):
    def fail(_model_id):
        raise RuntimeError("private checkpoint path")

    monkeypatch.setattr(api, "AUTH_MODE", "none")
    monkeypatch.setattr(api, "_loaded_model", fail)
    headers = [(b"content-type", b"multipart/form-data; boundary=media")]
    status, _, data = request("POST", "/v1/evaluations", _upload_body(b"\x89PNG\r\n\x1a\n"), headers)
    assert status == 500
    assert data == {"code": 1600, "error": "Inference failed", "detail": None}


def test_upload_timeout_has_distinct_code_and_historical_status(monkeypatch):
    from deploy.web.exceptions import UploadTimeoutError

    async def timeout(*_args, **_kwargs):
        raise UploadTimeoutError("Upload read exceeded 1 seconds")

    monkeypatch.setattr(api, "AUTH_MODE", "none")
    monkeypatch.setattr(api, "_save_upload_to_temp", timeout)
    headers = [(b"content-type", b"multipart/form-data; boundary=media")]
    status, _, data = request("POST", "/v1/evaluations", _upload_body(b"\x89PNG\r\n\x1a\n"), headers)
    assert (status, data["code"]) == (413, 1503)


def test_visualization_failure_hides_underlying_exception(monkeypatch):
    def fail(_model_id):
        raise RuntimeError("private model data")

    monkeypatch.setattr(api, "AUTH_MODE", "none")
    monkeypatch.setattr(api, "EVALUATION_STORE_BACKEND", "sqlite")
    monkeypatch.setattr(api, "_loaded_model", fail)
    headers = [(b"content-type", b"multipart/form-data; boundary=media")]
    status, _, data = request("POST", "/v1/visualizations", _upload_body(b"\x89PNG\r\n\x1a\n"), headers)
    assert status == 500
    assert data == {"code": 1601, "error": "Visualization failed", "detail": None}
