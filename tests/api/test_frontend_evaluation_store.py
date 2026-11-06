import asyncio
import hashlib

import pytest

api = pytest.importorskip("deploy.web.api")


def _response():
    return api.EvaluationResponse(
        evaluation_id="eval-test",
        filename="sample.jpg",
        model_id="iqa",
        media_type="image",
        task_type="iqa",
        model_name="test-model",
        backbone="test-backbone",
        dataset="TID2013",
        score=0.8,
        mos_score=4.2,
        mos_min=0.0,
        mos_max=5.0,
        media_size="16x16",
        latency_ms=12.5,
    )


def test_sqlite_store_round_trip_contains_expected_jsonl_fields(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "EVALUATION_STORE_BACKEND", "sqlite")
    database = tmp_path / "evaluations.sqlite3"
    legacy = tmp_path / "evaluations.jsonl"
    monkeypatch.setattr(api, "FRONTEND_EVALUATION_DB_PATH", database)
    monkeypatch.setattr(api, "LEGACY_FRONTEND_EVALUATION_LOG_PATH", legacy)

    media = tmp_path / "sample.jpg"
    media.write_bytes(b"media")
    api._append_frontend_evaluation_log(_response(), media)

    records = api._read_frontend_evaluation_log(10)

    assert len(records) == 1
    assert set(records[0]) == {
        "timestamp",
        "file_name",
        "file_hash",
        "task_type",
        "model_used",
        "mos_score",
        "mos_interval",
        "inference_time_ms",
    }
    assert records[0]["file_name"] == "sample.jpg"
    assert records[0]["file_hash"] == hashlib.sha256(b"media").hexdigest()
    assert records[0]["mos_interval"] == [0.0, 5.0]


def test_legacy_jsonl_is_imported_once(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "EVALUATION_STORE_BACKEND", "sqlite")
    database = tmp_path / "evaluations.sqlite3"
    legacy = tmp_path / "evaluations.jsonl"
    monkeypatch.setattr(api, "FRONTEND_EVALUATION_DB_PATH", database)
    monkeypatch.setattr(api, "LEGACY_FRONTEND_EVALUATION_LOG_PATH", legacy)
    legacy.write_text(
        '{"timestamp":"2026-08-21T00:00:00+00:00","file_name":"old.jpg",'
        '"file_hash":"abc","task_type":"iqa","model_used":"old-model",'
        '"mos_score":3.0,"mos_interval":[0.0,5.0],"inference_time_ms":9.0}\n',
        encoding="utf-8",
    )

    api._initialize_frontend_evaluation_store()
    api._initialize_frontend_evaluation_store()

    records = api._read_frontend_evaluation_log(10)
    assert len(records) == 1
    assert records[0]["file_name"] == "old.jpg"


def test_sqlite_connection_sets_explicit_busy_timeout_and_page_limit(tmp_path) -> None:
    store = api.SQLiteEvaluationStore(tmp_path / "evaluations.sqlite3", 1024 * 1024, 7.5)
    connection = store.connection()
    assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 7500
    page_size = connection.execute("PRAGMA page_size").fetchone()[0]
    max_pages = connection.execute("PRAGMA max_page_count").fetchone()[0]
    assert max_pages * page_size <= 1024 * 1024 + page_size


def test_sqlite_store_uses_wal_strict_schema_and_rejects_invalid_records(tmp_path) -> None:
    store = api.SQLiteEvaluationStore(tmp_path / "evaluations.sqlite3", 1024 * 1024, 5)
    store.initialize()
    with store.connection() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        schema = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", ("frontend_evaluations",)
        ).fetchone()[0]
        assert schema.upper().endswith("STRICT")
    with pytest.raises(Exception):
        store.append(
            api.EvaluationRecord(
                session_id="session",
                timestamp="now",
                file_name="bad.jpg",
                file_hash="bad",
                task_type="iqa",
                model_used="model",
                mos_min=0,
                mos_max=5,
                inference_time_ms=1,
            )
        )


def test_evaluation_store_can_be_disabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "EVALUATION_STORE_BACKEND", "none")
    media = tmp_path / "sample.jpg"
    media.write_bytes(b"media")

    api._append_frontend_evaluation_log(_response(), media)

    assert api._read_frontend_evaluation_log(10, "any-session") == []
    assert not (tmp_path / "evaluations.sqlite3").exists()


def test_frontend_settings_follow_selected_mode(monkeypatch) -> None:
    request = api.Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/v1/frontend-settings",
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        }
    )

    monkeypatch.setattr(api, "AUTH_MODE", "none")
    monkeypatch.setattr(api, "EVALUATION_STORE_BACKEND", "none")
    no_storage = asyncio.run(api.frontend_settings(request))
    assert no_storage.auth_required is False
    assert no_storage.authenticated is True
    assert no_storage.evaluation_history_enabled is False
    assert no_storage.visualizations_enabled is False

    monkeypatch.setattr(api, "AUTH_MODE", "api_key")
    monkeypatch.setattr(api, "EVALUATION_STORE_BACKEND", "sqlite")
    with_storage = asyncio.run(api.frontend_settings(request))
    assert with_storage.auth_required is True
    assert with_storage.authenticated is False
    assert with_storage.evaluation_history_enabled is True
    assert with_storage.visualizations_enabled is True


def test_auth_rate_limiter_rejects_excessive_attempts(monkeypatch) -> None:
    monkeypatch.setattr(api, "AUTH_MODE", "api_key")
    monkeypatch.setattr(api, "LOGIN_ATTEMPTS", 2)
    monkeypatch.setattr(api, "LOGIN_WINDOW_SECONDS", 900)
    api._login_attempts.clear()
    request = api.Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/auth/login",
            "headers": [(b"x-real-ip", b"192.0.2.10")],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        }
    )
    api._check_login_rate_limit(request)
    api._check_login_rate_limit(request)
    with pytest.raises(api.RateLimitError) as error:
        api._check_login_rate_limit(request)
    assert error.value.http_status_code == 429
    api._login_attempts.clear()


def test_generated_file_lookup_is_contained_and_direct(tmp_path) -> None:
    root = tmp_path / "reports"
    generated_file = root / "viz-1" / "gradcam.png"
    generated_file.parent.mkdir(parents=True)
    generated_file.write_bytes(b"png")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")

    assert api._find_generated_file(root.resolve(), "viz-1/gradcam.png") == generated_file.resolve()
    assert api._find_generated_file(root.resolve(), "../outside.png") is None
