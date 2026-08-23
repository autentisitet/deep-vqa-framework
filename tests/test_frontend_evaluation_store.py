import asyncio
import hashlib

import pytest

api = pytest.importorskip("deploy.api")


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


def test_sqlite_connection_sets_explicit_busy_timeout_and_page_limit(monkeypatch, tmp_path) -> None:
    database = tmp_path / "evaluations.sqlite3"
    monkeypatch.setattr(api, "FRONTEND_EVALUATION_DB_PATH", database)
    monkeypatch.setattr(api, "SQLITE_BUSY_TIMEOUT_SECONDS", 7.5)
    monkeypatch.setattr(api, "SQLITE_MAX_DATABASE_BYTES", 1024 * 1024)

    connection = api._frontend_evaluation_connection()
    try:
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 7500
        page_size = connection.execute("PRAGMA page_size").fetchone()[0]
        max_pages = connection.execute("PRAGMA max_page_count").fetchone()[0]
        assert max_pages * page_size <= 1024 * 1024 + page_size
    finally:
        connection.close()


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
        store.append(api.EvaluationRecord(
            session_id="session", timestamp="now", file_name="bad.jpg", file_hash="bad",
            task_type="iqa", model_used="model", mos_min=0, mos_max=5, inference_time_ms=1,
        ))


def test_evaluation_store_can_be_disabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "EVALUATION_STORE_BACKEND", "none")
    media = tmp_path / "sample.jpg"
    media.write_bytes(b"media")

    api._append_frontend_evaluation_log(_response(), media)

    assert api._read_frontend_evaluation_log(10, "any-session") == []
    assert not (tmp_path / "evaluations.sqlite3").exists()


def test_upload_copy_enforces_size_limit_and_read_timeout(monkeypatch, tmp_path) -> None:
    class FakeUpload:
        def __init__(self, chunks):
            self.chunks = iter(chunks)

        async def read(self, _size):
            return next(self.chunks, b"")

    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 3)
    with pytest.raises(api.UploadLimitError):
        asyncio.run(api._save_upload_to_temp(FakeUpload([b"12", b"345"]), ".jpg"))

    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)
    monkeypatch.setattr(api, "UPLOAD_READ_TIMEOUT_SECONDS", 0.001)

    class SlowUpload:
        async def read(self, _size):
            await asyncio.sleep(0.01)
            return b"1"

    with pytest.raises(api.UploadTimeoutError):
        asyncio.run(api._save_upload_to_temp(SlowUpload(), ".jpg"))
