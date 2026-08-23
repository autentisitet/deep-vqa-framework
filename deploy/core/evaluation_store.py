"""Pluggable, validated persistence for frontend evaluation history."""

from __future__ import annotations

import json
import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=256)
    timestamp: str
    file_name: str = Field(min_length=1, max_length=512)
    file_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_type: str = Field(pattern=r"^(iqa|vqa)$")
    model_used: str = Field(min_length=1, max_length=256)
    mos_score: float | None = None
    mos_min: float
    mos_max: float
    inference_time_ms: float = Field(ge=0)


SCHEMA = """
CREATE TABLE IF NOT EXISTS frontend_evaluations (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL CHECK(length(session_id) BETWEEN 1 AND 256),
    timestamp TEXT NOT NULL,
    file_name TEXT NOT NULL CHECK(length(file_name) BETWEEN 1 AND 512),
    file_hash TEXT NOT NULL CHECK(length(file_hash) = 64),
    task_type TEXT NOT NULL CHECK(task_type IN ('iqa', 'vqa')),
    model_used TEXT NOT NULL,
    mos_score REAL,
    mos_min REAL NOT NULL,
    mos_max REAL NOT NULL CHECK(mos_max > mos_min),
    inference_time_ms REAL NOT NULL CHECK(inference_time_ms >= 0)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_frontend_evaluations_timestamp
    ON frontend_evaluations (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_frontend_evaluations_session_timestamp
    ON frontend_evaluations (session_id, timestamp DESC);
"""


class SQLiteEvaluationStore:
    def __init__(self, path: Path, max_bytes: int, busy_timeout_seconds: float):
        self.path = path
        self.max_bytes = max_bytes
        self.busy_timeout_seconds = busy_timeout_seconds

    def connection(self) -> sqlite3.Connection:
        if self.max_bytes <= 0 or self.busy_timeout_seconds <= 0:
            raise ValueError("SQLite limits must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=self.busy_timeout_seconds)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA foreign_keys = ON")
        busy_timeout_ms = int(self.busy_timeout_seconds * 1000)
        connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
        max_pages = max(1, (self.max_bytes + page_size - 1) // page_size)
        if int(connection.execute("PRAGMA page_count").fetchone()[0]) > max_pages:
            connection.close()
            raise RuntimeError("SQLite database exceeds configured size limit")
        connection.execute(f"PRAGMA max_page_count = {max_pages}")
        return connection

    def initialize(self, legacy_path: Path | None = None) -> None:
        with self.connection() as connection:
            existing = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                ("frontend_evaluations",),
            ).fetchone()
            if existing and "STRICT" not in (existing[0] or "").upper():
                connection.execute("ALTER TABLE frontend_evaluations RENAME TO frontend_evaluations_legacy")
            connection.executescript(SCHEMA)
            if existing and "STRICT" not in (existing[0] or "").upper():
                rows = connection.execute(
                    "SELECT session_id,timestamp,file_name,file_hash,task_type,model_used,mos_score,mos_min,mos_max,inference_time_ms FROM frontend_evaluations_legacy"
                ).fetchall()
                for row in rows:
                    try:
                        self._insert(connection, EvaluationRecord(**dict(row)))
                    except (TypeError, ValueError):
                        continue
                connection.execute("DROP TABLE frontend_evaluations_legacy")
            if legacy_path and not connection.execute("SELECT 1 FROM frontend_evaluations LIMIT 1").fetchone():
                self._import_legacy(connection, legacy_path)

    @staticmethod
    def _import_legacy(connection: sqlite3.Connection, path: Path) -> None:
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                raw = json.loads(line)
                interval = raw["mos_interval"]
                record = EvaluationRecord(
                    session_id="legacy", timestamp=raw["timestamp"], file_name=raw["file_name"],
                    file_hash=(raw["file_hash"] if len(str(raw["file_hash"])) == 64 else hashlib.sha256(str(raw["file_hash"]).encode()).hexdigest()),
                    task_type=raw["task_type"], model_used=raw["model_used"],
                    mos_score=raw.get("mos_score"), mos_min=float(interval[0]), mos_max=float(interval[1]),
                    inference_time_ms=float(raw["inference_time_ms"]),
                )
                SQLiteEvaluationStore._insert(connection, record)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    @staticmethod
    def _insert(connection: sqlite3.Connection, record: EvaluationRecord) -> None:
        connection.execute(
            """INSERT INTO frontend_evaluations
            (session_id,timestamp,file_name,file_hash,task_type,model_used,mos_score,mos_min,mos_max,inference_time_ms)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                record.session_id,
                record.timestamp,
                record.file_name,
                record.file_hash,
                record.task_type,
                record.model_used,
                record.mos_score,
                record.mos_min,
                record.mos_max,
                record.inference_time_ms,
            ),
        )

    def append(self, record: EvaluationRecord) -> None:
        with self.connection() as connection:
            self._insert(connection, record)

    def read(self, limit: int, session_id: str | None = None) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query = "SELECT timestamp,file_name,file_hash,task_type,model_used,mos_score,mos_min,mos_max,inference_time_ms FROM frontend_evaluations"
        params: tuple[Any, ...] = ()
        if session_id is not None:
            query += " WHERE session_id = ?"
            params = (session_id,)
        query += " ORDER BY id DESC LIMIT ?"
        params += (limit,)
        with self.connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                "timestamp": row["timestamp"],
                "file_name": row["file_name"],
                "file_hash": row["file_hash"],
                "task_type": row["task_type"],
                "model_used": row["model_used"],
                "mos_score": row["mos_score"],
                "mos_interval": [row["mos_min"], row["mos_max"]],
                "inference_time_ms": row["inference_time_ms"],
            }
            for row in rows
        ]
