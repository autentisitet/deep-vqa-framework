"""Shared SQLite connection configuration and lifecycle.

Both the evaluation store and the account store need identical WAL /
synchronous / busy-timeout / size-cap setup. Keeping it in one module avoids
copy-paste drift and gives a single place to manage connection lifetime.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


def configure_connection(connection: sqlite3.Connection, max_bytes: int, busy_timeout_seconds: float) -> None:
    """Apply the shared PRAGMA settings and database size cap to a connection."""
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_seconds * 1000)}")
    page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
    max_pages = max(1, (max_bytes + page_size - 1) // page_size)
    if int(connection.execute("PRAGMA page_count").fetchone()[0]) > max_pages:
        connection.close()
        raise RuntimeError("SQLite database exceeds configured size limit")
    connection.execute(f"PRAGMA max_page_count = {max_pages}")


def open_sqlite(path: Path, max_bytes: int, busy_timeout_seconds: float) -> sqlite3.Connection:
    """Open and configure a new SQLite connection for ``path``."""
    if max_bytes <= 0 or busy_timeout_seconds <= 0:
        raise ValueError("SQLite limits must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=busy_timeout_seconds)
    configure_connection(connection, max_bytes, busy_timeout_seconds)
    return connection


class SQLiteDatabase:
    """One long-lived connection per path, opened lazily and reused.

    The returned connection is shared; callers must not close it. The
    application runs SQLite work on the single event-loop thread, so a shared
    connection is safe without a per-operation lock.
    """

    def __init__(self, max_bytes: int, busy_timeout_seconds: float):
        self.max_bytes = max_bytes
        self.busy_timeout_seconds = busy_timeout_seconds
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None
        self._path: Path | None = None
        self._initialized_path: Path | None = None

    def connection(self, path: Path) -> sqlite3.Connection:
        """Return the shared connection for ``path``, opening it lazily."""
        with self._lock:
            if self._connection is None or self._path != path:
                if self._connection is not None:
                    self._connection.close()
                self._connection = open_sqlite(path, self.max_bytes, self.busy_timeout_seconds)
                self._path = path
            return self._connection

    def mark_initialized(self, path: Path) -> None:
        """Record that ``path`` has its schema initialized."""
        self._initialized_path = path

    def is_initialized(self, path: Path) -> bool:
        """Whether ``path`` has been marked schema-initialized."""
        return self._initialized_path == path


__all__ = ["SQLiteDatabase", "configure_connection", "open_sqlite"]
