"""Account and session storage for the ``accounts`` authentication mode.

Passwords are hashed with scrypt from the standard library: it is memory-hard,
available without adding a dependency, and the stored format records its
parameters so a later migration to argon2id can still verify old hashes.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .exceptions import AccountAlreadyExistsError, AccountNotFoundError, AccountValidationError
from .sqlite import SQLiteDatabase

EMAIL_PATTERN = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,189}\.[A-Za-z]{2,}$")
MIN_PASSWORD_LENGTH = 8
# Bounded so an unauthenticated caller cannot trigger unbounded scrypt work.
MAX_PASSWORD_LENGTH = 128

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_DKLEN = 32
_SALT_BYTES = 16

# Verification still runs for unknown accounts so login timing does not reveal
# whether an email is registered.
_DUMMY_HASH = (
    f"scrypt$16384$8$1${base64.b64encode(b'0123456789abcdef').decode()}${base64.b64encode(b'0' * _DKLEN).decode()}"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
) STRICT;
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id);
"""


def normalize_email(value: str) -> str:
    """Return a canonical email or raise ``AccountValidationError``."""
    email = value.strip().lower()
    if not EMAIL_PATTERN.fullmatch(email):
        raise AccountValidationError("Enter a valid email address")
    return email


def validate_password(value: str) -> str:
    """Return the password or raise ``AccountValidationError`` when outside policy."""
    if len(value) < MIN_PASSWORD_LENGTH:
        raise AccountValidationError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(value) > MAX_PASSWORD_LENGTH:
        raise AccountValidationError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters")
    return value


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """Derive a self-describing scrypt hash for storage."""
    salt = salt or os.urandom(_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_DKLEN,
    )
    return "$".join(
        (
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(derived).decode("ascii"),
        )
    )


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verification of a password against a stored hash."""
    if len(password) > MAX_PASSWORD_LENGTH:
        return False
    try:
        scheme, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64, validate=True)
        expected = base64.b64decode(digest_b64, validate=True)
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
    except (ValueError, TypeError, MemoryError):
        return False
    return hmac.compare_digest(derived, expected)


def _session_token_hash(token: str) -> str:
    """Return the non-reversible digest used to store a session token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AccountStore:
    """SQLite-backed users and sessions with the same limits as history storage."""

    def __init__(self, path: Path, max_bytes: int, busy_timeout_seconds: float):
        self.path = path
        self.max_bytes = max_bytes
        self.busy_timeout_seconds = busy_timeout_seconds
        self._db = SQLiteDatabase(max_bytes, busy_timeout_seconds)

    def connection(self) -> sqlite3.Connection:
        """Return the shared connection for this store's database path."""
        self._db.max_bytes = self.max_bytes
        self._db.busy_timeout_seconds = self.busy_timeout_seconds
        return self._db.connection(self.path)

    def initialize(self) -> None:
        """Create the account schema and migrate it once, if needed."""
        self._db.max_bytes = self.max_bytes
        self._db.busy_timeout_seconds = self.busy_timeout_seconds
        if self._db.is_initialized(self.path):
            return
        with self.connection() as connection:
            connection.executescript(SCHEMA)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(users)")}
            if "disabled" not in columns:
                connection.execute("ALTER TABLE users ADD COLUMN disabled INTEGER NOT NULL DEFAULT 0")
        self._db.mark_initialized(self.path)

    def register(self, email: str, password: str) -> tuple[str, str]:
        """Create an account and return ``(user_id, email)``."""
        canonical = normalize_email(email)
        validate_password(password)
        user_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            with self.connection() as connection:
                connection.execute(
                    "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (user_id, canonical, hash_password(password), created_at),
                )
        except sqlite3.IntegrityError as exc:
            raise AccountAlreadyExistsError("This email is already registered") from exc
        return user_id, canonical

    def authenticate(self, email: str, password: str) -> tuple[str, str] | None:
        """Return ``(user_id, canonical_email)`` for valid credentials."""
        try:
            canonical = normalize_email(email)
        except ValueError:
            # Run the same work as a real check so a malformed address is not
            # distinguishable from a wrong password by response time.
            verify_password(password, _DUMMY_HASH)
            return None
        with self.connection() as connection:
            row = connection.execute(
                "SELECT id, password_hash, disabled FROM users WHERE email = ?",
                (canonical,),
            ).fetchone()
        if row is None:
            verify_password(password, _DUMMY_HASH)
            return None
        if not verify_password(password, row["password_hash"]) or bool(row["disabled"]):
            return None
        return str(row["id"]), canonical

    def set_disabled(self, email: str, disabled: bool) -> bool:
        """Enable or disable an account; disabling also revokes its sessions."""
        canonical = normalize_email(email)
        self.initialize()
        with self.connection() as connection:
            cursor = connection.execute("UPDATE users SET disabled = ? WHERE email = ?", (int(disabled), canonical))
            if cursor.rowcount == 0:
                raise AccountNotFoundError("Account not found")
            if disabled:
                connection.execute(
                    "DELETE FROM sessions WHERE user_id = (SELECT id FROM users WHERE email = ?)", (canonical,)
                )
        return disabled

    def reset_password(self, email: str, password: str) -> None:
        """Replace an account's password and revoke all of its sessions."""
        canonical = normalize_email(email)
        validate_password(password)
        self.initialize()
        with self.connection() as connection:
            cursor = connection.execute(
                "UPDATE users SET password_hash = ? WHERE email = ?", (hash_password(password), canonical)
            )
            if cursor.rowcount == 0:
                raise AccountNotFoundError("Account not found")
            connection.execute(
                "DELETE FROM sessions WHERE user_id = (SELECT id FROM users WHERE email = ?)", (canonical,)
            )

    def create_session(self, user_id: str, max_age_seconds: int) -> str:
        """Create a session and return the raw token (only its digest is stored)."""
        token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (
                    _session_token_hash(token),
                    user_id,
                    now.isoformat(),
                    (now + timedelta(seconds=max_age_seconds)).isoformat(),
                ),
            )
        return token

    def resolve_session(self, token: str) -> tuple[str, str] | None:
        """Return ``(user_id, email)`` for a live session token, or ``None``."""
        if not token:
            return None
        digest = _session_token_hash(token)
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT sessions.user_id AS user_id, users.email AS email "
                "FROM sessions JOIN users ON users.id = sessions.user_id "
                "WHERE sessions.token_hash = ? AND sessions.expires_at > ?",
                (digest, now),
            ).fetchone()
            if row is None:
                connection.execute("DELETE FROM sessions WHERE token_hash = ?", (digest,))
                return None
        return str(row["user_id"]), str(row["email"])

    def delete_session(self, token: str) -> None:
        """Delete one session row by its raw token."""
        if not token:
            return
        with self.connection() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (_session_token_hash(token),))

    def purge_expired_sessions(self) -> int:
        """Delete every expired session and return how many were removed."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as connection:
            cursor = connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            return cursor.rowcount or 0


__all__ = [
    "AccountStore",
    "hash_password",
    "normalize_email",
    "validate_password",
    "verify_password",
]
