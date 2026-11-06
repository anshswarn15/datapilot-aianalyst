"""Small local account store for the public demo login flow."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import sqlite3
import time
from pathlib import Path


class AuthError(ValueError):
    """Safe, user-facing authentication error."""


class AuthStore:
    def __init__(self, path: Path, session_secret: str):
        self.path = path
        self.session_secret = session_secret.encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )""")

    def _connection(self):
        return sqlite3.connect(self.path)

    @staticmethod
    def _validate(username: str, password: str) -> tuple[str, str]:
        username, password = username.strip(), password
        if not 3 <= len(username) <= 64 or not all(c.isalnum() or c in "._-" for c in username):
            raise AuthError("Use 3–64 letters, numbers, dots, hyphens, or underscores for the username.")
        if len(password) < 8:
            raise AuthError("Password must contain at least 8 characters.")
        return username, password

    @staticmethod
    def _hash(password: str, salt: bytes | None = None) -> str:
        salt = salt or os.urandom(16)
        digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
        return base64.b64encode(salt + digest).decode()

    @staticmethod
    def _matches(password: str, saved: str) -> bool:
        raw = base64.b64decode(saved.encode())
        return hmac.compare_digest(AuthStore._hash(password, raw[:16]), saved)

    def create_user(self, username: str, password: str) -> str:
        username, password = self._validate(username, password)
        try:
            with self._connection() as con:
                con.execute("INSERT INTO users VALUES (?, ?, ?)", (username, self._hash(password), int(time.time())))
        except sqlite3.IntegrityError:
            raise AuthError("That username is already in use.") from None
        return username

    def verify(self, username: str, password: str) -> str | None:
        with self._connection() as con:
            row = con.execute("SELECT username, password_hash FROM users WHERE username = ?", (username.strip(),)).fetchone()
        return row[0] if row and self._matches(password, row[1]) else None

    def bootstrap(self, username: str, password: str) -> None:
        if username and password and not self.verify(username, password):
            try:
                self.create_user(username, password)
            except AuthError:
                pass

    def issue_token(self, username: str) -> str:
        payload = f"{username}|{int(time.time()) + 12 * 60 * 60}"
        signature = hmac.new(self.session_secret, payload.encode(), hashlib.sha256).hexdigest()
        return base64.urlsafe_b64encode(f"{payload}|{signature}".encode()).decode()

    def verify_token(self, token: str | None) -> str | None:
        if not token:
            return None
        try:
            username, expiry, signature = base64.urlsafe_b64decode(token.encode()).decode().rsplit("|", 2)
            payload = f"{username}|{expiry}"
            expected = hmac.new(self.session_secret, payload.encode(), hashlib.sha256).hexdigest()
            return username if hmac.compare_digest(signature, expected) and int(expiry) >= time.time() else None
        except Exception:  # noqa: BLE001 - malformed cookies are simply anonymous
            return None
