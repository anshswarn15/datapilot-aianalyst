"""In-memory sessions: one per browser tab. Holds uploaded tables, SQL engine and chat history."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from app.config import Settings
from app.core.ingestion import IngestionError, LoadedTable, load_csv
from app.core.sql_engine import SQLEngine


class SessionNotFound(KeyError):
    pass


@dataclass
class Session:
    id: str
    settings: Settings
    engine: SQLEngine
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    tables: dict[str, LoadedTable] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)  # LLM-facing {"role","content"} pairs
    turns: list[dict] = field(default_factory=list)    # UI/export-facing records with artifacts
    lock: threading.Lock = field(default_factory=threading.Lock)  # one chat at a time per session

    def touch(self) -> None:
        self.last_used = time.time()

    def add_csv(self, filename: str, content: bytes) -> LoadedTable:
        if len(self.tables) >= self.settings.max_files_per_session:
            raise IngestionError(f"Limit of {self.settings.max_files_per_session} files per session reached.")
        table = load_csv(
            filename, content,
            max_bytes=self.settings.max_upload_bytes, max_rows=self.settings.max_rows_per_file,
            taken_names=set(self.tables),
        )
        self.tables[table.name] = table
        self.engine.register(table.name, table.df)
        return table

    def remove_table(self, name: str) -> None:
        if name not in self.tables:
            raise KeyError(name)
        del self.tables[name]
        self.engine.drop(name)

    def reset_chat(self) -> None:
        self.history.clear()
        self.turns.clear()


class SessionStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        self.cleanup()
        s = Session(
            id=uuid.uuid4().hex, settings=self.settings,
            engine=SQLEngine(self.settings.sql_row_limit, self.settings.sql_timeout_s),
        )
        with self._lock:
            self._sessions[s.id] = s
        return s

    def get(self, session_id: str) -> Session:
        with self._lock:
            s = self._sessions.get(session_id)
        if s is None:
            raise SessionNotFound(session_id)
        s.touch()
        return s

    def delete(self, session_id: str) -> None:
        with self._lock:
            s = self._sessions.pop(session_id, None)
        if s:
            s.engine.close()

    def cleanup(self) -> int:
        cutoff = time.time() - self.settings.session_ttl_minutes * 60
        with self._lock:
            stale = [sid for sid, s in self._sessions.items() if s.last_used < cutoff]
        for sid in stale:
            self.delete(sid)
        return len(stale)

    def __len__(self) -> int:
        return len(self._sessions)
