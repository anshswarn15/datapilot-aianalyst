"""A locked-down, read-only SQL engine over the uploaded data (SQLite, in-memory).

Safety layers (defence in depth, because the SQL is written by an LLM):
  1. Static check: a single SELECT / WITH statement only.
  2. SQLite *authorizer*: the engine itself refuses anything except reads.
  3. Progress handler: queries are aborted after a timeout.
  4. Row cap: never materialise more than `row_limit` rows.
Results are cached (LRU) because LLMs frequently re-issue identical queries.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import sqlite3
import statistics
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd

from app.core.utils import jsonable


class SQLValidationError(ValueError):
    pass


class SQLExecutionError(RuntimeError):
    pass


# sqlite3 authorizer action codes: SELECT=21 READ=20 FUNCTION=31 RECURSIVE=33
_ALLOWED_ACTIONS = {21, 20, 31, 33}
_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]
    truncated: bool
    elapsed_ms: float = 0.0
    cached: bool = False

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)

    def to_dict(self, max_rows: int | None = None) -> dict:
        rows = self.rows if max_rows is None else self.rows[:max_rows]
        return {
            "columns": self.columns, "rows": rows, "row_count": self.row_count,
            "truncated": self.truncated or (max_rows is not None and self.row_count > max_rows),
        }


# ---- custom SQL functions that SQLite lacks but analysts (and LLMs) expect --------------
class _Stdev:
    def __init__(self):
        self.vals: list[float] = []

    def step(self, v):
        try:
            if v is not None:
                self.vals.append(float(v))
        except (TypeError, ValueError):
            pass

    def finalize(self):
        return statistics.stdev(self.vals) if len(self.vals) > 1 else None


class _Median(_Stdev):
    def finalize(self):
        return statistics.median(self.vals) if self.vals else None


def _date_trunc(unit, value):
    if value is None or unit is None:
        return None
    try:
        d = dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
    unit = str(unit).lower()
    if unit == "year":
        d = d.replace(month=1, day=1)
    elif unit == "quarter":
        d = d.replace(month=3 * ((d.month - 1) // 3) + 1, day=1)
    elif unit == "month":
        d = d.replace(day=1)
    elif unit == "week":
        d = d - dt.timedelta(days=d.weekday())
    elif unit != "day":
        return None
    return d.isoformat()


def _safe_math(fn):
    def wrapper(*args):
        try:
            if any(a is None for a in args):
                return None
            return fn(*[float(a) for a in args])
        except (ValueError, OverflowError, TypeError):
            return None

    return wrapper


class SQLEngine:
    def __init__(self, row_limit: int = 500, timeout_s: int = 10, cache_size: int = 128):
        self.row_limit = row_limit
        self.timeout_s = timeout_s
        self._cache: OrderedDict[tuple, QueryResult] = OrderedDict()
        self._cache_size = cache_size
        self.cache_hits = 0
        self._lock = threading.RLock()
        self._con = sqlite3.connect(":memory:", check_same_thread=False)
        self._tables: list[str] = []
        self._register_functions()
        self._con.set_authorizer(self._authorizer)

    # -- setup -------------------------------------------------------------------------
    def _register_functions(self) -> None:
        c = self._con
        for name in ("STDEV", "STDDEV"):
            c.create_aggregate(name, 1, _Stdev)
        c.create_aggregate("MEDIAN", 1, _Median)
        c.create_function("DATE_TRUNC", 2, _date_trunc, deterministic=True)
        for name, fn, n in (("SQRT", math.sqrt, 1), ("LN", math.log, 1), ("EXP", math.exp, 1),
                            ("POWER", math.pow, 2), ("FLOOR", math.floor, 1), ("CEIL", math.ceil, 1)):
            c.create_function(name, n, _safe_math(fn), deterministic=True)

    @staticmethod
    def _authorizer(action, *_):
        return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY

    def register(self, name: str, df: pd.DataFrame) -> None:
        out = df.copy()
        for col in out.columns:
            s = out[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                nn = s.dropna()
                has_time = bool((nn.dt.normalize() != nn).any()) if len(nn) else False
                out[col] = s.dt.strftime("%Y-%m-%d %H:%M:%S" if has_time else "%Y-%m-%d")
            elif pd.api.types.is_bool_dtype(s):
                out[col] = s.astype("int64")
        with self._lock:
            self._con.set_authorizer(None)  # loading data needs write access; queries do not
            try:
                out.to_sql(name, self._con, index=False, if_exists="replace")
            finally:
                self._con.set_authorizer(self._authorizer)
            if name not in self._tables:
                self._tables.append(name)
            self._cache.clear()

    def drop(self, name: str) -> None:
        with self._lock:
            self._con.set_authorizer(None)
            try:
                self._con.execute(f'DROP TABLE IF EXISTS "{name}"')
            finally:
                self._con.set_authorizer(self._authorizer)
            self._tables = [t for t in self._tables if t != name]
            self._cache.clear()

    # -- querying ----------------------------------------------------------------------
    @staticmethod
    def validate(sql: str) -> str:
        if not isinstance(sql, str) or not sql.strip():
            raise SQLValidationError("Empty query.")
        cleaned = _COMMENTS.sub(" ", sql).strip().rstrip(";").strip()
        if ";" in cleaned:
            raise SQLValidationError("Only a single SQL statement is allowed.")
        if not re.match(r"(?is)^\(*\s*(select|with)\b", cleaned):
            raise SQLValidationError("Only read-only SELECT queries (optionally starting with WITH) are allowed.")
        return cleaned

    def query(self, sql: str, limit: int | None = None) -> QueryResult:
        sql = self.validate(sql)
        limit = min(limit or self.row_limit, self.row_limit)
        key = (sql, limit)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                self.cache_hits += 1
                return QueryResult(hit.columns, hit.rows, hit.truncated, 0.0, cached=True)

            start = time.monotonic()
            deadline = start + self.timeout_s
            self._con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 20000)
            try:
                cur = self._con.execute(sql)
                columns = [d[0] for d in (cur.description or [])]
                fetched = cur.fetchmany(limit + 1)
            except sqlite3.OperationalError as exc:
                if "interrupted" in str(exc).lower():
                    raise SQLExecutionError(f"Query exceeded the {self.timeout_s}s time limit.") from None
                raise SQLExecutionError(str(exc)) from None
            except sqlite3.Error as exc:
                raise SQLExecutionError(str(exc)) from None
            finally:
                self._con.set_progress_handler(None, 0)

            result = QueryResult(
                columns=columns,
                rows=[[jsonable(v) for v in row] for row in fetched[:limit]],
                truncated=len(fetched) > limit,
                elapsed_ms=round((time.monotonic() - start) * 1000, 2),
            )
            self._cache[key] = result
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return result

    def close(self) -> None:
        with self._lock:
            self._con.close()
