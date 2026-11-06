"""CSV upload validation and cleaning.

Goals: never crash on messy real-world CSVs, always tell the user *what* was changed,
and produce tidy snake_case tables that an LLM can query reliably.
"""
from __future__ import annotations

import csv
import io
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class IngestionError(ValueError):
    """Raised when a file cannot be accepted. The message is safe to show to end users."""


SQL_RESERVED = {
    "order", "group", "select", "from", "where", "table", "index", "key", "limit", "by", "desc", "asc",
    "values", "default", "check", "join", "union", "all", "and", "or", "not", "null", "in", "is", "as",
    "on", "case", "when", "then", "else", "end", "having", "primary", "references", "transaction",
}
_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
_DATE_LIKE = re.compile(
    r"^\d{4}-\d{1,2}-\d{1,2}|^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|^\d{4}/\d{1,2}/\d{1,2}|^[A-Za-z]{3,9}\.? \d{1,2},? \d{4}"
)
_NOT_NUMERIC_NAME = re.compile(r"(^|_)(id|zip|zipcode|postcode|phone|code|sku)$", re.I)
_CURRENCY = re.compile(r"[,$€£₹\s]")
_EMPTY_EXPORT_COLUMN = re.compile(r"^unnamed(_\d+)?$", re.I)


@dataclass
class LoadedTable:
    name: str
    filename: str
    df: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    column_map: dict[str, str] = field(default_factory=dict)  # original -> clean

    @property
    def rows(self) -> int:
        return len(self.df)


def to_snake(name: object, fallback: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", str(name).strip())
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    if not s:
        return fallback
    if s[0].isdigit():
        s = "c_" + s
    return s + "_" if s in SQL_RESERVED else s


def _unique(name: str, taken: set[str]) -> str:
    candidate, i = name, 2
    while candidate in taken:
        candidate = f"{name}_{i}"
        i += 1
    return candidate


def _decode(content: bytes) -> str:
    for enc in _ENCODINGS:
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("latin-1", errors="replace")  # pragma: no cover


def _sniff_delimiter(text: str) -> str:
    sample = text[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _is_text(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _try_numeric(s: pd.Series, name: str) -> pd.Series | None:
    if _NOT_NUMERIC_NAME.search(name):
        return None
    mask = s.notna()
    if not mask.any():
        return None
    raw = s[mask].astype(str)
    if raw.str.match(r"^\s*0\d+").any():  # leading zeros -> identifier, keep as text
        return None
    conv = pd.to_numeric(raw.str.replace(_CURRENCY, "", regex=True), errors="coerce")
    if conv.notna().mean() < 0.98:
        return None
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out[mask] = conv
    if out.notna().all() and (out % 1 == 0).all():
        return out.astype("int64")
    return out


def _try_bool(s: pd.Series) -> pd.Series | None:
    nn = s.dropna().astype(str).str.lower()
    if nn.empty or not nn.isin({"true", "false"}).all():
        return None
    return s.map(lambda v: np.nan if pd.isna(v) else str(v).lower() == "true").astype("boolean") if s.isna().any() \
        else s.astype(str).str.lower().eq("true")


def _try_datetime(s: pd.Series) -> pd.Series | None:
    non_null = s.dropna().astype(str)
    if non_null.empty:
        return None
    sample = non_null.head(200)
    if sample.str.match(_DATE_LIKE).mean() < 0.9:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, format="ISO8601", errors="coerce")
        if parsed.notna().sum() / len(non_null) < 0.95:
            parsed = pd.to_datetime(s, format="mixed", errors="coerce")
    if parsed.notna().sum() / len(non_null) < 0.95:
        return None
    return parsed


def load_csv(
    filename: str,
    content: bytes,
    *,
    max_bytes: int,
    max_rows: int,
    taken_names: set[str] | None = None,
) -> LoadedTable:
    taken_names = taken_names or set()
    notes: list[str] = []

    # ---- 1. file-level validation -------------------------------------------------
    if not filename or not filename.lower().endswith((".csv", ".tsv", ".txt")):
        raise IngestionError(f"'{filename}': only .csv / .tsv files are supported.")
    if not content or not content.strip():
        raise IngestionError(f"'{filename}': the file is empty.")
    if len(content) > max_bytes:
        raise IngestionError(f"'{filename}': file is larger than the {max_bytes // (1024 * 1024)} MB limit.")
    if b"\x00" in content[:4096]:
        raise IngestionError(f"'{filename}': this looks like a binary file, not a CSV.")

    text = _decode(content)
    delimiter = _sniff_delimiter(text)

    # ---- 2. parse -----------------------------------------------------------------
    def _read(**kw) -> pd.DataFrame:
        return pd.read_csv(io.StringIO(text), sep=delimiter, skipinitialspace=True, low_memory=False, dtype=str, **kw)

    try:
        df = _read()
    except pd.errors.EmptyDataError:
        raise IngestionError(f"'{filename}': no columns could be parsed.") from None
    except pd.errors.ParserError:
        try:
            df = _read(on_bad_lines="skip")
            notes.append("Some malformed rows (wrong number of fields) were skipped.")
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"'{filename}': could not parse CSV ({exc}).") from None

    # ---- 3. structural checks -----------------------------------------------------
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if df.shape[1] == 0 or df.shape[0] == 0:
        raise IngestionError(f"'{filename}': the file has a header but no usable data rows.")
    if df.shape[0] > max_rows:
        raise IngestionError(f"'{filename}': {df.shape[0]:,} rows exceeds the limit of {max_rows:,}.")
    if df.shape[1] > 300:
        raise IngestionError(f"'{filename}': too many columns ({df.shape[1]}); the limit is 300.")
    df = df.reset_index(drop=True)

    # ---- 4. clean column names ----------------------------------------------------
    col_map: dict[str, str] = {}
    used: set[str] = set()
    for i, col in enumerate(df.columns):
        clean = _unique(to_snake(col, f"column_{i + 1}"), used)
        used.add(clean)
        col_map[str(col)] = clean
    renamed = {k: v for k, v in col_map.items() if k != v}
    df.columns = list(col_map.values())
    if renamed:
        notes.append(f"{len(renamed)} column name(s) were normalised to snake_case for SQL.")

    # Spreadsheet exports often contain trailing blank columns named "Unnamed: 9".
    # Retain genuine sparse business fields, but remove these import artifacts when
    # they are almost entirely empty so they cannot pollute profiling or dashboards.
    artifact_columns = [
        col for col in df.columns
        if _EMPTY_EXPORT_COLUMN.fullmatch(col) and df[col].isna().mean() >= 0.95
    ]
    if artifact_columns:
        df = df.drop(columns=artifact_columns)
        notes.append(f"{len(artifact_columns)} mostly empty spreadsheet-export column(s) were removed.")

    # ---- 5. clean values & infer types -------------------------------------------
    for col in df.columns:
        s = df[col]
        if not _is_text(s):
            continue
        s = s.astype("object").where(s.isna(), s.astype(str).str.strip())
        s = s.replace("", np.nan)
        as_num = _try_numeric(s, col)
        if as_num is not None:
            df[col] = as_num
            notes.append(f"Column '{col}' was converted from text to numbers.")
            continue
        as_bool = _try_bool(s)
        if as_bool is not None:
            df[col] = as_bool
            continue
        as_dt = _try_datetime(s)
        if as_dt is not None:
            df[col] = as_dt
            notes.append(f"Column '{col}' was parsed as dates.")
            continue
        df[col] = s

    dup = int(df.duplicated().sum())
    if dup:
        notes.append(f"{dup} exact duplicate row(s) detected (kept as-is; see data quality report).")

    stem = re.sub(r"\.(csv|tsv|txt)$", "", filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1], flags=re.I)
    table = _unique(to_snake(stem, "table"), taken_names)
    return LoadedTable(name=table, filename=filename, df=df, warnings=notes, column_map=col_map)
