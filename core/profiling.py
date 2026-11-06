"""Schema profiling (fed to the LLM and the UI) and data-quality checks."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from app.core.utils import jsonable, round_num

_ID_NAME = re.compile(r"(^|_)(id|uuid|guid|key|index)$", re.I)
_NON_NEGATIVE_HINT = re.compile(r"(qty|quantity|price|revenue|amount|cost|age|count|total|sales|units)", re.I)


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    n = s.nunique(dropna=True)
    if n <= 50 or (len(s) and n / len(s) <= 0.5):
        return "categorical"
    return "text"


def is_id_like(name: str, s: pd.Series) -> bool:
    if _ID_NAME.search(name):
        return True
    if pd.api.types.is_integer_dtype(s) and len(s) > 20 and s.is_unique and s.is_monotonic_increasing:
        return True
    return False


def _numeric_stats(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {}
    return {
        "min": round_num(s.min()), "max": round_num(s.max()), "mean": round_num(s.mean()),
        "median": round_num(s.median()), "std": round_num(s.std()) if len(s) > 1 else None,
    }


def profile_table(df: pd.DataFrame) -> dict:
    cols = []
    for name in df.columns:
        s = df[name]
        kind = column_kind(s)
        info: dict = {
            "name": name, "kind": kind, "dtype": str(s.dtype),
            "nulls": int(s.isna().sum()), "null_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)), "is_id": is_id_like(name, s),
        }
        if kind == "numeric":
            info["stats"] = _numeric_stats(s)
        elif kind == "datetime":
            info["stats"] = {"min": jsonable(s.min()), "max": jsonable(s.max())}
        elif kind in ("categorical", "text"):
            top = s.value_counts().head(5)
            info["top_values"] = [{"value": str(k), "count": int(v)} for k, v in top.items()]
        cols.append(info)
    sample = [{k: jsonable(v) for k, v in row.items()} for row in df.head(5).to_dict("records")]
    return {"rows": int(len(df)), "columns": cols, "sample": sample}


def _g(v) -> str:
    return "?" if v is None else f"{v:g}"


def _d(ts) -> str:
    if pd.isna(ts):
        return "?"
    return ts.strftime("%Y-%m-%d") if ts == ts.normalize() else ts.strftime("%Y-%m-%d %H:%M:%S")


def schema_prompt(tables: dict) -> str:
    """Compact, LLM-friendly description of every loaded table (exact category values included)."""
    blocks = []
    for name, t in tables.items():
        df = t.df
        lines = [f"TABLE {name}  ({len(df):,} rows, from file '{t.filename}')"]
        for col in df.columns:
            s = df[col]
            kind = column_kind(s)
            nulls = int(s.isna().sum())
            extra = f", {nulls} nulls" if nulls else ""
            if kind == "numeric":
                st = _numeric_stats(s)
                tag = "integer" if pd.api.types.is_integer_dtype(s) else "number"
                idtag = " [identifier]" if is_id_like(col, s) else ""
                lines.append(f"  - {col}: {tag}{idtag}, range {_g(st.get('min'))}..{_g(st.get('max'))}, mean {_g(st.get('mean'))}{extra}")
            elif kind == "datetime":
                lines.append(f"  - {col}: date (TEXT 'YYYY-MM-DD[ HH:MM:SS]'), range {_d(s.min())}..{_d(s.max())}{extra}")
            elif kind == "boolean":
                lines.append(f"  - {col}: boolean (0/1){extra}")
            elif kind == "categorical":
                vals = s.value_counts().index.tolist()
                shown = ", ".join(repr(str(v)[:30]) for v in vals[:12])
                more = f", ... ({len(vals)} distinct)" if len(vals) > 12 else ""
                lines.append(f"  - {col}: category, values: {shown}{more}{extra}")
            else:
                ex = ", ".join(repr(str(v)[:25]) for v in s.dropna().head(3))
                lines.append(f"  - {col}: text, e.g. {ex}{extra}")
        aggregate_notes = []
        for col in df.columns:
            s = df[col]
            if not _is_text(s):
                continue
            values = s.dropna().astype(str).unique().tolist()
            flagged = [v for v in values if re.search(r"(^all$|(?:grand|industry)[ _-]?total$|.+_total$)", v, re.I)]
            if flagged:
                aggregate_notes.append(f"{col}={flagged[:6]}")
        if aggregate_notes:
            lines.append("  ! HIERARCHY WARNING: aggregate category values detected (" + "; ".join(aggregate_notes) + "). "
                         "Never sum aggregate rows together with their detail rows. Choose exactly one reporting level.")
        if "rme_size_grp" in df.columns:
            coded_bands = [str(v) for v in df["rme_size_grp"].dropna().unique()
                           if re.fullmatch(r"[a-z]_\d+(?:-\d+|\+)?", str(v), re.I)]
            if coded_bands:
                lines.append("  ! SIZE-BAND NOTE: rme_size_grp codes describe employee bands: a_0 means 0 employees; "
                             "b_1-5 means 1–5 employees; a trailing '+' means that many or more employees.")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def quality_report(df: pd.DataFrame, table: str) -> dict:
    issues: list[dict] = []

    def add(sev: str, col: str | None, issue: str, detail: str, count: int | None = None):
        issues.append({"severity": sev, "column": col, "issue": issue, "detail": detail, "count": count})

    n = len(df)
    dups = int(df.duplicated().sum())
    if dups:
        add("medium", None, "Duplicate rows", f"{dups} fully identical rows ({dups / n:.1%} of data).", dups)

    for col in df.columns:
        s = df[col]
        nulls = int(s.isna().sum())
        if nulls:
            pct = nulls / n
            sev = "high" if pct > 0.30 else "medium" if pct > 0.05 else "low"
            add(sev, col, "Missing values", f"{nulls} missing ({pct:.1%}).", nulls)
        if s.nunique(dropna=True) == 1 and n > 1:
            add("low", col, "Constant column", f"Every value is {str(s.dropna().iloc[0])!r}; carries no information.")
        if _is_text(s):
            txt = s.dropna().astype(str)
            ws = int((txt != txt.str.strip()).sum())
            if ws:
                add("low", col, "Leading/trailing whitespace", f"{ws} values have stray spaces.", ws)
            if 0 < txt.nunique() <= 200:
                norm = txt.str.strip().str.lower()
                if norm.nunique() < txt.nunique():
                    groups = txt.groupby(norm).agg(lambda x: sorted(set(x)))
                    clash = [g for g in groups if len(g) > 1][:3]
                    add("medium", col, "Inconsistent categories",
                        "Same value spelled differently: " + "; ".join(" / ".join(repr(v) for v in g) for g in clash))
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if _NON_NEGATIVE_HINT.search(col):
                neg = int((s < 0).sum())
                if neg:
                    add("medium", col, "Negative values", f"{neg} negative values in a column that is normally >= 0 (returns/refunds or errors?).", neg)
        elif pd.api.types.is_datetime64_any_dtype(s):
            fut = int((s > pd.Timestamp.today()).sum())
            if fut:
                add("low", col, "Future dates", f"{fut} dates are in the future.", fut)

    weights = {"high": 15, "medium": 7, "low": 2}
    score = max(0, 100 - sum(weights[i["severity"]] for i in issues))
    order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: order[i["severity"]])
    return {"table": table, "rows": n, "columns": df.shape[1], "score": score, "issues": issues}


def _is_text(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
