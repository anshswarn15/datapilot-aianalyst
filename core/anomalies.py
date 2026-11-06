"""Explainable statistical anomaly detection.

Every flagged value comes with the *rule* that flagged it, the expected range and a
plain-English reason, so the LLM (and the user) can explain it instead of just reporting it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.profiling import is_id_like
from app.core.utils import jsonable, round_num

METHODS = ("iqr", "zscore", "mad")
MIN_GROUP_SIZE = 8


def _fmt(x: float) -> str:
    return f"{x:,.2f}" if abs(x) < 1e6 else f"{x:,.0f}"


def _bounds(values: pd.Series, method: str, threshold: float | None):
    """Return (lower, upper, center, scale, description) or None if the spread is degenerate."""
    v = values.dropna()
    if len(v) < MIN_GROUP_SIZE:
        return None
    median = float(v.median())
    if method == "iqr":
        q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
        iqr = q3 - q1
        if iqr > 0:
            k = threshold or 1.5
            return (q1 - k * iqr, q3 + k * iqr, median, iqr,
                    f"IQR rule: outside Q1 {_fmt(q1)} - {k} x IQR / Q3 {_fmt(q3)} + {k} x IQR (IQR = {_fmt(iqr)})")
        method = "mad"  # spread collapsed: fall back to MAD
    if method == "zscore":
        mean, std = float(v.mean()), float(v.std())
        if std == 0 or np.isnan(std):
            return None
        k = threshold or 3.0
        return (mean - k * std, mean + k * std, mean, std,
                f"z-score rule: more than {k} standard deviations from the mean {_fmt(mean)} (std = {_fmt(std)})")
    mad = float((v - median).abs().median())
    if mad == 0:
        return None
    k = threshold or 3.5
    spread = k * mad / 0.6745
    return (median - spread, median + spread, median, mad / 0.6745,
            f"modified z-score rule (MAD): more than {k} robust deviations from the median {_fmt(median)}")


def detect_anomalies(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    group_by: str | None = None,
    method: str = "iqr",
    threshold: float | None = None,
    max_results: int = 25,
) -> dict:
    method = (method or "iqr").lower()
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Use one of: {', '.join(METHODS)}.")
    if group_by and group_by not in df.columns:
        raise ValueError(f"group_by column '{group_by}' not found. Columns: {list(df.columns)}")

    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Unknown column(s): {missing}. Columns: {list(df.columns)}")
        candidates = columns
    else:
        candidates = [c for c in df.columns if not is_id_like(c, df[c])]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if not numeric:
        raise ValueError("No numeric columns available for anomaly detection.")

    context_cols_all = list(df.columns)
    findings: list[dict] = []
    col_summaries: list[dict] = []

    for col in numeric:
        series = pd.to_numeric(df[col], errors="coerce")
        parts = [(None, series)] if not group_by else [
            (g, series.loc[idx]) for g, idx in df.groupby(group_by, dropna=True).groups.items()
        ]
        flagged_total, last_bounds = 0, None
        for gval, part in parts:
            b = _bounds(part, method, threshold)
            if b is None:
                continue
            lo, hi, center, scale, rule = b
            last_bounds = b
            mask = (part < lo) | (part > hi)
            for idx, value in part[mask].items():
                direction = "above" if value > hi else "below"
                edge = hi if value > hi else lo
                severity = abs(value - edge) / scale if scale else 0.0
                where = f" within {group_by}={gval!r}" if group_by else ""
                reason = (f"{col} = {_fmt(float(value))} is {direction} the expected range "
                          f"[{_fmt(lo)}, {_fmt(hi)}]{where}; typical value (median/mean) is {_fmt(center)}. {rule}.")
                ctx_cols = [c for c in context_cols_all if c != col][:7]
                findings.append({
                    "row_index": int(idx), "column": col, "value": jsonable(value),
                    "expected_min": round_num(lo, 2), "expected_max": round_num(hi, 2),
                    "direction": direction, "severity": round(float(severity), 2),
                    "group": None if gval is None else jsonable(gval), "reason": reason,
                    "row_context": {c: jsonable(df.at[idx, c]) for c in ctx_cols},
                })
            flagged_total += int(mask.sum())
        col_summaries.append({
            "column": col, "flagged": flagged_total,
            "pct_flagged": round(100 * flagged_total / max(series.notna().sum(), 1), 2),
            "expected_min": None if (group_by or last_bounds is None) else round_num(last_bounds[0], 2),
            "expected_max": None if (group_by or last_bounds is None) else round_num(last_bounds[1], 2),
            "median": round_num(series.median(), 2),
        })

    findings.sort(key=lambda f: f["severity"], reverse=True)
    return {
        "method": method, "threshold": threshold, "group_by": group_by,
        "rows_analyzed": int(len(df)), "columns_analyzed": numeric,
        "total_flagged": len(findings), "columns": col_summaries,
        "anomalies": findings[:max_results],
        "shown": min(len(findings), max_results),
        "other_flags": {"duplicate_rows": int(df.duplicated().sum())},
        "note": "Statistical outliers are *unusual*, not necessarily *wrong*: verify against business context.",
    }
