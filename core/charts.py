"""Turn a SQL result into a library-neutral chart spec (rendered by Chart.js in the UI)."""
from __future__ import annotations

import pandas as pd

from app.core.utils import jsonable

CHART_TYPES = ("bar", "horizontal_bar", "line", "area", "pie", "doughnut", "scatter")


class ChartError(ValueError):
    pass


def _resolve(name: str | None, columns: list[str]) -> str | None:
    if name is None:
        return None
    lowered = {c.lower(): c for c in columns}
    if name in columns:
        return name
    if name.lower() in lowered:
        return lowered[name.lower()]
    raise ChartError(f"Column '{name}' is not in the query result. Available columns: {columns}")


def _label(v) -> str:
    return "(missing)" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)


def _num(series: pd.Series) -> list:
    return [jsonable(v) for v in pd.to_numeric(series, errors="coerce")]


def build_chart(chart_type: str, columns: list[str], rows: list[list], x: str, y, series: str | None = None,
                title: str | None = None, max_categories: int = 30) -> dict:
    chart_type = (chart_type or "").lower()
    if chart_type not in CHART_TYPES:
        raise ChartError(f"chart_type must be one of {list(CHART_TYPES)}")
    if not rows:
        raise ChartError("The query returned no rows, so there is nothing to chart.")
    df = pd.DataFrame(rows, columns=columns)
    x = _resolve(x, columns)
    ys = [y] if isinstance(y, str) else list(y or [])
    ys = [_resolve(c, columns) for c in ys]
    if not ys:
        raise ChartError("At least one y column is required.")
    series = _resolve(series, columns)
    truncated = False
    spec = {"type": chart_type, "title": title or f"{', '.join(ys)} by {x}", "x_label": x,
            "y_label": ", ".join(ys), "labels": [], "datasets": []}

    if chart_type == "scatter":
        if len(df) > 1000:
            df, truncated = df.sample(1000, random_state=0), True
        groups = [(None, df)] if not series else list(df.groupby(series))
        for g, part in groups:
            xs, vs = _num(part[x]), _num(part[ys[0]])
            spec["datasets"].append({
                "label": str(g) if g is not None else f"{ys[0]} vs {x}",
                "data": [{"x": a, "y": b} for a, b in zip(xs, vs) if a is not None and b is not None],
            })
    elif chart_type in ("pie", "doughnut"):
        d = df[[x, ys[0]]].copy()
        d[ys[0]] = pd.to_numeric(d[ys[0]], errors="coerce")
        d[x] = d[x].map(_label)
        d = d.dropna().groupby(x, sort=False)[ys[0]].sum().sort_values(ascending=False)
        if len(d) > 10:
            d = pd.concat([d.head(9), pd.Series({"Other": d.iloc[9:].sum()})])
            truncated = True
        spec["labels"] = [str(k) for k in d.index]
        spec["datasets"] = [{"label": ys[0], "data": [jsonable(v) for v in d.values]}]
    else:
        if series:
            df[x] = df[x].map(_label)
            labels = list(dict.fromkeys(df[x]))
            for g, part in df.groupby(series, sort=False):
                lookup = dict(zip(part[x], _num(part[ys[0]])))
                spec["datasets"].append({"label": str(g), "data": [lookup.get(l) for l in labels]})
        else:
            labels = df[x].map(_label).tolist()
            for col in ys:
                spec["datasets"].append({"label": col, "data": _num(df[col])})
        cap = 500 if chart_type in ("line", "area") else max_categories
        if len(labels) > cap:
            labels, truncated = labels[:cap], True
            for ds in spec["datasets"]:
                ds["data"] = ds["data"][:cap]
        spec["labels"] = labels
    spec["truncated"] = truncated
    return spec
