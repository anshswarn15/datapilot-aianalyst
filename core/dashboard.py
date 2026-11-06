"""Rule-based auto-dashboard: KPIs and charts inferred from column types (no LLM call, instant)."""
from __future__ import annotations

import re

import pandas as pd

from app.core.charts import build_chart
from app.core.profiling import column_kind, is_id_like
from app.core.utils import jsonable

_MEASURE_PRIORITY = ["revenue", "sales", "amount", "profit", "total", "cost", "quantity", "qty", "units", "price"]
_ADDITIVE = re.compile(r"(revenue|sales|amount|profit|total|quantity|qty|units|cost)", re.I)
_YEAR_NAME = re.compile(r"(^|_)(year|fiscal_year|calendar_year)($|_)", re.I)
_SIZE_GROUP = re.compile(r"^[a-z]_(\d+)(?:-(\d+)|(\+))?$", re.I)


def _measure_rank(name: str) -> int:
    for i, hint in enumerate(_MEASURE_PRIORITY):
        if hint in name.lower():
            return i
    return len(_MEASURE_PRIORITY)


def _agg(name: str) -> str:
    return "sum" if _ADDITIVE.search(name) else "mean"


def _is_year_measure(name: str, series: pd.Series) -> bool:
    """Exclude calendar years from KPI/measure selection.

    Year is often imported as an integer, but an "average year by industry"
    chart is not an actionable business metric. It remains available to the
    agent and can still be used as a grouping/filtering field.
    """
    if _YEAR_NAME.search(name):
        return True
    values = series.dropna()
    return bool(len(values) and values.between(1900, 2200).mean() >= 0.98 and values.nunique() <= 200)


def _size_group_label(value: object) -> str:
    """Turn coded employee bands such as ``b_1-5`` into a readable label."""
    text = str(value)
    match = _SIZE_GROUP.fullmatch(text)
    if not match:
        return text
    low, high, plus = match.groups()
    if plus:
        return f"{low}+ employees"
    if high:
        return f"{low}–{high} employees"
    return f"{low} employees"


def _chart(kind, df, x, y, title, series=None):
    return build_chart(kind, list(df.columns), df.values.tolist(), x, y, series=series, title=title)


def _enterprise_survey_dashboard(name: str, df: pd.DataFrame) -> dict | None:
    """Build a meaningful dashboard for Statistics NZ-style long-form survey data.

    The file contains several hierarchy levels at once: individual industries,
    industry totals, and a national grand total. A generic mean across all rows
    would mix those levels and produce misleading charts, so this view uses one
    explicit level for every metric.
    """
    required = {"year", "variable", "value", "unit", "rme_size_grp", "industry_code_anzsic"}
    if not required <= set(df.columns):
        return None

    data = df.copy()
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data = data.dropna(subset=["value", "year"])
    if data.empty or not {"all", "j_Grand_Total", "i_Industry_Total"} <= set(data["industry_code_anzsic"].astype(str)) | set(data["rme_size_grp"].astype(str)):
        return None

    latest_year = int(data["year"].max())
    activity = data[(data["variable"] == "Activity unit") & (data["unit"] == "COUNT")]
    national = activity[(activity["industry_code_anzsic"].astype(str) == "all") &
                        (activity["rme_size_grp"].astype(str) == "j_Grand_Total")]
    if national.empty:
        return None

    latest_total = national[national["year"] == latest_year]
    kpis = [{"label": "Rows", "value": int(len(df))}, {"label": "Latest year", "value": latest_year}]
    if not latest_total.empty:
        kpis.append({"label": "Activity units", "value": int(latest_total["value"].iloc[0])})
    employees = data[(data["variable"] == "Rolling mean employees") & (data["unit"] == "COUNT") &
                     (data["industry_code_anzsic"].astype(str) == "all") &
                     (data["rme_size_grp"].astype(str) == "j_Grand_Total") &
                     (data["year"] == latest_year)]
    if not employees.empty:
        kpis.append({"label": "Rolling mean employees", "value": int(employees["value"].iloc[0])})

    charts = []
    trend = national.groupby("year", as_index=False)["value"].sum().sort_values("year")
    if len(trend) > 1:
        charts.append(_chart("line", trend, "year", "value", "National activity units by year"))

    # Sum individual industries only; exclude the supplied subtotal/grand-total rows.
    size = activity[(activity["year"] == latest_year) & (activity["industry_code_anzsic"].astype(str) != "all") &
                    ~activity["rme_size_grp"].astype(str).isin(["i_Industry_Total", "j_Grand_Total"])]
    size = size.groupby("rme_size_grp", as_index=False)["value"].sum()
    if not size.empty:
        size["size_group"] = size["rme_size_grp"].map(_size_group_label)
        charts.append(_chart("bar", size, "size_group", "value", f"Activity units by size group ({latest_year})"))

    industry = activity[(activity["year"] == latest_year) & (activity["rme_size_grp"].astype(str) == "i_Industry_Total") &
                        (activity["industry_code_anzsic"].astype(str) != "all")]
    label = "industry_name_anzsic" if "industry_name_anzsic" in industry.columns else "industry_code_anzsic"
    industry = industry.groupby(label, as_index=False)["value"].sum().sort_values("value", ascending=False).head(10)
    if not industry.empty:
        charts.append(_chart("horizontal_bar", industry, label, "value", f"Largest industries by activity units ({latest_year})"))

    return {"table": name, "kpis": kpis, "charts": charts}


def build_dashboard(tables: dict) -> dict:
    sections = []
    for name, t in tables.items():
        df: pd.DataFrame = t.df
        survey_section = _enterprise_survey_dashboard(name, df)
        if survey_section:
            sections.append(survey_section)
            continue
        kinds = {c: column_kind(df[c]) for c in df.columns}
        measures = sorted(
            [c for c in df.columns if kinds[c] == "numeric" and not is_id_like(c, df[c])
             and not _is_year_measure(c, df[c])], key=_measure_rank
        )[:3]
        dims = [c for c in df.columns if kinds[c] == "categorical" and not is_id_like(c, df[c])
                and 2 <= df[c].nunique() <= 25][:2]
        dates = [c for c in df.columns if kinds[c] == "datetime"]

        kpis = [{"label": "Rows", "value": int(len(df))}]
        for m in measures:
            agg = _agg(m)
            label = f"Total {m}" if agg == "sum" else f"Average {m}"
            kpis.append({"label": label, "value": jsonable(round(float(getattr(df[m], agg)()), 2))})

        charts = []
        main = measures[0] if measures else None
        if dates and main:
            d = df[[dates[0], main]].dropna()
            d = d.assign(period=d[dates[0]].dt.strftime("%Y-%m")).groupby("period")[main].agg(_agg(main)).reset_index()
            if len(d) > 1:
                charts.append(_chart("line", d, "period", main, f"{main} by month ({_agg(main)})"))
        for dim in dims:
            if main:
                d = df.groupby(dim)[main].agg(_agg(main)).sort_values(ascending=False).head(10).reset_index()
                charts.append(_chart("bar", d, dim, main, f"{main} by {dim} ({_agg(main)})"))
            else:
                d = df[dim].value_counts().head(10).rename_axis(dim).reset_index(name="count")
                charts.append(_chart("bar", d, dim, "count", f"Rows by {dim}"))
        if dims and main and df[dims[0]].nunique() <= 6:
            d = df.groupby(dims[0])[main].sum().reset_index()
            charts.append(_chart("doughnut", d, dims[0], main, f"Share of {main} by {dims[0]}"))
        if len(measures) >= 2:
            d = df[measures[:2]].dropna().head(800)
            charts.append(_chart("scatter", d, measures[0], measures[1], f"{measures[1]} vs {measures[0]}"))
        sections.append({"table": name, "kpis": kpis, "charts": charts})
    return {"sections": sections}
