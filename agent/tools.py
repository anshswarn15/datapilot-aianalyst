"""Tool definitions (JSON schemas the LLM sees) and the executor that runs them safely.

Each tool returns a `ToolOutcome`:
  * llm_text  - compact JSON fed back to the model (kept small to save tokens)
  * artifact  - rich object shown in the UI (table / chart / anomaly report ...)
Errors are returned to the model as {"error": ...} so it can self-correct instead of crashing the turn.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import pandas as pd

from app.core.anomalies import detect_anomalies
from app.core.charts import CHART_TYPES, ChartError, build_chart
from app.core.forecasting import FREQ, forecast_series
from app.core.profiling import profile_table, quality_report
from app.core.session import Session
from app.core.sql_engine import SQLExecutionError, SQLValidationError
from app.core.utils import clip_text

log = logging.getLogger(__name__)

LLM_ROWS = 40      # rows of a result echoed back to the model
UI_ROWS = 200      # rows of a result shown in the UI table
MAX_TOOL_TEXT = 7000

TOOL_SCHEMAS: list[dict] = [
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run ONE read-only SQLite SELECT (or WITH ... SELECT) over the loaded tables and return the rows.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The SQL query."},
            "purpose": {"type": "string", "description": "Short title of what this query computes (shown to the user)."},
        }, "required": ["query", "purpose"]}}},
    {"type": "function", "function": {
        "name": "create_chart",
        "description": "Run a SQL query and render the result as a chart for the user.",
        "parameters": {"type": "object", "properties": {
            "chart_type": {"type": "string", "enum": list(CHART_TYPES)},
            "query": {"type": "string", "description": "SQL producing the data to plot (aggregate in SQL, ORDER BY the x axis)."},
            "x": {"type": "string", "description": "Result column for the x axis / labels."},
            "y": {"type": "array", "items": {"type": "string"}, "description": "Result column(s) with numeric values."},
            "series": {"type": "string", "description": "Optional result column that splits the data into multiple series."},
            "title": {"type": "string"},
        }, "required": ["chart_type", "query", "x", "y", "title"]}}},
    {"type": "function", "function": {
        "name": "detect_anomalies",
        "description": "Find statistically unusual numeric values in a table, with the rule and expected range for each. "
                       "Omit `columns` to scan every numeric non-ID column.",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "group_by": {"type": "string", "description": "Optional category column: detect outliers within each group."},
            "method": {"type": "string", "enum": ["iqr", "zscore", "mad"], "description": "Default iqr (robust to skew)."},
        }, "required": ["table"]}}},
    {"type": "function", "function": {
        "name": "data_quality_report",
        "description": "Check missing values, duplicates, inconsistent category spellings, negative values etc. Omit `table` for all tables.",
        "parameters": {"type": "object", "properties": {"table": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "forecast",
        "description": "Forecast a time series. The SQL must return one date column and one numeric column (already aggregated).",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "date_column": {"type": "string"}, "value_column": {"type": "string"},
            "periods": {"type": "integer", "description": "How many future periods (1-36)."},
            "frequency": {"type": "string", "enum": list(FREQ)},
        }, "required": ["query", "date_column", "value_column"]}}},
    {"type": "function", "function": {
        "name": "describe_table",
        "description": "Detailed column statistics and sample rows for one table.",
        "parameters": {"type": "object", "properties": {"table": {"type": "string"}}, "required": ["table"]}}},
]


@dataclass
class ToolOutcome:
    llm_text: str
    ok: bool = True
    summary: str = ""
    artifact: dict | None = None


def _err(msg: str) -> ToolOutcome:
    return ToolOutcome(json.dumps({"error": msg}), ok=False, summary=msg[:160])


class ToolExecutor:
    def __init__(self, session: Session):
        self.s = session

    def execute(self, name: str, args: dict) -> ToolOutcome:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return _err(f"Unknown tool '{name}'. Available: {[t['function']['name'] for t in TOOL_SCHEMAS]}")
        if "__parse_error__" in args:
            return _err("Tool arguments were not valid JSON. Re-send the call with valid JSON arguments.")
        try:
            return handler(**args)
        except TypeError as exc:  # wrong / missing arguments
            return _err(f"Bad arguments for {name}: {exc}")
        except (SQLValidationError, SQLExecutionError, ChartError, ValueError, KeyError) as exc:
            return _err(str(exc).strip("'\""))
        except Exception as exc:  # noqa: BLE001 - a tool bug must never kill the whole turn
            log.exception("tool_crash name=%s", name)
            return _err(f"Internal error in {name}: {type(exc).__name__}")

    # ---- helpers --------------------------------------------------------------------
    def _table(self, name: str):
        t = self.s.tables.get(name)
        if t is None:
            raise ValueError(f"Table '{name}' not found. Loaded tables: {list(self.s.tables)}")
        return t

    @staticmethod
    def _txt(obj) -> str:
        return clip_text(json.dumps(obj, default=str), MAX_TOOL_TEXT)

    # ---- tools ----------------------------------------------------------------------
    def _tool_run_sql(self, query: str, purpose: str = "") -> ToolOutcome:
        res = self.s.engine.query(query)
        payload = res.to_dict(LLM_ROWS)
        payload["note"] = "Result truncated at the row cap." if res.truncated else ""
        artifact = {"type": "sql", "title": purpose or "Query result", "query": query.strip(),
                    **res.to_dict(UI_ROWS), "elapsed_ms": res.elapsed_ms, "cached": res.cached}
        return ToolOutcome(self._txt(payload), summary=f"{res.row_count} row(s)" + (" (cached)" if res.cached else ""), artifact=artifact)

    def _tool_create_chart(self, chart_type: str, query: str, x: str, y, title: str = "", series: str | None = None) -> ToolOutcome:
        res = self.s.engine.query(query)
        spec = build_chart(chart_type, res.columns, res.rows, x, y, series=series, title=title or None)
        preview = res.to_dict(10)
        artifact = {"type": "chart", "chart": spec, "query": query.strip()}
        text = self._txt({"status": "chart displayed to the user", "chart_type": chart_type,
                          "points": res.row_count, "data_preview": preview["rows"], "columns": res.columns})
        return ToolOutcome(text, summary=f"{chart_type} chart, {res.row_count} point(s)", artifact=artifact)

    def _tool_detect_anomalies(self, table: str, columns: list[str] | None = None,
                               group_by: str | None = None, method: str = "iqr") -> ToolOutcome:
        t = self._table(table)
        report = detect_anomalies(t.df, columns=columns, group_by=group_by, method=method)
        llm = {k: v for k, v in report.items() if k != "anomalies"}
        llm["top_anomalies"] = [
            {k: a[k] for k in ("row_index", "column", "value", "expected_min", "expected_max", "severity", "group", "row_context")}
            for a in report["anomalies"][:12]
        ]
        return ToolOutcome(self._txt(llm), summary=f"{report['total_flagged']} flagged in '{table}'",
                           artifact={"type": "anomalies", "table": table, "report": report})

    def _tool_data_quality_report(self, table: str | None = None) -> ToolOutcome:
        names = [table] if table else list(self.s.tables)
        reports = [quality_report(self._table(n).df, n) for n in names]
        return ToolOutcome(self._txt(reports), summary=", ".join(f"{r['table']}: {r['score']}/100" for r in reports),
                           artifact={"type": "quality", "reports": reports})

    def _tool_forecast(self, query: str, date_column: str, value_column: str,
                       periods: int = 6, frequency: str = "month") -> ToolOutcome:
        res = self.s.engine.query(query)
        df = res.to_df()
        for col in (date_column, value_column):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not in query result {list(df.columns)}")
        fc = forecast_series(df[date_column], df[value_column], periods, frequency)
        hist, fut = fc["history"], fc["forecast"]
        labels = [h["period"] for h in hist] + [f["period"] for f in fut]
        pad = [None] * len(hist)
        spec = {
            "type": "line", "title": f"{value_column}: history and {len(fut)}-period forecast",
            "x_label": date_column, "y_label": value_column, "labels": labels, "truncated": False,
            "datasets": [
                {"label": "Actual", "data": [h["value"] for h in hist] + [None] * len(fut)},
                {"label": "Forecast", "style": "dashed",
                 "data": [None] * (len(hist) - 1) + [hist[-1]["value"]] + [f["value"] for f in fut]},
                {"label": "Upper (95%)", "style": "band", "data": pad + [f["upper"] for f in fut]},
                {"label": "Lower (95%)", "style": "band", "data": pad + [f["lower"] for f in fut]},
            ],
        }
        llm = {k: v for k, v in fc.items() if k != "history"}
        llm["history_points"] = len(hist)
        llm["last_actuals"] = hist[-6:]
        return ToolOutcome(self._txt(llm), summary=f"{fc['method']}, trend {fc['direction']}",
                           artifact={"type": "forecast", "chart": spec, "details": {k: v for k, v in fc.items() if k not in ('history',)}, "query": query.strip()})

    def _tool_describe_table(self, table: str) -> ToolOutcome:
        p = profile_table(self._table(table).df)
        return ToolOutcome(self._txt(p), summary=f"{p['rows']} rows, {len(p['columns'])} columns")
