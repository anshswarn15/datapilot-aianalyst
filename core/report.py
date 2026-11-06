"""Export a session as a Markdown report."""
from __future__ import annotations

from datetime import datetime


def _md_table(columns: list[str], rows: list[list], limit: int = 15) -> str:
    esc = lambda v: "" if v is None else str(v).replace("|", "\\|")
    out = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    out += ["| " + " | ".join(esc(v) for v in r) + " |" for r in rows[:limit]]
    if len(rows) > limit:
        out.append(f"\n_... {len(rows) - limit} more rows not shown_")
    return "\n".join(out)


def build_markdown_report(session) -> str:
    lines = ["# DataPilot analysis report", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", "", "## Datasets"]
    for name, t in session.tables.items():
        lines.append(f"- **{name}** ({t.rows:,} rows, {t.df.shape[1]} columns) - from `{t.filename}`")
    for i, turn in enumerate(session.turns, 1):
        lines += ["", f"## {i}. {turn['question']}", "", turn["answer"], ""]
        for a in turn["artifacts"]:
            kind = a["type"]
            if kind == "sql":
                lines += [f"**Query - {a.get('title', '')}**", "```sql", a["query"], "```", _md_table(a["columns"], a["rows"]), ""]
            elif kind in ("chart", "forecast"):
                c = a["chart"]
                lines += [f"**Chart:** {c['title']} ({c['type']})", ""]
                if a.get("query"):
                    lines += ["```sql", a["query"], "```", ""]
            elif kind == "anomalies":
                r = a["report"]
                lines += [f"**Anomalies in `{a['table']}`** - {r['total_flagged']} flagged by {r['method']} rule", ""]
                lines += [f"- row {x['row_index']}: {x['reason']}" for x in r["anomalies"][:10]] + [""]
            elif kind == "quality":
                for q in a["reports"]:
                    lines += [f"**Data quality `{q['table']}`: {q['score']}/100**"]
                    lines += [f"- [{x['severity']}] {x['column'] or 'table'}: {x['issue']} - {x['detail']}" for x in q["issues"]] + [""]
    return "\n".join(lines)
