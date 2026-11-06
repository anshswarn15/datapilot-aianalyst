"""HTTP API. Deliberately thin: validation + wiring only. Business logic lives in app/core and app/agent."""
import json
import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from app.core.dashboard import build_dashboard
from app.core.ingestion import IngestionError
from app.core.profiling import profile_table, quality_report
from app.core.report import build_markdown_report
from app.core.session import Session, SessionNotFound
from app.core.sql_engine import SQLExecutionError, SQLValidationError
from app.schemas import ChatRequest, SQLRequest

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _session(request: Request, sid: str) -> Session:
    try:
        return request.app.state.store.get(sid)
    except SessionNotFound:
        raise HTTPException(404, "Session not found or expired. Refresh the page to start a new session.")


def _table_payload(t) -> dict:
    return {"name": t.name, "filename": t.filename, "rows": t.rows, "warnings": t.warnings,
            "column_map": t.column_map, "profile": profile_table(t.df)}


@router.get("/health")
def health(request: Request):
    s = request.app.state.settings
    return {"status": "ok", "llm_configured": s.llm_configured, "provider": s.llm_provider,
            "model": s.llm_model, "active_sessions": len(request.app.state.store)}


@router.post("/sessions")
def create_session(request: Request):
    s = request.app.state.store.create()
    return {"session_id": s.id}


@router.get("/sessions/{sid}")
def get_session(sid: str, request: Request):
    s = _session(request, sid)
    return {"session_id": s.id, "tables": [_table_payload(t) for t in s.tables.values()],
            "turns": s.turns, "llm_configured": request.app.state.settings.llm_configured}


@router.delete("/sessions/{sid}")
def delete_session(sid: str, request: Request):
    request.app.state.store.delete(sid)
    return {"deleted": True}


@router.post("/sessions/{sid}/files")
def upload_files(sid: str, request: Request, files: list[UploadFile] = File(...)):
    s = _session(request, sid)
    max_bytes = request.app.state.settings.max_upload_bytes
    results = []
    for f in files:
        try:
            content = f.file.read(max_bytes + 1)  # never read more than limit+1 bytes into memory
            table = s.add_csv(f.filename or "upload.csv", content)
            results.append({"filename": f.filename, "status": "loaded", **_table_payload(table)})
            log.info("upload_ok session=%s table=%s rows=%d", s.id[:8], table.name, table.rows)
        except IngestionError as exc:
            log.info("upload_rejected session=%s file=%s reason=%s", s.id[:8], f.filename, exc)
            results.append({"filename": f.filename, "status": "rejected", "error": str(exc)})
        except Exception:  # noqa: BLE001 - one bad upload must not hide other file results
            log.exception("upload_crash session=%s file=%s", s.id[:8], f.filename)
            results.append({"filename": f.filename, "status": "rejected",
                            "error": f"'{f.filename}': the file could not be processed."})
    return {"results": results}


@router.post("/sessions/{sid}/load-samples")
def load_samples(sid: str, request: Request):
    s = _session(request, sid)
    sample_dir = request.app.state.settings.sample_dir
    loaded_files = {t.filename for t in s.tables.values()}
    results = []
    for path in sorted(sample_dir.glob("*.csv")):
        if path.name in loaded_files:
            continue
        try:
            results.append({"filename": path.name, "status": "loaded", **_table_payload(s.add_csv(path.name, path.read_bytes()))})
        except IngestionError as exc:
            results.append({"filename": path.name, "status": "rejected", "error": str(exc)})
    return {"results": results}


@router.delete("/sessions/{sid}/files/{table}")
def remove_file(sid: str, table: str, request: Request):
    s = _session(request, sid)
    try:
        s.remove_table(table)
    except KeyError:
        raise HTTPException(404, f"Table '{table}' not found.")
    return {"removed": table}


@router.delete("/sessions/{sid}/files")
def clear_files(sid: str, request: Request):
    """Remove every uploaded table and its related conversation from the session."""
    s = _session(request, sid)
    removed = list(s.tables)
    for name in removed:
        s.remove_table(name)
    s.reset_chat()
    return {"removed": removed}


@router.post("/sessions/{sid}/chat")
def chat(sid: str, body: ChatRequest, request: Request):
    s = _session(request, sid)
    if not s.lock.acquire(blocking=False):
        raise HTTPException(409, "Another question is still being processed in this session.")
    try:
        final = None
        for event in request.app.state.agent.run(s, body.message):
            if event["type"] == "error":
                raise HTTPException(502, event["message"])
            if event["type"] == "final":
                final = event["turn"]
        return final
    finally:
        s.lock.release()


@router.post("/sessions/{sid}/chat/stream")
def chat_stream(sid: str, body: ChatRequest, request: Request):
    """Server-Sent Events: live progress (tool calls, partial artifacts) followed by the final turn."""
    s = _session(request, sid)
    if not s.lock.acquire(blocking=False):
        raise HTTPException(409, "Another question is still being processed in this session.")

    def events():
        try:
            for event in request.app.state.agent.run(s, body.message):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception:  # noqa: BLE001
            log.exception("stream_crash session=%s", s.id[:8])
            yield f"data: {json.dumps({'type': 'error', 'message': 'Unexpected server error while answering.'})}\n\n"
        finally:
            s.lock.release()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/sessions/{sid}/sql")
def run_sql(sid: str, body: SQLRequest, request: Request):
    """Let the user run (or tweak) the generated SQL themselves - same read-only guard as the agent."""
    s = _session(request, sid)
    try:
        return s.engine.query(body.query).to_dict(200)
    except (SQLValidationError, SQLExecutionError) as exc:
        raise HTTPException(400, str(exc))


@router.get("/sessions/{sid}/dashboard")
def dashboard(sid: str, request: Request):
    s = _session(request, sid)
    if not s.tables:
        raise HTTPException(400, "Upload a CSV first.")
    return build_dashboard(s.tables)


@router.get("/sessions/{sid}/quality")
def quality(sid: str, request: Request):
    s = _session(request, sid)
    return {"reports": [quality_report(t.df, n) for n, t in s.tables.items()]}


@router.post("/sessions/{sid}/reset-chat")
def reset_chat(sid: str, request: Request):
    _session(request, sid).reset_chat()
    return {"reset": True}


@router.get("/sessions/{sid}/export.md")
def export_markdown(sid: str, request: Request):
    s = _session(request, sid)
    return Response(build_markdown_report(s), media_type="text/markdown",
                    headers={"Content-Disposition": 'attachment; filename="datapilot_report.md"'})
