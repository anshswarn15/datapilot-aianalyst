"""The agent loop: LLM <-> tools, emitting events so the UI can stream progress.

Flow per user message:
    build prompt (system + live schema + history) -> LLM
      -> if tool calls: execute them, append results, loop (max N steps)
      -> else: final answer
Events yielded: status | tool_call | tool_result | artifact | final | error
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Iterator

from app.agent.tools import TOOL_SCHEMAS, ToolExecutor
from app.config import Settings
from app.core.profiling import schema_prompt
from app.core.session import Session
from app.llm.client import LLMClient, LLMError
from app.llm.prompts import NO_DATA_MESSAGE, SYSTEM_PROMPT

log = logging.getLogger(__name__)
_FOLLOWUPS = re.compile(r"```followups\s*(.*?)```", re.S | re.I)


def split_followups(text: str) -> tuple[str, list[str]]:
    """Separate the trailing ```followups block from the answer text."""
    m = _FOLLOWUPS.search(text)
    if not m:
        return text.strip(), []
    items = [re.sub(r"^[-*\d.\s]+", "", ln).strip() for ln in m.group(1).splitlines()]
    return _FOLLOWUPS.sub("", text).strip(), [i for i in items if i][:4]


class DataAnalystAgent:
    def __init__(self, llm: LLMClient, settings: Settings):
        self.llm = llm
        self.settings = settings

    def run(self, session: Session, user_message: str) -> Iterator[dict]:
        turn_id = uuid.uuid4().hex[:8]
        started = time.monotonic()

        if not session.tables:
            yield self._finish(session, turn_id, user_message, NO_DATA_MESSAGE, [], [], [], started, {})
            return

        system = SYSTEM_PROMPT.format(schema=schema_prompt(session.tables))
        history = session.history[-2 * self.settings.history_turns:]
        messages: list[dict] = [{"role": "system", "content": system}, *history,
                                {"role": "user", "content": user_message}]
        executor = ToolExecutor(session)
        artifacts: list[dict] = []
        steps: list[dict] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}
        final_text: str | None = None

        yield {"type": "status", "message": "Thinking..."}
        try:
            for step in range(self.settings.max_agent_steps + 1):
                last_chance = step == self.settings.max_agent_steps
                resp = self.llm.chat(messages, None if last_chance else TOOL_SCHEMAS)
                usage["llm_calls"] += 1
                usage["prompt_tokens"] += resp.usage.get("prompt_tokens", 0)
                usage["completion_tokens"] += resp.usage.get("completion_tokens", 0)

                if not resp.tool_calls or last_chance:
                    final_text = resp.content or ""
                    break

                messages.append({
                    "role": "assistant", "content": resp.content or "",
                    "tool_calls": [{"id": c.id, "type": "function",
                                    "function": {"name": c.name, "arguments": c.raw_arguments}} for c in resp.tool_calls],
                })
                for call in resp.tool_calls:
                    yield {"type": "tool_call", "name": call.name, "args": call.arguments}
                    t0 = time.monotonic()
                    outcome = executor.execute(call.name, call.arguments)
                    ms = round((time.monotonic() - t0) * 1000)
                    log.info("tool name=%s ok=%s ms=%s session=%s", call.name, outcome.ok, ms, session.id[:8])
                    steps.append({"tool": call.name, "args": call.arguments, "ok": outcome.ok,
                                  "summary": outcome.summary, "ms": ms})
                    yield {"type": "tool_result", "name": call.name, "ok": outcome.ok, "summary": outcome.summary, "ms": ms}
                    if outcome.artifact:
                        artifacts.append(outcome.artifact)
                        yield {"type": "artifact", "artifact": outcome.artifact}
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": outcome.llm_text})
        except LLMError as exc:
            log.warning("agent_llm_error session=%s err=%s", session.id[:8], exc)
            yield {"type": "error", "message": str(exc)}
            return

        answer, followups = split_followups(final_text or "")
        if not answer:
            answer = "I ran the analysis but could not produce a written summary. Please look at the results above or rephrase the question."
        yield self._finish(session, turn_id, user_message, answer, followups, artifacts, steps, started, usage)

    # ------------------------------------------------------------------------------------
    def _finish(self, session, turn_id, question, answer, followups, artifacts, steps, started, usage) -> dict:
        elapsed = round((time.monotonic() - started) * 1000)
        queries = [a["query"] for a in artifacts if a.get("query")]
        remembered = answer + (("\n\n[SQL used this turn]\n" + "\n".join(queries[-3:])) if queries else "")
        session.history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": remembered}])
        del session.history[:-2 * max(self.settings.history_turns, 1) * 2]  # hard cap on memory
        turn = {"id": turn_id, "question": question, "answer": answer, "followups": followups,
                "artifacts": artifacts, "steps": steps, "elapsed_ms": elapsed, "usage": usage, "created": time.time()}
        session.turns.append(turn)
        log.info("turn_done session=%s ms=%s tools=%d usage=%s", session.id[:8], elapsed, len(steps), json.dumps(usage))
        return {"type": "final", "turn": turn}
