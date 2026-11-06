"""Thin LLM abstraction. One implementation covers any OpenAI-compatible endpoint
(OpenAI, Gemini, Groq, Anthropic's compat layer, Ollama, OpenRouter, vLLM...).
The agent only depends on the `LLMClient` protocol, so tests use a fake and
adding a native SDK later is a single new class."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

from app.config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Any failure talking to the model. Message is safe to show to users."""


class LLMNotConfigured(LLMError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = "{}"


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse: ...


class OpenAICompatClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    def _get(self):
        if not self.settings.llm_api_key:
            raise LLMNotConfigured(
                "No LLM API key configured. Copy .env.example to .env, set LLM_PROVIDER and the matching API key, then restart."
            )
        if self._client is None:
            from openai import OpenAI  # imported lazily so tests/tools work without the SDK

            self._client = OpenAI(
                api_key=self.settings.llm_api_key, base_url=self.settings.llm_base_url,
                timeout=self.settings.llm_timeout_s, max_retries=3,
            )
        return self._client

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        client = self._get()
        kwargs: dict = {"model": self.settings.llm_model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.settings.llm_temperature is not None:
            kwargs["temperature"] = self.settings.llm_temperature
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - SDK raises many types; normalise them
            name = type(exc).__name__
            log.error("llm_error type=%s detail=%s", name, str(exc)[:300])
            if "Authentication" in name or "PermissionDenied" in name:
                raise LLMError("The LLM provider rejected the API key. Check LLM_API_KEY / provider settings.") from exc
            if "RateLimit" in name:
                raise LLMError("The LLM provider rate limit was hit. Wait a moment and try again.") from exc
            if "Timeout" in name or "Connection" in name:
                raise LLMError("Could not reach the LLM provider (timeout / network). Please retry.") from exc
            raise LLMError(f"LLM request failed ({name}): {str(exc)[:200]}") from exc

        msg = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw)
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {"__parse_error__": raw[:200]}
            calls.append(ToolCall(id=tc.id or f"call_{len(calls)}", name=tc.function.name, arguments=args, raw_arguments=raw))
        usage = {}
        if getattr(resp, "usage", None):
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
        return LLMResponse(content=msg.content, tool_calls=calls, usage=usage)
