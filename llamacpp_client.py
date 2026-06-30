"""Lean llama.cpp server inference client optimized for the cluster branch."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_alias_map: dict[str, int] = {}
_registry: dict[str, str] = {}


def set_alias_map(mapping: dict[str, int]) -> None:
    """Register alias → port mapping for the llama.cpp server."""
    global _alias_map
    _alias_map = dict(mapping)


def _base_url(alias: str) -> str:
    host = os.environ.get("LLAMACPP_HOST", "127.0.0.1")
    port = os.getenv("LLM_MODEL_PORT")
    if port:
        port = int(port)
    else:
        port = _alias_map.get(alias, 8080)
    return f"http://{host}:{port}"


def load_models(aliases: list[str]) -> None:
    """Health-check the llama.cpp server on startup to ensure it is responsive."""
    health_timeout = float(os.environ.get("LLAMACPP_HEALTH_TIMEOUT", "5"))

    for alias in aliases:
        base_url = _base_url(alias)
        print(f"[llama.cpp] Checking {alias} → {base_url} …")
        try:
            resp = requests.get(f"{base_url}/health", timeout=health_timeout)
            resp.raise_for_status()
            _registry[alias] = base_url
            print(f"[llama.cpp] {alias} ready ✓")
        except requests.RequestException as exc:
            raise RuntimeError(
                f"llama.cpp server for '{alias}' not reachable at {base_url}: {exc}"
            )


def _lc_role(msg: BaseMessage) -> str:
    from langchain_core.messages import AIMessage as AI
    from langchain_core.messages import SystemMessage as SM

    if isinstance(msg, SM):
        return "system"
    if isinstance(msg, AI):
        return "assistant"
    return "user"


class ChatLlamaCppServer(BaseChatModel):
    """Drop-in LangChain chat model targeting a local containerized llama-server."""

    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        base_url = _registry.get(self.model)
        if not base_url:
            raise RuntimeError(
                f"Model '{self.model}' not loaded. Call load_models() first."
            )

        formatted_messages = [
            {"role": _lc_role(m), "content": str(m.content)} for m in messages
        ]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        timeout = float(os.environ.get("LLAMACPP_TIMEOUT", "300"))
        resp = requests.post(
            f"{base_url}/v1/chat/completions", json=payload, timeout=timeout
        )
        resp.raise_for_status()

        text: str = resp.json()["choices"][0]["message"]["content"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @property
    def _llm_type(self) -> str:
        return "llamacpp-server"
