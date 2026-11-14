"""Minimal LLM client abstraction so agents can be tested without network access."""

from __future__ import annotations

import os
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str, max_tokens: int = 300) -> str: ...


class AnthropicClient:
    """Calls the Anthropic Messages API. Reads ``ANTHROPIC_API_KEY`` from the environment.

    The model can be changed with the ``CURIO_MODEL`` environment variable.
    """

    DEFAULT_MODEL = "claude-sonnet-5-5"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        import anthropic  # lazy: optional dependency

        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model or os.environ.get("CURIO_MODEL", self.DEFAULT_MODEL)

    def complete(self, system: str, prompt: str, max_tokens: int = 300) -> str:
        response = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


def default_llm() -> LLMClient | None:
    """Return an Anthropic client if a key is configured, otherwise ``None`` (offline mode)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        return AnthropicClient()
    except ImportError:
        return None
