import os

from core.llm.base import LLMProvider


def get_provider(name: str | None = None, **kwargs) -> LLMProvider:
    """Return an LLM provider by name (default from $LLM_PROVIDER, else 'anthropic').

    Copilot is imported lazily so the Anthropic path never needs the Copilot SDK.
    """
    name = (name or os.environ.get("LLM_PROVIDER", "anthropic")).lower()
    if name in ("anthropic", "claude"):
        from core.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(**kwargs)
    if name == "copilot":
        from core.llm.copilot_provider import CopilotProvider

        return CopilotProvider(**kwargs)
    raise ValueError(
        f"Unknown LLM provider: {name!r} (expected 'anthropic', 'claude', or 'copilot')"
    )


__all__ = ["LLMProvider", "get_provider"]
