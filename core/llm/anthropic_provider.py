"""Anthropic Claude provider — structured JD extraction via tool_use."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from core.llm.base import LLMProvider, T

DEFAULT_STRUCTURED_MODEL = "claude-sonnet-4-6"
DEFAULT_TEXT_MODEL = "claude-sonnet-4-6"


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    return t.strip()


def _tool_input(response) -> dict:
    """Pull the structured tool_use payload (preferred) or fall back to text JSON."""
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and isinstance(block.input, dict):
            return block.input
    texts = [b.text for b in response.content if getattr(b, "type", None) == "text" and b.text]
    if texts:
        return json.loads(_strip_fences(texts[0]))
    raise ValueError("Anthropic returned no structured content")


class AnthropicProvider(LLMProvider):
    """LLMProvider backed by the Anthropic Messages API.

    Reads ``ANTHROPIC_API_KEY`` from the environment (same pattern as Sourcing_Apify /
    contra6_source2). Structured calls force a single tool_use whose input_schema is the
    target Pydantic model's JSON schema — reliable for complex shapes like JobRoleSchema.
    """

    def __init__(
        self,
        structured_model: str = DEFAULT_STRUCTURED_MODEL,
        text_model: str = DEFAULT_TEXT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        env_path: str = "./.env",
        api_key: str | None = None,
    ):
        load_dotenv(env_path)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        from anthropic import Anthropic

        self.client = Anthropic(api_key=key)
        self.structured_model = structured_model
        self.text_model = text_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_text(self, prompt, system=None, temperature=None, model=None) -> str:
        kwargs: dict = {
            "model": model or self.text_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)
        texts = [b.text for b in response.content if getattr(b, "type", None) == "text" and b.text]
        if texts:
            return "".join(texts)
        raise ValueError("Anthropic returned an empty response")

    def generate_structured(
        self,
        prompt,
        schema: type[T],
        system=None,
        temperature=None,
        model=None,
        *,
        cache_system: bool = False,
    ) -> T:
        tool_name = "emit_result"
        json_schema = schema.model_json_schema()
        # Anthropic tools expect a top-level object schema; strip Pydantic title noise.
        json_schema.pop("title", None)

        system_parts = [
            system or "",
            "You MUST call the emit_result tool with a payload that validates against the tool schema. "
            "Do not return free-form prose.",
        ]
        system_text = "\n\n".join(p for p in system_parts if p)

        # Prompt caching: mark the shared system block (instructions + JD) as ephemeral
        # so batch narrate calls reuse the prefix across candidates.
        if cache_system and system_text:
            system_arg: str | list[dict] = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_arg = system_text

        response = self.client.messages.create(
            model=model or self.structured_model,
            max_tokens=self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            system=system_arg,
            tools=[
                {
                    "name": tool_name,
                    "description": f"Emit a structured {schema.__name__} result",
                    "input_schema": json_schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
        )
        payload = _tool_input(response)
        return schema.model_validate(payload)
