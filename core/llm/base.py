from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Provider-agnostic LLM interface (the model-side analogue of CandidateAdapter).

    Concrete providers wrap a specific backend (Anthropic Claude, GitHub Copilot SDK, ...).
    Callers depend only on this interface, so swapping model providers is a config
    change, not a code change.
    """

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> str:
        """Return a free-text completion."""
        raise NotImplementedError

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system: str | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> T:
        """Return a validated instance of `schema`."""
        raise NotImplementedError
