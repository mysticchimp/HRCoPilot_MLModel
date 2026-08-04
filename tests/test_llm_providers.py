import json
import types as pytypes
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from core.llm import get_provider
from core.llm.base import LLMProvider


class Person(BaseModel):
    name: str
    age: int


# ----------------------------- factory -----------------------------
def test_get_provider_anthropic():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        with patch("anthropic.Anthropic", return_value=MagicMock()):
            provider = get_provider("anthropic")
    from core.llm.anthropic_provider import AnthropicProvider

    assert isinstance(provider, AnthropicProvider)


def test_get_provider_claude_alias():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        with patch("anthropic.Anthropic", return_value=MagicMock()):
            provider = get_provider("claude")
    from core.llm.anthropic_provider import AnthropicProvider

    assert isinstance(provider, AnthropicProvider)


def test_get_provider_copilot_is_lazy():
    from core.llm.copilot_provider import CopilotProvider

    provider = get_provider("copilot")
    assert isinstance(provider, CopilotProvider)


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError):
        get_provider("bogus")


def test_get_provider_rejects_gemini():
    with pytest.raises(ValueError, match="gemini"):
        get_provider("gemini")


# ------------------------- AnthropicProvider --------------------------
def _anthropic(tool_input=None, text=None):
    fake_client = MagicMock()
    blocks = []
    if tool_input is not None:
        blocks.append(pytypes.SimpleNamespace(type="tool_use", input=tool_input, name="emit_result"))
    if text is not None:
        blocks.append(pytypes.SimpleNamespace(type="text", text=text))
    fake_client.messages.create.return_value = pytypes.SimpleNamespace(content=blocks)
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        with patch("anthropic.Anthropic", return_value=fake_client):
            from core.llm.anthropic_provider import AnthropicProvider

            return AnthropicProvider(), fake_client


def test_anthropic_generate_text_uses_text_model():
    provider, client = _anthropic(text="hello world")
    assert provider.generate_text("hi", system="sys", temperature=0.4) == "hello world"
    assert client.messages.create.call_args.kwargs["model"] == "claude-sonnet-4-6"


def test_anthropic_generate_structured_uses_tool_use():
    provider, client = _anthropic(tool_input={"name": "Ada", "age": 36})
    out = provider.generate_structured("who", Person, system="sys")
    assert isinstance(out, Person) and out.name == "Ada" and out.age == 36
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "emit_result"}


def test_anthropic_requires_api_key():
    from core.llm.anthropic_provider import AnthropicProvider

    with patch.dict("os.environ", {}, clear=True):
        with patch("dotenv.load_dotenv"):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                AnthropicProvider()


# ------------------------- CopilotProvider -------------------------
class _FakeSession:
    def __init__(self, tools):
        self._tools = tools or []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def on(self, cb):
        pass

    async def send_and_wait(self, prompt, timeout=60.0):
        if self._tools:  # structured path: simulate the model calling emit_result
            invocation = pytypes.SimpleNamespace(
                arguments={"name": "Grace", "age": 45}, tool_name="emit_result", tool_call_id="1", session_id="s"
            )
            await self._tools[0].handler(invocation)
            return pytypes.SimpleNamespace(data=pytypes.SimpleNamespace(content="Recorded."))
        return pytypes.SimpleNamespace(data=pytypes.SimpleNamespace(content="copilot text reply"))


class _FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def create_session(self, **kwargs):
        return _FakeSession(kwargs.get("tools"))


def test_copilot_generate_text(tmp_path):
    from core.llm import copilot_provider

    with patch.object(copilot_provider, "CopilotClient", _FakeClient):
        provider = copilot_provider.CopilotProvider(base_directory=str(tmp_path), working_directory=str(tmp_path))
        out = provider.generate_text("hi", system="sys")
    assert out == "copilot text reply"


def test_copilot_generate_structured_via_tool_call(tmp_path):
    from core.llm import copilot_provider

    with patch.object(copilot_provider, "CopilotClient", _FakeClient):
        provider = copilot_provider.CopilotProvider(base_directory=str(tmp_path), working_directory=str(tmp_path))
        out = provider.generate_structured("who", Person, system="sys")
    assert isinstance(out, Person) and out.name == "Grace" and out.age == 45


class _FakeSessionWrapped(_FakeSession):
    async def send_and_wait(self, prompt, timeout=60.0):
        # SDK wraps whole-object tool args under a single "$RESULT" key
        invocation = pytypes.SimpleNamespace(
            arguments={"$RESULT": {"name": "Env", "age": 7}}, tool_name="emit_result", tool_call_id="1", session_id="s"
        )
        await self._tools[0].handler(invocation)
        return pytypes.SimpleNamespace(data=pytypes.SimpleNamespace(content="Recorded."))


class _FakeClientWrapped(_FakeClient):
    async def create_session(self, **kwargs):
        return _FakeSessionWrapped(kwargs.get("tools"))


def test_copilot_structured_unwraps_result_envelope(tmp_path):
    from core.llm import copilot_provider

    with patch.object(copilot_provider, "CopilotClient", _FakeClientWrapped):
        provider = copilot_provider.CopilotProvider(base_directory=str(tmp_path), working_directory=str(tmp_path))
        out = provider.generate_structured("who", Person, system="sys")
    assert isinstance(out, Person) and out.name == "Env" and out.age == 7


class _FakeSessionJsonStringWrapped(_FakeSession):
    async def send_and_wait(self, prompt, timeout=60.0):
        invocation = pytypes.SimpleNamespace(
            arguments={"$RESULT": json.dumps({"name": "Json", "age": 8})},
            tool_name="emit_result",
            tool_call_id="1",
            session_id="s",
        )
        await self._tools[0].handler(invocation)
        return pytypes.SimpleNamespace(data=pytypes.SimpleNamespace(content="Recorded."))


class _FakeClientJsonStringWrapped(_FakeClient):
    async def create_session(self, **kwargs):
        return _FakeSessionJsonStringWrapped(kwargs.get("tools"))


def test_copilot_structured_unwraps_json_string_envelope(tmp_path):
    from core.llm import copilot_provider

    with patch.object(copilot_provider, "CopilotClient", _FakeClientJsonStringWrapped):
        provider = copilot_provider.CopilotProvider(
            base_directory=str(tmp_path), working_directory=str(tmp_path)
        )
        out = provider.generate_structured("who", Person, system="sys")
    assert isinstance(out, Person) and out.name == "Json" and out.age == 8


# --------------------------- delegation ----------------------------
class _FakeProvider(LLMProvider):
    def __init__(self, text="TEXT", structured=None):
        self._text = text
        self._structured = structured
        self.calls = []

    def generate_text(self, prompt, system=None, temperature=None, model=None):
        self.calls.append(("text", prompt, system, temperature))
        return self._text

    def generate_structured(self, prompt, schema, system=None, temperature=None, model=None):
        self.calls.append(("structured", prompt, system, temperature))
        return self._structured


def test_process_jd_delegates_to_provider():
    from core.jd_extraction import process_jd
    from models.data_models import Company, JobRoleSchema, Skill
    from models.enums import ImportanceLevel

    jd = JobRoleSchema(
        role="X", company=Company(name="C"), responsibilities=["r"],
        skills=[Skill(skill="s", priority=ImportanceLevel.ESSENTIAL, proficiency_level=None)],
    )
    fake = _FakeProvider(structured=jd)
    # Disable hash cache so this always hits the provider.
    assert process_jd("some jd text", provider=fake, cache_dir="0") is jd
    assert fake.calls[0][0] == "structured"


def test_process_jd_empty_raises():
    from core.jd_extraction import process_jd

    with pytest.raises(ValueError):
        process_jd("   ", provider=_FakeProvider())


def test_generate_email_delegates_and_keeps_temperature():
    from core.email_generator import generate_email

    fake = _FakeProvider(text="EMAIL")
    assert generate_email("jd", "{}", provider=fake) == "EMAIL"
    assert fake.calls[0][0] == "text" and fake.calls[0][3] == 0.4


def test_generate_jd_from_profile_delegates():
    from core.jd_generation import generate_jd_from_profile
    from models.candidate import CandidateProfile

    fake = _FakeProvider(text="JD TEXT")
    profile = CandidateProfile(candidate_id="C1", job_title="HR Assistant", skills=["Payroll"])
    out = generate_jd_from_profile(profile, provider=fake)
    assert out == "JD TEXT"
    assert fake.calls[0][0] == "text"
    assert "HR Assistant" in fake.calls[0][1]  # profile serialized into the prompt
