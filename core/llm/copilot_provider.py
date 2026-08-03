import asyncio
import concurrent.futures
import json
import os

from copilot import CopilotClient, PermissionHandler, Tool, ToolResult
from pydantic import ValidationError

from core.llm.base import LLMProvider, T

# All Copilot data (session state, config) is isolated to this project-local dir
# instead of ~/.copilot or the project root.
DEFAULT_COPILOT_DIR = ".ai-recruiter"


def _ensure_ca_bundle() -> None:
    """Point OpenSSL at certifi's CA bundle when the interpreter lacks one.

    The SDK downloads its CLI runtime over HTTPS via urllib; some macOS Python
    builds ship without a populated CA store, which raises
    CERTIFICATE_VERIFY_FAILED. Setting SSL_CERT_FILE fixes it without touching
    the system trust store.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


def _run_sync(coro):
    """Run an async coroutine from sync code, safely even inside a running loop (e.g. Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


class CopilotProvider(LLMProvider):
    """LLMProvider backed by the GitHub Copilot SDK.

    Auth defaults to the logged-in GitHub user (pass `github_token` to override).
    The SDK has no native structured-output flag, so `generate_structured` uses a
    single tool call whose parameters are the target schema's JSON schema and reads
    back the validated arguments. `temperature` is accepted for interface parity but
    ignored (the SDK does not expose it).
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4.5",
        github_token: str | None = None,
        timeout: float = 120.0,
        base_directory: str | None = None,
        working_directory: str | None = None,
    ):
        self.model = model
        self.github_token = github_token
        self.timeout = timeout
        # base_directory -> COPILOT_HOME, which holds BOTH auth and session state.
        # Default None keeps the SDK default (~/.copilot) so the EXISTING Copilot
        # login is reused; pointing it at a fresh dir would require re-authenticating
        # there. Session state therefore lives under ~/.copilot, not the project root.
        self.base_directory = os.path.abspath(base_directory) if base_directory else None
        # working_directory -> the runtime process cwd. Default to an isolated,
        # git-ignored subfolder so nothing the runtime writes lands in the project root.
        self.working_directory = os.path.abspath(working_directory or os.path.join(os.getcwd(), DEFAULT_COPILOT_DIR))

    def _client_kwargs(self) -> dict:
        _ensure_ca_bundle()
        os.makedirs(self.working_directory, exist_ok=True)
        kwargs = {"working_directory": self.working_directory}
        if self.base_directory:
            os.makedirs(self.base_directory, exist_ok=True)
            kwargs["base_directory"] = self.base_directory
        if self.github_token:
            kwargs["github_token"] = self.github_token
        return kwargs

    def generate_text(self, prompt, system=None, temperature=None, model=None) -> str:
        return _run_sync(self._agenerate_text(prompt, system, model or self.model))

    def generate_structured(self, prompt, schema: type[T], system=None, temperature=None, model=None) -> T:
        return _run_sync(self._agenerate_structured(prompt, schema, system, model or self.model))

    async def _agenerate_text(self, prompt, system, model) -> str:
        async with CopilotClient(**self._client_kwargs()) as client:
            session_kwargs = {
                "on_permission_request": PermissionHandler.approve_all,
                "model": model,
                "available_tools": [],  # pure text generation: no agentic tools
            }
            if system:
                session_kwargs["system_message"] = {"mode": "append", "content": system}
            async with await client.create_session(**session_kwargs) as session:
                event = await session.send_and_wait(prompt, timeout=self.timeout)
                content = getattr(getattr(event, "data", None), "content", None) if event else None
                if not content:
                    raise ValueError("Copilot returned an empty response")
                return content

    async def _agenerate_structured(self, prompt, schema: type[T], system, model) -> T:
        captured: dict = {}

        async def handler(invocation) -> ToolResult:
            captured["arguments"] = invocation.arguments
            return ToolResult(text_result_for_llm="Recorded.")

        tool = Tool(
            name="emit_result",
            description="Return the final answer as structured data matching the required schema.",
            parameters=schema.model_json_schema(),
            handler=handler,
        )
        directive = (
            "\n\nCall the `emit_result` tool exactly once with the fully populated "
            "structured result. Do not reply with prose."
        )

        async with CopilotClient(**self._client_kwargs()) as client:
            session_kwargs = {
                "on_permission_request": PermissionHandler.approve_all,
                "model": model,
                "tools": [tool],
                "available_tools": ["emit_result"],
            }
            if system:
                session_kwargs["system_message"] = {"mode": "append", "content": system}
            async with await client.create_session(**session_kwargs) as session:
                await session.send_and_wait(prompt + directive, timeout=self.timeout)

        if "arguments" not in captured:
            raise ValueError("Copilot did not call emit_result; no structured output produced")
        arguments = captured["arguments"]
        # Models sometimes wrap the whole payload under a single key (the key name
        # is inconsistent: "result", "$RESULT", ...). Try direct validation first,
        # then unwrap a single-key envelope.
        try:
            return schema.model_validate(arguments)
        except ValidationError:
            if isinstance(arguments, dict) and len(arguments) == 1:
                inner = next(iter(arguments.values()))
                if isinstance(inner, dict):
                    return schema.model_validate(inner)
                if isinstance(inner, str):
                    parsed = json.loads(inner)
                    if isinstance(parsed, dict):
                        return schema.model_validate(parsed)
            raise
