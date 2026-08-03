"""Live smoke test for the GitHub Copilot LLM provider.

Run it from the project root (uses your logged-in GitHub user for auth):

    uv run python scripts/smoke_copilot.py

On first run the SDK downloads its CLI runtime. All Copilot session state/config
is isolated to the git-ignored ./.ai-recruiter directory.
"""

import os
import sys

# Make the project root importable when this file is run directly (python scripts/...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.jd_extraction import load_sample_jd, process_jd
from core.llm import get_provider

MODEL = "claude-opus-4.7"  # or "gpt-5", etc.


def main() -> None:
    provider = get_provider("copilot", model=MODEL)
    print(f"Copilot session/data dir: {provider.base_directory}\n")

    # 1) Free-text generation (low risk)
    print("=== 1. text generation ===")
    try:
        text = provider.generate_text("Say hello to a candidate in exactly one sentence.")
        print(text.strip(), "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[text generation FAILED] {type(exc).__name__}: {exc}\n")

    # 2) Structured extraction on the deeply-nested JobRoleSchema (higher risk)
    print("=== 2. structured JD extraction (JobRoleSchema) ===")
    try:
        jd = process_jd(load_sample_jd("./jd/sample_hr_assistant_jd.txt"), provider=provider)
        print("role:", jd.role)
        print("skills:", [s.skill for s in jd.skills][:8])
        print("STRUCTURED OUTPUT OK")
    except Exception as exc:  # noqa: BLE001
        print(f"[structured extraction FAILED] {type(exc).__name__}: {exc}")
        print("-> If this failed on the nested schema, we'll add a prompt+JSON-parse fallback.")


if __name__ == "__main__":
    main()
