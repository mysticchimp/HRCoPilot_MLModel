#!/usr/bin/env python3
"""POST the real 10-candidate fixture to production /score (investigation trigger)."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(os.environ.get("SCORE_FIXTURE", ROOT / ".ai-recruiter" / "real_score_batch_10.json"))
URL = os.environ.get("SCORING_API_URL", "https://contra6-scoring-api.onrender.com/score")


def main() -> int:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload = {
        "jd_text": data["jd_text"],
        "candidates": [
            {"candidate_id": c["candidate_id"], "raw_profile": c["raw_profile"]}
            for c in data["candidates"]
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    print(f"POST {URL} fixture={FIXTURE} n={len(payload['candidates'])} bytes={len(body)}")
    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = resp.read()
            print(f"HTTP {resp.status} bytes={len(out)}")
            print(out[:500].decode("utf-8", errors="replace"))
            return 0
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read()[:500]}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
