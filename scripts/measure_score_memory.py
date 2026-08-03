"""Measure /score RSS + latency with warmed models (4 candidates).

Starts a local uvicorn, waits for model warm-up, samples process RSS
(idle → during request → after), and prints duration.

    COPILOT_SKIP_CLI_DOWNLOAD=1 JD_CACHE_PATH=jd/parsed/hr_assistant_prime_ac.json \\
      uv run python scripts/measure_score_memory.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PORT = int(os.environ.get("SCORE_PORT", "8091"))
BASE = f"http://127.0.0.1:{PORT}"
JD_CACHE = os.environ.get("JD_CACHE_PATH", "jd/parsed/hr_assistant_prime_ac.json")

# 4 nested Apify-shaped candidates (same style as sanity_score_api)
CANDIDATES = [
    {
        "candidate_id": "cand-roxanna",
        "raw_profile": {
            "publicIdentifier": "roxanna",
            "linkedinUrl": "https://www.linkedin.com/in/roxanna",
            "fullName": "Roxanna Ghassemlou",
            "headline": "HR Manager | Dubai",
            "about": "HR professional with 4+ years in employee relations and onboarding.",
            "location": {
                "linkedinText": "Dubai, United Arab Emirates",
                "parsed": {"city": "Dubai", "country": "United Arab Emirates", "countryCode": "AE", "text": "Dubai, UAE"},
            },
            "experience": [
                {
                    "position": "HR Manager",
                    "companyName": "Distinct Group",
                    "duration": "5 mos",
                    "description": "Led HR ops and onboarding.",
                    "startDate": {"text": "Mar 2026"},
                    "endDate": {"text": "Present"},
                },
                {
                    "position": "HR Assistant",
                    "companyName": "Prime Focus",
                    "duration": "2 yrs 3 mos",
                    "description": "Payroll and manufacturing-site HR admin.",
                    "startDate": {"text": "Jan 2024"},
                    "endDate": {"text": "Mar 2026"},
                },
            ],
            "education": [{"degree": "MSc", "fieldOfStudy": "Psychology", "schoolName": "Sussex", "endDate": {"year": 2021}}],
            "skills": [{"name": "Onboarding"}, {"name": "Payroll"}, {"name": "HR Policies"}],
            "languages": [{"name": "English"}, {"name": "Tagalog"}],
        },
    },
    {
        "candidate_id": "cand-entry-hr",
        "raw_profile": {
            "publicIdentifier": "aisha",
            "linkedinUrl": "https://www.linkedin.com/in/aisha",
            "fullName": "Aisha Khan",
            "headline": "HR Assistant | Dubai",
            "about": "HR Assistant focused on recruitment and onboarding.",
            "location": {
                "parsed": {"city": "Dubai", "country": "United Arab Emirates", "countryCode": "AE", "text": "Dubai, UAE"},
            },
            "experience": [
                {
                    "position": "HR Assistant",
                    "companyName": "Gulf Manufacturing LLC",
                    "duration": "1 yr 6 mos",
                    "description": "Interviews, files, onboarding.",
                    "startDate": {"text": "Jan 2025"},
                    "endDate": {"text": "Present"},
                }
            ],
            "education": [{"degree": "BBA", "fieldOfStudy": "HR", "schoolName": "UAEU", "endDate": {"year": 2024}}],
            "skills": [{"name": "Onboarding"}, {"name": "Recruitment"}, {"name": "MS Office"}],
            "languages": [{"name": "English"}, {"name": "Arabic"}],
        },
    },
    {
        "candidate_id": "cand-coordinator",
        "raw_profile": {
            "publicIdentifier": "leena",
            "linkedinUrl": "https://www.linkedin.com/in/leena",
            "fullName": "Leena E",
            "headline": "HR Coordinator",
            "about": "HR coordination across manufacturing sites in the UAE.",
            "location": {
                "parsed": {"city": "Abu Dhabi", "country": "United Arab Emirates", "countryCode": "AE", "text": "Abu Dhabi, UAE"},
            },
            "experience": [
                {
                    "position": "HR Coordinator",
                    "companyName": "Industrial Co",
                    "duration": "3 yrs",
                    "description": "Employee relations, onboarding, admin.",
                    "startDate": {"text": "Jan 2023"},
                    "endDate": {"text": "Present"},
                }
            ],
            "education": [{"degree": "BA", "fieldOfStudy": "Business", "schoolName": "Zayed", "endDate": {"year": 2022}}],
            "skills": [{"name": "Employee Relations"}, {"name": "Onboarding"}, {"name": "HR Administration"}],
            "languages": [{"name": "English"}],
        },
    },
    {
        "candidate_id": "cand-unrelated",
        "raw_profile": {
            "publicIdentifier": "dev",
            "linkedinUrl": "https://www.linkedin.com/in/dev",
            "fullName": "Dev Patel",
            "headline": "Software Engineer",
            "about": "Backend engineer building APIs.",
            "location": {
                "parsed": {"city": "Bengaluru", "country": "India", "countryCode": "IN", "text": "Bengaluru, India"},
            },
            "experience": [
                {
                    "position": "Software Engineer",
                    "companyName": "Acme Tech",
                    "duration": "3 yrs",
                    "description": "Microservices and data pipelines.",
                    "startDate": {"text": "Jan 2023"},
                    "endDate": {"text": "Present"},
                }
            ],
            "education": [{"degree": "B.Tech", "fieldOfStudy": "CS", "schoolName": "IIT", "endDate": {"year": 2022}}],
            "skills": [{"name": "Python"}, {"name": "Go"}],
            "languages": [{"name": "English"}],
        },
    },
]


def _rss_mb(pid: int) -> float:
    """Process tree RSS in MiB (macOS/Linux ``ps``)."""
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
        kb = float(out.strip() or "0")
        return kb / 1024.0
    except Exception:  # noqa: BLE001
        return 0.0


def _tree_rss_mb(pid: int) -> float:
    import psutil

    p = psutil.Process(pid)
    rss = p.memory_info().rss
    for c in p.children(recursive=True):
        try:
            rss += c.memory_info().rss
        except Exception:  # noqa: BLE001
            pass
    return rss / (1024 * 1024)


def _wait_ready(timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=5) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") == "ok" and data.get("models_ready"):
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    raise TimeoutError("server did not become ready")


def _post_score(jd_text: str) -> tuple[dict, float]:
    payload = json.dumps({"jd_text": jd_text, "candidates": CANDIDATES}).encode()
    req = urllib.request.Request(
        f"{BASE}/score", data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode())
    return body, time.perf_counter() - t0


def main():
    try:
        import psutil  # noqa: F401
    except ImportError:
        print("installing psutil for RSS sampling...", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "psutil"], cwd=str(ROOT)
        )
        import psutil  # noqa: F401

    jd_path = ROOT / "jd" / "sample_hr_assistant_jd.txt"
    jd_text = jd_path.read_text(encoding="utf-8") if jd_path.exists() else "HR Assistant Dubai onboarding payroll."

    env = os.environ.copy()
    env["COPILOT_SKIP_CLI_DOWNLOAD"] = "1"
    env["JD_CACHE_PATH"] = JD_CACHE
    env.setdefault("LLM_PROVIDER", "anthropic")

    cmd = [
        sys.executable, "-m", "uvicorn", "api.main:app",
        "--host", "127.0.0.1", "--port", str(PORT),
    ]
    print(f"starting: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    peak = {"idle": 0.0, "during": 0.0, "after": 0.0}
    stop_sample = False

    def sampler():
        while not stop_sample and proc.poll() is None:
            try:
                peak["during"] = max(peak["during"], _tree_rss_mb(proc.pid))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.05)

    try:
        _wait_ready()
        peak["idle"] = _tree_rss_mb(proc.pid)
        print(f"idle RSS: {peak['idle']:.1f} MB", flush=True)

        import threading

        peak["during"] = peak["idle"]
        t = threading.Thread(target=sampler, daemon=True)
        t.start()

        body1, dur1 = _post_score(jd_text)
        print(f"request #1: {dur1:.2f}s count={body1.get('count')}", flush=True)

        body2, dur2 = _post_score(jd_text)
        print(f"request #2: {dur2:.2f}s count={body2.get('count')}", flush=True)

        stop_sample = True
        t.join(timeout=2)
        time.sleep(0.5)
        peak["after"] = _tree_rss_mb(proc.pid)

        print("\n=== MEMORY / LATENCY ===")
        print(f"idle_rss_mb:    {peak['idle']:.1f}")
        print(f"peak_during_mb: {peak['during']:.1f}")
        print(f"after_rss_mb:   {peak['after']:.1f}")
        print(f"req1_seconds:   {dur1:.2f}")
        print(f"req2_seconds:   {dur2:.2f}")
        print(f"candidates:     {len(CANDIDATES)}")
        assert body2.get("count") == len(CANDIDATES)
    finally:
        stop_sample = True
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        # drain a bit of server log on failure
        if proc.stdout:
            try:
                leftover = proc.stdout.read()
                if leftover:
                    print("--- server log (tail) ---", flush=True)
                    print("\n".join(leftover.strip().splitlines()[-40:]), flush=True)
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
