"""Measure /score RSS + latency with warmed models.

Default fixture: REAL richest-10 LinkedIn profiles (nested Apify shape) at
``.ai-recruiter/real_score_batch_10.json``. Override with SCORE_FIXTURE=.

    COPILOT_SKIP_CLI_DOWNLOAD=1 JD_CACHE_PATH=jd/parsed/hr_assistant_prime_ac.json \\
      uv run python scripts/measure_score_memory.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PORT = int(os.environ.get("SCORE_PORT", "8091"))
BASE = f"http://127.0.0.1:{PORT}"
JD_CACHE = os.environ.get("JD_CACHE_PATH", "jd/parsed/hr_assistant_prime_ac.json")
FIXTURE = Path(os.environ.get("SCORE_FIXTURE", str(ROOT / ".ai-recruiter" / "real_score_batch_10.json")))


def _load_payload() -> tuple[str, list]:
    if not FIXTURE.exists():
        raise SystemExit(f"Missing fixture {FIXTURE}")
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    jd = data.get("jd_text") or (ROOT / "jd" / "sample_hr_assistant_jd.txt").read_text(encoding="utf-8")
    cands = data["candidates"]
    print(f"fixture={FIXTURE} candidates={len(cands)} jd_chars={len(jd)}", flush=True)
    return jd, cands


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


def _post_score(jd_text: str, candidates: list) -> tuple[dict, float]:
    payload = json.dumps({"jd_text": jd_text, "candidates": candidates}).encode()
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

    jd_text, candidates = _load_payload()

    env = os.environ.copy()
    env["COPILOT_SKIP_CLI_DOWNLOAD"] = "1"
    env["JD_CACHE_PATH"] = JD_CACHE
    env.setdefault("LLM_PROVIDER", "anthropic")
    env["TOKENIZERS_PARALLELISM"] = "false"

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

        body1, dur1 = _post_score(jd_text, candidates)
        print(f"request #1: {dur1:.2f}s count={body1.get('count')}", flush=True)

        body2, dur2 = _post_score(jd_text, candidates)
        print(f"request #2: {dur2:.2f}s count={body2.get('count')}", flush=True)

        stop_sample = True
        t.join(timeout=2)
        time.sleep(0.5)
        peak["after"] = _tree_rss_mb(proc.pid)

        print("\n=== MEMORY / LATENCY (real profiles) ===")
        print(f"idle_rss_mb:    {peak['idle']:.1f}")
        print(f"peak_during_mb: {peak['during']:.1f}")
        print(f"after_rss_mb:   {peak['after']:.1f}")
        print(f"req1_seconds:   {dur1:.2f}")
        print(f"req2_seconds:   {dur2:.2f}")
        print(f"candidates:     {len(candidates)}")
        assert body2.get("count") == len(candidates)
    finally:
        stop_sample = True
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        if proc.stdout:
            try:
                leftover = proc.stdout.read()
                if leftover:
                    print("--- server log (tail) ---", flush=True)
                    print("\n".join(leftover.strip().splitlines()[-50:]), flush=True)
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
