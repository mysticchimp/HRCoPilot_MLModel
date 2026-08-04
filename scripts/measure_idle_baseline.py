"""Measure idle (warm+prime) RSS for baseline-reduction levers.

Runs each combo in a fresh subprocess so prior model loads don't contaminate RSS.

    COPILOT_SKIP_CLI_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
      uv run python scripts/measure_idle_baseline.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMBOS = [
    # label, BASE_EMBEDDING_DTYPE, SIMILARITY_MODEL, optional Qwen max_seq override via env
    ("A_control_fp32_mpnet+qwen_L1024", "fp32", "qwen", None),
    ("B_fp16_mpnet+qwen_L1024", "fp16", "qwen", None),
    ("C_fp32_mpnet_only", "fp32", "mpnet-only", None),
    ("D_fp16_mpnet_only", "fp16", "mpnet-only", None),
    ("E_fp16_mpnet+qwen_L512", "fp16", "qwen", "512"),
]


WORKER = r"""
import os, sys, json, copy
os.environ.setdefault("COPILOT_SKIP_CLI_DOWNLOAD", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.environ["REPO_ROOT"])

dtype = os.environ["BASE_EMBEDDING_DTYPE"]
sim = os.environ["SIMILARITY_MODEL"]
qwen_l = os.environ.get("QWEN_MAX_SEQ")

from core.mem_trace import rss_mb
rss0 = rss_mb()

import models.mappings as m
from core.model_cache import warm_scoring_models

cfg = m.similarity_model_config
if qwen_l and cfg:
    cfg = dict(cfg)
    cfg["max_seq_length"] = int(qwen_l)

base, sim_spec, _ = warm_scoring_models(
    similarity_model_config=cfg,
    apply_env_overrides=True,
)
bp = next(base.parameters())
base_dtype = str(bp.dtype).replace("torch.", "")
base_dev = str(bp.device)
sim_key = None if sim_spec is None else sim_spec.model_key
sim_dtype = None
sim_l = None
if sim_spec is not None:
    sp = next(sim_spec.model.parameters())
    sim_dtype = str(sp.dtype).replace("torch.", "")
    sim_l = getattr(sim_spec.model, "max_seq_length", None)

out = {
    "rss_before_mb": round(rss0, 1),
    "rss_after_warm_prime_mb": round(rss_mb(), 1),
    "base_dtype": base_dtype,
    "base_device": base_dev,
    "sim_key": sim_key,
    "sim_dtype": sim_dtype,
    "sim_max_seq_length": sim_l,
    "BASE_EMBEDDING_DTYPE": dtype,
    "SIMILARITY_MODEL": sim,
    "QWEN_MAX_SEQ": qwen_l,
}
print(json.dumps(out))
"""


def run_combo(label: str, dtype: str, sim: str, qwen_l: str | None) -> dict:
    env = os.environ.copy()
    env["REPO_ROOT"] = str(ROOT)
    env["COPILOT_SKIP_CLI_DOWNLOAD"] = "1"
    env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")
    env["TRANSFORMERS_OFFLINE"] = env.get("TRANSFORMERS_OFFLINE", "1")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["BASE_EMBEDDING_DTYPE"] = dtype
    env["SIMILARITY_MODEL"] = sim
    if qwen_l:
        env["QWEN_MAX_SEQ"] = qwen_l
    else:
        env.pop("QWEN_MAX_SEQ", None)

    proc = subprocess.run(
        [sys.executable, "-c", WORKER],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "label": label,
            "error": proc.stderr[-2000:] or proc.stdout[-2000:],
            "returncode": proc.returncode,
        }
    # Last JSON line of stdout
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        return {"label": label, "error": f"no json in stdout: {proc.stdout[-1000:]}"}
    data = json.loads(lines[-1])
    data["label"] = label
    return data


def main() -> None:
    print("Measuring idle baseline RSS per combo (fresh process each)...\n")
    rows = []
    for label, dtype, sim, qwen_l in COMBOS:
        print(f"→ {label} ...", flush=True)
        row = run_combo(label, dtype, sim, qwen_l)
        rows.append(row)
        if "error" in row:
            print(f"  ERROR: {row['error'][:400]}")
        else:
            print(
                f"  baseline={row['rss_after_warm_prime_mb']:.1f} MB  "
                f"base={row['base_dtype']}  sim={row['sim_key'] or 'None'}  "
                f"sim_dtype={row['sim_dtype']} L={row['sim_max_seq_length']}"
            )

    print("\n=== SUMMARY ===")
    print(f"{'combo':<40} {'baseline_MB':>12} {'base':>8} {'sim':>12}")
    for r in rows:
        if "error" in r:
            print(f"{r['label']:<40} {'ERR':>12}")
            continue
        sim = "qwen" if r.get("sim_key") else "mpnet-only"
        print(
            f"{r['label']:<40} {r['rss_after_warm_prime_mb']:12.1f} "
            f"{r['base_dtype']:>8} {sim:>12}"
        )


if __name__ == "__main__":
    main()
