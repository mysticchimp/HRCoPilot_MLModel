"""Blindly adjudicate a frozen candidate cohort against the real JD.

The frozen top-50 union is presented anonymously and in shuffled order to independent
LLM judges. Source ranks, fit_0_10 labels, and candidate identities are hidden. Judges
score only evidence present in the LinkedIn profile against an explicit rubric derived
from the real Prime Focus Group posting.

Outputs are staged until validation and explicit promotion:
    evals/judgments/staging/t3/blind_judgments_hr_assistant.csv
    evals/judgments/staging/t3/blind_ranking_comparison_hr_assistant.json

This is a silver LLM adjudication, not human ground truth.
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import signal
import shutil
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pydantic import BaseModel, Field

from core.adapters.linkedin_adapter import LinkedInAdapter
from core.language_normalization import normalize_language
from core.llm import get_provider
from evals.metrics import ndcg_at_k

LINKEDIN_CSV = "data/Raw_dataset_linkedin-profile-search_HR_Assistant_2026-07-06_16-31-55-822.csv"
COMPARISON_CSV = "evals/results/pipeline_vs_llm_hr_assistant.csv"
JD_PATH = "jd/HR Assistant — Prime Focus Group (Prime AC).md"
JUDGMENTS_OUT = "evals/judgments/blind_judgments_hr_assistant.csv"
SUMMARY_OUT = "evals/judgments/blind_ranking_comparison_hr_assistant.json"
GOLD_FIXTURE = "evals/fixtures/linkedin/_gold_hr_assistant.json"
BASELINE_OUT = "evals/results/baseline_linkedin.json"
PANEL_PATH = "evals/judgments/t3_panel_hr_assistant.json"
STAGING_DIR = "evals/judgments/staging/t3"
ARCHIVE_DIR = "evals/judgments/archive/pre_t3_2026-07-30"
CHECKPOINT_ROOT = ".ai-recruiter/t3_blind_judge_checkpoints"
DEFAULT_JUDGES = ["claude-opus-4.8", "gpt-5.5"]
SECTION_MAXIMA = {
    "role_level_fit": 15,
    "uae_compliance_payroll_pro": 22,
    "core_hr_operations": 15,
    "hr_systems_office_tools": 8,
    "industrial_context": 10,
    "tenure_continuity": 8,
    "career_relevance_mix": 7,
    "preferred_signals": 10,
    "evidence_quality": 5,
}
SECTION_COLUMNS = list(SECTION_MAXIMA)
AGREEMENT_GATES = {
    "overall_spearman": 0.80,
    "section_spearman": 0.60,
    "section_mae": 2.0,
}
PreferredSignal = Literal[
    "arabic",
    "cipd_or_equivalent",
    "tagalog_or_filipino",
    "english",
    "government_portals",
]


class SectionScores(BaseModel):
    role_level_fit: float = Field(ge=0, le=15)
    uae_compliance_payroll_pro: float = Field(ge=0, le=22)
    core_hr_operations: float = Field(ge=0, le=15)
    hr_systems_office_tools: float = Field(ge=0, le=8)
    industrial_context: float = Field(ge=0, le=10)
    tenure_continuity: float = Field(ge=0, le=8)
    career_relevance_mix: float = Field(ge=0, le=7)
    preferred_signals: float = Field(ge=0, le=10)
    evidence_quality: float = Field(ge=0, le=5)


class SectionEvidence(BaseModel):
    role_level_fit: bool
    uae_compliance_payroll_pro: bool
    core_hr_operations: bool
    hr_systems_office_tools: bool
    industrial_context: bool
    tenure_continuity: bool
    career_relevance_mix: bool
    preferred_signals: bool
    evidence_quality: bool


class CandidateJudgment(BaseModel):
    candidate_key: str
    section_scores: SectionScores
    section_evidence_present: SectionEvidence
    credited_preferred_signals: list[PreferredSignal]
    strengths: list[str]
    gaps: list[str]
    rationale: str


class JudgeResult(BaseModel):
    evaluations: list[CandidateJudgment]


def _clean(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _history(raw: dict, limit=15):
    rows = []
    seen = set()
    for prefix in ["currentPosition/0"] + [f"experience/{i}" for i in range(15)]:
        item = {
            "position": _clean(raw.get(f"{prefix}/position")),
            "company": _clean(raw.get(f"{prefix}/companyName")),
            "duration": _clean(raw.get(f"{prefix}/duration")),
            "employment_type": _clean(raw.get(f"{prefix}/employmentType")),
            "start_date": _clean(raw.get(f"{prefix}/startDate/text")),
            "end_date": _clean(raw.get(f"{prefix}/endDate/text")),
            "is_current_record": prefix == "currentPosition/0",
            "location": _clean(raw.get(f"{prefix}/location")),
            "description": _clean(raw.get(f"{prefix}/description")),
        }
        key = (item["position"], item["company"], item["duration"])
        if not item["position"] or key in seen:
            continue
        seen.add(key)
        if item["description"]:
            item["description"] = item["description"][:1000]
        rows.append(item)
        if len(rows) >= limit:
            break
    return rows


def _profile_payload(profile, key):
    return {
        "candidate_key": key,
        "current_title": profile.job_title,
        "current_company": _clean(profile.raw.get("currentPosition/0/companyName")),
        "derived_seniority": profile.seniority,
        "derived_total_years": profile.years_experience,
        "location": profile.location.model_dump(exclude_none=True) if profile.location else None,
        "about": profile.summary[:1200] if profile.summary else None,
        "work_history": _history(profile.raw),
        "skills": profile.skills[:40],
        "education": [e.model_dump(exclude_none=True) for e in profile.education],
        "languages": [x.model_dump(exclude_none=True) for x in (profile.languages or [])],
        "certifications": (profile.certifications or [])[:15],
    }


SYSTEM = """You are a rigorous senior recruiter conducting a blind candidate-fit audit.
Judge only explicit evidence in each supplied profile against the supplied job description.
Do not infer unlisted skills, languages, sector experience, legal knowledge, or systems.
Missing evidence is not proof of absence, but it cannot receive positive credit.
Apply the same rubric to every candidate. Do not reward fame, followers, writing quality,
or demographic attributes. The candidate keys are random and reveal nothing.
Credit Tagalog or Filipino only when the supplied languages field explicitly declares it.
Never infer language, nationality, or ethnicity from a name or any demographic proxy.
Return every candidate exactly once."""

RUBRIC = """Score out of 100 using these additive sections:
- Role and level fit (0-15): hands-on HR assistant/admin/coordinator; 1-4 years is ideal.
  Penalize clearly supervisory/managerial or unrelated profiles; do not automatically
  penalize extra total career years if recent role evidence is junior and hands-on.
- UAE compliance / payroll / PRO depth (0-22): UAE Labour Law, WPS/payroll, visas,
  labour cards, Emirates ID, MOHRE/Tasheel/GDRFA, gratuity/end-of-service.
- Core HR operations (0-15): records, attendance/leave/overtime, onboarding/offboarding,
  recruitment coordination, employee queries, confidentiality.
- HR systems and office tools (0-8): HRIS/payroll systems, Excel/MS Office.
- Industrial / blue-collar workforce context (0-10): manufacturing, HVAC, MEP,
  construction, contracting, facilities, building materials, labour-camp/site workforce.
- Tenure and continuity (0-8): Reward a stable, coherent job history as a recruiter would —
    consider average time per role and whether the candidate stayed long enough to deliver.
    Do not over-penalize junior candidates (1-4 years), contract roles, or a short current
    role (they have not left yet). Judge only from the per-role raw facts provided.
- Career relevance mix (0-7): Reward time in genuinely HR/people roles over adjacent
    admin/coordination/operations; give partial credit to adjacent roles. Judge from the
    supplied titles and durations.
- Preferred signals (0-10): Arabic, CIPD/equivalent HR certification, explicit English,
    hands-on government portals, and Tagalog/Filipino to support the Filipino factory
    workforce. Credit Tagalog/Filipino only from an explicit listed language or stated
    proficiency; never infer it from a name, nationality, ethnicity, or demographic proxy.
    Score only what is evidenced.
- Evidence quality (0-5): profile provides concrete, role-relevant evidence rather than
  only generic labels.
Return each bounded section grade, whether explicit evidence was present for each section,
the preferred-signal enum values actually credited, and evidence-backed strengths/gaps.
Do not return or independently choose an overall score or verdict; Python sums the nine
section grades and derives the verdict deterministically.
Rank by job fit, not general candidate quality."""


def build_prompt(jd_text, payloads, batch_number, batch_count):
    return (
        f"Evaluate exactly these {len(payloads)} anonymized candidates against this exact job description. "
        f"This is blind batch {batch_number}/{batch_count}; use the absolute rubric so scores remain "
        "comparable across batches. Return each supplied candidate_key exactly once and no other keys.\n\n"
        f"=== JOB DESCRIPTION ===\n{jd_text}\n\n"
        f"=== FIXED RUBRIC ===\n{RUBRIC}\n\n"
        "=== ANONYMIZED CANDIDATES (shuffled; no source ranks shown) ===\n"
        + json.dumps(payloads, indent=2, ensure_ascii=False)
    )


def _overall_score(judgment: CandidateJudgment) -> float:
    return float(sum(judgment.section_scores.model_dump().values()))


def _verdict(score: float) -> Literal["strong", "qualified", "borderline", "weak"]:
    if score >= 75:
        return "strong"
    if score >= 60:
        return "qualified"
    if score >= 40:
        return "borderline"
    return "weak"


def _explicit_languages(payload: dict) -> set[str]:
    names = set()
    for item in payload.get("languages") or []:
        name = item.get("language") if isinstance(item, dict) else None
        if name:
            names.add(normalize_language(name))
    return names


def _validate_judgments(model, expected, result, payload_by_key):
    got = [e.candidate_key for e in result.evaluations]
    if len(got) != len(set(got)):
        raise ValueError(f"{model}: duplicate candidate keys in judge output")
    missing, extra = expected - set(got), set(got) - expected
    if missing or extra:
        raise ValueError(f"{model}: missing={sorted(missing)}, extra={sorted(extra)}")
    for evaluation in result.evaluations:
        credited = evaluation.credited_preferred_signals
        if len(credited) != len(set(credited)):
            raise ValueError(f"{model}: duplicate preferred-signal credits for {evaluation.candidate_key}")
        if "tagalog_or_filipino" in credited:
            languages = _explicit_languages(payload_by_key[evaluation.candidate_key])
            if "filipino" not in languages:
                raise ValueError(
                    f"{model}: phantom Tagalog/Filipino credit for {evaluation.candidate_key}"
                )


class CallBudget:
    def __init__(self, limit: int, state_path: Path, prior_used: int = 0):
        self.limit = limit
        self.state_path = state_path
        self._lock = threading.Lock()
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.used = int(state["used"])
            self.attempts = list(state.get("attempts", []))
        else:
            self.used = prior_used
            self.attempts = []
            if prior_used:
                self.attempts.append({
                    "model": "recovery",
                    "label": "calls completed before persistent ledger",
                    "count": prior_used,
                })
            self._persist()
        if self.used > self.limit:
            raise ValueError(f"recorded live calls ({self.used}) exceed cap ({self.limit})")
        print(f"live call budget: {self.used}/{self.limit} already used", flush=True)

    def _persist(self) -> None:
        _atomic_json(
            self.state_path,
            {"limit": self.limit, "used": self.used, "attempts": self.attempts},
        )

    def claim(self, model: str, label: str) -> None:
        with self._lock:
            if self.used >= self.limit:
                raise RuntimeError(f"live call cap reached ({self.limit}); checkpoints preserved")
            self.used += 1
            self.attempts.append({"model": model, "label": label})
            self._persist()
            print(f"  live call {self.used}/{self.limit}: {model} {label}", flush=True)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _isolated_judge_worker(model, prompt, timeout, output_path):
    """Run one paid call in its own process group and serialize the outcome."""
    try:
        if hasattr(os, "setsid"):
            os.setsid()
        provider = get_provider("copilot", model=model, timeout=timeout)
        result = provider.generate_structured(prompt, JudgeResult, system=SYSTEM)
        payload = {"status": "ok", "result": result.model_dump(mode="json")}
    except BaseException as exc:  # noqa: BLE001
        payload = {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    _atomic_json(Path(output_path), payload)


def _terminate_isolated_process(process) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except (ProcessLookupError, PermissionError):
        process.terminate()
    process.join(5)
    if process.is_alive():
        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.join(5)


def _generate_structured_isolated(model, prompt, timeout, hard_timeout):
    """Return one JudgeResult or kill the entire call process group at the deadline."""
    Path(".ai-recruiter").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="t3-judge-", dir=".ai-recruiter") as directory:
        output_path = Path(directory) / "result.json"
        context = mp.get_context("spawn")
        process = context.Process(
            target=_isolated_judge_worker,
            args=(model, prompt, timeout, str(output_path)),
        )
        process.start()
        process.join(hard_timeout)
        if process.is_alive():
            _terminate_isolated_process(process)
            process.close()
            raise TimeoutError(
                f"{model}: hard timeout after {hard_timeout}s; isolated process group killed"
            )
        exit_code = process.exitcode
        process.close()
        if not output_path.exists():
            raise RuntimeError(f"{model}: isolated judge exited {exit_code} without a result")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            raise RuntimeError(
                f"{model}: {payload.get('error_type', 'RemoteError')}: {payload.get('message', '')}"
            )
        return JudgeResult.model_validate(payload["result"])


def _checkpoint_result(path: Path, result: JudgeResult) -> None:
    _atomic_json(path, result.model_dump(mode="json"))


def _load_checkpoint(path: Path, model: str, expected: set[str], payload_by_key: dict) -> JudgeResult | None:
    if not path.exists():
        return None
    try:
        result = JudgeResult.model_validate_json(path.read_text(encoding="utf-8"))
        _validate_judgments(model, expected, result, payload_by_key)
        return result
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"  {model}: ignoring invalid checkpoint {path.name}: {exc}", flush=True)
        return None


def _call_judge(
    model,
    prompt,
    expected,
    payload_by_key,
    budget,
    label,
    timeout,
    hard_timeout,
):
    budget.claim(model, label)
    result = _generate_structured_isolated(model, prompt, timeout, hard_timeout)
    _validate_judgments(model, expected, result, payload_by_key)
    return result


def preflight_one(
    model,
    jd_text,
    payload,
    run_dir: Path,
    budget: CallBudget,
    timeout=360,
    hard_timeout=420,
):
    expected = {payload["candidate_key"]}
    payload_by_key = {payload["candidate_key"]: payload}
    checkpoint = run_dir / model / "preflight.json"
    cached = _load_checkpoint(checkpoint, model, expected, payload_by_key)
    if cached is not None:
        print(f"  {model}: preflight checkpoint valid", flush=True)
        return
    result = _call_judge(
        model,
        build_prompt(jd_text, [payload], 1, 1),
        expected,
        payload_by_key,
        budget,
        "preflight",
        timeout,
        hard_timeout,
    )
    _checkpoint_result(checkpoint, result)
    print(f"  {model}: preflight complete", flush=True)


def _split_batch(batch, enabled, split_size):
    if not enabled:
        return [batch]
    if split_size < 1:
        raise ValueError("split_size must be at least 1")
    return [batch[index:index + split_size] for index in range(0, len(batch), split_size)]


def judge_one(
    model,
    jd_text,
    payloads,
    run_dir,
    budget,
    batch_size=7,
    retries=1,
    timeout=360,
    hard_timeout=420,
    split_batches=(),
    split_size=4,
):
    shuffled = payloads[:]
    random.Random(f"judge:{model}:t3-20260730").shuffle(shuffled)
    batches = [shuffled[i:i + batch_size] for i in range(0, len(shuffled), batch_size)]
    evaluations = []
    for index, batch in enumerate(batches, start=1):
        expected = {p["candidate_key"] for p in batch}
        payload_by_key = {p["candidate_key"]: p for p in batch}
        checkpoint = run_dir / model / f"batch_{index:03d}.json"
        cached = _load_checkpoint(checkpoint, model, expected, payload_by_key)
        if cached is not None:
            evaluations.extend(cached.evaluations)
            print(f"  {model}: batch {index}/{len(batches)} checkpoint valid", flush=True)
            continue
        parts = _split_batch(batch, index in set(split_batches), split_size)
        part_evaluations = []
        for part_index, part in enumerate(parts, start=1):
            part_expected = {p["candidate_key"] for p in part}
            part_payload_by_key = {p["candidate_key"]: p for p in part}
            part_checkpoint = (
                checkpoint
                if len(parts) == 1
                else run_dir / model / f"batch_{index:03d}_part_{part_index:02d}.json"
            )
            part_cached = _load_checkpoint(
                part_checkpoint, model, part_expected, part_payload_by_key
            )
            if part_cached is not None:
                part_evaluations.extend(part_cached.evaluations)
                print(
                    f"  {model}: batch {index}/{len(batches)} part "
                    f"{part_index}/{len(parts)} checkpoint valid",
                    flush=True,
                )
                continue

            last_error = None
            for attempt in range(1, retries + 2):
                try:
                    part_label = f"batch {index}/{len(batches)}"
                    if len(parts) > 1:
                        part_label += f" part {part_index}/{len(parts)}"
                    result = _call_judge(
                        model,
                        build_prompt(jd_text, part, index, len(batches)),
                        part_expected,
                        part_payload_by_key,
                        budget,
                        f"{part_label} attempt {attempt}",
                        timeout,
                        hard_timeout,
                    )
                    _checkpoint_result(part_checkpoint, result)
                    part_evaluations.extend(result.evaluations)
                    print(f"  {model}: {part_label} complete", flush=True)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    print(
                        f"  {model}: {part_label} attempt {attempt} failed: "
                        f"{type(exc).__name__}",
                        flush=True,
                    )
            else:
                raise RuntimeError(
                    f"{model}: batch {index}/{len(batches)} part "
                    f"{part_index}/{len(parts)} failed after retries"
                ) from last_error

        combined_batch = JudgeResult(evaluations=part_evaluations)
        _validate_judgments(model, expected, combined_batch, payload_by_key)
        if len(parts) > 1:
            _checkpoint_result(checkpoint, combined_batch)
        evaluations.extend(combined_batch.evaluations)
        print(f"  {model}: batch {index}/{len(batches)} complete", flush=True)

    combined = JudgeResult(evaluations=evaluations)
    _validate_judgments(
        model,
        {p["candidate_key"] for p in payloads},
        combined,
        {p["candidate_key"]: p for p in payloads},
    )
    return model, combined


def rank_percentile(series):
    # Highest score -> 1.0, lowest -> 0.0; robust to judge scale differences.
    if len(series) <= 1:
        return pd.Series([1.0] * len(series), index=series.index)
    return series.rank(method="average", ascending=True).sub(1).div(len(series) - 1)


def ranking_metrics(name, ranked_ids, consensus_relevance, consensus_top, k):
    ids = ranked_ids[:k]
    return {
        "ranking": name,
        "k": k,
        "ndcg": round(ndcg_at_k(ids, consensus_relevance, k), 4),
        "top_k_overlap_with_consensus": len(set(ids) & set(consensus_top[:k])),
        "mean_consensus_score": round(sum(consensus_relevance[x] for x in ids) / len(ids), 2),
    }


def _current_union(comparison: pd.DataFrame, top_n: int) -> tuple[list[str], list[str], list[str]]:
    pipeline_ids = comparison.sort_values("pipeline_rank")["candidate_id"].tolist()
    llm_ids = comparison.sort_values("llm_rank")["candidate_id"].tolist()
    union_ids = list(dict.fromkeys(pipeline_ids[:top_n] + llm_ids[:top_n]))
    return pipeline_ids, llm_ids, union_ids


def _freeze_panel(comparison: pd.DataFrame, top_n: int, panel_path: str) -> None:
    pipeline_ids, llm_ids, union_ids = _current_union(comparison, top_n)
    payload = {
        "description": "Frozen T3 silver Judge-grade anchor panel",
        "source": COMPARISON_CSV,
        "top_n": top_n,
        "candidate_count": len(union_ids),
        "candidate_ids": union_ids,
        "pipeline_top20_covered": set(pipeline_ids[:20]).issubset(union_ids),
        "llm_top20_covered": set(llm_ids[:20]).issubset(union_ids),
    }
    _atomic_json(Path(panel_path), payload)
    print(f"froze {len(union_ids)} candidates in {panel_path}")


def _load_panel(panel_path: str) -> list[str]:
    path = Path(panel_path)
    if not path.exists():
        raise FileNotFoundError(f"frozen panel missing: run with --freeze-panel first ({panel_path})")
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate_ids = payload.get("candidate_ids") or []
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("frozen panel contains duplicate candidate IDs")
    if payload.get("candidate_count") != len(candidate_ids):
        raise ValueError("frozen panel candidate_count does not match candidate_ids")
    return candidate_ids


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive_current() -> None:
    archive = Path(ARCHIVE_DIR)
    manifest_path = archive / "manifest.json"
    if manifest_path.exists():
        print(f"archive already present: {ARCHIVE_DIR}")
        return
    sources = [
        JUDGMENTS_OUT,
        SUMMARY_OUT,
        GOLD_FIXTURE,
        BASELINE_OUT,
        "tests/test_eval_regression.py",
        "models/mappings.py",
    ]
    manifest = {"description": "Pre-T3 silver-anchor artifacts", "files": {}}
    for source_name in sources:
        source = Path(source_name)
        if not source.exists():
            continue
        destination = archive / source_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest["files"][source_name] = _sha256(destination)
    _atomic_json(manifest_path, manifest)
    print(f"archived {len(manifest['files'])} artifacts in {ARCHIVE_DIR}")


def _run_fingerprint(jd_text, payloads, judges, batch_size, panel_ids) -> str:
    material = {
        "rubric": RUBRIC,
        "system": SYSTEM,
        "schema": CandidateJudgment.model_json_schema(),
        "jd": jd_text,
        "payloads": payloads,
        "judges": judges,
        "batch_size": batch_size,
        "panel_ids": panel_ids,
        "shuffle_seed": "t3-20260730",
    }
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _judgment_rows(results, key_to_id):
    rows = []
    for model, result in results:
        for evaluation in result.evaluations:
            row = {
                "candidate_key": evaluation.candidate_key,
                "candidate_id": key_to_id[evaluation.candidate_key],
                "judge": model,
                "overall_score": _overall_score(evaluation),
                "verdict": _verdict(_overall_score(evaluation)),
                "credited_preferred_signals": json.dumps(
                    evaluation.credited_preferred_signals, ensure_ascii=False
                ),
                "strengths": json.dumps(evaluation.strengths, ensure_ascii=False),
                "gaps": json.dumps(evaluation.gaps, ensure_ascii=False),
                "rationale": evaluation.rationale,
            }
            for section, score in evaluation.section_scores.model_dump().items():
                row[f"{section}_score"] = score
            for section, present in evaluation.section_evidence_present.model_dump().items():
                row[f"{section}_evidence_present"] = present
            rows.append(row)
    return pd.DataFrame(rows)


def _pair_metrics(pivot, left, right, candidate_ids=None):
    paired = pivot[[left, right]].dropna()
    if candidate_ids is not None:
        paired = paired.loc[paired.index.intersection(candidate_ids)]
    rho = paired[left].corr(paired[right], method="spearman") if len(paired) >= 2 else float("nan")
    mae = (paired[left] - paired[right]).abs().mean() if len(paired) else float("nan")
    return {
        "n": int(len(paired)),
        "spearman": None if pd.isna(rho) else round(float(rho), 4),
        "mae": None if pd.isna(mae) else round(float(mae), 4),
    }


def _agreement(judged, judges, structural_ids):
    left, right = judges
    overall = judged.pivot(index="candidate_id", columns="judge", values="overall_score")
    sections = {}
    for section in SECTION_COLUMNS:
        score_col = f"{section}_score"
        score_pivot = judged.pivot(index="candidate_id", columns="judge", values=score_col)
        fixed_ids = structural_ids if section in {"tenure_continuity", "career_relevance_mix"} else None
        fixed = _pair_metrics(score_pivot, left, right, fixed_ids)

        evidence_col = f"{section}_evidence_present"
        evidence = judged.pivot(index="candidate_id", columns="judge", values=evidence_col)
        joint_evidence_ids = evidence.index[(evidence[left].astype(bool)) & (evidence[right].astype(bool))]
        evidence_slice = _pair_metrics(score_pivot, left, right, joint_evidence_ids)
        sections[section] = {"fixed_subset": fixed, "joint_evidence_subset": evidence_slice}
    return {"overall": _pair_metrics(overall, left, right), "sections": sections}


def _agreement_validation(agreement):
    overall = agreement["overall"]
    overall_pass = (
        overall["spearman"] is not None
        and overall["spearman"] >= AGREEMENT_GATES["overall_spearman"]
    )
    section_gates = {}
    for section in ("tenure_continuity", "career_relevance_mix", "preferred_signals"):
        metrics = agreement["sections"][section]["fixed_subset"]
        section_gates[section] = {
            **metrics,
            "pass": (
                metrics["spearman"] is not None
                and metrics["spearman"] >= AGREEMENT_GATES["section_spearman"]
                and metrics["mae"] is not None
                and metrics["mae"] <= AGREEMENT_GATES["section_mae"]
            ),
        }
    return {
        "thresholds": AGREEMENT_GATES,
        "overall_pass": overall_pass,
        "section_gates": section_gates,
        "all_section_gates_pass": all(item["pass"] for item in section_gates.values()),
        "all_pass": overall_pass,
    }


def _join_judge_details(output, judged, judges):
    for model in judges:
        model_rows = judged[judged["judge"] == model].set_index("candidate_id")
        for section in SECTION_COLUMNS:
            output[f"{model}_{section}_score"] = model_rows[f"{section}_score"]
            output[f"{model}_{section}_evidence_present"] = model_rows[
                f"{section}_evidence_present"
            ]
        for column in ("verdict", "credited_preferred_signals", "strengths", "gaps", "rationale"):
            output[f"{model}_{column}"] = model_rows[column]
    for section in SECTION_COLUMNS:
        columns = [f"{model}_{section}_score" for model in judges]
        output[f"{section}_mean_score"] = output[columns].mean(axis=1)
    return output


def _promote_staged(staging_dir: str) -> None:
    staging = Path(staging_dir)
    staged_judgments = staging / Path(JUDGMENTS_OUT).name
    staged_summary = staging / Path(SUMMARY_OUT).name
    if not staged_judgments.exists() or not staged_summary.exists():
        raise FileNotFoundError(f"complete staged outputs not found in {staging}")
    summary = json.loads(staged_summary.read_text(encoding="utf-8"))
    if not summary.get("validation", {}).get("all_pass"):
        raise ValueError("staged regrade did not pass the anchor-level safety gates")
    _archive_current()
    Path(JUDGMENTS_OUT).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(staged_judgments, JUDGMENTS_OUT)
    shutil.copy2(staged_summary, SUMMARY_OUT)
    print(f"promoted validated judgments to {JUDGMENTS_OUT} and {SUMMARY_OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES)
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--batch-size", type=int, default=7)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--top-n", type=int, default=20,
                    help="take top-N of each ranking when freezing the panel")
    ap.add_argument("--panel", default=PANEL_PATH)
    ap.add_argument("--freeze-panel", action="store_true")
    ap.add_argument("--staging-dir", default=STAGING_DIR)
    ap.add_argument("--call-cap", type=int, default=32)
    ap.add_argument("--prior-live-calls", type=int, default=0,
                    help="one-time seed used only when this fingerprint has no call ledger")
    ap.add_argument("--timeout", type=int, default=360)
    ap.add_argument("--opus-timeout", type=int, default=None)
    ap.add_argument("--hard-timeout", type=int, default=None)
    ap.add_argument("--opus-hard-timeout", type=int, default=None)
    ap.add_argument("--opus-split-batches", type=int, nargs="*", default=[])
    ap.add_argument("--split-size", type=int, default=4)
    ap.add_argument("--promote", action="store_true",
                    help="promote already-staged outputs; makes no live calls")
    args = ap.parse_args()

    if args.promote:
        _promote_staged(args.staging_dir)
        return

    comparison = pd.read_csv(COMPARISON_CSV)
    pipeline_ids, llm_ids, current_union = _current_union(comparison, args.top_n)
    if args.freeze_panel:
        _freeze_panel(comparison, args.top_n, args.panel)
        return
    if args.judges != DEFAULT_JUDGES:
        ap.error(f"T3 requires the pinned judge panel: {' '.join(DEFAULT_JUDGES)}")
    panel_ids = _load_panel(args.panel)
    if not set(pipeline_ids[:20]).issubset(panel_ids):
        raise ValueError("frozen panel does not cover the current pipeline top 20")
    if not set(llm_ids[:20]).issubset(panel_ids):
        raise ValueError("frozen panel does not cover the current LLM top 20")

    profiles = LinkedInAdapter().to_profiles(LINKEDIN_CSV)
    by_id = {p.candidate_id: p for p in profiles}
    missing_profiles = set(panel_ids) - set(by_id)
    if missing_profiles:
        raise ValueError(f"frozen panel candidates missing from source data: {sorted(missing_profiles)}")
    key_to_id = {f"J{i + 1:02d}": cid for i, cid in enumerate(sorted(panel_ids))}
    id_to_key = {cid: key for key, cid in key_to_id.items()}
    payloads = [_profile_payload(by_id[cid], id_to_key[cid]) for cid in panel_ids]
    with open(JD_PATH) as fh:
        jd_text = fh.read()

    print(f"blind pool: {len(panel_ids)} frozen candidates from {args.panel}")
    print(f"judges: {', '.join(args.judges)} | source ranks/fit labels hidden")
    _archive_current()
    fingerprint = _run_fingerprint(jd_text, payloads, args.judges, args.batch_size, panel_ids)
    run_dir = Path(CHECKPOINT_ROOT) / fingerprint
    budget = CallBudget(
        args.call_cap,
        run_dir / "call_budget.json",
        prior_used=args.prior_live_calls,
    )

    def model_timeout(model):
        if model == "claude-opus-4.8" and args.opus_timeout is not None:
            return args.opus_timeout
        return args.timeout

    def model_hard_timeout(model):
        if model == "claude-opus-4.8" and args.opus_hard_timeout is not None:
            return args.opus_hard_timeout
        if args.hard_timeout is not None:
            return args.hard_timeout
        return model_timeout(model) + 60

    che_id = "che-ibardelosa-a538072a1"
    preflight_payload = next((p for p in payloads if key_to_id[p["candidate_key"]] == che_id), payloads[0])
    print(f"run fingerprint: {fingerprint}")
    print("running exact-schema preflights")
    for model in args.judges:
        preflight_one(
            model,
            jd_text,
            preflight_payload,
            run_dir,
            budget,
            timeout=model_timeout(model),
            hard_timeout=model_hard_timeout(model),
        )

    results = []
    if args.sequential:
        for model in args.judges:
            results.append(
                judge_one(
                    model,
                    jd_text,
                    payloads,
                    run_dir,
                    budget,
                    args.batch_size,
                    args.retries,
                    timeout=model_timeout(model),
                    hard_timeout=model_hard_timeout(model),
                    split_batches=(args.opus_split_batches if model == "claude-opus-4.8" else ()),
                    split_size=args.split_size,
                )
            )
            print(f"  {model}: complete", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=len(args.judges)) as pool:
            futures = {
                pool.submit(
                    judge_one, model, jd_text, payloads, run_dir, budget,
                    args.batch_size, args.retries, model_timeout(model),
                    model_hard_timeout(model),
                    (args.opus_split_batches if model == "claude-opus-4.8" else ()),
                    args.split_size,
                ): model
                for model in args.judges
            }
            for future in as_completed(futures):
                results.append(future.result())
                print(f"  {futures[future]}: complete", flush=True)

    judged = _judgment_rows(results, key_to_id)

    score_wide = judged.pivot(index="candidate_id", columns="judge", values="overall_score")
    for model in args.judges:
        score_wide[f"{model}_percentile"] = rank_percentile(score_wide[model])
    score_wide["judge_mean_score"] = score_wide[args.judges].mean(axis=1)
    score_wide["consensus_percentile"] = score_wide[[f"{m}_percentile" for m in args.judges]].mean(axis=1)
    score_wide["consensus_rank"] = score_wide["consensus_percentile"].rank(method="min", ascending=False).astype(int)

    base_cols = comparison.set_index("candidate_id")[["pipeline_rank", "llm_rank", "llm_fit_0_10", "current_title"]]
    output = base_cols.join(score_wide, how="inner")
    output = _join_judge_details(output, judged, args.judges).sort_values("consensus_rank")

    structural_ids = {
        key_to_id[payload["candidate_key"]]
        for payload in payloads
        if sum(bool(role.get("duration")) for role in payload["work_history"]) >= 2
    }
    agreement = _agreement(judged, args.judges, structural_ids)
    validation = _agreement_validation(agreement)
    validation["anti_inference_pass"] = True
    validation["che_explicit_language_count"] = len(_explicit_languages(preflight_payload))
    validation["panel_coverage_pass"] = len(output) == len(panel_ids) == len(set(output.index))
    validation["all_pass"] = (
        validation["all_pass"]
        and validation["anti_inference_pass"]
        and validation["panel_coverage_pass"]
        and validation["che_explicit_language_count"] == 0
    )

    staging = Path(args.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    staged_judgments = staging / Path(JUDGMENTS_OUT).name
    staged_summary = staging / Path(SUMMARY_OUT).name
    output.to_csv(staged_judgments)

    # Judge agreement: score/rank correlation across all blindly reviewed candidates.
    left, right = args.judges
    judge_agreement = {
        f"{left}__{right}_spearman": agreement["overall"]["spearman"],
    }

    consensus_relevance = output["judge_mean_score"].to_dict()
    consensus_top = output.sort_values(["consensus_rank", "judge_mean_score"], ascending=[True, False]).index.tolist()
    metrics = []
    for k in (10, 20):
        metrics.append(ranking_metrics("pipeline", pipeline_ids, consensus_relevance, consensus_top, k))
        metrics.append(ranking_metrics("llm_scored", llm_ids, consensus_relevance, consensus_top, k))

    summary = {
        "disclaimer": "Blind consensus of LLM judges; silver evaluation, not human ground truth.",
        "jd": JD_PATH,
        "judges": args.judges,
        "rubric_version": "t3-tenure-relevance-language-v1",
        "run_fingerprint": fingerprint,
        "actual_live_calls": budget.used,
        "blind_union_size": len(panel_ids),
        "structural_agreement_subset_size": len(structural_ids),
        "judge_agreement": judge_agreement,
        "agreement": agreement,
        "validation": validation,
        "metrics": metrics,
        "consensus_top20": [
            {
                "consensus_rank": int(output.loc[cid, "consensus_rank"]),
                "candidate_id": cid,
                "title": output.loc[cid, "current_title"],
                "judge_mean_score": round(float(output.loc[cid, "judge_mean_score"]), 2),
                "pipeline_rank": int(output.loc[cid, "pipeline_rank"]),
                "llm_rank": int(output.loc[cid, "llm_rank"]),
            }
            for cid in consensus_top[:20]
        ],
    }
    with open(staged_summary, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("\n=== judge agreement ===")
    for metric, value in judge_agreement.items():
        print(f"{metric}: {value:.3f}")
    print("\n=== alignment with blind consensus ===")
    print(f"{'ranking':12} {'k':>3} {'NDCG':>7} {'overlap':>8} {'mean fit':>9}")
    for row in metrics:
        print(f"{row['ranking']:12} {row['k']:>3} {row['ndcg']:7.4f} "
              f"{row['top_k_overlap_with_consensus']:>3}/{row['k']:<4} {row['mean_consensus_score']:9.2f}")

    print("\n=== blind-consensus top 20 ===")
    print(f"{'#':>3} {'candidate':<28} {'score':>6} {'pipe#':>6} {'LLM#':>6}")
    for row in summary["consensus_top20"]:
        print(f"{row['consensus_rank']:>3} {row['candidate_id'][:28]:<28} {row['judge_mean_score']:6.1f} "
              f"{row['pipeline_rank']:>6} {row['llm_rank']:>6}")

    print("\n=== section agreement gates ===")
    for section, result in validation["section_gates"].items():
        print(
            f"{section}: rho={result['spearman']} mae={result['mae']} "
            f"n={result['n']} pass={result['pass']}"
        )
    print(f"\nstaged {staged_judgments}\nstaged {staged_summary}")
    print(f"validation all_pass={validation['all_pass']}")
    if not validation["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
