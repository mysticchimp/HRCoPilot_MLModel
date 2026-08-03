"""Fast offline weight / normalization calibration sweep.

The ranking weights only affect the FINAL linear combination
(`core.scoring.calculate_total_score`). The four component score vectors
(title / skill / qualification / similarity) are identical across every weight
config, and with soft-title (`hard=False`) the candidate pool is constant too.

So we run the expensive scorers ONCE per eval case, then sweep weights +
min-max normalization as cheap re-combinations. `calculate_total_score` is
reused verbatim, so the ranking math matches production exactly.

    uv run python scripts/calibrate_weights.py --n-per-group 5 --top 8

The precomputed component vectors use the exact production champion stack:
all-mpnet for title/skill, hybrid skill matching, and the isolated Qwen model
for similarity. Before any sweep, the recombined incumbent ranking is asserted
to match ``evals.runner.rank_candidates`` exactly for every committed case.
"""

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sentence_transformers import SentenceTransformer

from core.adapters.linkedin_adapter import LinkedInAdapter
from core.data import profiles_to_dataframe
from core.embedding import build_similarity_spec, embed_profiles
from core.filtering import filter_by_job_title
from core.language_normalization import normalize_language
from core.scoring import (
    calculate_attrition_score,
    calculate_education_relevance_score,
    calculate_experience_relevance_score,
    calculate_experience_score,
    calculate_industry_score,
    calculate_language_score,
    calculate_location_score,
    calculate_qualification_score,
    calculate_seniority_score,
    calculate_similarity_score,
    calculate_skill_score,
    calculate_total_score,
)
from evals.cases import load_fixture, sample_seeds
from evals.metrics import hit_at_k, ndcg_at_k, rank_of, reciprocal_rank
from evals.runner import PipelineConfig, aggregate, rank_candidates
from models.mappings import (
    candidate_score_weights,
    education_hr_certs,
    education_neutral,
    similarity_model_config,
)
from tests.test_eval_regression import FLOORS as REGRESSION_FLOORS

LINKEDIN_CSV = "data/Raw_dataset_linkedin-profile-search_HR_Assistant_2026-07-06_16-31-55-822.csv"
CACHE = ".ai-recruiter/emb_linkedin_v2.pkl"
JUDGMENTS_CSV = "evals/judgments/blind_judgments_hr_assistant.csv"
JUDGMENT_SUMMARY = "evals/judgments/blind_ranking_comparison_hr_assistant.json"
C5_OUT = "evals/results/t3_c5_reablation.json"
COMP_COLS = ["candidate_id", "title_score", "skill_score", "qualification_score",
             "seniority_score", "experience_score", "industry_score", "language_score",
             "location_score", "attrition_score", "experience_relevance_score", "education_relevance_score", "similarity_score"]
C5_COMPONENTS = [
    "attrition_score",
    "experience_relevance_score",
    "education_relevance_score",
    "language_score",
]
C5_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
C5_SECTION_GATES = {
    "attrition_score": "tenure_continuity",
    "experience_relevance_score": "career_relevance_mix",
    "education_relevance_score": "preferred_signals",
    "language_score": "preferred_signals",
}


def _metric(summary, name):
    if name.startswith("ndcg@"):
        return summary.get(f"raw_{name}", summary.get(name, 0.0))
    return summary.get(name, 0.0)


def _truncate_floor(value):
    return math.floor(value * 100 + 1e-12) / 100


def _reset_floors(incumbent):
    floors = dict(REGRESSION_FLOORS)
    floors["ndcg@5"] = _truncate_floor(_metric(incumbent, "ndcg@5"))
    floors["ndcg@10"] = _truncate_floor(_metric(incumbent, "ndcg@10"))
    return floors


def _floor_results(summary, floors):
    values = {
        name: {
            "value": _metric(summary, name),
            "floor": floor,
            "pass": _metric(summary, name) >= floor - 1e-12,
        }
        for name, floor in floors.items()
    }
    return {"pass": all(item["pass"] for item in values.values()), "metrics": values}


def _json_list(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = json.loads(value)
    return parsed if isinstance(parsed, list) else []


def _credit_count(judgments, judges, accepted):
    output = pd.Series(0.0, index=judgments.index)
    for judge in judges:
        column = f"{judge}_credited_preferred_signals"
        if column not in judgments:
            raise ValueError(f"judgment artifact missing structured credits: {column}")
        output += judgments[column].map(
            lambda value: float(sum(signal in accepted for signal in _json_list(value)))
        )
    return output / len(judges)


def _spearman(left, right):
    value = left.corr(right, method="spearman")
    return None if pd.isna(value) else float(value)


def _cohort_summary(frame, mask):
    selected = frame[mask]
    return {
        "n": int(len(selected)),
        "mean_education_component": (
            None if selected.empty else float(selected["education_relevance_score"].mean())
        ),
        "mean_preferred_section": (
            None if selected.empty else float(selected["preferred_signals_mean_score"].mean())
        ),
    }


def load_cases(profiles, n_per_group):
    """Reproduce the offline eval set used by scripts/run_eval.py (no LLM calls)."""
    seeds = sample_seeds(profiles, n_per_group, group_key=lambda p: p.seniority)
    cases = [c for c in (load_fixture("linkedin", s.candidate_id) for s in seeds) if c]
    gold = load_fixture("linkedin", "_gold_hr_assistant")
    if gold:
        cases.append(gold)
    return cases


def precompute_components(
    cases,
    df_full,
    model,
    skill_mode="hybrid",
    semantic_threshold=None,
    sim_spec=None,
):
    """Run the expensive scorers once per case (soft-title pool is config-invariant).

    skill_mode/semantic_threshold change the skill_score column, so this must be
    re-run per skill config (unlike weight sweeps, which only re-combine columns).
    """
    out = []
    for case in cases:
        df = df_full.copy()
        df = filter_by_job_title(df, case.parsed_jd.role, 0.4, model=model, mode="hybrid", hard=False)
        df = calculate_skill_score(df, case.parsed_jd, False, 0.25, match_threshold=70,
                                   model=model, skill_mode=skill_mode, semantic_threshold=semantic_threshold)
        df = calculate_qualification_score(df, case.parsed_jd, False, 0.2)
        df = calculate_seniority_score(df, case.parsed_jd)
        df = calculate_experience_score(df, case.parsed_jd)
        df = calculate_industry_score(df, case.parsed_jd)
        df = calculate_language_score(df, case.parsed_jd)
        df = calculate_location_score(df, case.parsed_jd)
        df = calculate_attrition_score(df, case.parsed_jd)
        df = calculate_experience_relevance_score(df, case.parsed_jd)
        df = calculate_education_relevance_score(df, case.parsed_jd)
        df = calculate_similarity_score(
            df,
            case.parsed_jd,
            sim_spec.model if sim_spec else model,
            query_instruction=sim_spec.query_instruction if sim_spec else None,
        )
        out.append((case, df[COMP_COLS].copy()))
    return out


def ranked_config(precomputed, weights, normalize):
    rankings = {}
    for case, comp in precomputed:
        df = calculate_total_score(
            comp.copy(), case.parsed_jd, weights=weights, normalize=normalize
        )
        rankings[case.case_id] = df.sort_values(
            "total_score", ascending=False
        )["candidate_id"].tolist()
    return rankings


def eval_config(precomputed, weights, normalize, ks=(1, 3, 5, 10)):
    """Cheap re-combination: reuse calculate_total_score, rank, score metrics."""
    per_case = []
    raw_gold = {5: [], 10: []}
    for case, comp in precomputed:
        df = calculate_total_score(comp.copy(), case.parsed_jd, weights=weights, normalize=normalize)
        ranked = df.sort_values("total_score", ascending=False)["candidate_id"].tolist()
        relevant = {cid for cid, grade in case.relevance.items() if grade > 0}
        row = {"case_id": case.case_id, "source": case.source, "n_ranked": len(ranked)}
        for k in ks:
            row[f"hit@{k}"] = hit_at_k(ranked, relevant, k)
        row["rr"] = reciprocal_rank(ranked, relevant)
        if case.seed_id:
            row["seed_rank"] = rank_of(ranked, case.seed_id)
        if case.source == "llm_scored":
            for k in (5, 10):
                value = ndcg_at_k(ranked, case.relevance, k)
                row[f"ndcg@{k}"] = value
                raw_gold[k].append(value)
        per_case.append(row)
    summary = aggregate(per_case)
    for k, values in raw_gold.items():
        if values:
            summary[f"raw_ndcg@{k}"] = sum(values) / len(values)
    return summary


def assert_runner_parity(precomputed, df_full, model, sim_spec, weights=None):
    """Fail closed unless fast recombination reproduces the production runner."""
    fast_rankings = ranked_config(precomputed, weights, normalize=False)
    config = PipelineConfig(
        title_mode="hybrid",
        title_hard=False,
        skill_mode="hybrid",
        weights=weights,
        normalize_components=False,
    )
    for case, _ in precomputed:
        runner_ranking = rank_candidates(
            df_full, case.parsed_jd, model, config, sim_spec=sim_spec
        )
        fast_ranking = fast_rankings[case.case_id]
        if runner_ranking != fast_ranking:
            mismatch = next(
                (
                    index
                    for index, (runner_id, fast_id) in enumerate(
                        zip(runner_ranking, fast_ranking), start=1
                    )
                    if runner_id != fast_id
                ),
                min(len(runner_ranking), len(fast_ranking)) + 1,
            )
            raise AssertionError(
                f"calibrator parity failed for {case.case_id} at rank {mismatch}"
            )
    print(f"calibrator parity: PASS ({len(precomputed)} cases, exact rankings)")


def _construct_report(precomputed, df_full, judgments_path, summary_path):
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if summary.get("rubric_version") != "t3-tenure-relevance-language-v1":
        raise ValueError("C5 re-ablation requires the promoted T3 rubric artifact")
    judges = summary.get("judges") or []
    if len(judges) != 2:
        raise ValueError("C5 re-ablation requires exactly two pinned judges")

    judgments = pd.read_csv(judgments_path)
    required = {
        "candidate_id",
        "tenure_continuity_mean_score",
        "career_relevance_mix_mean_score",
        "preferred_signals_mean_score",
    }
    missing = required - set(judgments.columns)
    if missing:
        raise ValueError(f"T3 judgment artifact missing columns: {sorted(missing)}")

    gold_components = next(
        comp for case, comp in precomputed if case.source == "llm_scored"
    )
    profile_inputs = df_full[
        ["candidate_id", "certifications", "languages"]
    ].copy()
    frame = (
        gold_components.merge(judgments, on="candidate_id", how="inner")
        .merge(profile_inputs, on="candidate_id", how="left")
    )
    if len(frame) != len(judgments):
        raise ValueError(
            f"component/judgment coverage mismatch: {len(frame)} vs {len(judgments)}"
        )

    frame["language_credit"] = _credit_count(
        frame,
        judges,
        {"english", "arabic", "tagalog_or_filipino"},
    )
    frame["certification_credit"] = _credit_count(
        frame,
        judges,
        {"cipd_or_equivalent"},
    )
    correlations = {
        "attrition_score": _spearman(
            frame["attrition_score"], frame["tenure_continuity_mean_score"]
        ),
        "experience_relevance_score": _spearman(
            frame["experience_relevance_score"],
            frame["career_relevance_mix_mean_score"],
        ),
        "education_relevance_score": _spearman(
            frame["education_relevance_score"], frame["certification_credit"]
        ),
        "language_score": _spearman(
            frame["language_score"], frame["language_credit"]
        ),
    }

    has_hr_cert = frame["certifications"].map(
        lambda values: any(
            token in str(certification).lower()
            for certification in (values or [])
            for token in education_hr_certs
        )
    )
    degree_only = (
        (frame["education_relevance_score"] > education_neutral) & ~has_hr_cert
    )
    explicit_tagalog = frame["languages"].map(
        lambda values: any(normalize_language(language) == "filipino" for language in (values or []))
    )

    validation = summary.get("validation") or {}
    section_gates = validation.get("section_gates") or {}
    component_gates = {}
    for component, section in C5_SECTION_GATES.items():
        correlation = correlations[component]
        component_gates[component] = {
            "section": section,
            "overall_agreement_pass": bool(validation.get("all_pass")),
            "section_agreement_pass": bool(section_gates.get(section, {}).get("pass")),
            "construct_spearman": correlation,
            "construct_direction_pass": correlation is not None and correlation > 0,
        }
        component_gates[component]["pass"] = all(
            (
                component_gates[component]["overall_agreement_pass"],
                component_gates[component]["section_agreement_pass"],
                component_gates[component]["construct_direction_pass"],
            )
        )

    return {
        "n_judged": int(len(frame)),
        "correlations": correlations,
        "component_gates": component_gates,
        "education_partial_anchor": {
            "caveat": "The rubric directly grades HR certification, not degree relevance.",
            "certification_bearing": _cohort_summary(frame, has_hr_cert),
            "degree_only": _cohort_summary(frame, degree_only),
        },
        "language_evidence": {
            "explicit_language_count": int(frame["languages"].map(bool).sum()),
            "explicit_tagalog_or_filipino_count": int(explicit_tagalog.sum()),
            "tagalog_validated": False,
            "caveat": "Any adoption is a general-language result; Tagalog-specific evidence is n=1.",
        },
    }


def _weights_from(base, updates):
    weights = dict(base)
    weights.update(updates)
    return weights


def c5_reablation(
    precomputed,
    df_full,
    judgments_path=JUDGMENTS_CSV,
    summary_path=JUDGMENT_SUMMARY,
    out_path=C5_OUT,
    grid=C5_GRID,
):
    construct = _construct_report(
        precomputed, df_full, judgments_path, summary_path
    )
    incumbent_weights = dict(candidate_score_weights)
    incumbent = eval_config(precomputed, incumbent_weights, normalize=False)
    floors = _reset_floors(incumbent)
    incumbent_floor_results = _floor_results(incumbent, floors)
    if not incumbent_floor_results["pass"]:
        raise AssertionError("new-label incumbent does not clear its reset regression floors")

    neutral_weights = dict(incumbent_weights)
    for component in C5_COMPONENTS:
        neutral_weights[component] = 0.0
    neutral = eval_config(precomputed, neutral_weights, normalize=False)
    neutral_ndcg10 = _metric(neutral, "ndcg@10")

    sweeps = {}
    best_by_component = {}
    for component in C5_COMPONENTS:
        rows = []
        component_gate = construct["component_gates"][component]
        for weight in grid:
            weights = _weights_from(neutral_weights, {component: weight})
            metrics = eval_config(precomputed, weights, normalize=False)
            delta = _metric(metrics, "ndcg@10") - neutral_ndcg10
            floor_results = _floor_results(metrics, floors)
            eligible = (
                weight > 0
                and delta > 0
                and floor_results["pass"]
                and component_gate["pass"]
            )
            rows.append(
                {
                    "weight": weight,
                    "metrics": metrics,
                    "ndcg10_delta_vs_neutral": delta,
                    "floors": floor_results,
                    "eligible": eligible,
                }
            )
        sweeps[component] = rows
        eligible_rows = [row for row in rows if row["eligible"]]
        if eligible_rows:
            best_by_component[component] = min(
                eligible_rows,
                key=lambda row: (-_metric(row["metrics"], "ndcg@10"), row["weight"]),
            )

    combinations = []
    eligible_components = sorted(best_by_component)
    for size in range(len(eligible_components) + 1):
        for subset_tuple in itertools.combinations(eligible_components, size):
            subset = set(subset_tuple)
            updates = {
                component: best_by_component[component]["weight"]
                for component in subset
            }
            weights = _weights_from(neutral_weights, updates)
            metrics = eval_config(precomputed, weights, normalize=False)
            floor_results = _floor_results(metrics, floors)
            leave_one_out = {}
            for component in subset:
                without = dict(updates)
                without.pop(component)
                without_metrics = eval_config(
                    precomputed,
                    _weights_from(neutral_weights, without),
                    normalize=False,
                )
                delta = _metric(metrics, "ndcg@10") - _metric(
                    without_metrics, "ndcg@10"
                )
                leave_one_out[component] = {
                    "ndcg10_delta": delta,
                    "pass": delta > 0,
                }
            valid = (
                floor_results["pass"]
                and _metric(metrics, "ndcg@10") >= neutral_ndcg10
                and all(item["pass"] for item in leave_one_out.values())
            )
            combinations.append(
                {
                    "components": sorted(subset),
                    "weights": updates,
                    "metrics": metrics,
                    "ndcg10_delta_vs_neutral": _metric(metrics, "ndcg@10") - neutral_ndcg10,
                    "floors": floor_results,
                    "leave_one_out": leave_one_out,
                    "valid": valid,
                }
            )

    valid_combinations = [combination for combination in combinations if combination["valid"]]
    if not valid_combinations:
        raise AssertionError("no C5-neutral or eligible subset clears the reset floors")
    selected = min(
        valid_combinations,
        key=lambda combination: (
            -_metric(combination["metrics"], "ndcg@10"),
            len(combination["components"]),
            sum(combination["weights"].values()),
            combination["components"],
        ),
    )
    recommended_c5_weights = {
        component: selected["weights"].get(component, 0.0)
        for component in C5_COMPONENTS
    }

    report = {
        "methodology": {
            "anchor": "silver Judge-grade anchor (n=1; circular, not human ground truth)",
            "control": "all four C5 components at raw weight 0",
            "grid": grid,
            "primary_metric": "unrounded NDCG@10",
            "strict_gain_required": True,
            "combination_rule": "best eligible subset with strict leave-one-out contribution",
            "floor_reset": "gold floors truncated down to 0.01 from the new-label incumbent; reverse floors unchanged",
        },
        "incumbent_weights": incumbent_weights,
        "incumbent_metrics": incumbent,
        "reset_floors": floors,
        "incumbent_floor_results": incumbent_floor_results,
        "neutral_weights": neutral_weights,
        "neutral_metrics": neutral,
        "construct_validity": construct,
        "sweeps": sweeps,
        "best_individual": best_by_component,
        "combinations": combinations,
        "selected": selected,
        "recommended_c5_weights": recommended_c5_weights,
    }
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("=== C5 re-ablation (common neutral control) ===")
    print(f"incumbent raw NDCG@10: {_metric(incumbent, 'ndcg@10'):.6f}")
    print(f"neutral raw NDCG@10:   {neutral_ndcg10:.6f}")
    for component in C5_COMPONENTS:
        winner = best_by_component.get(component)
        gate = construct["component_gates"][component]
        if winner:
            print(
                f"{component}: eligible w={winner['weight']:.3f} "
                f"delta={winner['ndcg10_delta_vs_neutral']:+.6f}"
            )
        else:
            print(
                f"{component}: REJECT (section/construct gate={gate['pass']} or no strict gain)"
            )
    print(f"selected C5 weights: {recommended_c5_weights}")
    print(f"wrote {out_path}")
    return report


def ablate_component(precomputed, component, grid):
    """1-D sweep of one component's weight, others held at champion (renormalized)."""
    print(f"=== 1-D ablation: {component} (base = champion weights, renormalized) ===")
    print(f"{'w':>6} {'mrr':>7} {'ndcg@10':>8} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'hit@10':>7}")
    base = dict(candidate_score_weights)
    for w in grid:
        weights = dict(base)
        weights[component] = w
        s = eval_config(precomputed, weights, normalize=False)
        print(
            f"{w:6.2f} {s.get('mrr', 0):7.4f} {s.get('ndcg@10', 0):8.4f} "
            f"{s.get('hit@1', 0):6.4f} {s.get('hit@3', 0):6.4f} "
            f"{s.get('hit@5', 0):6.4f} {s.get('hit@10', 0):7.4f}"
        )


def redundancy_2x2(precomputed, comp_a, comp_b, w):
    """2x2 comparison (neither / A only / B only / both) at a fixed weight each.

    Reveals whether two components are redundant: if 'B only' ~= 'both', then A
    adds nothing beyond B (and vice-versa).
    """
    print(f"=== redundancy 2x2: {comp_a} x {comp_b} @ {w} each (others = champion) ===")
    print(f"{'config':20} {'mrr':>7} {'ndcg@10':>8} {'hit@3':>6} {'hit@5':>6} {'hit@10':>7}")
    base = dict(candidate_score_weights)
    base[comp_a] = 0.0
    base[comp_b] = 0.0
    for label, a_on, b_on in [("neither", False, False), (f"{comp_a} only", True, False),
                              (f"{comp_b} only", False, True), ("both", True, True)]:
        weights = dict(base)
        if a_on:
            weights[comp_a] = w
        if b_on:
            weights[comp_b] = w
        s = eval_config(precomputed, weights, normalize=False)
        print(f"{label:20} {s['mrr']:7.4f} {s.get('ndcg@10', 0):8.4f} "
              f"{s['hit@3']:6.4f} {s['hit@5']:6.4f} {s['hit@10']:7.4f}")


def joint_sweep(precomputed, grid, top):
    """Exhaustive sweep over ALL six component weights (normalization off).

    Ranked by the leakage-free gold NDCG@10 (MRR as tie-break). The default grid
    contains every champion value {0.0, 0.05, 0.25, 0.45}, so the champion config
    is itself a grid point and we can report exactly how many configs beat it.
    """
    comps = ["title_score", "skill_score", "qualification_score",
             "similarity_score", "seniority_score", "experience_score"]
    rows = []
    for combo in itertools.product(grid, repeat=len(comps)):
        if sum(combo) == 0:
            continue
        s = eval_config(precomputed, dict(zip(comps, combo)), normalize=False)
        rows.append((round(s.get("ndcg@10", 0.0), 4), round(s["mrr"], 4),
                     round(s["hit@5"], 4), round(s["hit@10"], 4), combo))
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)

    champ = eval_config(precomputed, dict(candidate_score_weights), normalize=False)
    champ_ndcg = round(champ.get("ndcg@10", 0.0), 4)
    better = [r for r in rows if r[0] > champ_ndcg]

    print(f"swept {len(rows)} configs (grid={grid}); ranked by gold NDCG@10, MRR tie-break")
    print(f"CHAMPION: ndcg@10={champ_ndcg} mrr={round(champ['mrr'], 4)} "
          f"hit@5={round(champ['hit@5'], 4)} hit@10={round(champ['hit@10'], 4)}")
    print(f"configs strictly beating champion on gold NDCG@10: {len(better)}\n")
    print(f"{'ndcg@10':>8} {'mrr':>7} {'hit@5':>6} {'hit@10':>7}  t/sk/q/sim/sen/exp")
    for r in rows[:top]:
        w = "/".join(f"{x:.2f}" for x in r[4])
        print(f"{r[0]:8.4f} {r[1]:7.4f} {r[2]:6.4f} {r[3]:7.4f}  {w}")


def _fmt(r):
    return (f"mrr={r[0]:.4f} ndcg10={r[1]:.4f} hit3={r[2]:.4f} hit5={r[3]:.4f} "
            f"hit10={r[4]:.4f} norm={r[5]!s:5} t/s/sim={r[6]}/{r[7]}/{r[8]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-group", type=int, default=5)
    ap.add_argument("--grid", type=float, nargs="+", default=[0.1, 0.2, 0.35, 0.5])
    ap.add_argument("--qual-weight", type=float, default=0.05)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--ablate", default=None,
                    help="component to 1-D ablate (e.g. seniority_score); skips the full grid")
    ap.add_argument("--ablate-grid", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    ap.add_argument("--redundancy", nargs=2, metavar=("A", "B"), default=None,
                    help="2x2 comparison of two components (neither/A/B/both)")
    ap.add_argument("--redundancy-weight", type=float, default=0.05)
    ap.add_argument("--joint", action="store_true",
                    help="exhaustive sweep over all six component weights (incl. champion values)")
    ap.add_argument("--joint-grid", type=float, nargs="+", default=[0.0, 0.05, 0.25, 0.45])
    ap.add_argument("--c5-reablate", action="store_true",
                    help="run the preregistered T3 C5 common-control decision workflow")
    ap.add_argument("--c5-grid", type=float, nargs="+", default=C5_GRID)
    ap.add_argument("--judgments", default=JUDGMENTS_CSV)
    ap.add_argument("--judgment-summary", default=JUDGMENT_SUMMARY)
    ap.add_argument("--c5-out", default=C5_OUT)
    ap.add_argument("--skill-mode", choices=["fuzzy", "semantic", "hybrid"], default="hybrid",
                    help="skill matcher used when precomputing the skill_score column")
    ap.add_argument("--skill-semantic-threshold", type=float, default=None,
                    help="cosine floor for semantic/hybrid skill matches (default: mappings constant)")
    args = ap.parse_args()

    profiles = LinkedInAdapter().to_profiles(LINKEDIN_CSV)
    model = SentenceTransformer("all-mpnet-base-v2")
    sim_spec = build_similarity_spec(similarity_model_config, base_model=model)
    embedding_model = sim_spec.model if sim_spec else model
    cache = CACHE
    if sim_spec:
        slug = similarity_model_config["model_name"].replace("/", "_")
        cache = f".ai-recruiter/emb_linkedin_v2_{slug}.pkl"
    embed_profiles(
        profiles,
        embedding_model,
        cache_path=cache,
        model_key=sim_spec.model_key if sim_spec else None,
        doc_instruction=sim_spec.doc_instruction if sim_spec else None,
        batch_size=sim_spec.batch_size if sim_spec else 32,
    )
    df_full = profiles_to_dataframe(profiles)

    cases = load_cases(profiles, args.n_per_group)
    n_rev = sum(c.source == "reverse_match" for c in cases)
    n_gold = sum(c.source == "llm_scored" for c in cases)
    print(f"{len(cases)} cases ({n_rev} reverse + {n_gold} gold)")
    print(f"skill_mode={args.skill_mode} semantic_threshold={args.skill_semantic_threshold}")

    t = time.perf_counter()
    precomputed = precompute_components(cases, df_full, model,
                                        skill_mode=args.skill_mode,
                                        semantic_threshold=args.skill_semantic_threshold,
                                        sim_spec=sim_spec)
    print(f"components precomputed in {time.perf_counter() - t:.1f}s\n")
    if args.skill_mode == "hybrid" and args.skill_semantic_threshold is None:
        assert_runner_parity(precomputed, df_full, model, sim_spec)

    if args.c5_reablate:
        c5_reablation(
            precomputed,
            df_full,
            judgments_path=args.judgments,
            summary_path=args.judgment_summary,
            out_path=args.c5_out,
            grid=args.c5_grid,
        )
        return

    if args.redundancy:
        redundancy_2x2(precomputed, args.redundancy[0], args.redundancy[1], args.redundancy_weight)
        return

    if args.joint:
        joint_sweep(precomputed, args.joint_grid, args.top)
        return

    if args.ablate:
        ablate_component(precomputed, args.ablate, args.ablate_grid)
        return

    print("=== normalization @ champion weights (0.25/0.25/0.05/0.45) ===")
    for norm in (False, True):
        s = eval_config(precomputed, None, norm)
        print(f"norm={norm!s:5} mrr={s['mrr']:.4f} ndcg10={s.get('ndcg@10', 0):.4f} "
              f"hit3={s['hit@3']:.4f} hit5={s['hit@5']:.4f} hit10={s['hit@10']:.4f}")

    rows = []
    t = time.perf_counter()
    for norm in (True, False):
        for wt, ws, wsim in itertools.product(args.grid, repeat=3):
            w = {"title_score": wt, "skill_score": ws,
                 "qualification_score": args.qual_weight, "similarity_score": wsim}
            s = eval_config(precomputed, w, norm)
            rows.append((s["mrr"], s.get("ndcg@10", 0.0), s["hit@3"], s["hit@5"],
                         s["hit@10"], norm, wt, ws, wsim))
    print(f"\nswept {len(rows)} configs in {time.perf_counter() - t:.1f}s")

    print(f"\n=== TOP {args.top} by MRR ===")
    for r in sorted(rows, key=lambda x: (x[0], x[1]), reverse=True)[:args.top]:
        print(_fmt(r))
    print(f"\n=== TOP {args.top} by NDCG@10 ===")
    for r in sorted(rows, key=lambda x: (x[1], x[0]), reverse=True)[:args.top]:
        print(_fmt(r))


if __name__ == "__main__":
    main()
