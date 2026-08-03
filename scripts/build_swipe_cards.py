"""Emit the recruiter swipe-feed cards (C5 P7 / U2) — backend only.

Scores the LinkedIn pool against the HR-Assistant JD (no rerank) and serializes each
candidate into the grounded card contract (core.swipe.build_card), plus a `screen_me`
subset of low-data profiles routed to screening.

    COPILOT_SKIP_CLI_DOWNLOAD=1 uv run python scripts/build_swipe_cards.py [--top N]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer

from core.adapters.linkedin_adapter import LinkedInAdapter
from core.embedding import build_similarity_spec, embed_profiles
from core.swipe import build_card
from evals.cases import load_fixture
from models.data_models import JobRoleSchema
from models.mappings import similarity_model_config
from scripts.run_hr_assistant import JD_STORE, LINKEDIN_CSV, score_pool

CARDS_OUT = "evals/results/swipe_cards_hr_assistant.json"


def load_jd():
    if os.path.exists(JD_STORE):
        with open(JD_STORE) as fh:
            return JobRoleSchema.model_validate(json.load(fh))
    return load_fixture("linkedin", "_gold_hr_assistant").parsed_jd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=None, help="only emit the top-N cards")
    ap.add_argument("--out", default=CARDS_OUT)
    args = ap.parse_args()

    profiles = LinkedInAdapter().to_profiles(LINKEDIN_CSV)
    jd = load_jd()
    model = SentenceTransformer("all-mpnet-base-v2")
    sim_spec = build_similarity_spec(similarity_model_config, base_model=model)
    emb_model = sim_spec.model if sim_spec else model
    cache = ".ai-recruiter/emb_linkedin_v2.pkl"
    if sim_spec:
        slug = similarity_model_config["model_name"].replace("/", "_")
        cache = f".ai-recruiter/emb_linkedin_v2_{slug}.pkl"
    embed_profiles(
        profiles, emb_model, cache_path=cache,
        model_key=sim_spec.model_key if sim_spec else None,
        doc_instruction=sim_spec.doc_instruction if sim_spec else None,
        batch_size=sim_spec.batch_size if sim_spec else 32,
    )

    df = score_pool(profiles, jd, model, sim_spec=sim_spec)
    by_id = {p.candidate_id: p for p in profiles}
    rows = df if args.top is None else df.head(args.top)
    cards = [build_card(by_id[r["candidate_id"]], r.to_dict(), jd)
             for _, r in rows.iterrows() if r["candidate_id"] in by_id]
    screen_me = [c for c in cards if c["flags"]["data_completeness"] == "low"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"jd": jd.role, "count": len(cards), "cards": cards, "screen_me": screen_me}, fh, indent=2)

    print(f"wrote {len(cards)} cards ({len(screen_me)} low-data 'screen me') -> {args.out}")
    for c in cards[:3]:
        print(f"  #{c['rank']} {(c['name'] or '')[:26]:<26} score={c['total_score']} "
              f"signals={c['matched_signals'][:4]} flags={c['flags']}")


if __name__ == "__main__":
    main()
