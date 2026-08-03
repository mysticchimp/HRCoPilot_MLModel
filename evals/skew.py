from collections import Counter


def dataset_skew(profiles) -> dict:
    """Summarize dataset skew: title balance, field coverage, and text duplication.

    `unique_responsibilities` << n reveals degenerate (templated) profile text that
    collapses the semantic-embedding signal within a group.
    """
    n = len(profiles) or 1
    titles = Counter(p.job_title for p in profiles)

    def frac(pred) -> float:
        return round(sum(1 for p in profiles if pred(p)) / n, 3)

    return {
        "n": len(profiles),
        "unique_titles": len(titles),
        "top_titles": titles.most_common(8),
        "coverage": {
            "summary": frac(lambda p: bool(p.summary)),
            "responsibilities": frac(lambda p: bool(p.responsibilities)),
            "skills>0": frac(lambda p: len(p.skills) > 0),
            "education>0": frac(lambda p: len(p.education) > 0),
            "location": frac(lambda p: p.location is not None),
            "seniority": frac(lambda p: p.seniority is not None),
            "years_experience": frac(lambda p: p.years_experience is not None),
        },
        "text_uniqueness": {
            "unique_summary": len({p.summary for p in profiles if p.summary}),
            "unique_responsibilities": len({p.responsibilities for p in profiles if p.responsibilities}),
            "unique_skill_sets": len({tuple(sorted(p.skills)) for p in profiles}),
        },
        "seniority_dist": dict(Counter(p.seniority for p in profiles)),
    }
