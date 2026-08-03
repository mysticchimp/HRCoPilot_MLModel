import ast

def parse_list_str(list_str):
    if isinstance(list_str, str):
        try:
            return ast.literal_eval(list_str)
        except (ValueError, SyntaxError):
            return []
    return list_str if isinstance(list_str, list) else []

def candidate_to_json(row):
    return {
        "candidate_id": row["candidate_id"],
        "job_title": row.get("job_position_name"),
        "total_score": round(row.get("total_score", 0.0), 2),
        "scores": {
            "title_score": round(row.get("title_score", 0.0), 2),
            "skill_score": round(row.get("skill_score", 0.0), 2),
            "qualification_score": round(row.get("qualification_score", 0.0), 2),
            "semantic_score": round(row.get("similarity_score", 0.0), 2),
        },
        "matched": {
            "skills": row.get("matched_skills", []),
            "qualifications": row.get("matched_qualifications", []),
        },
        "career_objectives": row.get("career_objective", None),
        "responsibilities": row.get("responsibilities", None),
    }