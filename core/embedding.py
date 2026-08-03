import hashlib
import logging
import os
import pickle
from dataclasses import dataclass

import torch
from sentence_transformers import SentenceTransformer

from models.candidate import CandidateProfile
from models.data_models import JobRoleSchema

logger = logging.getLogger(__name__)


def log_truncation(model, texts: list[str], label: str) -> int:
    """Warn when inputs exceed the encoder's max_seq_length (content is silently
    truncated). Returns the number of truncated texts. A signal, not an error — as
    JDs/profiles grow past the cap this surfaces where the embedding loses information.
    """
    max_len = getattr(model, "max_seq_length", None)
    tok = getattr(model, "tokenizer", None)
    if not max_len or tok is None:
        return 0
    lengths = [len(tok.encode(t, add_special_tokens=True)) for t in texts]
    n_trunc = sum(1 for length in lengths if length > max_len)
    if n_trunc:
        logger.warning(
            "[truncation] %d/%d %s exceed max_seq_length=%d (longest=%d tokens) — "
            "content beyond the cap is dropped from the embedding",
            n_trunc, len(texts), label, max_len, max(lengths),
        )
    return n_trunc


def build_jd_embedding_input(jd_data: JobRoleSchema):
    parts = []

    if jd_data.role:
        parts.append(f"Role: {jd_data.role}")

    if jd_data.responsibilities:
        parts.append("Responsibilities: " + ", ".join(jd_data.responsibilities))

    if jd_data.role_objectives:
        parts.append("Career Objective: " + ", ".join(jd_data.role_objectives))

    if jd_data.skills:
        parts.append("Skills: " + ", ".join([s.skill for s in jd_data.skills]))

    if jd_data.technologies:
        parts.append("Technologies: " + ", ".join([t.technology for t in jd_data.technologies]))

    return " ".join(parts)


def build_candidate_embedding_input(profile: CandidateProfile) -> str:
    """Compose a candidate's embedding text from canonical fields.

    Symmetric with build_jd_embedding_input (Responsibilities / Career Objective /
    Skills) so candidate and JD vectors describe the same aspects. Skills are the
    key addition (5.1): they are the most candidate-specific signal and were
    previously absent from the 0.45-weight semantic component.
    """
    parts = []
    if profile.summary:
        parts.append(f"Career Objective: {profile.summary}")
    if profile.responsibilities:
        parts.append(f"Responsibilities: {profile.responsibilities}")
    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))
    return " ".join(parts)


def build_rerank_jd_text(jd: JobRoleSchema) -> str:
    """Rich JD document for the Stage-2 cross-encoder.

    Deliberately fuller than build_jd_embedding_input (the thin bi-encoder input): the
    cross-encoder has a long context and benefits from the whole structured ask —
    role, company, industries, objectives, responsibilities, skills (with priority),
    technologies, qualifications, experience, languages, location.
    """
    def _imp(x):
        return getattr(getattr(x, "priority", None), "value", None)

    parts = [f"Role: {jd.role}"]
    if jd.company and jd.company.name:
        extra = ", ".join(v for v in [jd.company.size, jd.company.stage] if v)
        parts.append(f"Company: {jd.company.name}" + (f" ({extra})" if extra else ""))
    if jd.industry:
        parts.append("Industries: " + ", ".join(jd.industry))
    if jd.role_objectives:
        parts.append("Objectives: " + "; ".join(jd.role_objectives))
    if jd.responsibilities:
        parts.append("Responsibilities: " + "; ".join(jd.responsibilities))
    if jd.skills:
        parts.append("Required skills: " + ", ".join(
            f"{s.skill} [{_imp(s)}]" if _imp(s) else s.skill for s in jd.skills))
    if jd.technologies:
        parts.append("Technologies: " + ", ".join(t.technology for t in jd.technologies))
    if jd.qualifications and jd.qualifications.education:
        parts.append("Education: " + "; ".join(
            " ".join(x for x in [e.degree, (f"in {e.field}" if e.field else None)] if x)
            for e in jd.qualifications.education))
    if jd.experience:
        ex = jd.experience
        bits = []
        if ex.level:
            bits.append(f"level {ex.level}")
        for label, rng in [("total", ex.years_total), ("relevant", ex.years_relevant)]:
            if rng and (rng.min is not None or rng.max is not None):
                bits.append(f"{label} years {rng.min if rng.min is not None else 0}-{rng.max if rng.max is not None else ''}")
        if ex.industry_experience:
            bits.append("industries " + ", ".join(i.industry for i in ex.industry_experience))
        if bits:
            parts.append("Experience: " + "; ".join(bits))
    if jd.language_proficiency:
        parts.append("Languages: " + ", ".join(f"{lp.language} ({lp.level})" for lp in jd.language_proficiency))
    if jd.location and (jd.location.cities or jd.location.countries):
        parts.append("Location: " + ", ".join((jd.location.cities or []) + (jd.location.countries or [])))
    return "\n".join(parts)


def build_rerank_candidate_text(profile: CandidateProfile) -> str:
    """Rich candidate document for the Stage-2 cross-encoder.

    Mirrors the blind-judge payload (title, seniority, tenure, location, about,
    responsibilities, skills, education, employers, languages, certifications) — much
    fuller than the thin bi-encoder build_candidate_embedding_input, since the CE can
    attend over the full evidence.
    """
    parts = []
    if profile.job_title:
        parts.append(f"Title: {profile.job_title}")
    if profile.seniority:
        parts.append(f"Seniority: {profile.seniority}")
    if profile.years_experience is not None:
        parts.append(f"Total experience: {profile.years_experience} years")
    if profile.location:
        loc = ", ".join(x for x in [profile.location.city, profile.location.country] if x)
        if loc:
            parts.append(f"Location: {loc}")
    if profile.summary:
        parts.append(f"About: {profile.summary}")
    if profile.responsibilities:
        parts.append(f"Responsibilities: {profile.responsibilities}")
    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))
    if profile.education:
        edu = "; ".join(
            " ".join(x for x in [e.degree, (f"in {e.field}" if e.field else None),
                                 (f"@ {e.school}" if e.school else None)] if x)
            for e in profile.education if (e.degree or e.field or e.school))
        if edu:
            parts.append(f"Education: {edu}")
    if profile.employers:
        parts.append("Employers: " + ", ".join(profile.employers))
    if profile.languages:
        langs = ", ".join(lang.language for lang in profile.languages if lang.language)
        if langs:
            parts.append(f"Languages: {langs}")
    if profile.certifications:
        parts.append("Certifications: " + ", ".join(profile.certifications))
    return "\n".join(parts)


def embed_profiles(
    profiles: list[CandidateProfile],
    model,
    cache_path: str | None = None,
    model_key: str | None = None,
    doc_instruction: str | None = None,
    batch_size: int = 32,
):
    """Populate profile_text + profile_embedding for each profile (batch encoded).

    When cache_path is given, embeddings are cached keyed by a content hash of the
    profile texts, so unchanged inputs are not re-encoded (fixes stale-cache issues).

    `model_key` (an embedding-model identifier) and `doc_instruction` are folded into
    the cache key so swapping the embedding model — or toggling instruction prefixes —
    never returns stale or wrong-dimension vectors from a previous model. Callers that
    compare models MUST pass a distinct `model_key` (and ideally a per-model cache_path).
    `doc_instruction`, when given, is prepended to each candidate profile (the document
    side) for instruction-tuned encoders.
    """
    for profile in profiles:
        profile.profile_text = build_candidate_embedding_input(profile)
    texts = [p.profile_text or "" for p in profiles]
    encode_texts = [f"{doc_instruction}{t}" for t in texts] if doc_instruction else texts
    log_truncation(model, encode_texts, "candidate profiles")

    key_material = f"{model_key or ''}\x1f{doc_instruction or ''}\x1f" + "||".join(texts)
    key = hashlib.md5(key_material.encode("utf-8")).hexdigest()
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("key") == key and len(cached.get("embeddings", [])) == len(profiles):
            for profile, emb in zip(profiles, cached["embeddings"]):
                profile.profile_embedding = emb
            return profiles

    encoded = model.encode(
        encode_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size
    )
    if torch.isnan(encoded).any():
        raise ValueError(
            "embed_profiles: model produced NaN embeddings — check device/dtype "
            "(some custom encoders NaN on Apple MPS; retry with device='cpu' + float32)."
        )
    embeddings = [vec.cpu().numpy().tolist() for vec in encoded]
    for profile, emb in zip(profiles, embeddings):
        profile.profile_embedding = emb

    if cache_path:
        with open(cache_path, "wb") as fh:
            pickle.dump({"key": key, "embeddings": embeddings}, fh)
    return profiles


@dataclass
class SimilaritySpec:
    """Optional isolated embedding model for `similarity_score` only (Option B).

    When None everywhere, similarity uses the base `model` — title/skill semantic legs
    are untouched. When given, ONLY the candidate profile embeddings and the JD embedding
    use `model`; `model_key` disambiguates the on-disk cache; the instructions are
    prepended for instruction-tuned encoders (query side = the JD, doc side = candidates).
    """
    model: SentenceTransformer
    model_key: str
    query_instruction: str | None = None
    doc_instruction: str | None = None
    batch_size: int = 32


def _resolve_load_dtype(dtype: str, device: str | None):
    """Map config dtype string → torch.dtype for construction-time loading.

    fp16 is applied via ``model_kwargs={"torch_dtype": torch.float16}`` at load —
    never post-load ``.half()``, which briefly doubles peak RSS (fp32 + fp16).
    """
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32" or (dtype == "auto" and device == "cpu"):
        return torch.float32
    return None


def build_similarity_spec(config: dict | None, base_model=None) -> "SimilaritySpec | None":
    """Construct a SimilaritySpec from a config dict, loading the model with the right
    device/dtype/max_seq. Returns None when config is falsy (similarity uses base_model).

    config keys: model_name, query_instruction, doc_instruction, dtype
    (auto|fp32|fp16|bf16), device (None=auto), max_seq_length, batch_size.
    Notes from measurement: some encoders (e.g. Jasper) NaN on Apple MPS — pin device='cpu';
    fp16 is loaded directly via torch_dtype (no post-load .half() memory spike).
    """
    if not config:
        return None
    name = config["model_name"]
    if base_model is not None and name == "all-mpnet-base-v2":
        model = base_model
    else:
        load_kwargs: dict = {"trust_remote_code": True}
        device = config.get("device")
        if device:
            load_kwargs["device"] = device
        dtype = config.get("dtype", "auto")
        load_dtype = _resolve_load_dtype(dtype, device)
        if load_dtype is not None:
            load_kwargs["model_kwargs"] = {"torch_dtype": load_dtype}
        model = SentenceTransformer(name, **load_kwargs)
        if config.get("max_seq_length"):
            model.max_seq_length = config["max_seq_length"]
    q = config.get("query_instruction")
    d = config.get("doc_instruction")
    return SimilaritySpec(
        model=model,
        model_key=f"{name}|q={q}|d={d}|L={config.get('max_seq_length')}",
        query_instruction=q,
        doc_instruction=d,
        batch_size=config.get("batch_size", 32),
    )