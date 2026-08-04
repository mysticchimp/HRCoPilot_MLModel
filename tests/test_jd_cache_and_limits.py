"""JD hash-keyed cache + /score soft batch limit."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException


def test_jd_hash_cache_hit_skips_llm(tmp_path: Path):
    from core.jd_extraction import jd_text_hash, process_jd
    from models.data_models import Company, JobRoleSchema, Skill
    from models.enums import ImportanceLevel

    jd = JobRoleSchema(
        role="HR Assistant",
        company=Company(name="C"),
        responsibilities=["r"],
        skills=[Skill(skill="payroll", priority=ImportanceLevel.ESSENTIAL, proficiency_level=None)],
    )
    text = "Unique JD text for hash cache test AAA"
    # Seed cache file
    key = jd_text_hash(text)
    path = tmp_path / f"{key}.json"
    path.write_text(jd.model_dump_json(), encoding="utf-8")

    fake = MagicMock()
    out = process_jd(text, provider=fake, cache_dir=str(tmp_path))
    assert out.role == "HR Assistant"
    fake.generate_structured.assert_not_called()


def test_jd_hash_cache_miss_writes_and_invalidates_on_text_change(tmp_path: Path):
    from core.jd_extraction import jd_text_hash, process_jd
    from models.data_models import Company, JobRoleSchema, Skill
    from models.enums import ImportanceLevel

    jd_a = JobRoleSchema(
        role="Role A",
        company=Company(name="C"),
        responsibilities=["r"],
        skills=[Skill(skill="s", priority=ImportanceLevel.ESSENTIAL, proficiency_level=None)],
    )
    jd_b = JobRoleSchema(
        role="Role B",
        company=Company(name="C"),
        responsibilities=["r"],
        skills=[Skill(skill="s", priority=ImportanceLevel.ESSENTIAL, proficiency_level=None)],
    )

    class _Prov:
        def __init__(self):
            self.n = 0

        def generate_structured(self, *a, **k):
            self.n += 1
            return jd_a if self.n == 1 else jd_b

    prov = _Prov()
    t1 = "JD version one"
    t2 = "JD version two — changed"
    assert jd_text_hash(t1) != jd_text_hash(t2)

    a1 = process_jd(t1, provider=prov, cache_dir=str(tmp_path))
    a2 = process_jd(t1, provider=prov, cache_dir=str(tmp_path))  # hit
    b1 = process_jd(t2, provider=prov, cache_dir=str(tmp_path))  # miss
    assert a1.role == "Role A" and a2.role == "Role A"
    assert b1.role == "Role B"
    assert prov.n == 2
    assert (tmp_path / f"{jd_text_hash(t1)}.json").exists()
    assert (tmp_path / f"{jd_text_hash(t2)}.json").exists()


def test_process_jd_delegates_with_cache_disabled():
    from core.jd_extraction import process_jd
    from models.data_models import Company, JobRoleSchema, Skill
    from models.enums import ImportanceLevel

    jd = JobRoleSchema(
        role="X",
        company=Company(name="C"),
        responsibilities=["r"],
        skills=[Skill(skill="s", priority=ImportanceLevel.ESSENTIAL, proficiency_level=None)],
    )

    class _Fake:
        def generate_structured(self, *a, **k):
            return jd

    assert process_jd("some jd text", provider=_Fake(), cache_dir="0").role == "X"


def test_score_batch_limit_rejects():
    import api.main as main

    with patch.object(main, "_SCORE_MAX_CANDIDATES", 5), patch.object(
        main, "_models_ready", True
    ), patch.object(main, "_embedding_model", object()):
        with pytest.raises(HTTPException) as exc:
            main._require_batch_fits(6)
        assert exc.value.status_code == 422
        assert "batch too large" in exc.value.detail
        main._require_batch_fits(5)  # no raise
