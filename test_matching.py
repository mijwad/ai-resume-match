"""
Tests for matching_ranking.py

Run:  python test_matching.py          (no extra dependencies besides numpy)
  or: pytest test_matching.py

A tiny deterministic bag-of-words embedder stands in for Sentence-BERT so the
matching logic can be verified without downloading the model. At the bottom,
`test_with_real_sbert` runs against Kavya's real module if it is installed.
"""

import hashlib
import re

import numpy as np

from matching_ranking import (
    cosine_similarity, cosine_similarity_matrix, similarity_to_percentage,
    extract_skills, compare_skills, rank_resumes, rank_from_vectors,
    shortlist, threshold_sweep, suggest_threshold, results_to_dataframe,
    find_related_skills,
)

DIM = 64


def fake_embed_one(text):
    """Hashed bag-of-words, L2-normalised (mimics Kavya's normalised output)."""
    vec = np.zeros(DIM)
    for tok in re.findall(r"[a-z0-9+#.]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n else vec


def fake_embed_many(texts):
    return np.vstack([fake_embed_one(t) for t in texts])


JD = "Machine Learning Engineer: Python, NLP, deep learning, SQL and Docker required."
RESUMES = [
    ("Strong", "Python developer. Machine learning and natural language processing projects. "
               "Deep learning with PyTorch. SQL databases. Docker deployments."),
    ("Frontend", "Frontend developer skilled in HTML, CSS, JavaScript and React."),
    ("Partial", "Data analyst experienced in Python and SQL with statistics."),
]


# ---------------- cosine similarity ---------------- #
def test_cosine_identical_and_orthogonal():
    assert abs(cosine_similarity([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-9
    assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-9
    assert abs(cosine_similarity([1, 0], [-1, 0]) + 1.0) < 1e-9


def test_cosine_handles_unnormalised_and_zero():
    assert abs(cosine_similarity([3, 0], [10, 0]) - 1.0) < 1e-9
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_cosine_dimension_mismatch_raises():
    try:
        cosine_similarity([1, 2], [1, 2, 3])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_matrix_matches_pairwise():
    rng = np.random.default_rng(0)
    m, v = rng.normal(size=(5, 8)), rng.normal(size=8)
    expected = [cosine_similarity(row, v) for row in m]
    assert np.allclose(cosine_similarity_matrix(m, v), expected)


def test_matrix_zero_row_is_zero():
    m = np.array([[0.0, 0.0], [1.0, 0.0]])
    sims = cosine_similarity_matrix(m, np.array([1.0, 0.0]))
    assert sims[0] == 0.0 and abs(sims[1] - 1.0) < 1e-9


# ---------------- percentage ---------------- #
def test_percentage_basic_and_clipping():
    assert similarity_to_percentage(0.62) == 62.0
    assert similarity_to_percentage(-0.3) == 0.0
    assert similarity_to_percentage(1.0) == 100.0


def test_percentage_calibration():
    assert abs(similarity_to_percentage(0.45, calibrate=(0.10, 0.80)) - 50.0) < 1e-9
    assert similarity_to_percentage(0.05, calibrate=(0.10, 0.80)) == 0.0
    assert similarity_to_percentage(0.95, calibrate=(0.10, 0.80)) == 100.0


# ---------------- skills ---------------- #
def test_extract_skills_aliases():
    skills = extract_skills("Worked on natural language processing and Sklearn models")
    assert "nlp" in skills and "scikit-learn" in skills


def test_extract_skills_no_substring_false_positives():
    # 'java' must not match inside 'javascript'; 'r' must not match random letters
    skills = extract_skills("Experienced in JavaScript and React.")
    assert "javascript" in skills and "java" not in skills
    assert "r" not in skills


def test_extract_skills_special_characters():
    skills = extract_skills("Proficient in C++, C#, and Node.js.")
    assert {"c++", "c#", "node.js"} <= set(skills)


def test_extract_skills_excel_verb_not_a_skill():
    assert "excel" not in extract_skills("I excel at teamwork.")
    assert "excel" not in extract_skills("Eager to excel in a fast team; she excels at sales.")
    assert "excel" in extract_skills("Built dashboards in Microsoft Excel.")


def test_extract_skills_bare_excel_is_detected():
    assert "excel" in extract_skills("Python, SQL, Excel")
    assert "excel" in extract_skills("Proficient in Excel and Power BI.")
    assert "excel" in extract_skills("Skills: excel")
    assert "excel" in extract_skills("I excel at teamwork. Also proficient in Excel.")


def test_extract_skills_empty_and_non_string():
    assert extract_skills("") == [] and extract_skills(None) == []


def test_compare_skills_sets():
    comp = compare_skills("Python and SQL developer who knows React.",
                          "Need Python, SQL and Docker.")
    assert comp["matching"] == ["python", "sql"]
    assert comp["missing"] == ["docker"]
    assert comp["extra"] == ["react"]


def test_jd_alias_matches_resume_alias():
    comp = compare_skills("Expert in NLP.", "Seeking natural language processing skills.")
    assert comp["matching"] == ["nlp"] and comp["missing"] == []


# ---------------- ranking ---------------- #
def test_ranking_order_and_fields():
    ranked = rank_resumes(RESUMES, JD, embed_one=fake_embed_one, embed_many=fake_embed_many)
    assert [r.name for r in ranked][0] == "Strong"
    assert [r.rank for r in ranked] == [1, 2, 3]
    scores = [r.match_pct for r in ranked]
    assert scores == sorted(scores, reverse=True)
    top = ranked[0]
    assert {"python", "nlp", "sql", "docker", "deep learning", "machine learning"} >= set(top.matching_skills)
    assert "react" not in top.matching_skills
    frontend = next(r for r in ranked if r.name == "Frontend")
    assert "python" in frontend.missing_skills and frontend.skill_coverage_pct == 0.0
    assert all(0 <= r.match_pct <= 100 for r in ranked)


def test_threshold_shortlisting():
    ranked = rank_resumes(RESUMES, JD, embed_one=fake_embed_one, embed_many=fake_embed_many,
                          threshold=0)
    assert all(r.shortlisted for r in ranked)
    ranked = rank_resumes(RESUMES, JD, embed_one=fake_embed_one, embed_many=fake_embed_many,
                          threshold=101)
    assert not any(r.shortlisted for r in ranked)
    assert shortlist(ranked, threshold=0, top_k=2) and len(shortlist(ranked, 0, top_k=2)) == 2


def test_threshold_sweep_is_monotonic():
    ranked = rank_resumes(RESUMES, JD, embed_one=fake_embed_one, embed_many=fake_embed_many)
    counts = [c for _, c in threshold_sweep(ranked)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == 3  # threshold 0 keeps everyone


def test_suggest_threshold_hits_target():
    ranked = rank_resumes(RESUMES, JD, embed_one=fake_embed_one, embed_many=fake_embed_many)
    t = suggest_threshold(ranked, 2)
    assert len(shortlist(ranked, t)) >= 2
    assert t == ranked[1].match_pct


def test_empty_resume_does_not_crash_batch():
    cands = RESUMES + [("Blank", "   ")]
    ranked = rank_resumes(cands, JD, embed_one=fake_embed_one, embed_many=fake_embed_many,
                          threshold=0)
    last = ranked[-1]
    assert last.name == "Blank" and not last.valid and not last.shortlisted and last.match_pct == 0.0
    assert [r.valid for r in ranked] == [True, True, True, False]


def test_candidate_input_formats():
    ranked = rank_resumes(
        ["Python and SQL", ("Named", "Docker and Python"), {"id": "X9", "name": "Dict", "text": "NLP"}],
        JD, embed_one=fake_embed_one, embed_many=fake_embed_many)
    assert {r.candidate_id for r in ranked} == {"C1", "C2", "X9"}


def test_weights_change_scores_and_validate():
    kw = dict(embed_one=fake_embed_one, embed_many=fake_embed_many)
    sem_only = rank_resumes(RESUMES, JD, semantic_weight=1, skill_weight=0, **kw)
    for r in sem_only:
        assert abs(r.match_pct - r.semantic_pct) < 0.01
    try:
        rank_resumes(RESUMES, JD, semantic_weight=0, skill_weight=0, **kw)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for zero weights")


def test_jd_without_known_skills_falls_back_to_semantic():
    ranked = rank_resumes([("A", "Gardening and cooking")], "We want a friendly person.",
                          embed_one=fake_embed_one, embed_many=fake_embed_many)
    assert ranked[0].match_pct == ranked[0].semantic_pct


def test_empty_inputs():
    assert rank_resumes([], JD, embed_one=fake_embed_one, embed_many=fake_embed_many) == []
    try:
        rank_resumes(RESUMES, "  ", embed_one=fake_embed_one, embed_many=fake_embed_many)
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty JD")


def test_vector_count_mismatch_raises():
    try:
        rank_from_vectors(RESUMES, np.ones((2, DIM)), np.ones(DIM), JD)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _concept_embed_many(texts):
    """docker/container -> axis 0, sql -> axis 1, anything else -> axis 2."""
    def one(t):
        t = t.lower()
        if "docker" in t or "container" in t:
            return [1.0, 0.0, 0.0]
        if "sql" in t:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]
    return np.array([one(t) for t in texts])


# ---------------- semantic related-skill detection ---------------- #
def test_related_skills_uses_embeddings():
    """A missing skill phrase close to a resume sentence is reported as related."""
    related = find_related_skills(
        ["docker", "sql"], "Packaged services into containers.\nLoves gardening trips.",
        _concept_embed_many, threshold=0.5)
    assert related == ["docker"]


def test_semantic_skills_adds_partial_credit():
    embed_many = _concept_embed_many
    kw = dict(embed_one=lambda t: embed_many([t])[0], embed_many=embed_many)
    resume = [("Ops", "Packaged services into containers for production use.")]
    jd = "Need Docker."
    base = rank_resumes(resume, jd, **kw)[0]
    sem = rank_resumes(resume, jd, semantic_skills=True, **kw)[0]
    assert base.skill_coverage_pct == 0.0 and base.related_skills == []
    assert sem.related_skills == ["docker"] and sem.skill_coverage_pct == 50.0
    assert "docker" not in sem.missing_skills
    assert sem.match_pct > base.match_pct


# ---------------- export ---------------- #
def test_dataframe_export():
    ranked = rank_resumes(RESUMES, JD, embed_one=fake_embed_one, embed_many=fake_embed_many)
    df = results_to_dataframe(ranked)
    assert len(df) == 3 and "match_pct" in df.columns
    assert isinstance(df.loc[0, "matching_skills"], str)


# ---------------- optional: real Sentence-BERT ---------------- #
def test_with_real_sbert():
    try:
        from nlp_embeddings import generate_embedding  # noqa: F401  (Kavya's module)
    except Exception as exc:  # not installed / no model offline
        print(f"  (skipped real-SBERT test: {type(exc).__name__})")
        return
    ranked = rank_resumes(RESUMES, JD)
    print("  real SBERT ranking:", [(r.name, r.match_pct) for r in ranked])
    assert ranked[0].name == "Strong"
    assert ranked[-1].name == "Frontend"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
