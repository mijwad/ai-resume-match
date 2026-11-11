"""
ResumeMatch - Matching & Ranking Algorithm
==========================================
Depends: nlp_embeddings.py (Kavya) - Sentence-BERT, 384-d L2-normalised vectors

Responsibilities
----------------
1. Cosine similarity between resume and job-description embeddings
2. Convert similarity into a matching percentage
3. Rank candidates and generate a shortlist (with threshold tuning)
4. Identify matching and missing skills

The embedding functions are *injectable*: by default they are imported lazily
from Kavya's `nlp_embeddings` module, but any function with the same signature
can be passed in (useful for testing without downloading the model).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from skills_taxonomy import SKILL_TAXONOMY

# --------------------------------------------------------------------------- #
# Configuration defaults (all overridable per call)
# --------------------------------------------------------------------------- #
DEFAULT_SEMANTIC_WEIGHT = 0.7   # weight of embedding similarity in final score
DEFAULT_SKILL_WEIGHT = 0.3      # weight of skill coverage in final score
DEFAULT_SHORTLIST_THRESHOLD = 50.0   # match % needed to be shortlisted
DEFAULT_SKILL_SIM_THRESHOLD = 0.50   # cosine for a skill to count as "related"
RELATED_SKILL_CREDIT = 0.5           # partial credit for a semantically related skill

# (minimum match %, label) - checked top to bottom
MATCH_LABELS: List[Tuple[float, str]] = [
    (75.0, "Strong match"),
    (60.0, "Good match"),
    (45.0, "Partial match"),
    (0.0, "Weak match"),
]

EmbedOne = Callable[[str], np.ndarray]
EmbedMany = Callable[[Sequence[str]], np.ndarray]


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class MatchResult:
    candidate_id: str
    name: str
    similarity: float            # raw cosine similarity, -1..1
    semantic_pct: float          # similarity converted to 0..100
    skill_coverage_pct: float    # share of JD skills the resume covers, 0..100
    match_pct: float             # final blended score, 0..100
    label: str
    matching_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    related_skills: List[str] = field(default_factory=list)  # missing, but semantically close
    extra_skills: List[str] = field(default_factory=list)    # in resume, not asked for in JD
    rank: int = 0
    shortlisted: bool = False
    valid: bool = True           # False if the resume text was empty/unusable

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 1. Cosine similarity
# --------------------------------------------------------------------------- #
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity of two 1-D vectors.

    Works on normalised *and* non-normalised vectors. Returns 0.0 if either
    vector has zero length (instead of raising / returning NaN).
    """
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Vector dimension mismatch: {a.shape} vs {b.shape}")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def cosine_similarity_matrix(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between every row of `matrix` (n, d) and `vec` (d,).
    Vectorised, so ranking thousands of resumes is a single matrix product.
    Zero-norm rows get a similarity of 0.0.
    """
    m = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    v = np.asarray(vec, dtype=np.float64).ravel()
    if m.shape[1] != v.shape[0]:
        raise ValueError(f"Dimension mismatch: matrix has {m.shape[1]}, vector has {v.shape[0]}")
    norms = np.linalg.norm(m, axis=1) * np.linalg.norm(v)
    dots = m @ v
    sims = np.zeros(m.shape[0], dtype=np.float64)
    nonzero = norms > 0
    sims[nonzero] = dots[nonzero] / norms[nonzero]
    return np.clip(sims, -1.0, 1.0)


# --------------------------------------------------------------------------- #
# 2. Matching percentage
# --------------------------------------------------------------------------- #
def similarity_to_percentage(
    similarity: float,
    calibrate: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Convert a cosine similarity into a 0-100 matching percentage.

    Default: negative values clip to 0, so 0.62 -> 62.0.

    `calibrate=(low, high)` linearly stretches that range to 0-100. Sentence-BERT
    similarities for resume-vs-JD text rarely go above ~0.8 or below ~0.1, so
    e.g. calibrate=(0.10, 0.80) spreads scores over the full scale.
    """
    if calibrate is not None:
        low, high = calibrate
        if high <= low:
            raise ValueError("calibrate must be (low, high) with high > low")
        scaled = (similarity - low) / (high - low)
    else:
        scaled = similarity
    return float(np.clip(scaled, 0.0, 1.0) * 100.0)


def label_for_score(match_pct: float) -> str:
    for minimum, label in MATCH_LABELS:
        if match_pct >= minimum:
            return label
    return MATCH_LABELS[-1][1]


# --------------------------------------------------------------------------- #
# 4. Skill extraction + matching / missing skills
# --------------------------------------------------------------------------- #
def _compile_alias_patterns(taxonomy: Dict[str, List[str]]) -> Dict[str, "re.Pattern"]:
    """
    One regex per canonical skill. The custom boundaries (instead of \\b) make
    'c++', 'c#' and 'node.js' work and stop 'java' matching inside 'javascript'.
    """
    patterns = {}
    for canonical, aliases in taxonomy.items():
        # Only the listed aliases are searched (the canonical name is NOT added
        # automatically) so ambiguous names like "r" or "go" can be restricted
        # to unambiguous phrasings such as "r programming". ("excel" is allowed
        # bare; its verb uses are removed by _strip_verb_excel before matching.)
        names = set(aliases)
        # longest first so multi-word aliases win; the (?:...) group is essential -
        # without it the boundary guards would only bind to the first/last alternative
        alts = "|".join(re.escape(n.lower()) for n in sorted(names, key=len, reverse=True))
        patterns[canonical] = re.compile(rf"(?<![a-z0-9+#.])(?:{alts})(?![a-z0-9+#])")
    return patterns


_DEFAULT_PATTERNS = _compile_alias_patterns(SKILL_TAXONOMY)

# "Excel" the software vs. "excel" the verb ("I excel at teamwork", "to excel
# in sales", "excelled at..."). Verb uses are blanked out before skill matching,
# so a bare "Excel" ("SQL, Excel, Power BI") still counts as the skill.
_VERB_EXCEL = re.compile(
    r"\b(?:i|we|to|can|will|who|they|always|consistently|and|also)\s+excel(?:s|led|ling)?\b"
    r"|\bexcel(?:s|led|ling)\s"
    r"|\bexcel\s+(?:at|in|under|when|during|as)\b"
)


def _strip_verb_excel(lowered: str) -> str:
    return _VERB_EXCEL.sub(" ", lowered)


def extract_skills(text: str, taxonomy: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Return the sorted list of canonical skills found in `text`."""
    if not isinstance(text, str) or not text.strip():
        return []
    patterns = _DEFAULT_PATTERNS if taxonomy is None else _compile_alias_patterns(taxonomy)
    lowered = _strip_verb_excel(text.lower())
    return sorted(skill for skill, pat in patterns.items() if pat.search(lowered))


def compare_skills(
    resume_text: str,
    jd_text: str,
    taxonomy: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    """
    Compare skills asked for in the JD against those on the resume.

    Returns {"matching": [...], "missing": [...], "extra": [...]}
      matching - required by JD and present on resume
      missing  - required by JD, absent from resume
      extra    - on resume but not mentioned in JD
    """
    resume_skills = set(extract_skills(resume_text, taxonomy))
    jd_skills = set(extract_skills(jd_text, taxonomy))
    return {
        "matching": sorted(jd_skills & resume_skills),
        "missing": sorted(jd_skills - resume_skills),
        "extra": sorted(resume_skills - jd_skills),
    }


def _split_chunks(text: str, min_words: int = 2) -> List[str]:
    """Split raw resume text into sentence / bullet-sized chunks."""
    parts = re.split(r"[\n\r.;•·|]+", text)
    return [p.strip() for p in parts if len(p.split()) >= min_words]


def find_related_skills(
    missing_skills: Sequence[str],
    resume_text: str,
    embed_many: EmbedMany,
    skill_embeddings: Optional[Dict[str, np.ndarray]] = None,
    threshold: float = DEFAULT_SKILL_SIM_THRESHOLD,
) -> List[str]:
    """
    Among JD skills missing by keyword, find those the resume covers *in other
    words* - i.e. a skill phrase whose embedding is close to some sentence of
    the resume. This is the semantic edge over plain keyword search.
    """
    if not missing_skills:
        return []
    chunks = _split_chunks(resume_text)
    if not chunks:
        return []

    chunk_vecs = embed_many(chunks)
    skill_embeddings = skill_embeddings if skill_embeddings is not None else {}
    to_embed = [s for s in missing_skills if s not in skill_embeddings]
    if to_embed:
        for s, v in zip(to_embed, embed_many(to_embed)):
            skill_embeddings[s] = v

    related = []
    for skill in missing_skills:
        sims = cosine_similarity_matrix(chunk_vecs, skill_embeddings[skill])
        if sims.size and float(sims.max()) >= threshold:
            related.append(skill)
    return sorted(related)


# --------------------------------------------------------------------------- #
# 3. Ranking + shortlist
# --------------------------------------------------------------------------- #
Candidate = Union[str, Tuple[str, str], Dict[str, str]]


def _normalise_candidates(candidates: Iterable[Candidate]) -> List[Dict[str, str]]:
    """Accept strings, (name, text) tuples or {'id','name','text'} dicts."""
    out = []
    for i, c in enumerate(candidates):
        if isinstance(c, str):
            out.append({"id": f"C{i + 1}", "name": f"Candidate {i + 1}", "text": c})
        elif isinstance(c, tuple) and len(c) == 2:
            out.append({"id": f"C{i + 1}", "name": str(c[0]), "text": c[1]})
        elif isinstance(c, dict):
            cid = str(c.get("id", f"C{i + 1}"))
            out.append({"id": cid, "name": str(c.get("name", cid)), "text": c.get("text", "")})
        else:
            raise TypeError(f"Unsupported candidate format at index {i}: {type(c).__name__}")
    return out


def rank_from_vectors(
    candidates: Sequence[Candidate],
    resume_vectors: np.ndarray,
    jd_vector: np.ndarray,
    jd_text: str,
    *,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    skill_weight: float = DEFAULT_SKILL_WEIGHT,
    calibrate: Optional[Tuple[float, float]] = None,
    threshold: float = DEFAULT_SHORTLIST_THRESHOLD,
    semantic_skills: bool = False,
    embed_many: Optional[EmbedMany] = None,
    skill_sim_threshold: float = DEFAULT_SKILL_SIM_THRESHOLD,
    taxonomy: Optional[Dict[str, List[str]]] = None,
    valid_mask: Optional[Sequence[bool]] = None,
) -> List[MatchResult]:
    """
    Score and rank candidates whose embeddings are already computed.

    final match % = semantic_weight * semantic_pct + skill_weight * skill_coverage_pct
    (weights are renormalised to sum to 1; if the JD names no recognisable
    skills, the final score is the semantic score alone).
    """
    cands = _normalise_candidates(candidates)
    vectors = np.atleast_2d(np.asarray(resume_vectors, dtype=np.float64))
    if len(cands) != vectors.shape[0]:
        raise ValueError(f"{len(cands)} candidates but {vectors.shape[0]} vectors")
    if semantic_weight < 0 or skill_weight < 0 or (semantic_weight + skill_weight) == 0:
        raise ValueError("Weights must be non-negative and not both zero")
    if semantic_skills and embed_many is None:
        raise ValueError("semantic_skills=True requires embed_many")

    total_w = semantic_weight + skill_weight
    w_sem, w_skill = semantic_weight / total_w, skill_weight / total_w

    sims = cosine_similarity_matrix(vectors, jd_vector)
    jd_skills = extract_skills(jd_text, taxonomy)
    skill_cache: Dict[str, np.ndarray] = {}

    results: List[MatchResult] = []
    for i, cand in enumerate(cands):
        valid = True if valid_mask is None else bool(valid_mask[i])
        if not valid:
            results.append(MatchResult(
                candidate_id=cand["id"], name=cand["name"], similarity=0.0,
                semantic_pct=0.0, skill_coverage_pct=0.0, match_pct=0.0,
                label="Invalid (empty resume)", missing_skills=list(jd_skills), valid=False,
            ))
            continue

        sim = float(sims[i])
        semantic_pct = similarity_to_percentage(sim, calibrate)
        comp = compare_skills(cand["text"], jd_text, taxonomy)

        related: List[str] = []
        if semantic_skills and comp["missing"]:
            related = find_related_skills(
                comp["missing"], cand["text"], embed_many, skill_cache, skill_sim_threshold
            )

        if jd_skills:
            covered = len(comp["matching"]) + RELATED_SKILL_CREDIT * len(related)
            skill_pct = 100.0 * covered / len(jd_skills)
            match_pct = w_sem * semantic_pct + w_skill * skill_pct
        else:
            skill_pct = 0.0
            match_pct = semantic_pct

        results.append(MatchResult(
            candidate_id=cand["id"], name=cand["name"],
            similarity=round(sim, 4),
            semantic_pct=round(semantic_pct, 2),
            skill_coverage_pct=round(skill_pct, 2),
            match_pct=round(match_pct, 2),
            label=label_for_score(match_pct),
            matching_skills=comp["matching"],
            missing_skills=[s for s in comp["missing"] if s not in related],
            related_skills=related,
            extra_skills=comp["extra"],
        ))

    # Sort: invalid last, then best score; stable on ties (original order kept)
    order = sorted(range(len(results)),
                   key=lambda k: (not results[k].valid, -results[k].match_pct, -results[k].similarity))
    ranked = [results[k] for k in order]
    for pos, r in enumerate(ranked, start=1):
        r.rank = pos
        r.shortlisted = r.valid and r.match_pct >= threshold
    return ranked


def _default_embedders() -> Tuple[EmbedOne, EmbedMany]:
    """Lazy import so this module can be imported/tested without the SBERT model."""
    from nlp_embeddings import generate_embedding, generate_embeddings
    return generate_embedding, generate_embeddings


def rank_resumes(
    candidates: Sequence[Candidate],
    job_description: str,
    *,
    embed_one: Optional[EmbedOne] = None,
    embed_many: Optional[EmbedMany] = None,
    **kwargs,
) -> List[MatchResult]:
    """
    End-to-end helper: raw resume texts + JD text -> ranked MatchResult list.

    Uses Kavya's Sentence-BERT functions unless custom `embed_one` /
    `embed_many` callables are supplied. Resumes with empty text are not
    embedded (Kavya's module raises on empty input); they are returned at the
    bottom of the ranking flagged `valid=False` instead of crashing the batch.
    Extra keyword arguments are forwarded to `rank_from_vectors`.
    """
    if embed_one is None or embed_many is None:
        default_one, default_many = _default_embedders()
        embed_one = embed_one or default_one
        embed_many = embed_many or default_many

    cands = _normalise_candidates(candidates)
    if not cands:
        return []
    if not isinstance(job_description, str) or not job_description.strip():
        raise ValueError("Job description is empty.")

    valid_mask = [isinstance(c["text"], str) and bool(c["text"].strip()) for c in cands]
    jd_vector = embed_one(job_description)

    dim = np.asarray(jd_vector).shape[-1]
    vectors = np.zeros((len(cands), dim), dtype=np.float64)
    valid_idx = [i for i, ok in enumerate(valid_mask) if ok]
    if valid_idx:
        vectors[valid_idx] = np.asarray(embed_many([cands[i]["text"] for i in valid_idx]))

    return rank_from_vectors(
        cands, vectors, jd_vector, job_description,
        embed_many=embed_many, valid_mask=valid_mask, **kwargs,
    )


# --------------------------------------------------------------------------- #
# Shortlist + threshold tuning
# --------------------------------------------------------------------------- #
def shortlist(results: Sequence[MatchResult], threshold: float = DEFAULT_SHORTLIST_THRESHOLD,
              top_k: Optional[int] = None) -> List[MatchResult]:
    """Candidates with match % >= threshold, best first, optionally capped at top_k."""
    chosen = [r for r in results if r.valid and r.match_pct >= threshold]
    chosen.sort(key=lambda r: r.rank)
    return chosen[:top_k] if top_k is not None else chosen


def threshold_sweep(results: Sequence[MatchResult],
                    thresholds: Optional[Iterable[float]] = None) -> List[Tuple[float, int]]:
    """
    How many candidates survive at each threshold - the data behind
    "threshold tuning". Returns [(threshold, count), ...].
    """
    if thresholds is None:
        thresholds = range(0, 101, 5)
    valid_scores = [r.match_pct for r in results if r.valid]
    return [(float(t), sum(s >= t for s in valid_scores)) for t in thresholds]


def suggest_threshold(results: Sequence[MatchResult], target_count: int) -> float:
    """
    Threshold that yields (about) `target_count` shortlisted candidates:
    the score of the target_count-th best valid candidate. Ties at that score
    may return slightly more than target_count.
    """
    if target_count < 1:
        raise ValueError("target_count must be >= 1")
    scores = sorted((r.match_pct for r in results if r.valid), reverse=True)
    if not scores:
        return 0.0
    return scores[min(target_count, len(scores)) - 1]


def results_to_dataframe(results: Sequence[MatchResult]):
    """Return results as a pandas DataFrame (handy for Streamlit tables / CSV export)."""
    import pandas as pd
    rows = []
    for r in results:
        d = r.to_dict()
        for key in ("matching_skills", "missing_skills", "related_skills", "extra_skills"):
            d[key] = ", ".join(d[key])
        rows.append(d)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Demo (needs Kavya's nlp_embeddings + sentence-transformers installed)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    jd = """
    We are hiring a Machine Learning Engineer. Requirements: Python, NLP,
    deep learning with PyTorch, SQL, and experience deploying models with Docker.
    """
    resumes = [
        ("Aarav", "Python developer with 3 years in natural language processing and machine learning. "
                  "Built BERT classifiers in PyTorch. Strong SQL. Deployed with Docker on AWS."),
        ("Meera", "Frontend developer skilled in HTML, CSS, JavaScript and React."),
        ("Rohan", "Data analyst experienced in Python, SQL, Excel dashboards and statistics."),
        ("Sneha", "Trained neural networks to analyse text and extract meaning from documents. "
                  "Comfortable with Python and Linux."),
    ]

    ranked = rank_resumes(resumes, jd, threshold=45)

    print(f"{'Rank':<5}{'Name':<8}{'Match%':>8}{'Cosine':>9}  {'Label':<14} Shortlist")
    for r in ranked:
        print(f"{r.rank:<5}{r.name:<8}{r.match_pct:>8.1f}{r.similarity:>9.3f}  {r.label:<14} "
              f"{'YES' if r.shortlisted else '-'}")
        print(f"      matching: {r.matching_skills}")
        print(f"      missing : {r.missing_skills}")
    print("\nThreshold sweep:", threshold_sweep(ranked, [30, 40, 50, 60, 70]))
