# ResumeMatch — Matching & Ranking Module

## Overview
Takes the Sentence-BERT embeddings produced by Kavya's NLP module and turns them
into a ranked, explainable shortlist: similarity score, match percentage,
matching skills and missing skills for every candidate.

## Files
| File | Purpose |
|---|---|
| `matching_ranking.py` | Cosine similarity, match %, ranking, shortlist, threshold tuning, skill gap |
| `skills_taxonomy.py` | Skill -> aliases dictionary (edit this to add skills) |
| `test_matching.py` | 28 tests (`python test_matching.py`) |

Place these next to `nlp_embeddings.py` (same folder).

## Setup
```bash
pip install -r requirements.txt
pip install sentence-transformers      # for Kavya's module
```

## Quick start
```python
from matching_ranking import rank_resumes, shortlist, threshold_sweep

candidates = [
    ("Aarav", "Python developer, NLP and machine learning ..."),
    ("Meera", "Frontend developer, HTML, CSS, React ..."),
]
ranked = rank_resumes(candidates, job_description, threshold=50)

for r in ranked:
    print(r.rank, r.name, r.match_pct, r.label)
    print("  matching:", r.matching_skills)
    print("  missing :", r.missing_skills)
```
Candidates can be plain strings, `(name, text)` tuples, or
`{"id": ..., "name": ..., "text": ...}` dicts.

## How the score works
1. **Cosine similarity** between resume and JD vectors (`cosine_similarity`,
   or the vectorised `cosine_similarity_matrix` for many resumes at once).
2. **Semantic %** = similarity clipped to 0–100.
   Optional `calibrate=(0.10, 0.80)` stretches SBERT's typical range to 0–100.
3. **Skill coverage %** = JD skills found on the resume / JD skills.
   Aliases are handled, so "natural language processing" matches "NLP".
4. **Final match %** = `0.7 × semantic % + 0.3 × skill coverage %`
   (change with `semantic_weight=` / `skill_weight=`). If the JD names no
   recognised skill, the final score is the semantic score alone.

| Match % | Label |
|---|---|
| ≥ 75 | Strong match |
| ≥ 60 | Good match |
| ≥ 45 | Partial match |
| < 45 | Weak match |

## Skills output (per candidate)
- `matching_skills` – required by JD and on the resume
- `missing_skills` – required by JD, not on the resume
- `extra_skills` – on the resume, not asked for
- `related_skills` – *(opt-in)* missing by keyword but a resume sentence is
  semantically close. Enable with `semantic_skills=True`; counts for half credit.
  Tune with `skill_sim_threshold` (default 0.50).

## Shortlist & threshold tuning
```python
shortlist(ranked, threshold=60, top_k=10)     # filtered list
threshold_sweep(ranked, [40, 50, 60, 70])     # [(40, 12), (50, 7), ...] candidates kept
suggest_threshold(ranked, target_count=5)     # threshold that keeps ~5 people
```
`threshold_sweep` is the data for a "threshold slider" in the Streamlit UI.

## Streamlit hand-off
```python
from matching_ranking import results_to_dataframe
st.dataframe(results_to_dataframe(ranked))
```
Each `MatchResult` also has `.to_dict()`.

## Edge cases handled
- Empty / blank resume: Kavya's `preprocess_text` raises on empty text, so these
  are skipped and returned at the bottom flagged `valid=False` (batch never crashes).
- Zero vectors, dimension mismatches, empty candidate list, empty JD.
- No false skill hits such as "java" inside "javascript", or "I excel at…" as Excel (a bare "Excel" is still detected as the skill).

## Notes / limitations
- Default thresholds (shortlist 50, label cut-offs, 0.7/0.3 weights, related-skill
  0.50) are sensible starting points, **not** tuned on real data. Tune them on the
  Kaggle resumes with a few JDs the team has labelled by hand.
- Skill detection is dictionary-based: skills absent from `skills_taxonomy.py`
  aren't detected. Add entries as needed.
