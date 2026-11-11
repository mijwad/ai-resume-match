#!/usr/bin/env python
"""
Demo script: Full ResumeMatch pipeline test
Run this to verify all three modules integrate correctly.
"""

import pandas as pd
import numpy as np

# Import all three modules
from nlp_embeddings import clean_text, generate_embedding, generate_embeddings
from matching_ranking import (
    rank_resumes,
    shortlist,
    threshold_sweep,
    suggest_threshold,
    results_to_dataframe,
    extract_skills,
    compare_skills,
    cosine_similarity,
    similarity_to_percentage,
)
from skills_taxonomy import SKILL_TAXONOMY

print("=" * 60)
print("ResumeMatch — Full Pipeline Demo")
print("=" * 60)

# 1. Load pre-extracted resumes (Devananda's module output)
print("\n1. Loading pre-extracted resumes...")
df = pd.read_csv("ResumeMatch/outputs/extracted_resumes.csv")
df = df[df["status"] == "Success"]
print(f"   Loaded {len(df)} successful resume extractions")

# Show category distribution
if "category" in df.columns:
    print(f"   Categories: {df['category'].nunique()}")
    print(f"   Top categories:\n{df['category'].value_counts().head(10)}")

# 2. Take a sample of resumes from different categories for demo
# Pick a few diverse categories
sample_categories = ["INFORMATION-TECHNOLOGY", "DATA-SCIENCE", "ENGINEERING", "FINANCE", "HEALTHCARE"]
available_cats = df["category"].unique()
demo_cats = [c for c in sample_categories if c in available_cats]
if not demo_cats:
    demo_cats = list(available_cats[:5])

print(f"\n2. Selecting demo resumes from categories: {demo_cats}")
demo_resumes = []
for cat in demo_cats:
    cat_df = df[df["category"] == cat].head(2)
    for _, row in cat_df.iterrows():
        demo_resumes.append({
            "id": row["filename"],
            "name": f"{cat}_{row['filename'][:20]}",
            "text": row["resume_text"]
        })

print(f"   Selected {len(demo_resumes)} demo resumes")

# 3. Define a job description (Kavya's NLP will process this)
jd_text = """We are hiring a Machine Learning Engineer.
Requirements: Python, NLP, deep learning with PyTorch, SQL, and experience deploying models with Docker.
Strong background in machine learning algorithms, neural networks, and natural language processing.
Experience with transformers, BERT, and sentence embeddings is a plus.
Familiarity with MLOps, model deployment, and cloud platforms (AWS/GCP/Azure) preferred."""

print("\n3. Job Description:")
print(f"   {jd_text[:100]}...")

# 4. Test NLP Module (Kavya's) - Generate embeddings
print("\n4. Testing NLP Module (Kavya) - Generating embeddings...")
jd_vector = generate_embedding(jd_text)
print(f"   JD vector shape: {jd_vector.shape}")

resume_texts = [r["text"] for r in demo_resumes]
resume_vectors = generate_embeddings(resume_texts)
print(f"   Resume vectors shape: {resume_vectors.shape}")

# 5. Test Matching & Ranking Module (Ridhima's)
print("\n5. Testing Matching & Ranking Module (Ridhima)...")
results = rank_resumes(
    demo_resumes,
    jd_text,
    semantic_weight=0.7,
    skill_weight=0.3,
    threshold=50,
    semantic_skills=False,
)

print(f"   Ranked {len(results)} candidates")

# 6. Display results
print("\n6. Results:")
print("-" * 80)
print(f"{'Rank':<5}{'Name':<30}{'Match%':>8}{'Semantic%':>10}{'Skill%':>8}{'Label':<15}{'Shortlist'}")
print("-" * 80)
for r in results:
    shortlist_str = "YES" if r.shortlisted else "NO"
    print(f"{r.rank:<5}{r.name[:29]:<30}{r.match_pct:>8.1f}{r.semantic_pct:>10.1f}{r.skill_coverage_pct:>8.1f}{r.label:<15}{shortlist_str}")

# 7. Show skill analysis for top candidate
top = results[0]
print(f"\n7. Top Candidate Skill Analysis: {top.name}")
print(f"   Matching skills: {top.matching_skills}")
print(f"   Missing skills: {top.missing_skills}")
print(f"   Extra skills: {top.extra_skills}")

# 8. Test threshold sweep
print("\n8. Threshold Sweep:")
sweep = threshold_sweep(results)
for threshold, count in sweep:
    print(f"   Threshold {threshold:>3}% -> {count} candidates shortlisted")

# 9. Suggest threshold for top 3
suggested = suggest_threshold(results, target_count=3)
print(f"\n9. Suggested threshold for top 3: {suggested:.1f}%")

# 10. Test skill extraction (standalone)
print("\n10. Skill Extraction Test:")
test_text = "Python developer with NLP, machine learning, SQL, Docker, and React experience."
skills = extract_skills(test_text)
print(f"   Text: {test_text}")
print(f"   Extracted: {skills}")

# Test skill comparison
comp = compare_skills(
    "Python, NLP, machine learning, SQL, Docker",
    "Need Python, SQL, Docker, and Kubernetes"
)
print(f"\n   Skill Comparison:")
print(f"     Matching: {comp['matching']}")
print(f"     Missing:  {comp['missing']}")
print(f"     Extra:    {comp['extra']}")

# 11. Test cosine similarity
print("\n11. Cosine Similarity Test:")
sim = cosine_similarity(jd_vector, resume_vectors[0])
pct = similarity_to_percentage(sim)
print(f"   JD vs Top Resume: cosine={sim:.4f}, percentage={pct:.1f}%")

# 12. Export results to DataFrame
print("\n12. DataFrame Export:")
df_results = results_to_dataframe(results)
print(f"   DataFrame shape: {df_results.shape}")
print(f"   Columns: {list(df_results.columns)}")

print("\n" + "=" * 60)
print("SUCCESS: All modules integrated successfully!")
print("=" * 60)
print("\nTo run the Streamlit app:")
print("  streamlit run app.py")