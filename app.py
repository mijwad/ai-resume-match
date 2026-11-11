"""
ResumeMatch — Streamlit Frontend & Integration
===============================================
Developed by: [Your Name] — Frontend & Integration

Integrates three modules:
1. Data Extraction (Devananda) — PDF text extraction via PyMuPDF
2. NLP & Embeddings (Kavya) — Sentence-BERT embeddings
3. Matching & Ranking (Ridhima) — Cosine similarity, skill matching, ranked shortlist

Run: streamlit run app.py
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Path to pre-extracted resumes (Devananda's output)
PREEXTRACTED_CSV = Path("ResumeMatch/outputs/extracted_resumes.csv")

# ─── Local module imports ──────────────────────────────────────────────
# These are the modules developed by teammates
try:
    from matching_ranking import (
        rank_resumes,
        shortlist,
        threshold_sweep,
        suggest_threshold,
        results_to_dataframe,
        MatchResult,
        cosine_similarity,
        similarity_to_percentage,
        extract_skills,
        compare_skills,
        DEFAULT_SHORTLIST_THRESHOLD,
        DEFAULT_SEMANTIC_WEIGHT,
        DEFAULT_SKILL_WEIGHT,
    )
    from nlp_embeddings import clean_text, generate_embedding, generate_embeddings
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    IMPORT_ERROR = str(e)

# ─── Sample Job Descriptions ────────────────────────────────────────────
SAMPLE_JDS = {
    "Machine Learning Engineer": """We are hiring a Machine Learning Engineer.
Requirements: Python, NLP, deep learning with PyTorch, SQL, and experience deploying models with Docker.
Strong background in machine learning algorithms, neural networks, and natural language processing.
Experience with transformers, BERT, and sentence embeddings is a plus.
Familiarity with MLOps, model deployment, and cloud platforms (AWS/GCP/Azure) preferred.""",

    "Data Scientist": """Looking for a Data Scientist to join our analytics team.
Requirements: Python, SQL, statistics, machine learning, data visualization.
Experience with pandas, numpy, scikit-learn, matplotlib, seaborn, plotly.
Strong analytical thinking and problem-solving skills.
Experience with A/B testing, experimental design, and statistical modeling.
Knowledge of Tableau or Power BI for dashboarding.""",

    "Full Stack Developer": """Seeking a Full Stack Developer for web application development.
Requirements: JavaScript, TypeScript, React, Node.js, REST API, SQL, Docker.
Experience with frontend frameworks (React, Vue, or Angular) and backend (Node.js, Python, or Go).
Database design with PostgreSQL or MongoDB.
CI/CD pipelines, Git, and cloud deployment (AWS/Azure/GCP).
Agile/Scrum experience preferred.""",

    "DevOps Engineer": """DevOps Engineer needed for infrastructure automation.
Requirements: AWS, Docker, Kubernetes, CI/CD, Linux, Terraform, Python/Bash.
Experience with container orchestration, infrastructure as code.
Monitoring and logging (Prometheus, Grafana, ELK stack).
Security best practices, networking, and cloud architecture.
Strong scripting skills in Python or Bash.""",

    "NLP Research Engineer": """NLP Research Engineer position for cutting-edge language models.
Requirements: Python, PyTorch, transformers, Hugging Face, sentence embeddings.
Research experience in NLP, LLMs, BERT, GPT, or similar architectures.
Publications in top NLP conferences (ACL, EMNLP, NAACL) a plus.
Experience with large-scale training, fine-tuning, and evaluation.
Strong software engineering practices.""",
}

# ─── Custom CSS for a polished look ────────────────────────────────────
st.markdown(
    """
<style>
    /* Global */
    :root {
        --primary: #2563eb;
        --primary-hover: #1d4ed8;
        --success: #16a34a;
        --warning: #eab308;
        --danger: #dc2626;
        --bg: #f8fafc;
        --card-bg: #ffffff;
        --text: #1e293b;
        --text-muted: #64748b;
        --border: #e2e8f0;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --primary: #3b82f6;
            --primary-hover: #60a5fa;
            --success: #22c55e;
            --warning: #facc15;
            --danger: #ef4444;
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #334155;
        }
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 2rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    .app-header h1 {
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--text);
        margin: 0 0 0.5rem;
    }
    .app-header p {
        color: var(--text-muted);
        font-size: 1.1rem;
        margin: 0;
    }

    /* Cards */
    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .metric-card:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }

    /* Status badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-strong { background: #dcfce7; color: #166534; }
    .badge-good { background: #dbeafe; color: #1e40af; }
    .badge-partial { background: #fef9c3; color: #854d0e; }
    .badge-weak { background: #fee2e2; color: #991b1b; }
    .badge-invalid { background: #f3f4f6; color: #6b7280; }
    @media (prefers-color-scheme: dark) {
        .badge-strong { background: #14532d; color: #bbf7d0; }
        .badge-good { background: #1e3a5f; color: #93c5fd; }
        .badge-partial { background: #713f12; color: #fde047; }
        .badge-weak { background: #7f1d1d; color: #fca5a5; }
        .badge-invalid { background: #374151; color: #9ca3af; }
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }
    .section-header h2 {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text);
        margin: 0;
    }

    /* Upload zone */
    .upload-zone {
        border: 2px dashed var(--border);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: var(--card-bg);
        transition: border-color 0.2s, background 0.2s;
    }
    .upload-zone:hover, .upload-zone.drag-active {
        border-color: var(--primary);
        background: rgba(37, 99, 235, 0.05);
    }

    /* Skill tags */
    .skill-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        margin: 0.15rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .skill-match { background: #dcfce7; color: #166534; }
    .skill-missing { background: #fee2e2; color: #991b1b; }
    .skill-related { background: #fef9c3; color: #854d0e; }
    .skill-extra { background: #e0e7ff; color: #3730a3; }
    @media (prefers-color-scheme: dark) {
        .skill-match { background: #14532d; color: #bbf7d0; }
        .skill-missing { background: #7f1d1d; color: #fca5a5; }
        .skill-related { background: #713f12; color: #fde047; }
        .skill-extra { background: #312e81; color: #c7d2fe; }
    }

    /* DataFrame styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Sidebar */
    .sidebar-section {
        padding: 1rem 0;
        border-bottom: 1px solid var(--border);
    }
    .sidebar-section:last-child {
        border-bottom: none;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem 0;
        color: var(--text-muted);
        font-size: 0.875rem;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Utility functions ─────────────────────────────────────────────────


def extract_text_from_pdf(pdf_file: UploadedFile) -> str:
    """Extract text from an uploaded PDF file using PyMuPDF."""
    try:
        import fitz

        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        return f"[Error extracting text: {e}]"


def extract_text_from_txt(pdf_file: UploadedFile) -> str:
    """Extract text from an uploaded text file."""
    try:
        return pdf_file.read().decode("utf-8").strip()
    except Exception as e:
        return f"[Error reading text file: {e}]"


def process_uploaded_files(files: List[UploadedFile]) -> pd.DataFrame:
    """Process multiple uploaded resume files and return a DataFrame."""
    records = []
    for f in files:
        name = f.name
        if name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(f)
        elif name.lower().endswith((".txt", ".md")):
            text = extract_text_from_txt(f)
        else:
            text = f"[Unsupported file type: {name}]"

        status = "Success" if text and not text.startswith("[Error") else "Failed"
        records.append(
            {
                "filename": name,
                "resume_text": text,
                "status": status,
            }
        )
    return pd.DataFrame(records)


def load_preextracted_resumes(max_resumes: int = 50, category: Optional[str] = None) -> pd.DataFrame:
    """Load resumes from Devananda's pre-extracted CSV."""
    if not PREEXTRACTED_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PREEXTRACTED_CSV)
        # Filter successful extractions
        df = df[df["status"] == "Success"].copy()
        if category and "category" in df.columns:
            df = df[df["category"] == category]
        # Limit for performance
        df = df.head(max_resumes)
        # Rename to match our expected format
        df = df.rename(columns={"filename": "filename", "resume_text": "resume_text"})
        df["status"] = "Success"
        return df[["filename", "resume_text", "status"]]
    except Exception as e:
        st.error(f"Error loading pre-extracted resumes: {e}")
        return pd.DataFrame()


def render_badge(label: str) -> str:
    """Render a styled badge for match labels."""
    label_lower = label.lower().replace(" ", "-")
    css_class = {
        "strong-match": "badge-strong",
        "good-match": "badge-good",
        "partial-match": "badge-partial",
        "weak-match": "badge-weak",
        "invalid": "badge-invalid",
    }.get(label_lower, "badge-invalid")
    return f'<span class="badge {css_class}">{label}</span>'


def render_skill_tags(skills: List[str], tag_type: str) -> str:
    """Render skills as colored tags."""
    if not skills:
        return '<span style="color: var(--text-muted); font-size: 0.875rem;">None</span>'
    css_map = {
        "matching": "skill-match",
        "missing": "skill-missing",
        "related": "skill-related",
        "extra": "skill-extra",
    }
    css_class = css_map.get(tag_type, "skill-match")
    tags = "".join(f'<span class="skill-tag {css_class}">{s}</span>' for s in skills)
    return tags


# ─── Main App ──────────────────────────────────────────────────────────

def main():
    # ─── Header ────────────────────────────────────────────────────────
    st.markdown(
        """
    <div class="app-header">
        <h1>📄 ResumeMatch</h1>
        <p>Semantic Resume-to-Job Description Matcher — Rank candidates by meaning, not just keywords</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ─── Module availability check ─────────────────────────────────────
    if not MODULES_AVAILABLE:
        st.error(
            f"""
        **Backend modules not found.** Please ensure the following files are in the same directory as `app.py`:
        - `matching_ranking.py` (Ridhima's module)
        - `nlp_embeddings.py` (Kavya's module)
        - `skills_taxonomy.py` (shared taxonomy)

        Error: `{IMPORT_ERROR}`

        You can install dependencies with:
        ```bash
        pip install -r requirements.txt
        pip install sentence-transformers streamlit plotly
        ```
        """
        )
        return

    # ─── Session state initialization ──────────────────────────────────
    if "results" not in st.session_state:
        st.session_state.results: List[MatchResult] = []
    if "jd_text" not in st.session_state:
        st.session_state.jd_text = ""
    if "resume_df" not in st.session_state:
        st.session_state.resume_df = pd.DataFrame()
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # ─── Sidebar: Configuration ────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-section"><h3>⚙️ Scoring Weights</h3></div>',
            unsafe_allow_html=True,
        )
        semantic_weight = st.slider(
            "Semantic Similarity Weight",
            0.0,
            1.0,
            DEFAULT_SEMANTIC_WEIGHT,
            0.05,
            help="Weight for cosine similarity between resume and JD embeddings",
        )
        skill_weight = st.slider(
            "Skill Coverage Weight",
            0.0,
            1.0,
            DEFAULT_SKILL_WEIGHT,
            0.05,
            help="Weight for keyword-based skill matching",
        )
        # Auto-normalize
        total = semantic_weight + skill_weight
        if total > 0:
            w_sem = semantic_weight / total
            w_skill = skill_weight / total
        else:
            w_sem, w_skill = 0.7, 0.3
        st.caption(f"Normalized: Semantic {w_sem:.0%} · Skills {w_skill:.0%}")

        st.markdown(
            '<div class="sidebar-section"><h3>🎯 Threshold Settings</h3></div>',
            unsafe_allow_html=True,
        )
        shortlist_threshold = st.slider(
            "Shortlist Threshold (%)",
            0,
            100,
            int(DEFAULT_SHORTLIST_THRESHOLD),
            5,
            help="Candidates above this match % are shortlisted",
        )

        st.markdown(
            '<div class="sidebar-section"><h3>🔍 Advanced Options</h3></div>',
            unsafe_allow_html=True,
        )
        semantic_skills = st.checkbox(
            "Enable Semantic Skill Detection",
            value=False,
            help="Detect skills mentioned with different phrasing (e.g., 'NLP' in JD matches 'natural language processing' in resume). Uses embeddings — slower but more thorough.",
        )
        calibrate_enabled = st.checkbox(
            "Calibrate Similarity Scores",
            value=False,
            help="Stretch SBERT's typical range (0.1–0.8) to 0–100 for wider score spread",
        )
        calibrate_range = None
        if calibrate_enabled:
            c_low = st.number_input("Calibrate Low", 0.0, 1.0, 0.10, 0.01)
            c_high = st.number_input("Calibrate High", 0.0, 1.0, 0.80, 0.01)
            if c_high > c_low:
                calibrate_range = (c_low, c_high)

        st.markdown(
            '<div class="sidebar-section"><h3>📊 Visualization Options</h3></div>',
            unsafe_allow_html=True,
        )
        show_skills_detail = st.checkbox("Show Skill Breakdown", value=True)
        show_threshold_chart = st.checkbox("Show Threshold Sweep Chart", value=True)
        show_similarity_dist = st.checkbox("Show Similarity Distribution", value=True)

        st.markdown("---")
        st.caption("Built by the ResumeMatch team • CSE-472 Project 11")

    # ─── Main Content: Input Section ───────────────────────────────────
    st.markdown(
        '<div class="section-header"><h2>📥 Input</h2></div>', unsafe_allow_html=True
    )

    col_jd, col_resumes = st.columns([1, 1], gap="large")

    with col_jd:
        st.markdown("#### Job Description")

        # Sample JD selector
        sample_jd = st.selectbox(
            "📋 Quick start with a sample JD",
            ["Custom..."] + list(SAMPLE_JDS.keys()),
            key="sample_jd_selector",
        )
        if sample_jd != "Custom...":
            st.session_state.jd_text = SAMPLE_JDS[sample_jd]
            jd_text = SAMPLE_JDS[sample_jd]
        else:
            jd_text = st.session_state.jd_text

        jd_text = st.text_area(
            "Paste the job description here",
            value=jd_text,
            height=300,
            placeholder="We are hiring a Machine Learning Engineer. Requirements: Python, NLP, deep learning with PyTorch, SQL, and experience deploying models with Docker...",
            key="jd_input",
        )
        st.session_state.jd_text = jd_text

        # JD stats
        if jd_text.strip():
            jd_skills = extract_skills(jd_text)
            st.caption(f"📝 {len(jd_text.split())} words · 🔍 {len(jd_skills)} recognized skills")
            if jd_skills:
                with st.expander("Recognized skills in JD"):
                    st.markdown(render_skill_tags(jd_skills, "matching"), unsafe_allow_html=True)

    with col_resumes:
        st.markdown("#### Resumes")

        # Tabs for different resume sources
        resume_tab1, resume_tab2 = st.tabs(["📤 Upload Files", "📂 Load from Dataset"])

        with resume_tab1:
            st.markdown(
                """
            <div class="upload-zone" id="upload-zone">
                <p style="margin: 0; color: var(--text-muted);">Drag & drop PDF or TXT files here</p>
                <p style="margin: 0.5rem 0 0; font-size: 0.875rem; color: var(--text-muted);">Or click to browse — multiple files supported</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            uploaded_files = st.file_uploader(
                "Upload resume files",
                type=["pdf", "txt", "md"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="resume_uploader",
            )

            if uploaded_files:
                with st.spinner(f"Extracting text from {len(uploaded_files)} resume(s)..."):
                    resume_df = process_uploaded_files(uploaded_files)
                    st.session_state.resume_df = resume_df

                success_count = (resume_df["status"] == "Success").sum()
                st.success(f"✅ Processed {len(resume_df)} files — {success_count} successful, {len(resume_df) - success_count} failed")

                with st.expander("Preview extracted resumes"):
                    for _, row in resume_df.iterrows():
                        status_icon = "✅" if row["status"] == "Success" else "❌"
                        preview = row["resume_text"][:200] + ("..." if len(row["resume_text"]) > 200 else "")
                        st.markdown(f"**{status_icon} {row['filename']}**")
                        st.code(preview, language=None)

        with resume_tab2:
            if PREEXTRACTED_CSV.exists():
                st.info(f"📊 Pre-extracted dataset found: {PREEXTRACTED_CSV}")

                # Load categories
                try:
                    full_df = pd.read_csv(PREEXTRACTED_CSV)
                    categories = sorted(full_df[full_df["status"] == "Success"]["category"].unique()) if "category" in full_df.columns else []
                except:
                    categories = []

                col_cat, col_num = st.columns(2)
                with col_cat:
                    selected_category = st.selectbox(
                        "Filter by category (optional)",
                        ["All"] + categories,
                        key="category_filter",
                    )
                with col_num:
                    max_resumes = st.number_input(
                        "Max resumes to load",
                        min_value=1,
                        max_value=200,
                        value=30,
                        step=5,
                        key="max_resumes_load",
                    )

                if st.button("📥 Load Resumes from Dataset", type="primary", use_container_width=True):
                    cat = None if selected_category == "All" else selected_category
                    with st.spinner(f"Loading resumes from dataset..."):
                        resume_df = load_preextracted_resumes(max_resumes=max_resumes, category=cat)
                        st.session_state.resume_df = resume_df

                    if not resume_df.empty:
                        st.success(f"✅ Loaded {len(resume_df)} resumes from dataset!")
                        with st.expander("Preview loaded resumes"):
                            for _, row in resume_df.head(10).iterrows():
                                preview = row["resume_text"][:200] + ("..." if len(row["resume_text"]) > 200 else "")
                                st.markdown(f"**{row['filename']}**")
                                st.code(preview, language=None)
                            if len(resume_df) > 10:
                                st.caption(f"... and {len(resume_df) - 10} more")
                    else:
                        st.warning("No resumes loaded. Check the dataset.")
            else:
                st.warning(f"Pre-extracted dataset not found at `{PREEXTRACTED_CSV}`")
                st.caption("Run `python ResumeMatch/resume_extractor.py` to generate it.")

    # ─── Process Button ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    can_process = (
        bool(st.session_state.jd_text.strip())
        and not st.session_state.resume_df.empty
        and (st.session_state.resume_df["status"] == "Success").any()
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        process_clicked = st.button(
            "🚀 Rank Candidates",
            type="primary",
            use_container_width=True,
            disabled=not can_process or st.session_state.processing,
        )

    if process_clicked:
        st.session_state.processing = True
        valid_resumes = st.session_state.resume_df[st.session_state.resume_df["status"] == "Success"]

        candidates = [
            {"id": f"C{i+1}", "name": row["filename"], "text": row["resume_text"]}
            for i, (_, row) in enumerate(valid_resumes.iterrows())
        ]

        with st.spinner("Generating embeddings and ranking candidates..."):
            try:
                results = rank_resumes(
                    candidates,
                    st.session_state.jd_text,
                    semantic_weight=w_sem,
                    skill_weight=w_skill,
                    threshold=shortlist_threshold,
                    semantic_skills=semantic_skills,
                    calibrate=calibrate_range,
                )
                st.session_state.results = results
                st.success(f"✅ Ranked {len(results)} candidates!")
            except Exception as e:
                st.error(f"Error during ranking: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                st.session_state.processing = False

    # ─── Results Section ──────────────────────────────────────────────
    if st.session_state.results:
        st.markdown(
            '<div class="section-header"><h2>📊 Results</h2></div>', unsafe_allow_html=True
        )

        results = st.session_state.results
        valid_results = [r for r in results if r.valid]
        shortlisted = [r for r in valid_results if r.shortlisted]

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{len(results)}</div>'
                f'<div class="metric-label">Total Candidates</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{len(valid_results)}</div>'
                f'<div class="metric-label">Valid Resumes</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{len(shortlisted)}</div>'
                f'<div class="metric-label">Shortlisted</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            avg_match = np.mean([r.match_pct for r in valid_results]) if valid_results else 0
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{avg_match:.1f}%</div>'
                f'<div class="metric-label">Avg Match Score</div></div>',
                unsafe_allow_html=True,
            )

        # ─── Results Table ─────────────────────────────────────────────
        st.markdown("### 📋 Ranked Candidates")

        df_results = results_to_dataframe(results)

        # Add badge column
        df_display = df_results.copy()
        df_display["Match"] = df_display.apply(
            lambda row: f"{row['match_pct']:.1f}% {render_badge(row['label'])}",
            axis=1,
        )
        df_display["Shortlisted"] = df_display["shortlisted"].apply(
            lambda x: "✅" if x else "—"
        )

        # Select columns to display
        display_cols = [
            "rank",
            "name",
            "Match",
            "semantic_pct",
            "skill_coverage_pct",
            "Shortlisted",
        ]
        if show_skills_detail:
            display_cols.extend(["matching_skills", "missing_skills", "related_skills", "extra_skills"])

        st.dataframe(
            df_display[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "name": st.column_config.TextColumn("Candidate", width="medium"),
                "Match": st.column_config.TextColumn("Match %", width="medium"),
                "semantic_pct": st.column_config.NumberColumn("Semantic %", format="%.1f%%", width="small"),
                "skill_coverage_pct": st.column_config.NumberColumn("Skills %", format="%.1f%%", width="small"),
                "Shortlisted": st.column_config.TextColumn("Shortlist", width="small"),
                "matching_skills": st.column_config.TextColumn("Matching Skills", width="large"),
                "missing_skills": st.column_config.TextColumn("Missing Skills", width="large"),
                "related_skills": st.column_config.TextColumn("Related Skills", width="large"),
                "extra_skills": st.column_config.TextColumn("Extra Skills", width="large"),
            },
        )

        # ─── Detailed Candidate View ───────────────────────────────────
        if show_skills_detail and valid_results:
            st.markdown("### 🔍 Candidate Details")
            selected_name = st.selectbox(
                "Select a candidate to view skill breakdown",
                options=[r.name for r in valid_results],
                index=0,
            )
            selected = next(r for r in valid_results if r.name == selected_name)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Matching Skills** ✅")
                st.markdown(render_skill_tags(selected.matching_skills, "matching"), unsafe_allow_html=True)

                st.markdown("**Related Skills (semantic match)** ⚡")
                st.markdown(render_skill_tags(selected.related_skills, "related"), unsafe_allow_html=True)

            with col2:
                st.markdown("**Missing Skills** ❌")
                st.markdown(render_skill_tags(selected.missing_skills, "missing"), unsafe_allow_html=True)

                st.markdown("**Extra Skills (not in JD)** ➕")
                st.markdown(render_skill_tags(selected.extra_skills, "extra"), unsafe_allow_html=True)

            # Score breakdown
            st.markdown("**Score Breakdown**")
            score_cols = st.columns(3)
            with score_cols[0]:
                st.metric("Semantic Similarity", f"{selected.semantic_pct:.1f}%", f"cosine: {selected.similarity:.3f}")
            with score_cols[1]:
                st.metric("Skill Coverage", f"{selected.skill_coverage_pct:.1f}%")
            with score_cols[2]:
                st.metric("Final Match", f"{selected.match_pct:.1f}%", selected.label)

        # ─── Visualizations ────────────────────────────────────────────
        if valid_results:
            st.markdown(
                '<div class="section-header"><h2>📈 Visualizations</h2></div>',
                unsafe_allow_html=True,
            )

            # Threshold sweep chart
            if show_threshold_chart:
                st.markdown("#### Threshold Sweep")
                sweep_data = threshold_sweep(results)
                sweep_df = pd.DataFrame(sweep_data, columns=["Threshold", "Candidates Shortlisted"])

                fig_threshold = px.line(
                    sweep_df,
                    x="Threshold",
                    y="Candidates Shortlisted",
                    markers=True,
                    title="Candidates Shortlisted vs. Threshold",
                    labels={"Threshold": "Match % Threshold", "Candidates Shortlisted": "Count"},
                )
                fig_threshold.add_vline(
                    x=shortlist_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Current: {shortlist_threshold}%",
                )
                fig_threshold.update_layout(
                    template="plotly_white",
                    height=350,
                    margin=dict(l=40, r=40, t=50, b=40),
                )
                st.plotly_chart(fig_threshold, use_container_width=True)

            # Similarity distribution
            if show_similarity_dist:
                st.markdown("#### Score Distribution")
                score_df = pd.DataFrame(
                    {
                        "Candidate": [r.name for r in valid_results],
                        "Match %": [r.match_pct for r in valid_results],
                        "Semantic %": [r.semantic_pct for r in valid_results],
                        "Skill %": [r.skill_coverage_pct for r in valid_results],
                        "Label": [r.label for r in valid_results],
                        "Shortlisted": [r.shortlisted for r in valid_results],
                    }
                )

                fig_dist = px.bar(
                    score_df.sort_values("Match %", ascending=True),
                    x="Match %",
                    y="Candidate",
                    orientation="h",
                    color="Label",
                    color_discrete_map={
                        "Strong match": "#16a34a",
                        "Good match": "#2563eb",
                        "Partial match": "#eab308",
                        "Weak match": "#dc2626",
                    },
                    title="Candidate Match Scores",
                    text="Match %",
                )
                fig_dist.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_dist.update_layout(
                    template="plotly_white",
                    height=300 + len(valid_results) * 30,
                    margin=dict(l=120, r=40, t=50, b=40),
                    showlegend=True,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            # Skill coverage vs Semantic similarity scatter
            st.markdown("#### Skill Coverage vs. Semantic Similarity")
            fig_scatter = px.scatter(
                score_df,
                x="Semantic %",
                y="Skill %",
                color="Label",
                size="Match %",
                hover_data=["Candidate", "Match %"],
                color_discrete_map={
                    "Strong match": "#16a34a",
                    "Good match": "#2563eb",
                    "Partial match": "#eab308",
                    "Weak match": "#dc2626",
                },
                title="Semantic vs. Skill-Based Matching",
            )
            fig_scatter.add_vline(x=50, line_dash="dot", line_color="gray")
            fig_scatter.add_hline(y=50, line_dash="dot", line_color="gray")
            fig_scatter.update_layout(
                template="plotly_white",
                height=400,
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Label distribution pie
            st.markdown("#### Match Quality Distribution")
            label_counts = score_df["Label"].value_counts().reset_index()
            label_counts.columns = ["Label", "Count"]
            fig_pie = px.pie(
                label_counts,
                values="Count",
                names="Label",
                color="Label",
                color_discrete_map={
                    "Strong match": "#16a34a",
                    "Good match": "#2563eb",
                    "Partial match": "#eab308",
                    "Weak match": "#dc2626",
                },
                hole=0.4,
            )
            fig_pie.update_layout(
                template="plotly_white",
                height=350,
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ─── Export Section ────────────────────────────────────────────
        st.markdown(
            '<div class="section-header"><h2>💾 Export</h2></div>', unsafe_allow_html=True
        )

        col_exp1, col_exp2, col_exp3 = st.columns(3)

        with col_exp1:
            # CSV export
            csv = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Full Results (CSV)",
                csv,
                "resumematch_results.csv",
                "text/csv",
                use_container_width=True,
            )

        with col_exp2:
            # Shortlist only CSV
            shortlist_df = df_results[df_results["shortlisted"] == True]
            if not shortlist_df.empty:
                shortlist_csv = shortlist_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Shortlist Only (CSV)",
                    shortlist_csv,
                    "resumematch_shortlist.csv",
                    "text/csv",
                    use_container_width=True,
                )
            else:
                st.button("📥 Download Shortlist (empty)", disabled=True, use_container_width=True)

        with col_exp3:
            # JSON export
            import json
            json_str = json.dumps([r.to_dict() for r in results], indent=2, default=str)
            st.download_button(
                "📥 Download Results (JSON)",
                json_str,
                "resumematch_results.json",
                "application/json",
                use_container_width=True,
            )

    # ─── Footer ────────────────────────────────────────────────────────
    st.markdown(
        """
    <div class="app-footer">
        ResumeMatch — Semantic Resume-to-JD Matcher<br>
        CSE-472 Project 11 • Team: Devananda (Data), Kavya (NLP), Ridhima (Matching), [Your Name] (Frontend)
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()