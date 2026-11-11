

import re
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load the model once and reuse it
model = SentenceTransformer(MODEL_NAME)


def clean_text(text):
    """Clean and normalize resume or job-description text."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_text(text):
    """Prepare non-empty text for Sentence-BERT."""
    cleaned = clean_text(text)

    if not cleaned:
        raise ValueError("Resume or job description text is empty.")

    return cleaned


def generate_embedding(text):
    """Convert text into a normalized 384-dimensional vector."""
    processed_text = preprocess_text(text)

    return model.encode(
        processed_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


def generate_embeddings(texts):
    """Generate embeddings for multiple resumes efficiently."""
    processed_texts = [preprocess_text(text) for text in texts]

    return model.encode(
        processed_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )


def process_resume_and_jd(resume_text, job_description):
    """Generate resume and job-description vectors."""
    resume_vector = generate_embedding(resume_text)
    jd_vector = generate_embedding(job_description)

    return resume_vector, jd_vector


if __name__ == "__main__":
    resume = "Python developer with experience in NLP and machine learning."
    jd = "Looking for a Python and machine learning engineer."

    resume_vector, jd_vector = process_resume_and_jd(resume, jd)

    print("Resume vector shape:", resume_vector.shape)
    print("Job description vector shape:", jd_vector.shape)

    sample_resumes = [
        "Python developer skilled in NLP and machine learning.",
        "Frontend developer skilled in HTML and CSS.",
        "Data analyst experienced in Python, SQL and statistics."
    ]

    vectors = generate_embeddings(sample_resumes)
    print("Multiple resume vectors shape:", vectors.shape)
    print("NLP module test completed successfully!")
