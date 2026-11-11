
from nlp_embeddings import (
    clean_text,
    generate_embedding,
    generate_embeddings,
    process_resume_and_jd
)


def prepare_resume(resume_text):
    """Prepare one resume for matching."""
    cleaned = clean_text(resume_text)
    return generate_embedding(cleaned)


def prepare_job_description(job_description):
    """Prepare a job description for matching."""
    cleaned = clean_text(job_description)
    return generate_embedding(cleaned)


if __name__ == "__main__":
    resume = """
    Python developer with machine learning, NLP,
    SQL and data analysis skills.
    """

    job_description = """
    We need a Python engineer with NLP,
    machine learning and SQL experience.
    """

    resume_vector, jd_vector = process_resume_and_jd(
        resume,
        job_description
    )

    print("Resume vector:", resume_vector.shape)
    print("Job description vector:", jd_vector.shape)
    print("NLP integration ready!")
