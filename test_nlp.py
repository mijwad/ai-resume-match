
from nlp_embeddings import (
    clean_text,
    generate_embedding,
    process_resume_and_jd
)

resume = """
Python developer with machine learning,
NLP, SQL and data analysis skills.
"""

job_description = """
Looking for a candidate skilled in Python,
machine learning and natural language processing.
"""

# Test text cleaning
cleaned = clean_text(resume)
print("Cleaned text:", cleaned)

# Test embedding generation
vector = generate_embedding(cleaned)
print("Embedding shape:", vector.shape)

# Test both inputs together
resume_vector, jd_vector = process_resume_and_jd(
    resume, job_description
)

print("Resume vector:", resume_vector.shape)
print("JD vector:", jd_vector.shape)
print("Integration test passed!")
