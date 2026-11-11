# ResumeMatch — NLP & Deep Learning Module

## Developed by
Kavya — NLP & Deep Learning

## Overview
This module cleans resume and job description text and uses Sentence-BERT to generate semantic embeddings.

## Technologies Used
- Python
- Sentence-Transformers
- Sentence-BERT (`all-MiniLM-L6-v2`)
- NumPy
- PyMuPDF

## Features
- Text cleaning and normalization
- Semantic embedding generation for resumes
- Semantic embedding generation for job descriptions
- Batch embedding generation for multiple resumes
- 384-dimensional normalized embedding vectors

## Setup Instructions

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the NLP Module

```bash
python nlp_embeddings.py
```

## Use in the Main Project

Import the functions from `nlp_embeddings.py`:

```python
from nlp_embeddings import (
    clean_text,
    preprocess_text,
    generate_embedding,
    generate_embeddings,
    process_resume_and_jd
)
```

Generate resume and job description embeddings:

```python
resume_vector, jd_vector = process_resume_and_jd(
    resume_text,
    job_description
)
```

The returned vectors can be passed to the similarity module to calculate resume–job description matching scores.

## Output
Each individual text embedding has 384 dimensions.

## Notes
The main project can use this module to generate embeddings. The similarity calculation and Streamlit frontend are handled by other team members.