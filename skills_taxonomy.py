"""
Skill taxonomy for ResumeMatch.

Maps a canonical skill name -> list of aliases / alternate phrasings.
Extending coverage is as simple as adding entries here. Aliases are what let
"natural language processing" in a JD match "NLP" in a resume.
"""

SKILL_TAXONOMY = {
    # ---- Programming languages ----
    "python": ["python", "python3"],
    "java": ["java"],
    "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp"],
    "c": ["c programming", "c language"],
    "r": ["r programming", "r language", "rstudio"],
    "go": ["golang", "go language"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite", "t-sql", "pl/sql"],
    "php": ["php"],
    "kotlin": ["kotlin"],
    "swift": ["swift"],
    "matlab": ["matlab"],
    "scala": ["scala"],

    # ---- Web / frontend / backend ----
    "html": ["html", "html5"],
    "css": ["css", "css3", "sass", "scss"],
    "react": ["react", "reactjs", "react.js"],
    "angular": ["angular", "angularjs"],
    "vue": ["vue", "vuejs", "vue.js"],
    "node.js": ["node.js", "nodejs", "node js"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "spring boot": ["spring boot", "springboot", "spring framework"],
    "rest api": ["rest api", "restful api", "rest apis", "restful apis"],
    "streamlit": ["streamlit"],

    # ---- ML / AI / NLP ----
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "neural network", "neural networks"],
    "nlp": ["nlp", "natural language processing", "text mining", "computational linguistics"],
    "computer vision": ["computer vision", "image processing", "opencv"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch", "torch"],
    "keras": ["keras"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "transformers": ["transformers", "bert", "sentence-bert", "sbert", "huggingface", "hugging face"],
    "word embeddings": ["word2vec", "glove", "word embeddings", "sentence embeddings", "embeddings"],
    "llm": ["llm", "llms", "large language model", "large language models", "generative ai", "gpt"],
    "reinforcement learning": ["reinforcement learning"],
    "statistics": ["statistics", "statistical analysis", "statistical modeling", "hypothesis testing"],

    # ---- Data ----
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "data analysis": ["data analysis", "data analytics", "data analyst", "exploratory data analysis", "eda"],
    "data visualization": ["data visualization", "data visualisation", "matplotlib", "seaborn", "plotly"],
    "data engineering": ["data engineering", "etl", "data pipeline", "data pipelines"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "excel": ["excel", "microsoft excel", "ms excel", "advanced excel", "excel spreadsheets", "excel dashboards"],
    "spark": ["spark", "pyspark", "apache spark"],
    "hadoop": ["hadoop"],
    "mongodb": ["mongodb", "mongo db"],
    "nosql": ["nosql", "cassandra", "dynamodb", "redis"],

    # ---- Cloud / DevOps ----
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git", "github", "gitlab", "version control"],
    "ci/cd": ["ci/cd", "cicd", "jenkins", "github actions", "continuous integration"],
    "linux": ["linux", "unix", "bash", "shell scripting"],
    "mlops": ["mlops", "model deployment"],

    # ---- Soft / general skills ----
    "communication": ["communication", "communication skills", "presentation skills"],
    "leadership": ["leadership", "team lead", "team leadership"],
    "teamwork": ["teamwork", "collaboration", "team player", "collaborative"],
    "problem solving": ["problem solving", "problem-solving", "analytical thinking"],
    "project management": ["project management", "agile", "scrum", "jira"],
}
