# ============================================================
# data_preprocessing.py
# Purpose: Clean and normalize resumes and job postings
# ============================================================

import pandas as pd
import json
import re
import spacy
from pathlib import Path
from tqdm import tqdm

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def basic_clean(text):
    """Remove special chars, URLs, and normalize whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def spacy_clean(nlp, texts):
    """Batch-clean text using spaCy (lemmatization + stopword removal)."""
    cleaned = []
    for doc in tqdm(nlp.pipe(texts, batch_size=100, n_process=4), total=len(texts), desc="spaCy cleaning"):
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        cleaned.append(" ".join(tokens))
    return cleaned

def extract_resume_text(row, field):
    """Safely extract resume sections."""
    val = row.get(field)
    if isinstance(val, list):
        return " ".join(map(str, val))
    elif isinstance(val, str):
        return val
    return ""

# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================
def main():
    # ============================================================
    # INITIALIZE spaCy
    # ============================================================
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    # ============================================================
    # DEFINE PATHS
    # ============================================================
    RESUME_PATH = Path("datasets/resumes/master_resumes.jsonl")
    JOB_PATH = Path("datasets/job-postings/job_postings.csv")
    OUTPUT_DIR = Path("datasets/cleaned")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ============================================================
    # CLEAN RESUMES
    # ============================================================
    print("\nCleaning resumes...")

    resumes = []
    with open(RESUME_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                resumes.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    resumes_df = pd.json_normalize(resumes)
    print(f"Loaded {len(resumes_df)} resumes")

    # ============================================================
    # EXTRACT & CLEAN RESUMES
    # ============================================================
    resumes_df["summary_clean"] = resumes_df.apply(lambda r: basic_clean(extract_resume_text(r, "personal_info.summary")), axis=1)
    resumes_df["experience_clean"] = resumes_df.apply(lambda r: basic_clean(extract_resume_text(r, "experience")), axis=1)
    resumes_df["education_clean"] = resumes_df.apply(lambda r: basic_clean(extract_resume_text(r, "education")), axis=1)
    resumes_df["skills_clean"] = resumes_df.apply(lambda r: basic_clean(extract_resume_text(r, "skills.technical.programming_languages")), axis=1)

    # ============================================================
    # COMBINE INTO SINGLE TEXT FOR MODELS
    # ============================================================
    resumes_df["combined_text"] = (
        resumes_df["summary_clean"].astype(str) + " " +
        resumes_df["experience_clean"].astype(str) + " " +
        resumes_df["education_clean"].astype(str) + " " +
        resumes_df["skills_clean"].astype(str)
    )

    # ============================================================
    # APPLY spaCy CLEANING TO COMBINED TEXT
    # ============================================================
    resumes_df["combined_text"] = spacy_clean(nlp, resumes_df["combined_text"].tolist())

    # ============================================================
    # ADD ID AND METADATA FIELDS
    # ============================================================
    resumes_df_out = pd.DataFrame({
        "resume_id": range(1, len(resumes_df) + 1),
        "name": resumes_df.get("personal_info.name", "Unknown"),
        "location": resumes_df.get("personal_info.location", "Unknown"),
        "summary_clean": resumes_df["summary_clean"],
        "experience_clean": resumes_df["experience_clean"],
        "education_clean": resumes_df["education_clean"],
        "skills_clean": resumes_df["skills_clean"],
        "combined_text": resumes_df["combined_text"]
    })

    print("Sample cleaned resume:")
    print(resumes_df_out.head(1).to_string(index=False))

    # ============================================================
    # CLEAN JOB POSTINGS
    # ============================================================
    print("\nCleaning job postings...")

    job_postings = pd.read_csv(JOB_PATH)
    print(f"Loaded {len(job_postings)} job postings")

    # ============================================================
    # ENSURE NEEDED COLUMNS EXIST
    # ============================================================
    for col in ["job_id", "title", "company_id", "description", "skills_desc",
                "formatted_experience_level", "formatted_work_type", "location"]:
        if col not in job_postings.columns:
            job_postings[col] = ""

    job_postings["description_clean"] = job_postings["description"].apply(basic_clean)
    job_postings["skills_clean"] = job_postings["skills_desc"].apply(basic_clean)

    # ============================================================
    # COMBINE MULTIPLE FIELDS FOR MODELING
    # ============================================================
    job_postings["combined_text"] = (
        job_postings["title"].astype(str) + " " +
        job_postings["description_clean"].astype(str) + " " +
        job_postings["skills_clean"].astype(str) + " " +
        job_postings["formatted_experience_level"].astype(str) + " " +
        job_postings["formatted_work_type"].astype(str) + " " +
        job_postings["location"].astype(str)
    )

    # ============================================================
    # APPLY spaCy CLEANING
    # ============================================================
    job_postings["combined_text"] = spacy_clean(nlp, job_postings["combined_text"].tolist())

    job_postings_out = job_postings[[
        "job_id", "title", "company_id",
        "description_clean", "skills_clean",
        "formatted_experience_level", "formatted_work_type",
        "location", "combined_text"
    ]]

    print("Sample cleaned job posting:")
    print(job_postings_out.head(1).to_string(index=False))

    # ============================================================
    # SAVE CLEANED FILES
    # ============================================================
    resumes_df_out.to_csv(OUTPUT_DIR / "cleaned_resumes.csv", index=False)
    job_postings_out.to_csv(OUTPUT_DIR / "cleaned_job_postings.csv", index=False)

    print("\nCleaning complete! Files saved to:")
    print(" - datasets/cleaned/cleaned_resumes.csv")
    print(" - datasets/cleaned/cleaned_job_postings.csv")

# ============================================================
# WINDOWS MULTIPROCESSING SAFE ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
