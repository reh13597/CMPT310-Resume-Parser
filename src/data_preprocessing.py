# ============================================================
# data_preprocessing.py
# Purpose: Clean and normalize resume and job posting text data
# ============================================================

import pandas as pd
import json
import re
from pathlib import Path

# Define dataset paths
RESUME_PATH = Path("datasets/resumes/master_resumes.jsonl")
JOB_PATH = Path("datasets/job-postings/job_postings.csv")

# ============================================================
# Load Resumes
# ============================================================
resumes = []
with open(RESUME_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            resumes.append(json.loads(line))
        except json.JSONDecodeError:
            continue

print(f"Loaded {len(resumes)} resumes")

# Convert to DataFrame
resumes_df = pd.json_normalize(resumes)

# Extract useful text fields
def extract_resume_text(row):
    parts = []
    # Combine text from common fields if they exist
    for field in ["experience", "education", "projects", "skills.technical.programming_languages"]:
        value = row.get(field)
        if isinstance(value, list):
            parts.extend(value)
        elif isinstance(value, str):
            parts.append(value)
    return " ".join(map(str, parts))

resumes_df["clean_text"] = resumes_df.apply(extract_resume_text, axis=1)

# Clean up text
resumes_df["clean_text"] = resumes_df["clean_text"].fillna("").apply(
    lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x.lower())
)

print("Sample cleaned resume text:\n", resumes_df["clean_text"].iloc[0][:300])

# ============================================================
# Load Job Postings
# ============================================================
job_postings = pd.read_csv(JOB_PATH)
print(f"Loaded {len(job_postings)} job postings")

# Keep essential fields
# We’ll use 'company_id' because there is no 'company_name' in the dataset
essential_cols = ["title", "company_id", "description", "skills_desc"]
job_postings = job_postings[essential_cols].fillna("")

# Combine text fields
job_postings["clean_text"] = (
    job_postings["title"].astype(str) + " " +
    job_postings["description"].astype(str) + " " +
    job_postings["skills_desc"].astype(str)
)

# Clean text
job_postings["clean_text"] = job_postings["clean_text"].apply(
    lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x.lower())
)

print("Sample cleaned job text:\n", job_postings["clean_text"].iloc[0][:300])

# ============================================================
# Save Cleaned Data
# ============================================================
Path("datasets/cleaned").mkdir(exist_ok=True)
resumes_df[["clean_text"]].to_csv("datasets/cleaned/cleaned_resumes.csv", index=False)
job_postings[["title", "company_id", "clean_text"]].to_csv("datasets/cleaned/cleaned_jobs.csv", index=False)

print("\n✅ Cleaning complete. Files saved to 'datasets/cleaned/'")

