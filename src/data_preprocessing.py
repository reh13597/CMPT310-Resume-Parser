import pandas as pd
import json

# Load resumes
with open("datasets/resumes/master_resumes.jsonl", "r", encoding="utf-8") as f:
    resumes = [json.loads(line) for line in f]
print(f"Loaded {len(resumes)} resumes")

# Load job postings
job_postings = pd.read_csv("datasets/job-postings/job_postings.csv")
print(f"Loaded {len(job_postings)} job postings")

print("\nSample resume keys:", resumes[0].keys())
print("\nSample job posting columns:", job_postings.columns.tolist())
