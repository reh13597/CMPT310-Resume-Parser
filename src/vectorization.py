# ============================================================
# vectorization.py
# Purpose: Vectorize cleaned resume and job posting datasets
#           using TF-IDF and compute similarity scores
# ============================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import joblib
from pathlib import Path

# ============================================================
# Paths
# ============================================================
DATA_DIR = Path("datasets/cleaned")
OUTPUT_DIR = Path("datasets/embeddings")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load the datasets
# ============================================================
print("Loading cleaned datasets...")
resumes_df = pd.read_csv(DATA_DIR / "cleaned_resumes.csv")
jobs_df = pd.read_csv(DATA_DIR / "cleaned_job_postings.csv")
print(f"Loaded {len(resumes_df)} resumes and {len(jobs_df)} job postings")

# ============================================================
# Combine the combined_text fields for a shared vocabulary
# ============================================================
all_texts = pd.concat([resumes_df["combined_text"], jobs_df["combined_text"]])

# ============================================================
# Initialize and fit the TF-IDF vectorizer
# ============================================================
print("\nFitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.75,
    sublinear_tf=True
)
vectorizer.fit(all_texts)

print("TF-IDF fitted successfully")
print("Vocabulary size:", len(vectorizer.get_feature_names_out()))

# ============================================================
# Transform datasets
# ============================================================
print("\nTransforming text into TF-IDF matrices...")
resumes_tfidf = vectorizer.transform(resumes_df["combined_text"])
jobs_tfidf = vectorizer.transform(jobs_df["combined_text"])

print("Resumes TF-IDF matrix:", resumes_tfidf.shape)
print("Jobs TF-IDF matrix:", jobs_tfidf.shape)

# ============================================================
# Compute cosine similarity
# ============================================================
print("\nComputing cosine similarity between resumes and jobs...")

# Each resume is compared to each job (rows = resumes, cols = jobs)
similarity_matrix = cosine_similarity(resumes_tfidf, jobs_tfidf)
print("Similarity matrix shape:", similarity_matrix.shape)

# Convert to DataFrame for inspection
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=[f"Resume {i+1}" for i in range(resumes_tfidf.shape[0])],
    columns=[f"Job {j+1}" for j in range(jobs_tfidf.shape[0])]
)

# ============================================================
# Extract Top N Matches for Each Resume
# ============================================================
N = 5
print(f"\nExtracting top {N} job matches for each resume...")

# For each resume (row), get top N job indices and scores
top_matches = []
for i, row in enumerate(similarity_df.values):
    top_indices = row.argsort()[-N:][::-1]  # top N job indices
    top_jobs = [
        {
            "resume_id": i + 1,
            "job_id": jobs_df.iloc[j]["job_id"],
            "job_title": jobs_df.iloc[j]["title"],
            "similarity": row[j]
        }
        for j in top_indices
    ]
    top_matches.extend(top_jobs)

# Convert to DataFrame
top_matches_df = pd.DataFrame(top_matches)

print("\nExample top matches:")
print(top_matches_df.head(10))


# ============================================================
# Save Results
# ============================================================
print("\nSaving vectorizer, embeddings, and similarity data...")

joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
sparse.save_npz(OUTPUT_DIR / "resumes_tfidf.npz", resumes_tfidf)
sparse.save_npz(OUTPUT_DIR / "jobs_tfidf.npz", jobs_tfidf)
similarity_df.to_csv(OUTPUT_DIR / "resume_job_similarity.csv", index=True)
top_matches_df.to_csv(OUTPUT_DIR / "top_matches.csv", index=True)

print("\nVectorization and similarity computation complete!")
print("Saved files:")
print(" - models/tfidf_vectorizer.pkl")
print(" - datasets/embeddings/resumes_tfidf.npz")
print(" - datasets/embeddings/jobs_tfidf.npz")
print(" - datasets/embeddings/resume_job_similarity.csv")
print(" - datasets/embeddings/top_matches.csv")
