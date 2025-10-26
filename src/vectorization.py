# ============================================================
# vectorization.py
# Purpose: Vectorize the datasets using TF-IDF
# ============================================================

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Load the datasets
# ============================================================
# resumes_df = pd.read_csv("datasets/cleaned/cleaned_resumes.csv")
# jobs_df = pd.read_csv("datasets/cleaned/cleaned_job_postings.csv")

# ============================================================
# Combine the combined_text fields from both datasets for fitting
# ============================================================
# all_texts = pd.concat([resumes_df['combined_text'], jobs_df['combined_text']])

# ============================================================
# Fit the TF-IDF vectorizer
# ============================================================
# vectorizer = TfidfVectorizer(
#     stop_words='english', # Removes common filler words like "the", "and", "a", etc.
#     max_features = 20000, # Keeps the 20k most common words
#     ngram_range = (1, 2), # Captures both single words and 2-word phrases
#     min_df = 5, # Filters out words that appear in less than 5 resumes/job postings
#     max_df = 0.75, # Filters out words that appear in >75% of resumes/job postings
#     sublinear_tf = True # Log-scale term frequencies for stability
# )

# vectorizer.fit(all_texts)

# ============================================================
# Transform each dataset
# ============================================================
# resumes_tfidf = vectorizer.transform(resumes_df['combined_text'])
# jobs_tfidf = vectorizer.transform(jobs_df['combined_text'])

# ============================================================
# Print some sanity checks
# ============================================================
# print("TF-IDF fitted vocabulary size:", len(vectorizer.get_feature_names_out()))
# print("Resumes TF-IDF matrix size:", resumes_tfidf.shape)
# print("Jobs TF-IDF matrix size:", jobs_tfidf.shape)


# """
# Cosine Similarity
# output:

# """

# def cosineSimilarity(jobMatrix, resumeMatrix):
#     similarityMatrix = cosine_similarity(resumeMatrix, jobMatrix)

#     return(similarityMatrix, similarityMatrix.shape)

# test, shape = cosineSimilarity(jobs_tfidf,  resumes_tfidf)

# df = pd.DataFrame(
#     test,
#     index= [f"Resume {i}" for i in range(resumes_tfidf.shape[0])],
#     columns=[f"Job {j}" for j in range(jobs_tfidf.shape[0])]
# )

# print(df) # similarity matrix in a dataframe
# print("Shape:", shape)

# """
#     Top N jobs for resume using a lambda function
# """

# # Argument for N here
# N = 5

# # Gives top 5 jobs per resume || Gives top 5 resumes per job
# scoresWithJobNames = lambda x: x.nlargest(N).to_dict()

# # axis 1 for top jobs for resumes, axis 0 for top resumes for job
# top_matches = df.apply(scoresWithJobNames, axis=1)

# print(top_matches)

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
print("📂 Loading cleaned datasets...")
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
print("\n🧠 Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.75,
    sublinear_tf=True
)
vectorizer.fit(all_texts)

print("✅ TF-IDF fitted successfully")
print("Vocabulary size:", len(vectorizer.get_feature_names_out()))

# ============================================================
# Transform datasets
# ============================================================
print("\n🔢 Transforming text into TF-IDF matrices...")
resumes_tfidf = vectorizer.transform(resumes_df["combined_text"])
jobs_tfidf = vectorizer.transform(jobs_df["combined_text"])

print("Resumes TF-IDF matrix:", resumes_tfidf.shape)
print("Jobs TF-IDF matrix:", jobs_tfidf.shape)

# ============================================================
# Compute cosine similarity
# ============================================================
print("\n📈 Computing cosine similarity between resumes and jobs...")

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
print(f"\n🏆 Extracting top {N} job matches for each resume...")

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

print("\n✅ Example top matches:")
print(top_matches_df.head(10))


# ============================================================
# Save Results
# ============================================================
print("\n💾 Saving vectorizer, embeddings, and similarity data...")

joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
sparse.save_npz(OUTPUT_DIR / "resumes_tfidf.npz", resumes_tfidf)
sparse.save_npz(OUTPUT_DIR / "jobs_tfidf.npz", jobs_tfidf)
similarity_df.to_csv(OUTPUT_DIR / "resume_job_similarity.csv", index=True)
top_matches_df.to_csv(OUTPUT_DIR / "top_matches.csv", index=True)

print("\n🎯 Vectorization and similarity computation complete!")
print("Saved files:")
print(" - models/tfidf_vectorizer.pkl")
print(" - datasets/embeddings/resumes_tfidf.npz")
print(" - datasets/embeddings/jobs_tfidf.npz")
print(" - datasets/embeddings/resume_job_similarity.csv")
print(" - datasets/embeddings/top_matches.csv")
