# ============================================================
# vectorization.py
# Purpose: Vectorize the datasets using TF-IDF
# ============================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# Load the datasets
# ============================================================
resumes_df = pd.read_csv("datasets/cleaned/cleaned_resumes.csv")
jobs_df = pd.read_csv("datasets/cleaned/cleaned_job_postings.csv")

# ============================================================
# Combine the combined_text fields from both datasets for fitting
# ============================================================
all_texts = pd.concat([resumes_df['combined_text'], jobs_df['combined_text']])

# ============================================================
# Fit the TF-IDF vectorizer
# ============================================================
vectorizer = TfidfVectorizer(
    stop_words='english', # Removes common filler words like "the", "and", "a", etc.
    max_features = 20000, # Keeps the 20k most common words
    ngram_range = (1, 2), # Captures both single words and 2-word phrases
    min_df = 5, # Filters out words that appear in less than 5 resumes/job postings
    max_df = 0.75, # Filters out words that appear in >75% of resumes/job postings
    sublinear_tf = True # Log-scale term frequencies for stability
)

vectorizer.fit(all_texts)

# ============================================================
# Transform each dataset
# ============================================================
resumes_tfidf = vectorizer.transform(resumes_df['combined_text'])
jobs_tfidf = vectorizer.transform(jobs_df['combined_text'])

# ============================================================
# Print some sanity checks
# ============================================================
print("TF-IDF fitted vocabulary size:", len(vectorizer.get_feature_names_out()))
print("Resumes TF-IDF matrix size:", resumes_tfidf.shape)
print("Jobs TF-IDF matrix size:", jobs_tfidf.shape)