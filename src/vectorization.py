# ============================================================
# vectorization.py
# Purpose: Vectorize the datasets using TF-IDF
# ============================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


"""
Cosine Similarity
output:

"""

def cosineSimilarity(jobMatrix, resumeMatrix):
    similarityMatrix = cosine_similarity(resumeMatrix, jobMatrix)

    return(similarityMatrix, similarityMatrix.shape)

test, shape = cosineSimilarity(jobs_tfidf,  resumes_tfidf)

df = pd.DataFrame(
    test,
    index= [f"Resume {i}" for i in range(resumes_tfidf.shape[0])],
    columns=[f"Job {j}" for j in range(jobs_tfidf.shape[0])]
)

print(df) # similarity matrix in a dataframe
print("Shape:", shape)

"""
    Top N jobs for resume using a lambda function
"""

# Argument for N here
N = 5

# Gives top 5 jobs per resume || Gives top 5 resumes per job
scoresWithJobNames = lambda x: x.nlargest(N).to_dict()

# axis 1 for top jobs for resumes, axis 0 for top resumes for job
top_matches = df.apply(scoresWithJobNames, axis=1)

print(top_matches)
