import pandas as pd
import re
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_skills(text):
    skill_list = [
        "python", "java", "c++", "sql", "javascript", "react", "angular",
        "node", "pandas", "numpy", "tensorflow", "keras", "docker", "aws",
        "excel", "machine learning", "deep learning", "nlp", "html", "css",
        "data", "analytics", "management", "communication"
    ]
    text = str(text).lower()
    return [s for s in skill_list if s in text]

if __name__ == "__main__":
    CLEAN_DIR = Path("datasets/cleaned")
    EMBED_DIR = Path("datasets/embeddings")
    OUTPUT_DIR = Path("datasets/recommendations")
    MODEL_DIR = Path("models")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load cleaned resumes and jobs
    resumes_df = pd.read_csv(CLEAN_DIR / "cleaned_resumes.csv")
    jobs_df = pd.read_csv(CLEAN_DIR / "cleaned_job_postings.csv")

    # Load the TF-IDF vectorizer for similarity (this is your unsupervised one)
    # If you already ran vectorization.py, you can reuse its vectorizer.
    # For now, we refit here for clarity.
    all_texts = pd.concat([resumes_df["combined_text"], jobs_df["combined_text"]])
    sim_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.75,
        sublinear_tf=True
    )
    sim_vectorizer.fit(all_texts)

    resumes_tfidf = sim_vectorizer.transform(resumes_df["combined_text"])
    jobs_tfidf = sim_vectorizer.transform(jobs_df["combined_text"])

    similarity_matrix = cosine_similarity(resumes_tfidf, jobs_tfidf)

    # Load binary fit model and its vectorizer
    fit_model = joblib.load(MODEL_DIR / "logreg_fit_model.pkl")
    fit_vectorizer = joblib.load(MODEL_DIR / "tfidf_fit_vectorizer.pkl")

    top_matches = []
    top_k = 5

    for i in range(resumes_df.shape[0]):
        sim_row = similarity_matrix[i]
        top_idx = sim_row.argsort()[-top_k:][::-1]

        resume_text_clean = basic_clean(resumes_df.iloc[i]["combined_text"])

        for j in top_idx:
            job_text_clean = basic_clean(jobs_df.iloc[j]["combined_text"])

            combined = resume_text_clean + " " + job_text_clean
            X_pair = fit_vectorizer.transform([combined])
            fit_prob = fit_model.predict_proba(X_pair)[0, 1]  # probability of Fit (class 1)

            matched_skills = sorted(
                set(extract_skills(resume_text_clean)) &
                set(extract_skills(job_text_clean))
            )

            top_matches.append({
                "resume_index": i,
                "resume_name": resumes_df.iloc[i].get("name", "Unknown"),
                "job_id": jobs_df.iloc[j].get("job_id", None),
                "job_title": jobs_df.iloc[j].get("title", ""),
                "company_id": jobs_df.iloc[j].get("company_id", ""),
                "similarity": sim_row[j],
                "fit_prob": fit_prob,
                "matched_skills": ", ".join(matched_skills),
                "matched_skill_count": len(matched_skills)
            })

    recs_df = pd.DataFrame(top_matches)

    # Normalize similarity and combine with fit probability
    min_sim, max_sim = recs_df["similarity"].min(), recs_df["similarity"].max()
    recs_df["normalized_similarity"] = (recs_df["similarity"] - min_sim) / (max_sim - min_sim + 1e-8)

    alpha, beta = 0.7, 0.3
    recs_df["fit_score"] = alpha * recs_df["normalized_similarity"] + beta * recs_df["fit_prob"]

    recs_df.sort_values(["resume_name", "fit_score"], ascending=[True, False], inplace=True)

    # Keep top 5 per resume
    final_recs = recs_df.groupby("resume_name").head(5).reset_index(drop=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "final_ranked_recommendations.csv"
    final_recs.to_csv(out_path, index=False)

    print(f"\nSaved final ranked recommendations to {out_path}")
    print(final_recs.head(10).to_string(index=False))
