import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================
# ESTABLISH DATASET PATHS
# ============================
VIS_DIR = Path("visualizations")
VIS_DIR.mkdir(exist_ok=True)

MODEL_DIR = Path("models")
DATA_DIR = Path("datasets/validation-set")
CLEAN_DIR = Path("datasets/cleaned")
RECOMMEND_DIR = Path("datasets/recommendations")


# ============================
# HELPER FUNCTIONS
# ============================
def basic_clean(text):
    import re
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\S+|https\\S+", " ", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def clean_label(label):
    label = str(label).strip().lower()
    if label in ["good fit", "potential fit"]:
        return 1
    elif label == "no fit":
        return 0
    return None


# ============================
# LOAD & CLEAN TRAIN + TEST DATA
# ============================
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

# Clean training data
train_df["label_clean"] = train_df["label"].apply(clean_label)
train_df["resume_text"] = train_df["resume_text"].apply(basic_clean)
train_df["job_description_text"] = train_df["job_description_text"].apply(basic_clean)
train_df["combined_text"] = train_df["resume_text"] + " " + train_df["job_description_text"]
y_train = train_df["label_clean"]

# Clean test data
test_df["label_clean"] = test_df["label"].apply(clean_label)
test_df["resume_text"] = test_df["resume_text"].apply(basic_clean)
test_df["job_description_text"] = test_df["job_description_text"].apply(basic_clean)
test_df["combined_text"] = test_df["resume_text"] + " " + test_df["job_description_text"]
y_test = test_df["label_clean"]


# ============================
# LOAD MAIN LOGISTIC REGRESSION MODEL
# ============================
log_model = joblib.load(MODEL_DIR / "logreg_fit_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "tfidf_fit_vectorizer.pkl")

X_train = vectorizer.transform(train_df["combined_text"])
X_test = vectorizer.transform(test_df["combined_text"])

y_pred_lr = log_model.predict(X_test)
y_prob_lr = log_model.predict_proba(X_test)[:, 1]


# ============================
# TRAIN RANDOM FOREST PROPERLY (ON TRAINING DATA)
# ============================
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)


# ============================
# (1) CONFUSION MATRICES (LOGISTIC REGRESSION AND RANDOM FOREST)
# ============================
plt.figure(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, display_labels=["No Fit","Fit"], cmap="Blues")
plt.title("Confusion Matrix – Logistic Regression")
plt.tight_layout()
plt.savefig(VIS_DIR / "conf_matrix_logreg.png")
plt.close()

plt.figure(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, display_labels=["No Fit","Fit"], cmap="Blues")
plt.title("Confusion Matrix – Random Forest")
plt.tight_layout()
plt.savefig(VIS_DIR / "conf_matrix_rf.png")
plt.close()


# ============================
# (2) LR vs RF MODEL COMPARISON
# ============================
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf)],
    "F1": [f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_rf)]
})

comparison.set_index("Model").plot(kind="bar", figsize=(7,5), color=["#2E86AB","#F18F01"])
plt.title("LR vs RF Model Performance")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(VIS_DIR / "model_comparison_lr_rf.png")
plt.close()


# ============================
# (3) FIT PROBABILITY HISTOGRAM (LR)
# ============================
plt.figure(figsize=(7,5))
sns.histplot(y_prob_lr, bins=20, kde=True, color="navy")
plt.title("Distribution of Fit Probabilities – Logistic Regression")
plt.xlabel("Fit Probability (0-1)")
plt.tight_layout()
plt.savefig(VIS_DIR / "fit_probability_histogram.png")
plt.close()


# ============================
# (4) SIMILARITY DISTRIBUTION (CLEANED RESUMES VS JOBS)
# ============================
resumes_df = pd.read_csv(CLEAN_DIR / "cleaned_resumes.csv")
jobs_df = pd.read_csv(CLEAN_DIR / "cleaned_job_postings.csv")

sim_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.75
)
sim_vectorizer.fit(pd.concat([resumes_df["combined_text"], jobs_df["combined_text"]]))

res_tfidf = sim_vectorizer.transform(resumes_df["combined_text"])
job_tfidf = sim_vectorizer.transform(jobs_df["combined_text"])

similarity_matrix = cosine_similarity(res_tfidf, job_tfidf)

plt.figure(figsize=(7,5))
sns.histplot(similarity_matrix.flatten(), bins=30, kde=True, color="green")
plt.title("Distribution of Resume–Job Similarity Scores")
plt.xlabel("Cosine Similarity")
plt.tight_layout()
plt.savefig(VIS_DIR / "similarity_distribution.png")
plt.close()


# ============================
# (5) RECOMMENDATION VISUALIZATIONS
# ============================
recs = pd.read_csv(RECOMMEND_DIR / "final_ranked_recommendations.csv")

sample_resumes = recs["resume_name"].unique()[:3]

for r in sample_resumes:
    sub = recs[recs["resume_name"] == r].head(5)

    plt.figure(figsize=(8,4))
    sns.barplot(data=sub, x="fit_score", y="job_title", palette="crest")
    plt.title(f"Top Job Recommendations for {r}")
    plt.xlabel("Fit Score")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"{r.replace(' ','_')}_top_recommendations.png")
    plt.close()


# ============================
# (6) SKILL OVERLAP BAR CHART
# ============================
for r in sample_resumes:
    sub = recs[recs["resume_name"] == r].head(5)

    plt.figure(figsize=(8,4))
    sns.barplot(data=sub, x="matched_skill_count", y="job_title", palette="viridis")
    plt.title(f"Skill Overlap for {r}'s Top Job Matches")
    plt.xlabel("Matched Skill Count")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"{r.replace(' ','_')}_skill_overlap.png")
    plt.close()


print("\n ALL visualizations saved in /visualizations/")
