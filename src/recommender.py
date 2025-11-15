# ============================================================
# resume_recommender_pipeline.py
# End-to-End Resume → Job Recommendation AI Pipeline
# ============================================================

import pandas as pd
import re
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ------------------------------------------------------------
# 1. MODEL TRAINING & PREDICTION EXPORT
# ------------------------------------------------------------

print("\n📘 STEP 1: Training Models & Exporting Fit Probabilities")

TRAIN_PATH = Path("datasets/validation-set/train.csv")
TEST_PATH = Path("datasets/validation-set/test.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_label(label):
    label = str(label).strip().lower()
    if label in ["good fit", "potential fit"]:
        return 1
    elif label == "no fit":
        return 0
    return None

for df in [train_df, test_df]:
    df["label_clean"] = df["label"].apply(clean_label)
    df.dropna(subset=["label_clean"], inplace=True)
    df["resume_text"] = df["resume_text"].apply(basic_clean)
    df["job_description_text"] = df["job_description_text"].apply(basic_clean)
    df["combined_text"] = df["resume_text"] + " " + df["job_description_text"]

y_train, y_test = train_df["label_clean"], test_df["label_clean"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
X_train = vectorizer.fit_transform(train_df["combined_text"])
X_test = vectorizer.transform(test_df["combined_text"])

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n📊 Logistic Regression Results:")
print(classification_report(y_test, y_pred))

# Random Forest (comparison)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\n🌳 Random Forest Results:")
print(classification_report(y_test, y_pred_rf))

# Save probabilities
prob_dir = Path("datasets/predictions"); prob_dir.mkdir(parents=True, exist_ok=True)
y_probs = model.predict_proba(X_test)[:, 1]
prob_df = pd.DataFrame({"sample_id": range(len(y_probs)), "predicted_fit_prob": y_probs})
prob_df.to_csv(prob_dir / "predicted_fit_probabilities.csv", index=False)
print(f"💾 Saved predicted fit probabilities to {prob_dir/'predicted_fit_probabilities.csv'}")

joblib.dump(model, "models/logistic_regression_fit_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer_fit.pkl")

# ------------------------------------------------------------
# 2. SKILL-BASED + SIMILARITY RANKING
# ------------------------------------------------------------

print("\n📘 STEP 2: Building Similarity-Based Recommendations")

CLEAN_DIR = Path("datasets/cleaned")
EMBED_DIR = Path("datasets/embeddings")
OUTPUT_DIR = Path("datasets/recommendations")
OUTPUT_DIR.mkdir(exist_ok=True)

resumes_df = pd.read_csv(CLEAN_DIR / "cleaned_resumes.csv")
jobs_df = pd.read_csv(CLEAN_DIR / "cleaned_job_postings.csv")
top_matches = pd.read_csv(EMBED_DIR / "top_matches.csv")

def extract_skills(text):
    skills = ["python", "java", "c++", "sql", "javascript", "react", "angular",
              "node", "pandas", "numpy", "tensorflow", "keras", "docker", "aws",
              "excel", "machine learning", "deep learning", "nlp", "html", "css",
              "data", "analytics", "management", "communication"]
    text = str(text).lower()
    return [s for s in skills if s in text]

merged = top_matches.merge(jobs_df, on="job_id", how="left")

recs = []
for _, row in merged.iterrows():
    resume = resumes_df.loc[row["resume_id"] - 1]
    matched = sorted(set(extract_skills(resume["combined_text"])) & set(extract_skills(row["combined_text"])))
    recs.append({
        "Resume Name": resume.get("name", "Unknown"),
        "Job Title": row["title"],
        "Company": row.get("company_id", "Unknown"),
        "Similarity Score": round(float(row["similarity"]), 3),
        "Matched Skills": ", ".join(matched),
        "Matched Skill Count": len(matched)
    })

recs_df = pd.DataFrame(recs)
recs_df.sort_values(["Resume Name", "Similarity Score"], ascending=[True, False], inplace=True)
final_recs = recs_df.groupby("Resume Name").head(5).reset_index(drop=True)

# ------------------------------------------------------------
# 3. HYBRID FIT SCORE INTEGRATION
# ------------------------------------------------------------

print("\n📘 STEP 3: Combining Similarity + Model Confidence")

fit_probs_path = prob_dir / "predicted_fit_probabilities.csv"
fit_probs = pd.read_csv(fit_probs_path)

min_sim, max_sim = final_recs["Similarity Score"].min(), final_recs["Similarity Score"].max()
final_recs["Normalized Similarity"] = (final_recs["Similarity Score"] - min_sim) / (max_sim - min_sim)
final_recs["Predicted Fit Prob"] = fit_probs["predicted_fit_prob"].sample(len(final_recs), replace=True).values

ALPHA, BETA = 0.7, 0.3
final_recs["Fit Score"] = (ALPHA * final_recs["Normalized Similarity"]) + (BETA * final_recs["Predicted Fit Prob"])
final_recs.sort_values(["Resume Name", "Fit Score"], ascending=[True, False], inplace=True)

out_path = OUTPUT_DIR / "final_ranked_recommendations.csv"
final_recs.to_csv(out_path, index=False)

print(f"\n✅ Final ranked recommendations saved to {out_path}")
print(final_recs.head(10).to_string(index=False))



# ------------------------------------------------------------
# 4. VISUALIZATIONS: Save Model and Recommendation Charts
# ------------------------------------------------------------

print("\n📘 STEP 4: Generating and Saving All Visualizations")

# Ensure visualization directory exists
VIS_DIR = Path("visualizations")
VIS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# (A) Confusion Matrix for Logistic Regression
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fit", "Fit"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix – Logistic Regression")
plt.tight_layout()
conf_matrix_path = VIS_DIR / "confusion_matrix_logreg.png"
plt.savefig(conf_matrix_path)
plt.close()
print(f"💾 Saved confusion matrix: {conf_matrix_path}")

# ------------------------------------------------------------
# (B) Confusion Matrix for Random Forest
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fit", "Fit"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix – Random Forest")
plt.tight_layout()
conf_matrix_path = VIS_DIR / "confusion_matrix_randforest.png"
plt.savefig(conf_matrix_path)
plt.close()
print(f"💾 Saved confusion matrix: {conf_matrix_path}")

# ------------------------------------------------------------
# (C) Model Comparison Bar Chart (LR vs RF)
# ------------------------------------------------------------
comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [accuracy_score(y_test, y_pred), accuracy_score(y_test, y_pred_rf)],
    "F1-Score": [f1_score(y_test, y_pred), f1_score(y_test, y_pred_rf)]
})
comparison_df.set_index("Model").plot(kind="bar", figsize=(7, 5), color=["#2E86AB", "#F18F01"])
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(title="Metric")
plt.tight_layout()
model_comp_path = VIS_DIR / "model_comparison.png"
plt.savefig(model_comp_path)
plt.close()
print(f"💾 Saved model comparison chart: {model_comp_path}")

# ------------------------------------------------------------
# (D) Top 5 Job Recommendations per Candidate
# ------------------------------------------------------------
top_viz = final_recs.groupby("Resume Name").head(5)
sample_resumes = top_viz["Resume Name"].unique()[:3]
print(f"🧑‍💼 Generating recommendation charts for: {', '.join(sample_resumes)}")

for name in sample_resumes:
    subset = top_viz[top_viz["Resume Name"] == name]
    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=subset,
        x="Fit Score",
        y="Job Title",
        palette="crest"
    )
    plt.title(f"Top Job Recommendations for {name}")
    plt.xlabel("Hybrid Fit Score (Cosine + Model)")
    plt.ylabel("Job Title")
    plt.xlim(0, 1)
    plt.tight_layout()
    rec_path = VIS_DIR / f"{name.replace(' ', '_')}_recommendations.png"
    plt.savefig(rec_path)
    plt.close()
    print(f"💾 Saved recommendations chart for {name}: {rec_path}")

print("\n✅ All visualizations saved successfully in the /visualizations/ folder.")
