import pandas as pd
import re
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_label(label):
    label = str(label).strip().lower()
    if label in ["good fit", "potential fit"]:
        return 1      # Fit
    elif label == "no fit":
        return 0      # No Fit
    return None

if __name__ == "__main__":
    TRAIN_PATH = Path("datasets/validation-set/train.csv")
    TEST_PATH = Path("datasets/validation-set/test.csv")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    for df in [train_df, test_df]:
        df["label_clean"] = df["label"].apply(clean_label)
        df.dropna(subset=["label_clean"], inplace=True)
        df["resume_text"] = df["resume_text"].apply(basic_clean)
        df["job_description_text"] = df["job_description_text"].apply(basic_clean)
        df["combined_text"] = df["resume_text"] + " " + df["job_description_text"]

    y_train, y_test = train_df["label_clean"], test_df["label_clean"]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train_df["combined_text"])
    X_test = vectorizer.transform(test_df["combined_text"])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nLogistic Regression (Fit vs No Fit):")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(model, models_dir / "logreg_fit_model.pkl")
    joblib.dump(vectorizer, models_dir / "tfidf_fit_vectorizer.pkl")

    print("\nSaved:")
    print(" - models/logreg_fit_model.pkl")
    print(" - models/tfidf_fit_vectorizer.pkl")
