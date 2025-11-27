# CMPT 310 AI/Machine Learning Project
# Resume-to-Job AI Matching System

## Project Overview
Our Resume Parser and Job Recommender System is an AI-driven Resume Parser and Job Recommender System that automates candidate–job matching. The system is built upon Natural Language Processing (NLP) for resume/job text extraction alongside TF-IDF vectorization in computation for candidate-job similarity via predicted numerical job-fit score for each candidate-job pair. The system takes resumes and job postings as input and outputs a ranked list of recommended jobs based on textual similarity and predicted compatibility.  

Datasets were sourced from HuggingFace:​

Resumes: https://huggingface.co/datasets/datasetmaster/resumes (mix of synthetic and real resumes in JSON format)
Job postings: Hugging Face LinkedIn Job Postings, which contains 33k real-world postings. We will sample or filter subsets (e.g., technical jobs) to keep training manageable.

Validation set: https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit (pairs of resumes and job descriptions with ground-truth "fit" labels, which allow us to evaluate model accuracy using metrics such as precision, recall, and F1-score).

The system utilized logistic regression and random forest for the classifiers. The pipeline combines 3 class labels (“good fit,” “potential fit,” “no fit”) into a binary classification problem, simplifying the learning task while preserving the core logic of the project.

## How to Run the Code
1. Create and Activate Virtual Environment
   ```
    python -m venv venv

    source venv/bin/activate   # Mac/Linux

    venv\Scripts\activate      # Windows
   ```

2. Install Dependencies
   ```
   pip install -r requirements.txt

   python -m spacy download en_core_web_sm
   ```

3. How to Run the Entire Pipeline
   ```
   python src/run_pipeline.py
   ```

4. Or Run Steps Individually
   
   A. Preprocess Raw Resumes + Job Postings
   ```
   python src/data_preprocessing.py
    ```
   B. Generate TF-IDF Embeddings + Similarity
   ```
   python src/vectorization.py
    ```
   C. Train Fit vs No Fit Classifier
   ```
   python src/train_fit_model.py
    ```
   D. Generate Job Recommendations
   ```
   python src/recommend_jobs.py
    ```
   E. Generate All Visualizations
   ```
   python src/generate_visualizations.py
   ```
## Project Structure
```
.
├── datasets
│   ├── cleaned
│   ├── embeddings
│   ├── job-postings
│   ├── predictions
│   ├── recommendations
│   ├── resumes
│   └── validation-set
├── models
│   ├── logistic_regression_fit_model.pkl
│   ├── tfidf_vectorizer_fit.pkl
│   └── tfidf_vectorizer.pkl
├── README.md
├── requirements.txt
├── src
│   ├── data_preprocessing.py
│   ├── generate_visualizations.py
│   ├── recommend_jobs.py
│   ├── run_pipeline.py
│   ├── train_fit_model.py
│   └── vectorization.py
└── visualizations
```
## Output Locations

Recommendations:
   ```
     datasets/recommendations/final_ranked_recommendations.csv
   ```
Visualizations:
   ```
     visualizations/
   ```
Saved Models:
   ```
      models/logreg_fit_model.pkl
   
      models/tfidf_fit_vectorizer.pkl
   ```

# Authors

Fizza Ali, Alex Guo, Macklin Tsang, Charlie Zhang
