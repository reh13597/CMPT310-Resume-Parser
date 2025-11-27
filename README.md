# CMPT 310 AI/Machine Learning Project
# Resume-to-Job AI Matching System

# Project Overview
This project builds an AI-powered Resume-to-Job Recommendation System that predicts whether a resume is a good match for a job posting and generates ranked job recommendations.

# How to Run the Code
1. Create and Activate Virtual Environment
   
    python -m venv venv

    source venv/bin/activate   # Mac/Linux

    venv\Scripts\activate      # Windows


2. Install Dependencies
   
   pip install -r requirements.txt

   python -m spacy download en_core_web_sm


3. Project Structure
   
   src/
   
     data_preprocessing.py
   
     vectorization.py
   
     train_fit_model.py
   
     recommend_jobs.py
   
     generate_visualizations.py
   
     run_pipeline.py

4. How to Run the Entire Pipeline
   
   python src/run_pipeline.py

5. Or Run Steps Individually
   
   A. Preprocess Raw Resumes + Job Postings
   
   python src/data_preprocessing.py
    
   B. Generate TF-IDF Embeddings + Similarity
   
   python src/vectorization.py
    
   C. Train Fit vs No Fit Classifier
   
   python src/train_fit_model.py
    
   D. Generate Job Recommendations
   
   python src/recommend_jobs.py
    
   E. Generate All Visualizations
   
   python src/generate_visualizations.py

6. Output Locations

   Recommendations:
   
     datasets/recommendations/final_ranked_recommendations.csv

   Visualizations:
   
     visualizations/

    Saved Models:
   
      models/logreg_fit_model.pkl
   
      models/tfidf_fit_vectorizer.pkl

# Authors

Fizza Ali, Alex Guo, Macklin Tsang, Charlie Zhang

