from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List

import math
import pandas as pd
from collections import defaultdict

CSV_PATH = "diseases.csv"

app = FastAPI()

# Global data structures
disease_counts = {}
symptom_counts = {}
all_symptoms = set()
total_rows = 0

class SymptomsRequest(BaseModel):
    symptoms: List[str]

@app.on_event("startup")
def startup_event():
    """
    On service startup, build or load the aggregated data 
    for the naive Bayes approach.
    """
    global disease_counts, symptom_counts, all_symptoms, total_rows
    (disease_counts, symptom_counts, all_symptoms, total_rows) = build_disease_statistics(
        CSV_PATH, 
        chunk_size=50000  # tweak chunk_size as needed
    )
    print("Disease model is ready.")

@app.get("/api/suggest_symptoms")
def suggest_symptoms(prefix: str = ""):
    """
    Returns a list of up to 10 symptoms that start with the given prefix.
    """
    prefix_lower = prefix.lower()
    matches = [s for s in all_symptoms if s.lower().startswith(prefix_lower)]
    return matches[:10]

@app.post("/api/predict_disease")
def get_disease_prediction(req: SymptomsRequest):
    """
    Returns the top diseases for a given list of symptoms,
    along with probabilities.
    """
    user_symptoms = req.symptoms
    predictions = predict_disease(
        user_symptoms, disease_counts, symptom_counts, total_rows, top_n=5
    )
    # Format response
    # predictions is [(disease, probability), ...]
    result = [
        {"disease": disease, "probability": prob}
        for disease, prob in predictions
    ]
    return JSONResponse(result)

@app.get("/")
def index():
    """
    Return a simple HTML page that uses Tailwind and 
    implements a minimal UI to test the endpoints.
    """
    with open("templates/index.html", "r") as f:
        return HTMLResponse(f.read())

import math

def predict_disease(user_symptoms, disease_counts, symptom_counts, total_rows, top_n=5):
    """
    Given a list of user symptoms, compute the probable diseases 
    using a simplified (Naive Bayes) approach.
    Returns the top_n diseases as a list of (disease, probability).
    """
    results = []
    
    for disease, d_count in disease_counts.items():
        # Prior
        p_d = d_count / total_rows
        # Start with log prior
        log_prob = math.log(p_d)
        
        # For each symptom the user has
        for symptom in user_symptoms:
            # Get P(symptom | disease)
            # If we don't have record, assume some minimal smoothing, e.g. 1e-5
            if symptom in symptom_counts[disease]:
                p_s_given_d = symptom_counts[disease][symptom] / d_count
                # Smoothing if necessary
                p_s_given_d = max(p_s_given_d, 1e-5)
            else:
                p_s_given_d = 1e-5
            
            log_prob += math.log(p_s_given_d)
        
        # We have a log probability, store it
        results.append((disease, log_prob))
    
    # Convert log probabilities to normalized probabilities
    # 1. exponentiate
    max_log = max(log_prob for _, log_prob in results)
    # to avoid overflow, subtract max first
    exps = [(d, math.exp(log_p - max_log)) for (d, log_p) in results]
    # 2. sum
    sum_exps = sum(v for _, v in exps)
    # 3. normalize
    final = [(d, v / sum_exps) for (d, v) in exps]
    # sort
    final.sort(key=lambda x: x[1], reverse=True)
    
    return final[:top_n]

import pandas as pd
from collections import defaultdict

def build_disease_statistics(csv_path: str, chunk_size=50000):
    """
    Reads the CSV in chunks and returns aggregated counts needed
    for a naive Bayes-like classifier.
    """
    # disease_counts[disease] = total rows for disease
    disease_counts = defaultdict(int)
    # symptom_counts[disease][symptom] = number of '1' for that disease
    symptom_counts = defaultdict(lambda: defaultdict(int))
    
    # We store a set of all symptoms for convenience
    all_symptoms = set()
    
    # We'll read in chunks for memory efficiency
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        # Let's assume the first column is "DiseaseName"
        # The remaining columns are the symptom booleans
        disease_col = chunk.columns[0]
        symptom_cols = chunk.columns[1:]
        for symptom in symptom_cols:
            all_symptoms.add(symptom)

        for _, row in chunk.iterrows():
            disease = row[disease_col]
            disease_counts[disease] += 1
            
            # For each symptom, if it's 1, increment counter
            for symptom in symptom_cols:
                if row[symptom] == 1:
                    symptom_counts[disease][symptom] += 1

    # Convert defaultdicts to normal dicts for convenience
    disease_counts = dict(disease_counts)
    symptom_counts = {d: dict(symptom_counts[d]) for d in symptom_counts}
    
    # total number of rows
    total_rows = sum(disease_counts.values())
    
    return disease_counts, symptom_counts, all_symptoms, total_rows

