# backend/utils/anomaly.py
from sklearn.ensemble import IsolationForest
import joblib
import numpy as np
import os

MODEL_PATH = "data/models/anomaly_iforest.pkl"

def train_iforest(X, contamination=0.01):
    iforest = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iforest.fit(X)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(iforest, MODEL_PATH)
    return iforest

def score_samples(X):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Train an IsolationForest model first.")
    iforest = joblib.load(MODEL_PATH)
    scores = iforest.decision_function(X)  # higher means more normal
    anomalies = iforest.predict(X)         # -1 anomaly, +1 normal
    return scores, anomalies
