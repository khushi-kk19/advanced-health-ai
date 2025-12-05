# backend/utils/shap_explain.py
import joblib
import shap
import numpy as np
import os

_model_cache = {}

def load_model(path="data/models/risk.pkl"):
    if path in _model_cache:
        return _model_cache[path]
    mdl = joblib.load(path)
    _model_cache[path] = mdl
    return mdl

def explain(features: dict, model_path="data/models/risk.pkl", feature_order=None):
    model = load_model(model_path)
    if feature_order is None:
        feature_order = sorted(features.keys())
    X = np.array([features[k] for k in feature_order]).reshape(1, -1)
    # if pipeline (scaler + rf)
    try:
        scaler = model.named_steps["scaler"]
        rf = model.named_steps["rf"]
        Xs = scaler.transform(X)
    except Exception:
        Xs = X
        rf = model
    explainer = shap.Explainer(rf, Xs)
    sv = explainer(Xs)
    base = float(sv.base_values[0]) if hasattr(sv, "base_values") else 0.0
    contribs = {feature_order[i]: float(sv.values[0][i]) for i in range(len(feature_order))}
    return {"base_value": base, "contributions": contribs}
