import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

# Flask setup
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

# Model directory
MODELS_DIR = os.environ.get("MODELS_DIR", "models")

# Lazy-loaded models
loaded_models = {}
feature_maps = {}

SUPPORTED_DISEASES = {
    "heart": "Heart Disease",
    "diabetes": "Diabetes",
    "parkinsons": "Parkinson's Disease",
    "breast_cancer": "Breast Cancer (Diagnostic)"
}

# ----------------------------
# Model Loader
# ----------------------------
def load_model_safe(key):
    if key in loaded_models:
        return loaded_models[key], feature_maps[key]
    model_path = os.path.join(MODELS_DIR, f"{key}_model.joblib")
    feats_path = os.path.join(MODELS_DIR, f"{key}_features.json")
    if not os.path.exists(model_path) or not os.path.exists(feats_path):
        return None, None
    model = joblib.load(model_path)
    with open(feats_path, "r") as f:
        feats = json.load(f)
    loaded_models[key] = model
    feature_maps[key] = feats
    return model, feats

# ----------------------------
# Home Page
# ----------------------------
@app.route("/", methods=["GET"])
def index():
    availability = {}
    for k in SUPPORTED_DISEASES.keys():
        model, feats = load_model_safe(k)
        availability[k] = (model is not None)
    return render_template("index.html", diseases=SUPPORTED_DISEASES, availability=availability)

# ----------------------------
# Predict from Manual Form
# ----------------------------
def _predict_from_form(disease_key, form):
    model, feats = load_model_safe(disease_key)
    if model is None:
        raise RuntimeError(f"Model for {disease_key} not found. Please train first.")

    # Handle both dict and list feature formats
    if isinstance(feats, dict) and "feature_order" in feats:
        feature_order = feats["feature_order"]
    elif isinstance(feats, list):
        feature_order = feats
    else:
        raise ValueError(f"Invalid feature file format for {disease_key}: {type(feats)}")

    # Build a row in correct order
    row = []
    for col in feature_order:
        val = form.get(col)
        if val is None or val == "":
            raise ValueError(f"Missing value for '{col}'.")
        try:
            row.append(float(val))
        except ValueError:
            raise ValueError(f"Invalid numeric value for '{col}': {val}")

    X = np.array([row])
    pred = model.predict(X)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = float(model.predict_proba(X)[0][int(pred)])
        except Exception:
            proba = None

    return int(pred), proba

# ----------------------------
# Predict from CSV Upload
# ----------------------------
def _predict_from_csv(disease_key, file_storage):
    model, feats = load_model_safe(disease_key)
    if model is None:
        raise RuntimeError(f"Model for {disease_key} not found. Please train first.")

    # Handle both dict and list feature formats
    if isinstance(feats, dict) and "feature_order" in feats:
        needed = feats["feature_order"]
    elif isinstance(feats, list):
        needed = feats
    else:
        raise ValueError(f"Invalid feature file format for {disease_key}: {type(feats)}")

    # Read CSV safely
    try:
        df = pd.read_csv(file_storage)
    except Exception:
        file_storage.seek(0)
        df = pd.read_csv(file_storage, sep=';')

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Invalid CSV format or empty file.")

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Expected columns: {needed}")

    X = df[needed].values
    preds = model.predict(X)
    return preds

# ----------------------------
# Main Prediction Route
# ----------------------------
@app.route("/predict/<disease_key>", methods=["POST"])
def predict(disease_key):
    if disease_key not in SUPPORTED_DISEASES:
        flash("Unknown disease type.", "danger")
        return redirect(url_for("index"))

    try:
        mode = request.form.get("mode", "form")

        if mode == "form":
            pred, proba = _predict_from_form(disease_key, request.form)
            label = "Positive / At Risk" if pred == 1 else "Negative / Low Risk"
            msg = f"Prediction for {SUPPORTED_DISEASES[disease_key]}: {label}"
            if proba is not None:
                msg += f" (confidence ~ {proba:.2f})"
            flash(msg, "success" if pred == 1 else "info")
            return redirect(url_for("index"))

        else:
            # CSV upload
            file = request.files.get("csvfile")
            if not file:
                raise ValueError("Please upload a CSV file.")
            preds = _predict_from_csv(disease_key, file)
            positives = int((preds == 1).sum())
            total = len(preds)
            msg = f"Batch prediction done for {SUPPORTED_DISEASES[disease_key]}: {positives}/{total} predicted Positive."
            flash(msg, "info")
            return redirect(url_for("index"))

    except Exception as e:
        flash(str(e), "danger")
        return redirect(url_for("index"))

# ----------------------------
# Run Flask App
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
