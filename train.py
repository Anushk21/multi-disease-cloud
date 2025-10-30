"""
Train models for multiple diseases and save them to ./models as joblib files.

Datasets:
- Diabetes (Pima): from GitHub (Plotly datasets)
- Heart Disease: uses local heart.csv (you must have data/heart.csv)
- Breast Cancer: from scikit-learn built-in dataset
- Parkinson's: from UCI repository

Run:
    python train.py

This will produce:
    models/diabetes_model.joblib, models/diabetes_features.json
    models/heart_model.joblib, models/heart_features.json
    models/breast_cancer_model.joblib, models/breast_cancer_features.json
    models/parkinsons_model.joblib, models/parkinsons_features.json
"""

import os
import json
import joblib
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# -------------------- Helper --------------------
def try_read_csv(urls):
    """Try reading multiple URLs until one works."""
    last_err = None
    for url in urls:
        try:
            return pd.read_csv(url)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err


# -------------------- Diabetes --------------------
def train_diabetes():
    print("\n=== Training Diabetes ===")
    urls = [
        "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        "https://raw.githubusercontent.com/selva86/datasets/master/PimaIndiansDiabetes.csv",
    ]
    df = try_read_csv(urls)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    target_col = "outcome" if "outcome" in df.columns else "diabetes"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    acc = accuracy_score(yte, pipe.predict(Xte))
    print(f"Diabetes test accuracy: {acc:.3f}")

    joblib.dump(pipe, os.path.join(MODELS_DIR, "diabetes_model.joblib"))
    with open(os.path.join(MODELS_DIR, "diabetes_features.json"), "w") as f:
        json.dump({"feature_order": list(X.columns)}, f, indent=2)


# -------------------- Heart Disease --------------------
def train_heart():
    print("\n=== Training Heart Disease ===")

    # Use local dataset
    csv_path = os.path.join("data", "heart.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Heart dataset not found at {csv_path}. Please place heart.csv inside a 'data' folder."
        )

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    target_col = "target"
    if target_col not in df.columns:
        target_candidates = [c for c in df.columns if c in ("num", "disease", "status")]
        if not target_candidates:
            raise ValueError("Could not find target column in heart dataset.")
        target_col = target_candidates[0]

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    acc = accuracy_score(yte, pipe.predict(Xte))
    print(f"Heart Disease test accuracy: {acc:.3f}")

    joblib.dump(pipe, os.path.join(MODELS_DIR, "heart_model.joblib"))
    with open(os.path.join(MODELS_DIR, "heart_features.json"), "w") as f:
        json.dump({"feature_order": list(X.columns)}, f, indent=2)


# -------------------- Breast Cancer --------------------
def train_breast_cancer():
    print("\n=== Training Breast Cancer (Wisconsin Diagnostic) ===")
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    acc = accuracy_score(yte, pipe.predict(Xte))
    print(f"Breast Cancer test accuracy: {acc:.3f}")

    joblib.dump(pipe, os.path.join(MODELS_DIR, "breast_cancer_model.joblib"))
    with open(os.path.join(MODELS_DIR, "breast_cancer_features.json"), "w") as f:
        json.dump({"feature_order": list(X.columns)}, f, indent=2)


# -------------------- Parkinson's --------------------
def train_parkinsons():
    print("\n=== Training Parkinson's Disease ===")
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
        "https://raw.githubusercontent.com/dsrscientist/DSData/master/parkisons.data"
    ]
    df = try_read_csv(urls)

    if df.shape[1] == 1:  # fallback if delimiter issue
        df = pd.read_csv(urls[0], sep=";")

    df.columns = [c.strip().lower() for c in df.columns]
    if "status" not in df.columns:
        raise ValueError("Expected 'status' column in Parkinson's dataset.")

    X = df.drop(columns=["status", "name"]) if "name" in df.columns else df.drop(columns=["status"])
    y = df["status"].astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=400, random_state=42))
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    acc = accuracy_score(yte, pipe.predict(Xte))
    print(f"Parkinson's test accuracy: {acc:.3f}")

    joblib.dump(pipe, os.path.join(MODELS_DIR, "parkinsons_model.joblib"))
    with open(os.path.join(MODELS_DIR, "parkinsons_features.json"), "w") as f:
        json.dump({"feature_order": list(X.columns)}, f, indent=2)


# -------------------- MAIN --------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_diabetes()
    train_heart()
    train_breast_cancer()
    train_parkinsons()
    print("\nAll models trained and saved to ./models ✅")

