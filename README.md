# Multi‑Disease Prediction (Cloud‑Ready)

Flask + scikit‑learn app that predicts **Heart Disease, Diabetes, Parkinson's, and Breast Cancer**.  
Deploy on **Google Cloud Run** (free) or run locally.

---

## 1) Setup (Local)

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train models
```bash
python train.py
```

> This downloads datasets (public mirrors / built‑ins), trains models, and saves them into `./models`.

### Run locally
```bash
python app.py
```
App runs at http://localhost:8080

---

## 2) Deploy to Google Cloud Run (Recommended & Free)

1. Create a Google Cloud project and enable Cloud Run + Cloud Build.
2. Authenticate:
```bash
gcloud auth login
gcloud config set project <your-project-id>
```
3. Build and push container:
```bash
gcloud builds submit --tag gcr.io/<your-project-id>/disease-app
```
4. Deploy:
```bash
gcloud run deploy disease-app --image gcr.io/<your-project-id>/disease-app --platform managed --allow-unauthenticated --region asia-south1
```
5. Open the URL Cloud Run gives you.

> Tip: You can also train locally, then upload the `models/` folder to your repo/container so the app has models at start.  
> Or mount from a bucket (advanced): put models in a GCS bucket and download on startup.

---

## 3) Using the App

- **Heart** and **Diabetes**: quick forms for single prediction + batch CSV upload.
- **Parkinson's** and **Breast Cancer**: CSV upload recommended (many features).
- Batch CSV must contain the **exact training feature columns** (order doesn't matter; app reorders).

To see expected columns: open files under `models/*_features.json` after training.

---

## 4) Project Notes

- Models are simple baselines (Logistic Regression / Random Forest) wrapped in scikit‑learn Pipelines with `StandardScaler` where appropriate.
- You can swap in advanced models later (XGBoost, LightGBM, etc.).
- For production, consider adding authentication and HTTPS (Cloud Run provides HTTPS by default).

---

## 5) Troubleshooting

- **Model not found**: run `python train.py` first to create `models/*.joblib` and `*_features.json`.
- **CSV columns mismatch**: open the corresponding `*_features.json` to see the expected columns.
- **Port issues locally**: set `PORT=5000 python app.py` and open http://localhost:5000

---

Made for your Cloud Computing project. Good luck! 🎉
