# ⚙️ PdM Cascade RF — Predictive Maintenance AI

Two-stage Random Forest cascade for industrial equipment failure prediction, with SHAP-validated explainability, a Streamlit dashboard, and a FastAPI inference service. Built on the AI4I 2020 industrial sensor dataset.

[![CI](https://github.com/Faiza-Bagban/pdm-cascade-rf/actions/workflows/ci.yml/badge.svg)](https://github.com/Faiza-Bagban/pdm-cascade-rf/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Live demos:**
🖥️ UI: [pdm-cascade-rf.streamlit.app](https://pdm-cascade-rf.streamlit.app/)
🔌 API docs: [pdm-cascade-rf-api.onrender.com/docs](https://pdm-cascade-rf-api.onrender.com/docs)

> Note: the API is on Render's free tier — it spins down when idle, so the first request after inactivity can take ~50s to wake up.

---

## Why a cascade, not one model

A single multiclass model has to learn 6 classes where one class ("No Failure") makes up 96.6% of the data. Every failure type gets drowned out during training regardless of how much SMOTE/class-weighting is applied afterward.

This project splits the problem instead:

1. **Stage 1 — binary failure detector.** Answers "will it fail?" Threshold-tuned to hit ≥90% recall on the failure class, because in maintenance, missing a real failure costs far more than a false alarm.
2. **Stage 2 — failure-type classifier.** Only runs if Stage 1 flags a failure. Trained on Stage 1's *out-of-fold* predicted-positive rows (not just ground-truth failures) — this exposes Stage 2 to Stage 1's real false-alarm distribution, so it learns to reject false alarms instead of blindly assigning a failure type to everything it receives.

This mirrors how real predictive-maintenance systems are architected in industry: alarm first, diagnose second.

## Results (held-out test set)

**Stage 1 — binary failure detection**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No Failure | 1.00 | 0.97 | 0.98 |
| Failure | 0.48 | 0.89 | 0.63 |

**End-to-end cascade — failure type**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| HDF | 0.92 | 1.00 | 0.96 | 23 |
| No Failure | 0.99 | 1.00 | 1.00 | 1930 |
| OSF | 0.89 | 1.00 | 0.94 | 16 |
| PWF | 0.95 | 1.00 | 0.97 | 18 |
| RNF | 0.00 | 0.00 | 0.00 | 4 |
| TWF | 0.00 | 0.00 | 0.00 | 9 |

Weighted F1: **0.99** · Macro F1 (5-fold CV, model selection metric): **0.66**

**On RNF/TWF:** recall is data-floor limited, not a modeling failure — RNF and TWF have only 19 and 46 total occurrences across all 10,000 rows. No amount of tuning manufactures signal that isn't in the data. RNF specifically is defined as *random* in the source dataset, meaning it has no learnable relationship to sensor readings by design.

Model selection used **Optuna** (40 trials, 5-fold stratified CV, macro-F1 objective) comparing Random Forest, XGBoost, and LightGBM — Random Forest won across all trials.

## Explainability (SHAP)

SHAP analysis on both cascade stages confirms the model recovered AI4I2020's actual underlying failure-mode definitions purely from sensor correlations, without ever being given the generative rules:

- `Wear_x_Torque` dominates the OSF and TWF SHAP contributions — matches their true definitions (wear × torque threshold failures)
- `Temp_diff_K` dominates the HDF contribution — matches HDF's true definition (heat dissipation failure)
- `Power_W` dominates the PWF contribution — matches PWF's true definition (power threshold failure)

This validates that predictions reflect real mechanism, not spurious pattern-matching.

## Architecture

```
data/ai4i2020.csv
      │
      ▼
Feature engineering (src/features.py)
  Power_W = Torque × angular velocity
  Temp_diff_K = Process temp − Air temp
  Wear_x_Torque = Tool wear × Torque
      │
      ▼
Stage 1: RandomForest (binary) ── threshold @ 90% recall
      │
      ├─ No Failure ──────────────► Final: No Failure
      │
      └─ Flagged failure
              │
              ▼
      Stage 2: RandomForest (multiclass)
      trained on Stage 1's OOF-flagged rows
              │
              ▼
      Final: {HDF, OSF, PWF, RNF, TWF, or rejected as No Failure}
```

## Tech stack

- **Modeling:** scikit-learn, imbalanced-learn (SMOTE, RandomOverSampler), Optuna, XGBoost, LightGBM
- **Explainability:** SHAP
- **Serving:** FastAPI + Pydantic (request validation, auto docs), Streamlit (dashboard)
- **Ops:** Docker, GitHub Actions CI (pytest smoke tests + Docker build verification)

## Project structure

```
pdm-cascade-rf/
├── app.py                      # Streamlit dashboard (4 tabs: live prediction, model performance, batch, about)
├── api.py                      # FastAPI inference service
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── data/
│   └── ai4i2020.csv
├── src/
│   ├── features.py             # feature engineering
│   ├── train_cascade.py        # two-stage cascade training
│   ├── tune.py                 # Optuna model comparison
│   └── explain.py              # SHAP analysis
├── tests/
│   └── test_smoke.py           # model + API smoke tests
└── .github/workflows/ci.yml    # pytest + Docker build on every push
```

## Run locally

```bash
git clone https://github.com/Faiza-Bagban/pdm-cascade-rf.git
cd pdm-cascade-rf
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt

# Train the cascade (or use the pretrained .joblib files already in the repo)
python -m src.train_cascade

# Run the dashboard
streamlit run app.py

# Run the API (separate terminal)
uvicorn api:app --reload
```

API docs available at `http://localhost:8000/docs` once running.

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (UCI Machine Learning Repository) — 10,000 rows, synthetic, reflects real industrial predictive-maintenance sensor patterns. Licensed CC BY 4.0.

## Limitations

- RNF/TWF recall is capped by sample size (19/46 total occurrences), not model quality
- Dataset is a single-snapshot design (each row is an independent sample, not a time series of one machine over time) — a production system would ingest streaming sensor data with temporal features, which this dataset doesn't support
- Free-tier deployment (Render) has cold-start latency after inactivity

## Author

Faiza Bagban — Final Year AIML

---
*Built as an end-to-end demonstration of the full ML engineering lifecycle: EDA → feature engineering → model comparison/tuning → explainability → serving → containerization → CI → deployment.*
