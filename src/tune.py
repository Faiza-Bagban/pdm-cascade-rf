"""Compare RandomForest, XGBoost, LightGBM with Optuna hyperparameter search.
SMOTE applied inside CV pipeline (not before split) -- avoids leakage.

Usage:
    python -m src.tune
"""
import re

import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.features import FEATURE_COLUMNS, engineer_features, load_data

df = load_data("data/ai4i2020.csv")
df = engineer_features(df)

X = df[FEATURE_COLUMNS]
X = X.rename(columns=lambda c: re.sub(r"[^\w]+", "_", c))  # LightGBM rejects [ ] etc.
y = df["Failure_Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

min_class = np.min(np.bincount(y_train_enc))
k_neighbors = max(1, min(5, min_class - 1))


def build_model(trial, model_type):
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 100, 500),
            max_depth=trial.suggest_int("rf_max_depth", 5, 30),
            min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 10),
            class_weight="balanced",
            random_state=42,
        )
    if model_type == "xgb":
        return XGBClassifier(
            n_estimators=trial.suggest_int("xgb_n_estimators", 100, 500),
            max_depth=trial.suggest_int("xgb_max_depth", 3, 12),
            learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
            eval_metric="mlogloss",
            random_state=42,
        )
    return LGBMClassifier(
        n_estimators=trial.suggest_int("lgbm_n_estimators", 100, 500),
        max_depth=trial.suggest_int("lgbm_max_depth", 3, 12),
        learning_rate=trial.suggest_float("lgbm_lr", 0.01, 0.3, log=True),
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )


def objective(trial):
    model_type = trial.suggest_categorical("model_type", ["rf", "xgb", "lgbm"])
    model = build_model(trial, model_type)

    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)),
        ("clf", model),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline, X_train, y_train_enc, cv=cv, scoring="f1_macro", n_jobs=-1
    )
    return scores.mean()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    print("\nBest macro F1 (CV):", study.best_value)
    print("Best params:", study.best_params)