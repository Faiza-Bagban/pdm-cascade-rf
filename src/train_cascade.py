"""Two-stage cascade for predictive maintenance -- v3, adds cascade-specific
confusion matrix and feature importance plots (v2 fixed train-serving skew).

Stage 1: binary failure detector, threshold-tuned for high recall.
Stage 2: retrained on Stage 1's OUT-OF-FOLD predicted-positive rows (not ground
         truth failures) -- exposes Stage 2 to Stage 1's real false-alarm
         distribution during training.

Usage:
    python -m src.train_cascade
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import cross_val_predict, train_test_split

from src.features import FEATURE_COLUMNS, engineer_features, load_data

RF_PARAMS = dict(n_estimators=171, max_depth=15, min_samples_split=6, random_state=42)
LABEL_ORDER = ["No Failure", "HDF", "OSF", "PWF", "RNF", "TWF"]


def pick_threshold(y_val, proba_val, min_recall=0.90):
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    precision, recall = precision[:-1], recall[:-1]
    ok = recall >= min_recall
    best_idx = np.argmax(np.where(ok, precision, -1)) if ok.any() else np.argmax(recall)
    return thresholds[best_idx], precision[best_idx], recall[best_idx]


def main():
    df = load_data("data/ai4i2020.csv")
    df = engineer_features(df)

    X = df[FEATURE_COLUMNS]
    y_type = df["Failure_Type"]

    X_train, X_test, y_train_type, y_test_type = train_test_split(
        X, y_type, test_size=0.2, stratify=y_type, random_state=42
    )
    y_train_bin = (y_train_type != "No Failure").astype(int)
    y_test_bin = (y_test_type != "No Failure").astype(int)

    # ---------- Stage 1: binary failure detector ----------
    X_tr, X_val, y_tr_bin, y_val_bin = train_test_split(
        X_train, y_train_bin, test_size=0.25, stratify=y_train_bin, random_state=42
    )
    stage1_probe = RandomForestClassifier(class_weight="balanced", **RF_PARAMS)
    stage1_probe.fit(X_tr, y_tr_bin)
    val_proba = stage1_probe.predict_proba(X_val)[:, 1]
    threshold, val_prec, val_rec = pick_threshold(y_val_bin, val_proba, min_recall=0.90)
    print(f"Stage 1 threshold: {threshold:.3f} (val precision={val_prec:.3f}, val recall={val_rec:.3f})")

    stage1 = RandomForestClassifier(class_weight="balanced", **RF_PARAMS)
    stage1.fit(X_train, y_train_bin)

    test_proba = stage1.predict_proba(X_test)[:, 1]
    test_pred_bin = (test_proba >= threshold).astype(int)
    print("\nStage 1 (binary failure detection) on held-out test:")
    print(classification_report(y_test_bin, test_pred_bin, target_names=["No Failure", "Failure"]))

    # ---------- Stage 2: retrain on Stage 1's OUT-OF-FOLD positives ----------
    oof_estimator = RandomForestClassifier(class_weight="balanced", **RF_PARAMS)
    oof_proba = cross_val_predict(
        oof_estimator, X_train, y_train_bin, cv=5, method="predict_proba", n_jobs=-1
    )[:, 1]
    oof_pred_bin = (oof_proba >= threshold).astype(int)

    stage2_mask = oof_pred_bin == 1
    X_stage2 = X_train[stage2_mask]
    y_stage2 = y_train_type[stage2_mask]

    print("\nStage 2 training set (Stage 1's OOF-flagged rows, includes false alarms):")
    print(y_stage2.value_counts().to_dict())

    ros = RandomOverSampler(random_state=42)
    X_stage2_res, y_stage2_res = ros.fit_resample(X_stage2, y_stage2)

    stage2 = RandomForestClassifier(class_weight="balanced", **RF_PARAMS)
    stage2.fit(X_stage2_res, y_stage2_res)

    # ---------- End-to-end cascade evaluation ----------
    stage2_test_pred = stage2.predict(X_test)
    final_pred = np.where(test_pred_bin == 0, "No Failure", stage2_test_pred)

    print("\nEnd-to-end cascade (Stage 1 -> Stage 2 combined) on held-out test:")
    print(classification_report(y_test_type, final_pred, zero_division=0))

    # ---------- Plots: cascade-specific, not stale single-model leftovers ----------
    cm = confusion_matrix(y_test_type, final_pred, labels=LABEL_ORDER)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - End-to-End Cascade")
    plt.tight_layout()
    plt.savefig("cascade_confusion_matrix.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pd.Series(stage1.feature_importances_, index=FEATURE_COLUMNS).sort_values().plot(
        kind="barh", ax=axes[0], title="Stage 1 feature importance (failure detection)"
    )
    pd.Series(stage2.feature_importances_, index=FEATURE_COLUMNS).sort_values().plot(
        kind="barh", ax=axes[1], title="Stage 2 feature importance (failure type)"
    )
    plt.tight_layout()
    plt.savefig("cascade_feature_importance.png")
    plt.close()
    print("\nSaved cascade_confusion_matrix.png, cascade_feature_importance.png")

    joblib.dump({"model": stage1, "threshold": threshold}, "stage1_model.joblib")
    joblib.dump(stage2, "stage2_model.joblib")
    print("Saved stage1_model.joblib, stage2_model.joblib")


if __name__ == "__main__":
    main()