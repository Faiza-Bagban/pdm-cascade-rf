"""Train predictive maintenance model on AI4I 2020 data (SMOTE + tuned RF).

Usage:
    python -m src.train
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.features import FEATURE_COLUMNS, engineer_features, load_data


def main(data_path="data/ai4i2020.csv", model_out="model.joblib"):
    df = load_data(data_path)
    df = engineer_features(df)

    X = df[FEATURE_COLUMNS]
    y = df["Failure_Type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # SMOTE only on training data -- never touch test set, that's leakage.
    min_class_count = y_train.value_counts().min()
    k = max(1, min(5, min_class_count - 1))
    print(f"Smallest training class has {min_class_count} samples, using k_neighbors={k}")

    smote = SMOTE(random_state=42, k_neighbors=k)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("Before SMOTE:", y_train.value_counts().to_dict())
    print("After SMOTE:", y_train_res.value_counts().to_dict())

    # Tuned via Optuna (40 trials, 5-fold CV, macro F1) -- RF beat XGBoost and LightGBM.
    model = RandomForestClassifier(
        n_estimators=171,
        max_depth=15,
        min_samples_split=6,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Failure Type (tuned RF + SMOTE)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    importances.plot(kind="barh", figsize=(7, 5), title="Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    joblib.dump(model, model_out)
    print(f"Model saved -> {model_out}")


if __name__ == "__main__":
    main()