"""SHAP explainability for both cascade stages.

Usage:
    python -m src.explain
"""
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

from src.features import FEATURE_COLUMNS, engineer_features, load_data

df = load_data("data/ai4i2020.csv")
df = engineer_features(df)
X = df[FEATURE_COLUMNS]
y_type = df["Failure_Type"]

_, X_test, _, y_test_type = train_test_split(
    X, y_type, test_size=0.2, stratify=y_type, random_state=42
)

stage1_bundle = joblib.load("stage1_model.joblib")
stage1 = stage1_bundle["model"]

explainer1 = shap.TreeExplainer(stage1)
shap_values1 = explainer1.shap_values(X_test)
if isinstance(shap_values1, list):
    shap_values1 = shap_values1[1]
elif shap_values1.ndim == 3:
    shap_values1 = shap_values1[:, :, 1]

plt.figure()
shap.summary_plot(shap_values1, X_test, show=False)
plt.title("SHAP - Stage 1 (failure detection)")
plt.tight_layout()
plt.savefig("shap_stage1.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved shap_stage1.png")

fail_mask = y_test_type != "No Failure"
X_test_fail = X_test[fail_mask]

stage2 = joblib.load("stage2_model.joblib")
explainer2 = shap.TreeExplainer(stage2)
shap_values2 = explainer2.shap_values(X_test_fail)

if not isinstance(shap_values2, list):
    n_classes = shap_values2.shape[2]
    shap_values2 = [shap_values2[:, :, i] for i in range(n_classes)]

plt.figure()
shap.summary_plot(shap_values2, X_test_fail, class_names=list(stage2.classes_), show=False)
plt.title("SHAP - Stage 2 (failure type)")
plt.tight_layout()
plt.savefig("shap_stage2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved shap_stage2.png")