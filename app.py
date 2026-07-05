import datetime as dt
import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features import FEATURE_COLUMNS, engineer_features

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- styling ---
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2rem;}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px 16px;
}
.verdict-ok {
    background: rgba(46,160,67,0.12);
    border: 1px solid rgba(46,160,67,0.4);
    border-radius: 12px;
    padding: 18px 20px;
    font-size: 1.1rem;
}
.verdict-bad {
    background: rgba(218,54,51,0.12);
    border: 1px solid rgba(218,54,51,0.4);
    border-radius: 12px;
    padding: 18px 20px;
    font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- models ---
@st.cache_resource
def load_models():
    stage1_bundle = joblib.load("stage1_model.joblib")
    stage2 = joblib.load("stage2_model.joblib")
    return stage1_bundle["model"], stage1_bundle["threshold"], stage2


stage1, threshold, stage2 = load_models()

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------- header ---
st.title("⚙️ Predictive Maintenance AI")
st.caption("Two-stage cascade (Random Forest) · SMOTE-balanced · AI4I 2020 industrial dataset")

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Model", "RF Cascade")
col_b.metric("Stage 1 recall target", "≥90%")
col_c.metric("Weighted F1", "0.99")
col_d.metric("Macro F1 (CV)", "0.66")

st.divider()

# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.header("🎛️ Sensor input")
    with st.form("predict_form"):
        air_temp = st.slider("Air temperature [K]", 295.0, 305.0, 300.0)
        process_temp = st.slider("Process temperature [K]", 305.0, 315.0, 310.0)
        rpm = st.slider("Rotational speed [rpm]", 1000, 3000, 1500)
        torque = st.slider("Torque [Nm]", 3.0, 80.0, 40.0)
        tool_wear = st.slider("Tool wear [min]", 0, 260, 100)
        machine_type = st.selectbox("Machine type", ["L", "M", "H"])
        submitted = st.form_submit_button(
            "🔍 Run prediction", use_container_width=True, type="primary"
        )

    st.divider()
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------- tabs ---
tab_predict, tab_perf, tab_batch, tab_about = st.tabs(
    ["🔍 Live prediction", "📊 Model performance", "📁 Batch prediction", "ℹ️ About"]
)

# ============================================================ TAB 1 ===
with tab_predict:
    if submitted:
        power = torque * (rpm * 2 * 3.14159 / 60)
        temp_diff = process_temp - air_temp
        wear_x_torque = tool_wear * torque

        input_row = {
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Power_W": power,
            "Temp_diff_K": temp_diff,
            "Wear_x_Torque": wear_x_torque,
            "Type_H": 1 if machine_type == "H" else 0,
            "Type_L": 1 if machine_type == "L" else 0,
            "Type_M": 1 if machine_type == "M" else 0,
        }
        X = pd.DataFrame([input_row])[FEATURE_COLUMNS]

        fail_proba = float(stage1.predict_proba(X)[0, 1])
        is_failure = fail_proba >= threshold

        final_verdict = "No Failure"
        type_proba, classes = None, None

        if is_failure:
            type_pred = stage2.predict(X)[0]
            type_proba = stage2.predict_proba(X)[0]
            classes = stage2.classes_
            final_verdict = type_pred

        st.session_state.history.insert(0, {
            "time": dt.datetime.now().strftime("%H:%M:%S"),
            "verdict": final_verdict,
            "fail_proba": f"{fail_proba:.1%}",
            "torque": torque,
            "tool_wear": tool_wear,
            "rpm": rpm,
        })

        col_left, col_right = st.columns([1, 1])

        with col_left:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fail_proba * 100,
                number={"suffix": "%"},
                title={"text": "Stage 1 · failure probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#d62728" if is_failure else "#2ca02c"},
                    "steps": [
                        {"range": [0, threshold * 100], "color": "rgba(46,160,67,0.15)"},
                        {"range": [threshold * 100, 100], "color": "rgba(218,54,51,0.15)"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "value": threshold * 100},
                },
            ))
            gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(gauge, use_container_width=True)

        with col_right:
            st.markdown("#### Final verdict")
            if final_verdict == "No Failure":
                st.markdown(
                    '<div class="verdict-ok">✅ <b>No Failure</b><br>'
                    'Machine operating within normal range.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="verdict-bad">⚠️ <b>{final_verdict}</b><br>'
                    'Maintenance recommended — Stage 2 confirmed failure type.</div>',
                    unsafe_allow_html=True,
                )

            if type_proba is not None:
                st.markdown("##### Stage 2 · type breakdown")
                fig = go.Figure(go.Bar(
                    x=type_proba, y=classes, orientation="h",
                    marker_color=["#2ca02c" if c == "No Failure" else "#d62728" for c in classes],
                ))
                fig.update_layout(height=220, margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Probability")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Derived features (fed to model)")
        st.dataframe(
            pd.DataFrame([{
                "Power (W)": round(power, 1),
                "Temp diff (K)": round(temp_diff, 2),
                "Wear × Torque": round(wear_x_torque, 1),
            }]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Set sensor values in the sidebar and click **Run prediction**.")

    if st.session_state.history:
        st.markdown("##### Session history")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download history as CSV",
            hist_df.to_csv(index=False).encode(),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

# ============================================================ TAB 2 ===
with tab_perf:
    st.markdown("#### Cascade performance (held-out test set)")

    stage1_metrics = pd.DataFrame({
        "Class": ["No Failure", "Failure"],
        "Precision": [1.00, 0.48],
        "Recall": [0.97, 0.89],
        "F1": [0.98, 0.63],
        "Support": [1930, 70],
    })
    st.markdown("**Stage 1 — binary failure detector**")
    st.dataframe(stage1_metrics, use_container_width=True, hide_index=True)

    cascade_metrics = pd.DataFrame({
        "Class": ["HDF", "No Failure", "OSF", "PWF", "RNF", "TWF"],
        "Precision": [0.92, 0.99, 0.89, 0.95, 0.00, 0.00],
        "Recall": [1.00, 1.00, 1.00, 1.00, 0.00, 0.00],
        "F1": [0.96, 1.00, 0.94, 0.97, 0.00, 0.00],
        "Support": [23, 1930, 16, 18, 4, 9],
    })
    st.markdown("**End-to-end cascade — failure type**")
    st.dataframe(cascade_metrics, use_container_width=True, hide_index=True)

    st.warning(
        "RNF and TWF recall is data-floor limited: only 19 and 46 total occurrences "
        "exist across all 10,000 rows. No amount of tuning manufactures signal that "
        "isn't in the data — documented as a dataset limitation, not a modeling failure."
    )

    # col1, col2 = st.columns(2)
    # if os.path.exists("confusion_matrix.png"):
    #     col1.image("confusion_matrix.png", caption="Confusion matrix")
    # if os.path.exists("feature_importance.png"):
    #     col2.image("feature_importance.png", caption="Feature importance")

    col1, col2 = st.columns(2)
    if os.path.exists("cascade_confusion_matrix.png"):
        col1.image("cascade_confusion_matrix.png", caption="Confusion matrix (end-to-end cascade)")
    if os.path.exists("cascade_feature_importance.png"):
        col2.image("cascade_feature_importance.png", caption="Feature importance (Stage 1 vs Stage 2)")

# ============================================================ TAB 3 ===
with tab_batch:
    st.markdown("#### Batch prediction from CSV")
    st.caption(
        "Upload a CSV with AI4I2020-style columns: Type, Air temperature [K], "
        "Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        raw.columns = [c.strip() for c in raw.columns]
        try:
            feat_df = engineer_features(raw)
            X_batch = feat_df[FEATURE_COLUMNS]

            batch_proba = stage1.predict_proba(X_batch)[:, 1]
            batch_is_fail = batch_proba >= threshold

            verdicts = ["No Failure"] * len(X_batch)
            if batch_is_fail.any():
                type_preds = stage2.predict(X_batch[batch_is_fail])
                idx = 0
                for i, flagged in enumerate(batch_is_fail):
                    if flagged:
                        verdicts[i] = type_preds[idx]
                        idx += 1

            result = raw.copy()
            result["failure_probability"] = batch_proba.round(4)
            result["predicted_verdict"] = verdicts

            st.success(f"Processed {len(result)} rows.")
            st.dataframe(result, use_container_width=True)
            st.download_button(
                "⬇️ Download predictions",
                result.to_csv(index=False).encode(),
                file_name="batch_predictions.csv",
                mime="text/csv",
            )
        except KeyError as e:
            st.error(f"Missing expected column in uploaded CSV: {e}")

# ============================================================ TAB 4 ===
with tab_about:
    st.markdown("""
#### Architecture

**Stage 1 — binary failure detector**
Random Forest, threshold tuned on validation data to hit ≥90% recall
(missing a real failure costs more than a false alarm).

**Stage 2 — failure-type classifier**
Random Forest trained only on Stage 1's out-of-fold flagged rows
(includes Stage 1's false alarms, not just ground-truth failures — this
lets Stage 2 learn to reject false alarms instead of blindly typing them).

#### Feature engineering
- `Power_W` = Torque × angular velocity (from rpm)
- `Temp_diff_K` = Process temp − Air temp
- `Wear_x_Torque` = Tool wear × Torque

These map directly to how AI4I2020's underlying failure rules are defined —
domain-informed, not arbitrary.

#### Dataset
AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository) — 10,000 rows,
synthetic, reflects real industrial predictive maintenance patterns.

#### Tech stack
Python · scikit-learn · imbalanced-learn · Optuna · SHAP · Streamlit · Plotly

---
Built by Faiza — Final Year AIML
""")