"""
Streamlit dashboard for monitoring MLflow experiments
and testing the FastAPI prediction endpoint.
"""
import streamlit as st
import requests
import mlflow
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, API_PORT

st.set_page_config(
    page_title="ML Platform Dashboard",
    page_icon="🧬",
    layout="wide"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
API_URL = f"http://localhost:{API_PORT}"

st.title("🧬 ML Platform Dashboard")
st.markdown(
    "Monitor MLflow experiments, compare models "
    "and test the prediction API in real time."
)

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Experiment Tracking",
    "🔮 Live Predictions",
    "🏗️ Architecture"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — MLflow experiment results
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 MLflow Experiment Results")

    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

        if not experiment:
            st.warning("No experiments found. Run train.py first.")
        else:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.roc_auc DESC"]
            )

            if runs:
                # Build results dataframe
                rows = []
                for run in runs:
                    rows.append({
                        "Model": run.data.params.get("model_name", "unknown"),
                        "ROC-AUC": round(run.data.metrics.get("roc_auc", 0), 4),
                        "F1 Score": round(run.data.metrics.get("f1", 0), 4),
                        "Accuracy": round(run.data.metrics.get("accuracy", 0), 4),
                        "Precision": round(run.data.metrics.get("precision", 0), 4),
                        "Recall": round(run.data.metrics.get("recall", 0), 4),
                        "CV ROC-AUC": round(run.data.metrics.get("cv_roc_auc_mean", 0), 4),
                        "Run ID": run.info.run_id[:8]
                    })

                df = pd.DataFrame(rows)

                # Highlight best model
                st.markdown("### 🏆 Model Comparison")
                st.dataframe(
                    df.style.highlight_max(
                        subset=["ROC-AUC", "F1 Score", "Accuracy"],
                        color="#d4edda"
                    ),
                    use_container_width=True
                )

                # Bar chart
                st.markdown("### 📈 Metrics Comparison")
                metrics_to_plot = ["ROC-AUC", "F1 Score", "Accuracy", "Precision", "Recall"]
                fig = go.Figure()
                for metric in metrics_to_plot:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=df["Model"],
                        y=df[metric]
                    ))
                fig.update_layout(
                    barmode="group",
                    title="Model Performance Comparison",
                    yaxis_range=[0, 1],
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                # Best model highlight
                best = df.iloc[0]
                st.success(
                    f"🏆 Best model: **{best['Model']}** "
                    f"— ROC-AUC: {best['ROC-AUC']} "
                    f"| F1: {best['F1 Score']} "
                    f"| Accuracy: {best['Accuracy']}"
                )

    except Exception as e:
        st.error(f"Error loading experiments: {e}")

# ════════════════════════════════════════════════════════════════════
# TAB 2 — Live API predictions
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔮 Live Predictions via REST API")

    # API health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        if health["model_loaded"]:
            st.success(
                f"✅ API online — Model: **{health['model_name']}** "
                f"| ROC-AUC: {health['roc_auc']:.4f}"
            )
        else:
            st.warning("⚠️ API online but model not loaded. Run train.py first.")
    except:
        st.error("❌ API offline. Run: `uvicorn api:app --reload` in a terminal.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎲 Random Sample Prediction")
        st.markdown("Generate a random genomic sample and predict.")

        sample_type = st.radio(
            "Sample type",
            ["Random", "Simulate Healthy", "Simulate Disease"]
        )

        if st.button("Generate & Predict", type="primary"):
            np.random.seed(None)

            if sample_type == "Simulate Healthy":
                features = np.random.normal(-0.5, 0.8, 50).tolist()
                for i in range(10):
                    features[i] = np.random.normal(-1.0, 0.5)
            elif sample_type == "Simulate Disease":
                features = np.random.normal(0.5, 1.2, 50).tolist()
                for i in range(10):
                    features[i] = np.random.normal(1.0, 0.5)
            else:
                features = np.random.normal(0, 1, 50).tolist()

            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"features": features},
                    timeout=10
                )
                result = response.json()

                st.markdown(f"### Prediction: {'🔴 Disease' if result['prediction'] == 1 else '🟢 Healthy'}")
                st.metric("Confidence level", result["confidence"])

                col_a, col_b = st.columns(2)
                col_a.metric(
                    "P(Disease)",
                    f"{result['probability_disease']:.1%}"
                )
                col_b.metric(
                    "P(Healthy)",
                    f"{result['probability_healthy']:.1%}"
                )

                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["probability_disease"] * 100,
                    title={"text": "Disease Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "red" if result["prediction"] == 1 else "green"},
                        "steps": [
                            {"range": [0, 40], "color": "#d4edda"},
                            {"range": [40, 60], "color": "#fff3cd"},
                            {"range": [60, 100], "color": "#f8d7da"}
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with col2:
        st.markdown("### ✏️ Manual Input")
        st.markdown("Enter custom gene expression values.")

        with st.expander("Gene expression inputs (first 10 genes)"):
            manual_features = []
            cols = st.columns(2)
            for i in range(10):
                with cols[i % 2]:
                    val = st.number_input(
                        f"gene_{i+1}",
                        value=0.0,
                        step=0.1,
                        format="%.2f"
                    )
                    manual_features.append(val)

            # Fill remaining 40 features with zeros
            manual_features.extend([0.0] * 40)

        if st.button("Predict from manual input"):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"features": manual_features},
                    timeout=10
                )
                result = response.json()
                st.markdown(
                    f"**Result:** {'🔴 Disease' if result['prediction'] == 1 else '🟢 Healthy'} "
                    f"({result['probability_disease']:.1%} disease probability)"
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ════════════════════════════════════════════════════════════════════
# TAB 3 — Architecture
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🏗️ Platform Architecture")
    st.markdown("""┌─────────────────────────────────────────────────────┐
│                   ML PLATFORM                        │
│                                                      │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │ generate_    │    │      train.py            │   │
│  │ data.py      │───▶│  4 models compared:      │   │
│  │              │    │  • Random Forest          │   │
│  │ Synthetic    │    │  • Gradient Boosting      │   │
│  │ genomic data │    │  • Logistic Regression   │   │
│  │ 1000 samples │    │  • SVM                   │   │
│  │ 50 features  │    └────────────┬─────────────┘   │
│  └──────────────┘                 │                  │
│                                   ▼                  │
│                    ┌──────────────────────────┐      │
│                    │      MLflow Tracking     │      │
│                    │  • Parameters logged     │      │
│                    │  • Metrics tracked       │      │
│                    │  • Models versioned      │      │
│                    │  • Artifacts stored      │      │
│                    └────────────┬─────────────┘      │
│                                 │                    │
│              ┌──────────────────┴───────────┐        │
│              ▼                              ▼        │
│  ┌───────────────────┐      ┌───────────────────┐   │
│  │    FastAPI        │      │  Streamlit        │   │
│  │    REST API       │      │  Dashboard        │   │
│  │  /predict         │◀─────│  • Experiment     │   │
│  │  /predict/batch   │      │    comparison     │   │
│  │  /experiments     │      │  • Live testing   │   │
│  │  /health          │      │  • Metrics viz    │   │
│  └───────────────────┘      └───────────────────┘   │
└─────────────────────────────────────────────────────┘""")

    st.markdown("### 🛠️ Tech Stack")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **ML & Tracking**
        - Scikit-learn
        - MLflow
        - NumPy / Pandas
        """)
    with col2:
        st.markdown("""
        **API & Serving**
        - FastAPI
        - Uvicorn
        - Pydantic
        """)
    with col3:
        st.markdown("""
        **Dashboard**
        - Streamlit
        - Plotly
        - Requests
        """)

st.markdown("---")
st.markdown(
    "Built with MLflow, FastAPI & Streamlit · "
    "[GitHub](https://github.com/HatimOMp/mlops-platform)"
)