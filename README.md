# 🧬 MLOps Platform — End-to-End ML Pipeline

A production-grade MLOps platform that covers the full machine learning lifecycle — from experiment tracking to model serving via a REST API — built with **MLflow**, **FastAPI** and **Streamlit**.

## 🎯 Project Overview

This project simulates a real-world ML engineering workflow applied to a **disease classification problem** (inspired by genomic data from a real biotech project). The focus is on the **infrastructure and engineering** around the model, not just the model itself.

**The platform covers 4 pillars of MLOps:**
1. **Experiment tracking** — all runs, parameters and metrics logged automatically
2. **Model versioning** — best model selected and stored via MLflow
3. **Model serving** — REST API with FastAPI for real-time and batch predictions
4. **Monitoring dashboard** — Streamlit interface for experiment comparison and live testing

## 🏗️ Architecture
┌─────────────────────────────────────────────────────┐
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
└─────────────────────────────────────────────────────┘

## 📊 Experiment Results

4 models trained and tracked automatically with MLflow:

| Model | ROC-AUC | F1 Score | Accuracy | CV ROC-AUC |
|-------|---------|----------|----------|------------|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SVM | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gradient Boosting | 0.9995 | 0.9754 | 0.9750 | 1.0000 |

> **Note:** Perfect scores are expected on this synthetic dataset which was designed with clear class separation to demonstrate the MLOps pipeline infrastructure.
> The engineering focus is on experiment tracking, model serving and deployment not on the dataset complexity.

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API and model status |
| POST | `/predict` | Single sample prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/experiments` | All MLflow runs |
| GET | `/docs` | Interactive API documentation |

### Example API call

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, -1.2, 0.8, ...]}'
```

### Example response

```json
{
  "prediction": 1,
  "label": "Disease",
  "probability_disease": 0.923,
  "probability_healthy": 0.077,
  "model_name": "RandomForest",
  "confidence": "High"
}
```

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/HatimOMp/mlops-platform
cd mlops-platform
```

**2. Create a virtual environment**
```bash
conda create -n mlops-env python=3.10
conda activate mlops-env
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Generate dataset**
```bash
python generate_data.py
```

**5. Train all models**
```bash
python train.py
```

**6. Start the API (Terminal 1)**
```bash
uvicorn api:app --reload --port 8000
```

**7. Start the dashboard (Terminal 2)**
```bash
streamlit run dashboard.py
```

**8. Open MLflow UI (Terminal 3)**
```bash
mlflow ui
```

Then open:
- Streamlit dashboard → http://localhost:8501
- FastAPI docs → http://localhost:8000/docs
- MLflow UI → http://localhost:5000

## 🐳 Docker

```bash
docker build -t mlops-platform .
docker run -p 8000:8000 mlops-platform
```

## 🗂️ Project Structure
mlops-platform/
│
├── config.py              # Central configuration
├── generate_data.py       # Synthetic genomic dataset generation
├── train.py               # Model training with MLflow tracking
├── api.py                 # FastAPI REST API for model serving
├── dashboard.py           # Streamlit monitoring dashboard
├── requirements.txt       # Dependencies
└── Dockerfile             # Container configuration

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Experiment Tracking | MLflow |
| Model Serving | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| ML Models | Scikit-learn |
| Data Processing | Pandas + NumPy |
| Containerization | Docker |
| API Schema | Pydantic |

## 💡 Key Engineering Decisions

**Why MLflow over manual logging?**
MLflow provides automatic parameter/metric logging, model versioning and a built-in UI — making experiment reproducibility trivial. Every run is tracked with full lineage.

**Why FastAPI over Flask?**
FastAPI generates automatic OpenAPI documentation, supports async natively, uses Pydantic for request validation and is significantly faster than Flask — making it the standard for ML APIs in production.

**Why Pydantic schemas?**
Strong typing on API inputs prevents silent failures. If a request has wrong types or missing fields, the API returns a clear 422 error rather than crashing at prediction time.

**Why a Pipeline (scaler + model)?**
Wrapping the scaler and model in a single sklearn Pipeline guarantees that the same preprocessing applied during training is always applied at inference time — preventing training/serving skew.

## 🔮 Potential Improvements

- **Model Registry** — promote best model to MLflow Model Registry with staging/production environments
- **A/B Testing** — route traffic between model versions to compare real-world performance
- **Monitoring** — add data drift detection with Evidently AI
- **CI/CD** — automate retraining and deployment with GitHub Actions
- **Kubernetes** — orchestrate multiple API replicas for high availability

## 👤 Author

**Hatim Omari** — [LinkedIn](https://www.linkedin.com/in/hatim-omari/) · [GitHub](https://github.com/HatimOMp)