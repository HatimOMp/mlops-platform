"""
FastAPI REST API for model serving.
Loads the best MLflow model and serves predictions.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from typing import List, Optional
import os
from config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME

# ── App setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="ML Platform API",
    description="REST API for disease classification model serving",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load best model from MLflow ───────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_best_model():
    """Load the best model from MLflow based on ROC-AUC."""
    client = mlflow.tracking.MlflowClient()

    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            return None, None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.roc_auc DESC"],
            max_results=1
        )

        if not runs:
            return None, None

        best_run = runs[0]
        model_uri = f"runs:/{best_run.info.run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        return model, best_run

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

model, best_run = load_best_model()

# ── Request / Response schemas ────────────────────────────────────────
class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description="Gene expression values (50 features)",
        min_items=50,
        max_items=50
    )

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_disease: float
    probability_healthy: float
    model_name: str
    confidence: str

class BatchPredictionRequest(BaseModel):
    samples: List[List[float]] = Field(
        ...,
        description="List of samples, each with 50 features"
    )

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str]
    roc_auc: Optional[float]
    experiment: str

# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
def root():
    return {
        "message": "ML Platform API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """Check API and model status."""
    if model is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_name=None,
            roc_auc=None,
            experiment=EXPERIMENT_NAME
        )

    roc_auc = best_run.data.metrics.get("roc_auc")
    model_name = best_run.data.params.get("model_name", "unknown")

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=model_name,
        roc_auc=roc_auc,
        experiment=EXPERIMENT_NAME
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(request: PredictionRequest):
    """Predict disease probability for a single sample."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first."
        )

    try:
        X = np.array(request.features).reshape(1, -1)
        feature_names = [f"gene_{i+1}" for i in range(50)]
        X_df = pd.DataFrame(X, columns=feature_names)

        prediction = int(model.predict(X_df)[0])
        probabilities = model.predict_proba(X_df)[0]

        prob_disease = float(probabilities[1])
        prob_healthy = float(probabilities[0])
        confidence = "High" if max(prob_disease, prob_healthy) > 0.8 else "Medium" \
            if max(prob_disease, prob_healthy) > 0.6 else "Low"

        return PredictionResponse(
            prediction=prediction,
            label="Disease" if prediction == 1 else "Healthy",
            probability_disease=prob_disease,
            probability_healthy=prob_healthy,
            model_name=best_run.data.params.get("model_name", "unknown"),
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Predictions"])
def predict_batch(request: BatchPredictionRequest):
    """Predict disease probability for multiple samples."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first."
        )

    try:
        feature_names = [f"gene_{i+1}" for i in range(50)]
        X_df = pd.DataFrame(request.samples, columns=feature_names)

        predictions = model.predict(X_df).tolist()
        probabilities = model.predict_proba(X_df).tolist()

        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                "prediction": int(pred),
                "label": "Disease" if pred == 1 else "Healthy",
                "probability_disease": prob[1],
                "probability_healthy": prob[0]
            })

        return {
            "results": results,
            "total": len(results),
            "disease_count": sum(1 for r in results if r["prediction"] == 1),
            "healthy_count": sum(1 for r in results if r["prediction"] == 0)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experiments", tags=["MLflow"])
def get_experiments():
    """Get all MLflow experiment runs."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

        if not experiment:
            return {"runs": []}

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.roc_auc DESC"]
        )

        return {
            "experiment": EXPERIMENT_NAME,
            "runs": [
                {
                    "run_id": r.info.run_id,
                    "model_name": r.data.params.get("model_name"),
                    "metrics": r.data.metrics,
                    "params": r.data.params,
                    "status": r.info.status
                }
                for r in runs
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))