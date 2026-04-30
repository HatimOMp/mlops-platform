"""
Training script with MLflow experiment tracking.
Trains multiple models and logs metrics, parameters and artifacts.
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from config import (
    MLFLOW_TRACKING_URI, EXPERIMENT_NAME,
    RANDOM_STATE, TEST_SIZE, DATA_PATH, MODEL_PATH
)

# ── Setup MLflow ─────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Load data ────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

# ── Plot confusion matrix ────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Healthy", "Disease"],
        yticklabels=["Healthy", "Disease"],
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    path = f"confusion_matrix_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# ── Train one model ──────────────────────────────────────────────────
def train_model(model_name, model, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # Build pipeline with scaler
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        # Cross validation
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=5, scoring="roc_auc"
        )
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] \
            if hasattr(model, "predict_proba") else None

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)

        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Log confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # Log model
        mlflow.sklearn.log_model(pipeline, "model")

        # Log metrics as JSON artifact
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact("metrics.json")
        os.remove("metrics.json")

        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return metrics, mlflow.active_run().info.run_id

# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Define models to compare
    models = [
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=RANDOM_STATE
            ),
            {"n_estimators": 100, "max_depth": 10}
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=RANDOM_STATE
            ),
            {"n_estimators": 100, "learning_rate": 0.1}
        ),
        (
            "LogisticRegression",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=RANDOM_STATE
            ),
            {"C": 1.0, "max_iter": 1000}
        ),
        (
            "SVM",
            SVC(
                C=1.0,
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE
            ),
            {"C": 1.0, "kernel": "rbf"}
        ),
    ]

    # Train all models
    results = []
    for model_name, model, params in models:
        print(f"\nTraining {model_name}...")
        metrics, run_id = train_model(
            model_name, model, params,
            X_train, X_test, y_train, y_test
        )
        results.append({
            "model": model_name,
            "run_id": run_id,
            **metrics
        })

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY — All models ranked by ROC-AUC:")
    results_sorted = sorted(
        results,
        key=lambda x: x.get("roc_auc", 0),
        reverse=True
    )
    for r in results_sorted:
        print(
            f"  {r['model']:<25} "
            f"ROC-AUC: {r.get('roc_auc', 0):.4f} | "
            f"F1: {r.get('f1', 0):.4f} | "
            f"Accuracy: {r.get('accuracy', 0):.4f}"
        )

    print(f"\n✅ Best model: {results_sorted[0]['model']}")
    print(f"Run MLflow UI with: mlflow ui")