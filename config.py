# ── Central configuration for the ML platform ───────────────────────
import os

# MLflow
MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "disease-classification"

# Model
RANDOM_STATE = 42
TEST_SIZE = 0.2

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# Paths
MODEL_PATH = "best_model"
DATA_PATH = "data/dataset.csv"