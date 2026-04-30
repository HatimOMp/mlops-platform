"""
Generates a synthetic genomic dataset simulating the Ziwig Biotech
use case — binary classification of disease vs healthy samples.
"""
import numpy as np
import pandas as pd
import os
from config import RANDOM_STATE

np.random.seed(RANDOM_STATE)

def generate_dataset(n_samples=1000, n_features=50):
    """
    Generate synthetic genomic expression data.
    Features simulate gene expression levels.
    Target: 0 = healthy, 1 = disease
    """
    # Healthy samples
    n_healthy = n_samples // 2
    healthy = np.random.normal(loc=0.0, scale=1.0, size=(n_healthy, n_features))

    # Disease samples — slightly different distribution
    n_disease = n_samples - n_healthy
    disease = np.random.normal(loc=0.5, scale=1.2, size=(n_disease, n_features))

    # Add some informative features with stronger signal
    for i in range(10):
        healthy[:, i] = np.random.normal(loc=-1.0, scale=0.5, size=n_healthy)
        disease[:, i] = np.random.normal(loc=1.0, scale=0.5, size=n_disease)

    # Combine
    X = np.vstack([healthy, disease])
    y = np.array([0] * n_healthy + [1] * n_disease)

    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]

    # Create DataFrame
    feature_names = [f"gene_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(n_samples=1000, n_features=50)
    df.to_csv("data/dataset.csv", index=False)
    print(f"Dataset generated: {df.shape}")
    print(f"Class distribution:\n{df['target'].value_counts()}")