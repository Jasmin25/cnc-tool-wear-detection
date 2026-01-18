"""
Data loading and preprocessing module for CNC Tool Wear Detection.
Handles data ingestion and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import os


# Define sensor feature groups
CF_FEATURES = [f"CF_Feature_{i}" for i in range(1, 6)]
VIB_FEATURES = [f"Vib_Feature_{i}" for i in range(1, 6)]
AE_FEATURES = [f"AE_Feature_{i}" for i in range(1, 6)]
SENSOR_FEATURES = CF_FEATURES + VIB_FEATURES + AE_FEATURES


def get_data_path() -> str:
    """Get the path to the data file."""
    # Try multiple possible locations
    possible_paths = [
        "data/tool_wear_dataset.csv",
        "../data/tool_wear_dataset.csv",
        os.path.join(os.path.dirname(__file__), "..", "data", "tool_wear_dataset.csv"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Default path
    return "data/tool_wear_dataset.csv"


def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the CNC tool wear dataset from CSV.

    Args:
        filepath: Path to the CSV file. If None, uses default path.

    Returns:
        DataFrame with the loaded data.
    """
    if filepath is None:
        filepath = get_data_path()

    df = pd.read_csv(filepath)

    # Add cycle index for time-series visualization
    df["cycle_index"] = range(len(df))

    return df


def preprocess_data(
    df: pd.DataFrame, fit_scaler: bool = True, scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Preprocess the sensor data for anomaly detection.

    Args:
        df: DataFrame containing the sensor features.
        fit_scaler: Whether to fit the scaler (True for training) or just transform.
        scaler: Pre-fitted scaler to use. If None and fit_scaler=False, raises error.

    Returns:
        Tuple of (scaled_features, scaler)
    """
    # Extract sensor features
    X = df[SENSOR_FEATURES].values

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit_scaler=False")
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


def get_healthy_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to only include healthy (normal) cycles.

    Args:
        df: Full DataFrame

    Returns:
        DataFrame with only Healthy wear class samples
    """
    return df[df["Wear_Class"] == "Healthy"].copy()


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate the loaded dataset and return summary statistics.

    Args:
        df: Loaded DataFrame

    Returns:
        Dictionary with validation results
    """
    results = {
        "total_samples": len(df),
        "n_features": len(SENSOR_FEATURES),
        "missing_values": df[SENSOR_FEATURES].isnull().sum().sum(),
        "wear_class_distribution": df["Wear_Class"].value_counts().to_dict(),
        "vb_mm_range": (df["VB_mm"].min(), df["VB_mm"].max()),
        "is_valid": True,
    }

    # Check for required columns
    required_cols = SENSOR_FEATURES + ["VB_mm", "Wear_Class"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        results["is_valid"] = False
        results["missing_columns"] = missing_cols

    return results
