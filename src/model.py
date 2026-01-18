"""
Anomaly Detection Model for CNC Tool Wear.
Uses Isolation Forest trained on healthy data (semi-supervised approach).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import joblib
import os

from .data_processing import SENSOR_FEATURES, preprocess_data, get_healthy_data


class ToolWearAnomalyDetector:
    """
    Anomaly detection model for CNC tool wear.

    Uses Isolation Forest trained on healthy (normal) tool operation data
    to detect anomalous cycles that may indicate tool wear or failure.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        """
        Initialize the anomaly detector.

        Args:
            n_estimators: Number of trees in the Isolation Forest.
            contamination: Expected proportion of anomalies in training data.
            random_state: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False

    def train(self, df: pd.DataFrame, train_on_healthy_only: bool = True) -> dict:
        """
        Train the anomaly detection model.

        Args:
            df: DataFrame with sensor data and Wear_Class column.
            train_on_healthy_only: If True, train only on Healthy cycles (semi-supervised).

        Returns:
            Dictionary with training statistics.
        """
        # Filter to healthy data if semi-supervised
        if train_on_healthy_only:
            train_df = get_healthy_data(df)
        else:
            train_df = df

        # Preprocess and scale features
        X_scaled, self.scaler = preprocess_data(train_df, fit_scaler=True)

        # Create and train Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.is_trained = True

        return {
            "training_samples": len(train_df),
            "n_features": len(SENSOR_FEATURES),
            "train_on_healthy_only": train_on_healthy_only,
            "contamination": self.contamination,
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies on new data.

        Args:
            df: DataFrame with sensor data.

        Returns:
            DataFrame with added anomaly prediction columns.
        """
        if not self.is_trained:
            raise ValueError(
                "Model must be trained before prediction. Call train() first."
            )

        # Preprocess using fitted scaler
        X_scaled, _ = preprocess_data(df, fit_scaler=False, scaler=self.scaler)

        # Get predictions and scores
        predictions = self.model.predict(X_scaled)  # 1 = normal, -1 = anomaly
        scores = self.model.decision_function(X_scaled)  # Lower = more anomalous

        # Add results to a copy of the DataFrame
        result_df = df.copy()
        result_df["anomaly_prediction"] = predictions
        result_df["anomaly_score"] = scores
        result_df["is_anomaly"] = predictions == -1

        return result_df

    def get_anomaly_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics about detected anomalies.

        Args:
            df: DataFrame with anomaly predictions (from predict()).

        Returns:
            Dictionary with anomaly statistics.
        """
        if "is_anomaly" not in df.columns:
            raise ValueError(
                "DataFrame must have anomaly predictions. Call predict() first."
            )

        total = len(df)
        anomalies = df["is_anomaly"].sum()

        summary = {
            "total_samples": total,
            "anomalies_detected": int(anomalies),
            "anomaly_rate": anomalies / total if total > 0 else 0,
            "by_wear_class": {},
        }

        # Breakdown by wear class if available
        if "Wear_Class" in df.columns:
            for wear_class in ["Healthy", "Moderate", "Worn"]:
                mask = df["Wear_Class"] == wear_class
                class_total = mask.sum()
                class_anomalies = df[mask]["is_anomaly"].sum() if class_total > 0 else 0
                summary["by_wear_class"][wear_class] = {
                    "total": int(class_total),
                    "anomalies": int(class_anomalies),
                    "rate": class_anomalies / class_total if class_total > 0 else 0,
                }

        return summary

    def get_flagged_cycles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get details of all flagged anomaly cycles.

        Args:
            df: DataFrame with anomaly predictions.

        Returns:
            DataFrame with only the anomaly rows.
        """
        if "is_anomaly" not in df.columns:
            raise ValueError(
                "DataFrame must have anomaly predictions. Call predict() first."
            )

        return df[df["is_anomaly"]].copy()

    def save_model(self, filepath: str):
        """Save the trained model and scaler to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "random_state": self.random_state,
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.n_estimators = model_data["n_estimators"]
        self.contamination = model_data["contamination"]
        self.random_state = model_data["random_state"]
        self.is_trained = True


def create_and_train_detector(
    df: pd.DataFrame, **kwargs
) -> Tuple[ToolWearAnomalyDetector, pd.DataFrame]:
    """
    Convenience function to create, train, and run prediction in one call.

    Args:
        df: Full DataFrame with sensor data.
        **kwargs: Additional arguments for ToolWearAnomalyDetector.

    Returns:
        Tuple of (trained detector, DataFrame with predictions).
    """
    detector = ToolWearAnomalyDetector(**kwargs)
    detector.train(df)
    result_df = detector.predict(df)
    return detector, result_df
