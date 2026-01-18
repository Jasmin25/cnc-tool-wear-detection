"""
Scoring Module for CNC Tool Wear Detection.
Provides a clean interface for model scoring operations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .data_processing import SENSOR_FEATURES, get_healthy_data


class ScoringEngine:
    """
    Wrapper for anomaly detection model scoring.
    Provides a stable interface for the operator console.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.healthy_mean: Optional[np.ndarray] = None
        self.healthy_std: Optional[np.ndarray] = None
        self.is_trained = False
        self.model_version = "iforest_v1"

    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train the model on healthy data.

        Args:
            df: DataFrame with sensor data and Wear_Class column.

        Returns:
            Dictionary with training statistics.
        """
        # Filter to healthy data for training
        healthy_df = get_healthy_data(df)

        if len(healthy_df) == 0:
            raise ValueError("No healthy data available for training")

        # Extract and scale features
        X = healthy_df[SENSOR_FEATURES].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Store healthy statistics for feature deviation analysis
        self.healthy_mean = np.mean(X_scaled, axis=0)
        self.healthy_std = np.std(X_scaled, axis=0)
        # Avoid division by zero
        self.healthy_std[self.healthy_std == 0] = 1.0

        # Train Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.is_trained = True

        return {
            "training_samples": len(healthy_df),
            "n_features": len(SENSOR_FEATURES),
            "contamination": self.contamination,
            "model_version": self.model_version,
        }

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Score cycles for anomalies.

        Args:
            df: DataFrame with sensor features.

        Returns:
            Array of anomaly scores (lower = more anomalous).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")

        X = df[SENSOR_FEATURES].values
        X_scaled = self.scaler.transform(X)

        return self.model.decision_function(X_scaled)

    def predict_is_anomaly(
        self, scores: np.ndarray, threshold: float = 0.0
    ) -> np.ndarray:
        """
        Predict whether each cycle is an anomaly based on scores.

        Args:
            scores: Array of anomaly scores.
            threshold: Score threshold (default 0.0 for Isolation Forest).

        Returns:
            Boolean array indicating anomalies.
        """
        return scores < threshold

    def score_cycles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score cycles and return DataFrame with predictions.

        Args:
            df: DataFrame with sensor features.

        Returns:
            DataFrame with added anomaly_score and is_anomaly columns.
        """
        result_df = df.copy()

        scores = self.score(df)
        result_df["anomaly_score"] = scores
        result_df["is_anomaly"] = self.predict_is_anomaly(scores)

        return result_df

    def get_top_deviating_features(
        self, row: pd.Series, k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get the top k features that deviate most from healthy baseline.

        Args:
            row: Series containing sensor feature values.
            k: Number of top features to return.

        Returns:
            List of (feature_name, z_score) tuples, sorted by deviation.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before computing deviations")

        # Get feature values and scale them
        X = row[SENSOR_FEATURES].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)[0]

        # Compute z-scores vs healthy baseline
        z_scores = np.abs((X_scaled - self.healthy_mean) / self.healthy_std)

        # Get top k features
        top_indices = np.argsort(z_scores)[-k:][::-1]

        return [(SENSOR_FEATURES[i], float(z_scores[i])) for i in top_indices]

    def explain_anomaly(self, row: pd.Series) -> str:
        """
        Generate a human-readable explanation for why a cycle was flagged.

        Args:
            row: Series containing the cycle data.

        Returns:
            String explanation of the anomaly.
        """
        top_features = self.get_top_deviating_features(row, k=3)

        if not top_features:
            return "Unusual sensor pattern detected"

        explanations = []
        for feature, z_score in top_features:
            if z_score > 2:
                explanations.append(f"{feature} ({z_score:.1f}σ)")

        if explanations:
            return "High deviation in: " + ", ".join(explanations)
        else:
            return "Slight deviations across multiple sensors"


def load_or_train_model(df: pd.DataFrame) -> ScoringEngine:
    """
    Load an existing model or train a new one.

    Args:
        df: Training DataFrame with sensor data.

    Returns:
        Trained ScoringEngine instance.
    """
    engine = ScoringEngine()
    engine.train(df)
    return engine
