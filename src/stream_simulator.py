"""
Stream Simulator for CNC Tool Wear Detection.
Simulates live data ingestion from the static dataset.
"""

import pandas as pd
from typing import Tuple, Optional

from .data_processing import load_data


def load_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the full dataset for streaming simulation.

    Args:
        filepath: Optional path to the dataset. Uses default if not provided.

    Returns:
        DataFrame with the full dataset.
    """
    return load_data(filepath)


def get_next_batch(
    df: pd.DataFrame, cursor: int, batch_size: int = 1
) -> Tuple[pd.DataFrame, int]:
    """
    Get the next batch of records from the dataset.

    Args:
        df: Full DataFrame.
        cursor: Current position in the dataset.
        batch_size: Number of records to retrieve.

    Returns:
        Tuple of (batch DataFrame, new cursor position).
    """
    if cursor >= len(df):
        # End of dataset - return empty batch
        return pd.DataFrame(), cursor

    end_idx = min(cursor + batch_size, len(df))
    batch = df.iloc[cursor:end_idx].copy()

    return batch, end_idx


def reset_cursor() -> int:
    """
    Reset the cursor to the beginning.

    Returns:
        Initial cursor position (0).
    """
    return 0


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get information about the dataset for display.

    Args:
        df: The dataset DataFrame.

    Returns:
        Dictionary with dataset information.
    """
    return {
        "total_cycles": len(df),
        "features": list(df.columns),
        "wear_classes": df["Wear_Class"].unique().tolist()
        if "Wear_Class" in df.columns
        else [],
        "vb_range": (float(df["VB_mm"].min()), float(df["VB_mm"].max()))
        if "VB_mm" in df.columns
        else (0, 0),
    }


def is_end_of_stream(df: pd.DataFrame, cursor: int) -> bool:
    """
    Check if we've reached the end of the dataset.

    Args:
        df: The dataset DataFrame.
        cursor: Current cursor position.

    Returns:
        True if at end of dataset.
    """
    return cursor >= len(df)


def get_progress(df: pd.DataFrame, cursor: int) -> float:
    """
    Get the streaming progress as a percentage.

    Args:
        df: The dataset DataFrame.
        cursor: Current cursor position.

    Returns:
        Progress percentage (0.0 to 100.0).
    """
    if len(df) == 0:
        return 100.0
    return (cursor / len(df)) * 100.0
