"""
Feedback Storage Layer for CNC Tool Wear Detection.
Handles persistence of operator feedback for model retraining.
"""

import pandas as pd
import os
from datetime import datetime
from typing import Optional

# Default paths
DEFAULT_FEEDBACK_PATH = "anomaly_feedback.csv"

# Feedback schema
FEEDBACK_COLUMNS = [
    "event_id",
    "cycle_index",
    "is_true_anomaly",
    "fault_type",
    "timestamp",
    "reviewer",
    "notes",
]


def init_feedback_file(path: str = DEFAULT_FEEDBACK_PATH) -> None:
    """
    Initialize the feedback file with headers if it doesn't exist.

    Args:
        path: Path to the feedback CSV file.
    """
    if not os.path.exists(path):
        df = pd.DataFrame(columns=FEEDBACK_COLUMNS)
        df.to_csv(path, index=False)


def append_feedback(
    cycle_index: int,
    is_true_anomaly: bool,
    notes: str = "",
    event_id: str = "",
    fault_type: str = "",
    reviewer: str = "",
    path: str = DEFAULT_FEEDBACK_PATH,
) -> pd.DataFrame:
    """
    Append a new feedback entry to the feedback file.

    Args:
        cycle_index: The cycle index being reviewed.
        is_true_anomaly: True if confirmed fault, False if false alarm.
        notes: Optional notes from the reviewer.
        event_id: Optional event ID for cross-reference.
        fault_type: Optional fault category (e.g., "Vibration spike", "Force anomaly").
        reviewer: Optional reviewer identifier.
        path: Path to the feedback CSV file.

    Returns:
        DataFrame with the new feedback entry.
    """
    init_feedback_file(path)

    # Create new feedback entry
    new_entry = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "cycle_index": cycle_index,
                "is_true_anomaly": is_true_anomaly,
                "fault_type": fault_type,
                "timestamp": datetime.now().isoformat(),
                "reviewer": reviewer,
                "notes": notes,
            }
        ]
    )

    # Load existing and append
    existing_df = pd.read_csv(path)
    combined_df = pd.concat([existing_df, new_entry], ignore_index=True)
    combined_df.to_csv(path, index=False)

    return new_entry


def load_feedback(path: str = DEFAULT_FEEDBACK_PATH) -> pd.DataFrame:
    """
    Load all feedback entries.

    Args:
        path: Path to the feedback CSV file.

    Returns:
        DataFrame with all feedback entries.
    """
    init_feedback_file(path)
    return pd.read_csv(path)


def get_feedback_summary(path: str = DEFAULT_FEEDBACK_PATH) -> dict:
    """
    Get summary statistics for feedback.

    Args:
        path: Path to the feedback CSV file.

    Returns:
        Dictionary with feedback statistics.
    """
    df = load_feedback(path)

    if df.empty:
        return {
            "total_feedback": 0,
            "confirmed_faults": 0,
            "false_alarms": 0,
            "fault_types": {},
        }

    return {
        "total_feedback": len(df),
        "confirmed_faults": int(df["is_true_anomaly"].sum()),
        "false_alarms": int((~df["is_true_anomaly"]).sum()),
        "fault_types": df["fault_type"].value_counts().to_dict()
        if "fault_type" in df.columns
        else {},
    }


def export_feedback(output_path: str, source_path: str = DEFAULT_FEEDBACK_PATH) -> bool:
    """
    Export feedback to a specified file path.

    Args:
        output_path: Destination path for the export.
        source_path: Source feedback file path.

    Returns:
        True if export was successful.
    """
    df = load_feedback(source_path)
    df.to_csv(output_path, index=False)
    return True
