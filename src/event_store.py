"""
Event Storage Layer for CNC Tool Wear Detection.
Handles persistence and retrieval of machining cycle events.
"""

import pandas as pd
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid

# Event log schema
EVENT_COLUMNS = [
    "event_id",
    "machine_id",
    "cycle_index",
    "ingest_ts",
    "anomaly_score",
    "is_anomaly_pred",
    "model_version",
    "review_status",
    "review_ts",
    "reviewer",
    "notes",
    # Raw sensor features will be added dynamically
]

# Review status enum values
REVIEW_STATUSES = ["UNREVIEWED", "CONFIRMED_FAULT", "FALSE_ALARM"]

# Default paths
DEFAULT_EVENTS_LOG_PATH = "data/events_log.csv"


def init_events_log(path: str = DEFAULT_EVENTS_LOG_PATH) -> None:
    """
    Initialize the events log file with headers if it doesn't exist.

    Args:
        path: Path to the events log CSV file.
    """
    if not os.path.exists(path):
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Create empty DataFrame with schema
        df = pd.DataFrame(columns=EVENT_COLUMNS)
        df.to_csv(path, index=False)


def generate_event_id(machine_id: str, cycle_index: int) -> str:
    """Generate a unique event ID."""
    return f"{machine_id}_{cycle_index}_{uuid.uuid4().hex[:8]}"


def append_events(
    df_new: pd.DataFrame,
    machine_id: str = "CNC-01",
    model_version: str = "iforest_v1",
    path: str = DEFAULT_EVENTS_LOG_PATH,
) -> pd.DataFrame:
    """
    Append new cycle events to the events log.

    Args:
        df_new: DataFrame with new cycle data including anomaly predictions.
        machine_id: Machine identifier.
        model_version: Version of the model used for scoring.
        path: Path to the events log CSV file.

    Returns:
        DataFrame of the appended events.
    """
    init_events_log(path)

    # Build event records
    events = []
    ingest_ts = datetime.now().isoformat()

    for _, row in df_new.iterrows():
        event = {
            "event_id": generate_event_id(machine_id, int(row["cycle_index"])),
            "machine_id": machine_id,
            "cycle_index": int(row["cycle_index"]),
            "ingest_ts": ingest_ts,
            "anomaly_score": float(row.get("anomaly_score", 0)),
            "is_anomaly_pred": bool(row.get("is_anomaly", False)),
            "model_version": model_version,
            "review_status": "UNREVIEWED",
            "review_ts": "",
            "reviewer": "",
            "notes": "",
        }

        # Add raw sensor features
        for col in row.index:
            if col.startswith(("CF_", "Vib_", "AE_")) or col in ["VB_mm", "Wear_Class"]:
                event[col] = row[col]

        events.append(event)

    events_df = pd.DataFrame(events)

    # Append to existing log
    existing_df = pd.read_csv(path)
    combined_df = pd.concat([existing_df, events_df], ignore_index=True)
    combined_df.to_csv(path, index=False)

    return events_df


def load_events(
    path: str = DEFAULT_EVENTS_LOG_PATH, filters: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Load events from the events log with optional filtering.

    Args:
        path: Path to the events log CSV file.
        filters: Dictionary of filter conditions:
            - anomalies_only: bool - Only show anomalous events
            - unreviewed_only: bool - Only show unreviewed events
            - cycle_range: tuple (min, max) - Filter by cycle index range
            - date_range: tuple (start, end) - Filter by ingest timestamp
            - machine_id: str - Filter by machine

    Returns:
        Filtered DataFrame of events.
    """
    init_events_log(path)

    df = pd.read_csv(path)

    if df.empty:
        return df

    # Ensure boolean column is properly typed (CSV reads as string)
    if "is_anomaly_pred" in df.columns:
        df["is_anomaly_pred"] = df["is_anomaly_pred"].astype(str).str.lower() == "true"

    if filters is None:
        return df

    # Apply filters
    if filters.get("anomalies_only", False):
        df = df[df["is_anomaly_pred"] == True]

    if filters.get("unreviewed_only", False):
        df = df[df["review_status"] == "UNREVIEWED"]

    if "cycle_range" in filters:
        min_cycle, max_cycle = filters["cycle_range"]
        df = df[(df["cycle_index"] >= min_cycle) & (df["cycle_index"] <= max_cycle)]

    if "machine_id" in filters:
        df = df[df["machine_id"] == filters["machine_id"]]

    if "review_status" in filters:
        df = df[df["review_status"] == filters["review_status"]]

    return df


def update_event(
    event_id: str, updates: Dict[str, Any], path: str = DEFAULT_EVENTS_LOG_PATH
) -> bool:
    """
    Update an event record in the events log.

    Args:
        event_id: The unique event ID to update.
        updates: Dictionary of field updates.
        path: Path to the events log CSV file.

    Returns:
        True if update was successful, False otherwise.
    """
    init_events_log(path)

    df = pd.read_csv(path)

    if df.empty or event_id not in df["event_id"].values:
        return False

    # Update the matching row
    for field, value in updates.items():
        if field in df.columns:
            df.loc[df["event_id"] == event_id, field] = value

    # Add review timestamp if review_status is being updated
    if "review_status" in updates:
        df.loc[df["event_id"] == event_id, "review_ts"] = datetime.now().isoformat()

    df.to_csv(path, index=False)
    return True


def get_event_by_id(
    event_id: str, path: str = DEFAULT_EVENTS_LOG_PATH
) -> Optional[pd.Series]:
    """
    Get a single event by its ID.

    Args:
        event_id: The unique event ID.
        path: Path to the events log CSV file.

    Returns:
        Series with event data or None if not found.
    """
    df = load_events(path)

    if df.empty:
        return None

    matches = df[df["event_id"] == event_id]
    if len(matches) == 0:
        return None

    return matches.iloc[0]


def get_unreviewed_anomalies(path: str = DEFAULT_EVENTS_LOG_PATH) -> pd.DataFrame:
    """
    Get all unreviewed anomalous events (the review queue).

    Args:
        path: Path to the events log CSV file.

    Returns:
        DataFrame of unreviewed anomalies, sorted newest first.
    """
    return load_events(
        path, filters={"anomalies_only": True, "unreviewed_only": True}
    ).sort_values("ingest_ts", ascending=False)


def get_event_statistics(path: str = DEFAULT_EVENTS_LOG_PATH) -> Dict[str, Any]:
    """
    Get summary statistics for events.

    Args:
        path: Path to the events log CSV file.

    Returns:
        Dictionary with event statistics.
    """
    df = load_events(path)

    if df.empty:
        return {
            "total_cycles": 0,
            "anomalies_detected": 0,
            "unreviewed_count": 0,
            "confirmed_faults": 0,
            "false_alarms": 0,
        }

    return {
        "total_cycles": len(df),
        "anomalies_detected": int(df["is_anomaly_pred"].sum()),
        "unreviewed_count": int((df["review_status"] == "UNREVIEWED").sum()),
        "confirmed_faults": int((df["review_status"] == "CONFIRMED_FAULT").sum()),
        "false_alarms": int((df["review_status"] == "FALSE_ALARM").sum()),
    }


def clear_events_log(path: str = DEFAULT_EVENTS_LOG_PATH) -> None:
    """
    Clear all events from the log (for reset functionality).

    Args:
        path: Path to the events log CSV file.
    """
    if os.path.exists(path):
        os.remove(path)
    init_events_log(path)
