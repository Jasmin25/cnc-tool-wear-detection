"""
CNC Tool Wear Anomaly Detection System
Streamlit Web Application

This application provides:
1. Visualization of sensor data over machining cycles
2. Anomaly detection using Isolation Forest
3. Interactive feedback collection for detected anomalies
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime
from typing import Optional

# Add src to path for imports
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_processing import (
    load_data,
    validate_data,
    SENSOR_FEATURES,
    CF_FEATURES,
    VIB_FEATURES,
    AE_FEATURES,
)
from src.model import ToolWearAnomalyDetector, create_and_train_detector


# Page configuration
st.set_page_config(
    page_title="CNC Tool Wear Detection",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Feedback file path
FEEDBACK_FILE = "anomaly_feedback.csv"


# ============================================================================
# Caching Functions
# ============================================================================


@st.cache_data
def load_and_validate_data():
    """Load and validate the dataset."""
    df = load_data()
    validation = validate_data(df)
    return df, validation


@st.cache_resource
def train_model(_df: pd.DataFrame, contamination: float = 0.05):
    """Train the anomaly detection model."""
    detector = ToolWearAnomalyDetector(contamination=contamination)
    detector.train(_df)
    return detector


# ============================================================================
# Visualization Functions
# ============================================================================


def create_wear_progression_chart(df: pd.DataFrame) -> go.Figure:
    """Create a time-series chart of wear progression with anomaly highlighting."""

    fig = go.Figure()

    # Normal points
    normal_df = df[~df["is_anomaly"]]
    fig.add_trace(
        go.Scatter(
            x=normal_df["cycle_index"],
            y=normal_df["VB_mm"],
            mode="markers",
            name="Normal",
            marker=dict(color="#2ecc71", size=6, opacity=0.7),
            hovertemplate="Cycle: %{x}<br>VB_mm: %{y:.4f}<br>Status: Normal<extra></extra>",
        )
    )

    # Anomaly points
    anomaly_df = df[df["is_anomaly"]]
    fig.add_trace(
        go.Scatter(
            x=anomaly_df["cycle_index"],
            y=anomaly_df["VB_mm"],
            mode="markers",
            name="Anomaly",
            marker=dict(color="#e74c3c", size=10, symbol="x", opacity=0.9),
            hovertemplate="Cycle: %{x}<br>VB_mm: %{y:.4f}<br>Status: ANOMALY<extra></extra>",
        )
    )

    fig.update_layout(
        title="Tool Wear Progression Over Machining Cycles",
        xaxis_title="Cycle Index",
        yaxis_title="Flank Wear VB (mm)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="closest",
        height=450,
    )

    return fig


def create_anomaly_score_chart(df: pd.DataFrame) -> go.Figure:
    """Create a chart showing anomaly scores over cycles."""

    fig = go.Figure()

    # Color by wear class
    color_map = {"Healthy": "#2ecc71", "Moderate": "#f39c12", "Worn": "#e74c3c"}

    for wear_class in ["Healthy", "Moderate", "Worn"]:
        class_df = df[df["Wear_Class"] == wear_class]
        fig.add_trace(
            go.Scatter(
                x=class_df["cycle_index"],
                y=class_df["anomaly_score"],
                mode="markers",
                name=wear_class,
                marker=dict(color=color_map[wear_class], size=5, opacity=0.6),
                hovertemplate=f"Cycle: %{{x}}<br>Score: %{{y:.4f}}<br>Class: {wear_class}<extra></extra>",
            )
        )

    # Add threshold line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold",
        annotation_position="right",
    )

    fig.update_layout(
        title="Anomaly Scores by Wear Class",
        xaxis_title="Cycle Index",
        yaxis_title="Anomaly Score (lower = more anomalous)",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=400,
    )

    return fig


def create_sensor_feature_chart(df: pd.DataFrame, feature: str) -> go.Figure:
    """Create a time-series chart for a specific sensor feature."""

    fig = go.Figure()

    # Normal points
    normal_df = df[~df["is_anomaly"]]
    fig.add_trace(
        go.Scatter(
            x=normal_df["cycle_index"],
            y=normal_df[feature],
            mode="markers",
            name="Normal",
            marker=dict(color="#3498db", size=4, opacity=0.5),
        )
    )

    # Anomaly points
    anomaly_df = df[df["is_anomaly"]]
    fig.add_trace(
        go.Scatter(
            x=anomaly_df["cycle_index"],
            y=anomaly_df[feature],
            mode="markers",
            name="Anomaly",
            marker=dict(color="#e74c3c", size=8, symbol="x"),
        )
    )

    fig.update_layout(
        title=f"{feature} Over Machining Cycles",
        xaxis_title="Cycle Index",
        yaxis_title=feature,
        height=350,
    )

    return fig


def create_class_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create a pie chart of wear class distribution."""

    counts = df["Wear_Class"].value_counts()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts.index,
                values=counts.values,
                marker_colors=["#2ecc71", "#f39c12", "#e74c3c"],
                hole=0.4,
            )
        ]
    )

    fig.update_layout(title="Wear Class Distribution", height=300)

    return fig


# ============================================================================
# Feedback Management Functions
# ============================================================================


def load_feedback() -> pd.DataFrame:
    """Load existing feedback from file."""
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame(
        columns=["cycle_index", "is_true_anomaly", "timestamp", "notes"]
    )


def save_feedback(feedback_df: pd.DataFrame):
    """Save feedback to file."""
    feedback_df.to_csv(FEEDBACK_FILE, index=False)


def add_feedback(cycle_index: int, is_true_anomaly: bool, notes: str = ""):
    """Add new feedback entry."""
    feedback_df = load_feedback()

    # Remove existing feedback for this cycle if any
    feedback_df = feedback_df[feedback_df["cycle_index"] != cycle_index]

    # Add new entry
    new_entry = pd.DataFrame(
        [
            {
                "cycle_index": cycle_index,
                "is_true_anomaly": is_true_anomaly,
                "timestamp": datetime.now().isoformat(),
                "notes": notes,
            }
        ]
    )

    feedback_df = pd.concat([feedback_df, new_entry], ignore_index=True)
    save_feedback(feedback_df)
    return feedback_df


# ============================================================================
# Main Application
# ============================================================================


def main():
    # Header
    st.title("🔧 CNC Tool Wear Anomaly Detection")
    st.markdown("""
    This system uses machine learning to detect anomalous tool wear patterns in CNC machining operations.
    Review the detected anomalies and provide feedback to help improve the model.
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    contamination = st.sidebar.slider(
        "Anomaly Sensitivity",
        min_value=0.01,
        max_value=0.30,
        value=0.10,
        step=0.01,
        help="Expected proportion of anomalies. Higher = more anomalies flagged.",
    )

    # Load data
    with st.spinner("Loading data..."):
        df, validation = load_and_validate_data()

    if not validation["is_valid"]:
        st.error(
            f"Data validation failed: Missing columns {validation.get('missing_columns', [])}"
        )
        return

    # Train model
    with st.spinner("Training anomaly detection model..."):
        detector = train_model(df, contamination=contamination)
        result_df = detector.predict(df)
        summary = detector.get_anomaly_summary(result_df)

    # Display summary metrics
    st.header("📊 Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Cycles", summary["total_samples"])

    with col2:
        st.metric("Anomalies Detected", summary["anomalies_detected"])

    with col3:
        st.metric("Anomaly Rate", f"{summary['anomaly_rate'] * 100:.1f}%")

    with col4:
        healthy_rate = summary["by_wear_class"]["Healthy"]["rate"] * 100
        st.metric(
            "False Alarm Rate",
            f"{healthy_rate:.1f}%",
            help="% of Healthy cycles flagged as anomalies",
        )

    # Main visualization
    st.header("📈 Wear Progression & Anomaly Detection")

    tab1, tab2, tab3 = st.tabs(
        ["Wear Progression", "Anomaly Scores", "Sensor Features"]
    )

    with tab1:
        fig = create_wear_progression_chart(result_df)
        st.plotly_chart(fig, width="stretch")

        st.markdown("""
        **Chart Interpretation:**
        - 🟢 **Green points**: Normal machining cycles
        - ❌ **Red X markers**: Detected anomalies
        - The Y-axis shows flank wear (VB_mm) - higher values indicate more worn tools
        """)

    with tab2:
        fig = create_anomaly_score_chart(result_df)
        st.plotly_chart(fig, width="stretch")

        st.markdown("""
        **Anomaly Score Interpretation:**
        - Points below the red dashed line (score < 0) are flagged as anomalies
        - Lower scores indicate more unusual sensor patterns
        - Colors show the actual wear class for validation
        """)

    with tab3:
        # Feature selector
        feature_groups = {
            "Cutting Force": CF_FEATURES,
            "Vibration": VIB_FEATURES,
            "Acoustic Emission": AE_FEATURES,
        }

        col1, col2 = st.columns([1, 3])
        with col1:
            group = st.selectbox("Feature Group", list(feature_groups.keys()))
            feature = st.selectbox("Feature", feature_groups[group])

        with col2:
            fig = create_sensor_feature_chart(result_df, feature)
            st.plotly_chart(fig, width="stretch")

    # Detection results by class
    st.header("🎯 Detection Results by Wear Class")

    col1, col2 = st.columns([1, 2])

    with col1:
        fig = create_class_distribution_chart(result_df)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("### Detection Performance")

        for wear_class, data in summary["by_wear_class"].items():
            color = {"Healthy": "🟢", "Moderate": "🟡", "Worn": "🔴"}[wear_class]
            st.markdown(f"""
            **{color} {wear_class}**: {data["anomalies"]} / {data["total"]} flagged ({data["rate"] * 100:.1f}%)
            """)

        st.info("""
        **Interpretation:**
        - Low Healthy detection rate = fewer false alarms ✓
        - High Worn detection rate = better at catching real issues ✓
        - Moderate cases may go either way (borderline wear)
        """)

    # Anomaly feedback section
    st.header("✅ Anomaly Review & Feedback")
    st.markdown("""
    Review the detected anomalies below. Mark whether each detection is a **true anomaly** 
    (real tool wear issue) or a **false alarm** to help improve future detection.
    """)

    # Get flagged anomalies
    anomalies_df = detector.get_flagged_cycles(result_df)

    if len(anomalies_df) == 0:
        st.success("No anomalies detected with current settings!")
    else:
        # Load existing feedback
        existing_feedback = load_feedback()

        # Display anomalies in expandable sections
        st.markdown(
            f"**{len(anomalies_df)} anomalies detected** - expand each to review and provide feedback:"
        )

        # Pagination for large number of anomalies
        items_per_page = 10
        total_pages = (len(anomalies_df) - 1) // items_per_page + 1

        if total_pages > 1:
            page = st.selectbox(
                "Page",
                range(1, total_pages + 1),
                format_func=lambda x: f"Page {x} of {total_pages}",
            )
        else:
            page = 1

        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(anomalies_df))

        page_anomalies = anomalies_df.iloc[start_idx:end_idx]

        for idx, row in page_anomalies.iterrows():
            cycle = int(row["cycle_index"])
            wear_class = row["Wear_Class"]
            vb_mm = row["VB_mm"]
            score = row["anomaly_score"]

            # Check if already has feedback
            has_feedback = cycle in existing_feedback["cycle_index"].values
            feedback_status = ""
            if has_feedback:
                fb = existing_feedback[existing_feedback["cycle_index"] == cycle].iloc[
                    0
                ]
                feedback_status = (
                    "✅ True Anomaly" if fb["is_true_anomaly"] else "❌ False Alarm"
                )

            with st.expander(
                f"Cycle {cycle} | Wear: {wear_class} | VB: {vb_mm:.4f}mm | Score: {score:.4f} {feedback_status}"
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**Sensor Readings:**")
                    # Show key sensor values
                    sensor_data = {"Feature": [], "Value": []}
                    for feat in ["CF_Feature_1", "Vib_Feature_1", "AE_Feature_1"]:
                        sensor_data["Feature"].append(feat)
                        sensor_data["Value"].append(f"{row[feat]:.4f}")

                    st.dataframe(pd.DataFrame(sensor_data), hide_index=True)

                with col2:
                    st.markdown("**Your Feedback:**")

                    feedback_key = f"feedback_{cycle}"
                    is_true_anomaly = st.radio(
                        "Is this a true anomaly?",
                        options=["Select...", "Yes - True Anomaly", "No - False Alarm"],
                        key=feedback_key,
                        horizontal=True,
                    )

                    notes = st.text_input("Notes (optional)", key=f"notes_{cycle}")

                    if st.button("Save Feedback", key=f"save_{cycle}"):
                        if is_true_anomaly == "Select...":
                            st.warning("Please select an option")
                        else:
                            is_true = is_true_anomaly == "Yes - True Anomaly"
                            add_feedback(cycle, is_true, notes)
                            st.success("Feedback saved!")
                            st.rerun()

        # Feedback summary
        if len(existing_feedback) > 0:
            st.markdown("---")
            st.subheader("📋 Feedback Summary")

            total_fb = len(existing_feedback)
            true_anomalies = existing_feedback["is_true_anomaly"].sum()
            false_alarms = total_fb - true_anomalies

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Feedback", total_fb)
            with col2:
                st.metric("True Anomalies", int(true_anomalies))
            with col3:
                st.metric("False Alarms", int(false_alarms))

            if st.checkbox("Show all feedback"):
                st.dataframe(existing_feedback, width="stretch")

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: gray;'>
        CNC Tool Wear Anomaly Detection System | Built with Streamlit & Scikit-learn
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
