"""
CNC Tool Wear Anomaly Detection System
Unified demo-first Streamlit Application.

This application provides:
1. Live monitoring simulation with anomaly alerts
2. Simplified analysis dashboard
3. Human-in-the-loop anomaly review
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_processing import (
    AE_FEATURES,
    CF_FEATURES,
    SENSOR_FEATURES,
    VIB_FEATURES,
    load_data,
    validate_data,
)
from src.model import ToolWearAnomalyDetector


st.set_page_config(
    page_title="CNC Tool Wear Detection",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEEDBACK_FILE = "anomaly_feedback.csv"


@st.cache_data
def load_and_validate_data():
    df = load_data()
    validation = validate_data(df)
    return df, validation


@st.cache_resource
def train_model(_df: pd.DataFrame, contamination: float = 0.05):
    detector = ToolWearAnomalyDetector(contamination=contamination)
    detector.train(_df)
    return detector


def create_wear_progression_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

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
    fig = go.Figure()

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
        height=350,
    )

    return fig


def create_sensor_feature_chart(df: pd.DataFrame, feature: str) -> go.Figure:
    fig = go.Figure()

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
        height=300,
    )

    return fig


def load_feedback() -> pd.DataFrame:
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame(
        columns=["cycle_index", "is_true_anomaly", "timestamp", "notes"]
    )


def save_feedback(feedback_df: pd.DataFrame):
    feedback_df.to_csv(FEEDBACK_FILE, index=False)


def add_feedback(cycle_index: int, is_true_anomaly: bool, notes: str = ""):
    feedback_df = load_feedback()
    feedback_df = feedback_df[feedback_df["cycle_index"] != cycle_index]

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


def build_demo_feedback(anomalies: pd.DataFrame) -> pd.DataFrame:
    demo_rows = []
    for idx, (_, row) in enumerate(anomalies.head(3).iterrows(), start=1):
        demo_rows.append(
            {
                "cycle_index": int(row["cycle_index"]),
                "is_true_anomaly": idx % 2 == 1,
                "timestamp": datetime.now().isoformat(),
                "notes": "Demo feedback entry",
            }
        )
    return pd.DataFrame(demo_rows)


def compute_top_feature(df: pd.DataFrame, scaler) -> pd.DataFrame:
    scaled = scaler.transform(df[SENSOR_FEATURES].values)
    top_feature_idx = np.argmax(np.abs(scaled), axis=1)
    top_feature_scores = scaled[np.arange(len(df)), top_feature_idx]
    df = df.copy()
    df["top_feature"] = [SENSOR_FEATURES[i] for i in top_feature_idx]
    df["top_feature_score"] = top_feature_scores
    return df


def initialize_session_state():
    defaults = {
        "stream_index": 0,
        "simulation_active": False,
        "autoplay": False,
        "demo_feedback_enabled": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_simulation_panel(result_df: pd.DataFrame, demo_anomalies: pd.DataFrame):
    st.subheader("🛰️ Live Monitoring Demo")
    st.caption("Alert → Triage → Outcome")

    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        st.metric("Current Cycle", st.session_state.stream_index)
    with col2:
        status = (
            "Anomaly" if st.session_state.stream_index in demo_anomalies else "Normal"
        )
        st.metric("Current Status", status)

    with col3:
        col_start, col_next, col_reset = st.columns(3)
        if col_start.button("▶ Start", use_container_width=True):
            st.session_state.simulation_active = True
        if col_next.button("Next Cycle", use_container_width=True):
            st.session_state.stream_index = min(
                st.session_state.stream_index + 1, len(result_df) - 1
            )
        if col_reset.button("↺ Reset", use_container_width=True):
            st.session_state.stream_index = 0
            st.session_state.simulation_active = False

    if st.session_state.simulation_active and st.session_state.autoplay:
        time.sleep(0.4)
        st.session_state.stream_index = min(
            st.session_state.stream_index + 1, len(result_df) - 1
        )
        st.rerun()


def render_anomaly_review(
    anomalies_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    highlighted_cycles: List[int],
):
    st.subheader("✅ Anomaly Review")
    st.caption("Triage anomalies and confirm outcomes.")

    if anomalies_df.empty:
        st.success("No anomalies detected with current settings.")
        return

    highlight_set = set(highlighted_cycles)
    for _, row in anomalies_df.iterrows():
        cycle = int(row["cycle_index"])
        wear_class = row["Wear_Class"]
        vb_mm = row["VB_mm"]
        score = row["anomaly_score"]
        top_feature = row.get("top_feature", "N/A")
        top_score = row.get("top_feature_score", 0.0)

        is_highlight = "⭐" if cycle in highlight_set else ""
        feedback_status = ""
        if cycle in feedback_df["cycle_index"].values:
            fb = feedback_df[feedback_df["cycle_index"] == cycle].iloc[0]
            feedback_status = (
                "✅ True Anomaly" if fb["is_true_anomaly"] else "❌ False Alarm"
            )

        with st.expander(
            f"{is_highlight} Cycle {cycle} | Wear: {wear_class} | VB: {vb_mm:.4f}mm | Score: {score:.4f} {feedback_status}",
            expanded=cycle in highlight_set,
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("**Why flagged:**")
                st.write(f"{top_feature} ({top_score:.2f} σ)")

                st.markdown("**Sensor Snapshot:**")
                sensor_data = {
                    "Feature": ["CF_Feature_1", "Vib_Feature_1", "AE_Feature_1"],
                    "Value": [
                        f"{row['CF_Feature_1']:.4f}",
                        f"{row['Vib_Feature_1']:.4f}",
                        f"{row['AE_Feature_1']:.4f}",
                    ],
                }
                st.dataframe(pd.DataFrame(sensor_data), hide_index=True)

            with col2:
                st.markdown("**Outcome:**")

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


def main():
    st.title("🔧 CNC Tool Wear Anomaly Detection")
    st.markdown(
        """
        A demo-first experience that blends live monitoring with lightweight analysis.
        Use the simulation to show real-time alerts, then review anomalies with context.
        """
    )

    initialize_session_state()

    st.sidebar.header("⚙️ Demo Configuration")
    demo_mode = st.sidebar.toggle("Demo Mode", value=True)
    contamination = st.sidebar.slider(
        "Anomaly Sensitivity",
        min_value=0.01,
        max_value=0.30,
        value=0.10,
        step=0.01,
        help="Expected proportion of anomalies. Higher = more anomalies flagged.",
    )

    st.sidebar.subheader("Simulation Controls")
    st.session_state.autoplay = st.sidebar.toggle("Auto-play", value=False)
    st.session_state.demo_feedback_enabled = st.sidebar.toggle(
        "Show demo feedback", value=True
    )

    with st.spinner("Loading data..."):
        df, validation = load_and_validate_data()

    if not validation["is_valid"]:
        st.error(
            f"Data validation failed: Missing columns {validation.get('missing_columns', [])}"
        )
        return

    with st.spinner("Training anomaly detection model..."):
        detector = train_model(df, contamination=contamination)
        result_df = detector.predict(df)
        result_df = compute_top_feature(result_df, detector.scaler)

    summary = detector.get_anomaly_summary(result_df)

    st.header("📊 Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cycles", summary["total_samples"])
    with col2:
        st.metric("Anomalies Detected", summary["anomalies_detected"])
    with col3:
        st.metric("Anomaly Rate", f"{summary['anomaly_rate'] * 100:.1f}%")

    anomalies_df = detector.get_flagged_cycles(result_df).sort_values(
        "anomaly_score"
    )
    demo_highlights = anomalies_df.head(5)["cycle_index"].astype(int).tolist()

    if demo_mode:
        render_simulation_panel(result_df, set(demo_highlights))

    st.header("📈 Tool Wear Overview")
    st.plotly_chart(create_wear_progression_chart(result_df), width="stretch")

    feedback_df = load_feedback()
    if demo_mode and st.session_state.demo_feedback_enabled and feedback_df.empty:
        feedback_df = build_demo_feedback(anomalies_df)

    render_anomaly_review(anomalies_df.head(8), feedback_df, demo_highlights)

    if not feedback_df.empty:
        st.markdown("---")
        st.subheader("📋 Feedback Summary")
        total_fb = len(feedback_df)
        true_anomalies = feedback_df["is_true_anomaly"].sum()
        false_alarms = total_fb - true_anomalies

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Feedback", total_fb)
        with col2:
            st.metric("True Anomalies", int(true_anomalies))
        with col3:
            st.metric("False Alarms", int(false_alarms))

        if st.checkbox("Show all feedback"):
            st.dataframe(feedback_df, width="stretch")

    with st.expander("🔬 Advanced Analysis"):
        st.markdown(
            "Explore detailed anomaly scores and sensor signals when you have time to dive deeper."
        )

        tab1, tab2 = st.tabs(["Anomaly Scores", "Sensor Features"])
        with tab1:
            st.plotly_chart(create_anomaly_score_chart(result_df), width="stretch")

        with tab2:
            feature_groups: Dict[str, List[str]] = {
                "Cutting Force": CF_FEATURES,
                "Vibration": VIB_FEATURES,
                "Acoustic Emission": AE_FEATURES,
            }

            col1, col2 = st.columns([1, 3])
            with col1:
                group = st.selectbox("Feature Group", list(feature_groups.keys()))
                feature = st.selectbox("Feature", feature_groups[group])

            with col2:
                st.plotly_chart(
                    create_sensor_feature_chart(result_df, feature), width="stretch"
                )

    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: gray;'>
        CNC Tool Wear Anomaly Detection System | Demo-first Streamlit Experience
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
