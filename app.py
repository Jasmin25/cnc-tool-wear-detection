"""
CNC Tool Wear Demo Application
Combines a live monitoring simulation with a simplified analysis view.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys

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
    page_title="CNC Tool Wear Demo",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEEDBACK_FILE = "anomaly_feedback.csv"


@st.cache_data
def load_and_validate_data() -> tuple[pd.DataFrame, dict]:
    df = load_data()
    validation = validate_data(df)
    return df, validation


@st.cache_resource
def train_model(_df: pd.DataFrame, contamination: float) -> ToolWearAnomalyDetector:
    detector = ToolWearAnomalyDetector(contamination=contamination)
    detector.train(_df)
    return detector


@st.cache_data
def get_demo_sequence(result_df: pd.DataFrame) -> List[int]:
    anomalies = (
        result_df[result_df["is_anomaly"]]
        .sort_values("anomaly_score")
        .head(3)
    )
    normals = (
        result_df[~result_df["is_anomaly"]]
        .sort_values("anomaly_score", ascending=False)
        .head(3)
    )

    sequence = []
    for normal_idx in normals["cycle_index"].tolist():
        sequence.append(int(normal_idx))

    for anomaly_idx in anomalies["cycle_index"].tolist():
        sequence.append(int(anomaly_idx))

    return sequence


def get_feature_explanation(
    row: pd.Series, detector: ToolWearAnomalyDetector
) -> str:
    if detector.scaler is None:
        return ""

    values = row[SENSOR_FEATURES].values.astype(float)
    scaled = (values - detector.scaler.mean_) / detector.scaler.scale_
    max_idx = int(np.argmax(np.abs(scaled)))
    feature = SENSOR_FEATURES[max_idx]
    score = scaled[max_idx]
    return f"{feature} ({score:.2f}σ)"


# Feedback management

def load_feedback() -> pd.DataFrame:
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame(
        columns=["cycle_index", "is_true_anomaly", "timestamp", "notes"]
    )


def save_feedback(feedback_df: pd.DataFrame) -> None:
    feedback_df.to_csv(FEEDBACK_FILE, index=False)


def add_feedback(cycle_index: int, is_true_anomaly: bool, notes: str = "") -> None:
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


# Visualization helpers

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
        height=420,
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
                hovertemplate=(
                    "Cycle: %{x}<br>Score: %{y:.4f}<br>Class: "
                    f"{wear_class}<extra></extra>"
                ),
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
        height=360,
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
        height=320,
    )

    return fig


def create_class_distribution_chart(df: pd.DataFrame) -> go.Figure:
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


def ensure_demo_state(sequence: List[int]) -> None:
    if "demo_index" not in st.session_state:
        st.session_state.demo_index = 0
    if "demo_history" not in st.session_state:
        st.session_state.demo_history = pd.DataFrame()
    if "demo_feedback" not in st.session_state:
        st.session_state.demo_feedback = {}
    if "demo_sequence" not in st.session_state:
        st.session_state.demo_sequence = sequence


def step_demo(result_df: pd.DataFrame) -> Optional[pd.Series]:
    sequence = st.session_state.demo_sequence
    idx = st.session_state.demo_index

    if idx >= len(sequence):
        return None

    cycle = sequence[idx]
    row = result_df[result_df["cycle_index"] == cycle].iloc[0]
    st.session_state.demo_history = pd.concat(
        [st.session_state.demo_history, pd.DataFrame([row])], ignore_index=True
    )
    st.session_state.demo_index += 1
    return row


def render_demo_mode(result_df: pd.DataFrame, detector: ToolWearAnomalyDetector) -> None:
    st.subheader("🚦 Demo Mode: Alert → Triage → Outcome")

    sequence = get_demo_sequence(result_df)
    ensure_demo_state(sequence)

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Controls")
        if st.button("Next Cycle", type="primary"):
            step_demo(result_df)
        if st.button("Reset Demo"):
            st.session_state.demo_index = 0
            st.session_state.demo_history = pd.DataFrame()
            st.session_state.demo_feedback = {}

        st.markdown("### Demo Progress")
        st.metric("Cycles Reviewed", len(st.session_state.demo_history))
        st.metric("Remaining", max(len(sequence) - st.session_state.demo_index, 0))

    with col1:
        st.markdown("### Live Monitor")
        if st.session_state.demo_history.empty:
            st.info("Click 'Next Cycle' to simulate the live feed.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.demo_history["cycle_index"],
                    y=st.session_state.demo_history["anomaly_score"],
                    mode="lines+markers",
                    marker=dict(size=8),
                    line=dict(color="#1f77b4"),
                )
            )
            fig.add_hline(y=0, line_dash="dash", line_color="#e74c3c")
            fig.update_layout(
                height=300,
                xaxis_title="Cycle Index",
                yaxis_title="Anomaly Score",
            )
            st.plotly_chart(fig, width="stretch")

    if not st.session_state.demo_history.empty:
        latest = st.session_state.demo_history.iloc[-1]
        explanation = get_feature_explanation(latest, detector)
        is_anomaly = bool(latest["is_anomaly"])

        st.markdown("---")
        st.markdown("### Step 1: Alert")
        if is_anomaly:
            st.error(
                f"Anomaly detected at cycle {int(latest['cycle_index'])} (score {latest['anomaly_score']:.4f})."
            )
        else:
            st.success(
                f"Cycle {int(latest['cycle_index'])} appears normal (score {latest['anomaly_score']:.4f})."
            )

        st.markdown("### Step 2: Triage")
        c1, c2, c3 = st.columns(3)
        c1.metric("VB_mm", f"{latest['VB_mm']:.4f}")
        c2.metric("Wear Class", latest["Wear_Class"])
        c3.metric("Top Deviation", explanation or "N/A")

        st.markdown("### Step 3: Outcome")
        col_confirm, col_false = st.columns(2)
        if col_confirm.button("✅ Confirm: True Anomaly"):
            st.session_state.demo_feedback[int(latest["cycle_index"])] = True
            st.success("Logged as True Anomaly (demo mode).")
        if col_false.button("❌ Mark as False Alarm"):
            st.session_state.demo_feedback[int(latest["cycle_index"])] = False
            st.info("Logged as False Alarm (demo mode).")

        st.markdown("#### Demo Feedback Summary")
        if st.session_state.demo_feedback:
            true_count = sum(st.session_state.demo_feedback.values())
            false_count = len(st.session_state.demo_feedback) - true_count
            st.write(
                f"True Anomalies: {true_count} | False Alarms: {false_count}"
            )
        else:
            st.caption("No feedback submitted yet.")


def render_analysis_mode(
    result_df: pd.DataFrame,
    detector: ToolWearAnomalyDetector,
    contamination: float,
) -> None:
    st.subheader("📊 Analysis Overview")

    summary = detector.get_anomaly_summary(result_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cycles", summary["total_samples"])
    with col2:
        st.metric("Anomalies Detected", summary["anomalies_detected"])
    with col3:
        st.metric("Anomaly Rate", f"{summary['anomaly_rate'] * 100:.1f}%")

    st.markdown("### Wear Progression")
    st.plotly_chart(create_wear_progression_chart(result_df), width="stretch")

    st.markdown("### Anomaly Review")
    anomalies = (
        result_df[result_df["is_anomaly"]]
        .sort_values("anomaly_score")
        .head(8)
    )
    feedback_df = load_feedback()

    for _, row in anomalies.iterrows():
        cycle = int(row["cycle_index"])
        explanation = get_feature_explanation(row, detector)
        has_feedback = cycle in feedback_df["cycle_index"].values
        status = ""
        if has_feedback:
            fb = feedback_df[feedback_df["cycle_index"] == cycle].iloc[0]
            status = "✅ True" if fb["is_true_anomaly"] else "❌ False Alarm"

        with st.expander(
            f"Cycle {cycle} | Score {row['anomaly_score']:.4f} | {status}"
        ):
            st.write(
                f"**Top deviation:** {explanation or 'N/A'} | **Wear:** {row['Wear_Class']} | **VB_mm:** {row['VB_mm']:.4f}"
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("Sensor Snapshot")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Feature": ["CF_Feature_1", "Vib_Feature_1", "AE_Feature_1"],
                            "Value": [
                                f"{row['CF_Feature_1']:.4f}",
                                f"{row['Vib_Feature_1']:.4f}",
                                f"{row['AE_Feature_1']:.4f}",
                            ],
                        }
                    ),
                    hide_index=True,
                )
            with col_b:
                choice = st.radio(
                    "Is this a true anomaly?",
                    options=["Select...", "Yes - True Anomaly", "No - False Alarm"],
                    key=f"feedback_{cycle}",
                )
                notes = st.text_input("Notes", key=f"notes_{cycle}")
                if st.button("Save Feedback", key=f"save_{cycle}"):
                    if choice == "Select...":
                        st.warning("Please select an option")
                    else:
                        add_feedback(cycle, choice == "Yes - True Anomaly", notes)
                        st.success("Feedback saved!")
                        st.rerun()

    with st.expander("Advanced Analysis"):
        st.caption(
            f"Model contamination setting: {contamination:.2f} (expected anomaly rate)."
        )

        st.plotly_chart(create_anomaly_score_chart(result_df), width="stretch")

        feature_groups = {
            "Cutting Force": CF_FEATURES,
            "Vibration": VIB_FEATURES,
            "Acoustic Emission": AE_FEATURES,
        }
        c1, c2 = st.columns([1, 3])
        with c1:
            group = st.selectbox("Feature Group", list(feature_groups.keys()))
            feature = st.selectbox("Feature", feature_groups[group])
        with c2:
            st.plotly_chart(create_sensor_feature_chart(result_df, feature), width="stretch")

        st.plotly_chart(create_class_distribution_chart(result_df), width="stretch")


# Main app

def main() -> None:
    st.title("🔧 CNC Tool Wear Demo")
    st.markdown(
        """
        A lightweight demo that combines a live monitoring simulation with a simplified analysis view.
        Use **Demo Mode** for a guided narrative, or switch to **Analysis Mode** for deeper exploration.
        """
    )

    st.sidebar.header("⚙️ Configuration")
    mode = st.sidebar.radio(
        "Mode",
        options=["Demo Mode", "Analysis Mode"],
        help="Demo Mode walks through Alert → Triage → Outcome with a curated sequence.",
    )

    contamination = st.sidebar.slider(
        "Anomaly Sensitivity",
        min_value=0.01,
        max_value=0.30,
        value=0.10,
        step=0.01,
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

    if mode == "Demo Mode":
        render_demo_mode(result_df, detector)
    else:
        render_analysis_mode(result_df, detector, contamination)

    st.markdown("---")
    st.caption("Demo-ready CNC Tool Wear Anomaly Detection | Streamlit + Scikit-learn")


if __name__ == "__main__":
    main()
