"""
CNC Tool Wear Detection - Operator Console
A streamlined interface for operators to monitor machine health and review anomalies.

Features:
- Live Console: Real-time monitoring with anomaly detection
- History & Analysis: Review past events and export data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.event_store import (
    init_events_log,
    append_events,
    load_events,
    update_event,
    get_unreviewed_anomalies,
    get_event_statistics,
    clear_events_log,
    DEFAULT_EVENTS_LOG_PATH,
)
from src.feedback_store import (
    append_feedback,
    load_feedback,
    get_feedback_summary,
    DEFAULT_FEEDBACK_PATH,
)
from src.scoring import ScoringEngine, load_or_train_model
from src.stream_simulator import (
    load_dataset,
    get_next_batch,
    reset_cursor,
    get_dataset_info,
    is_end_of_stream,
    get_progress,
)
from src.data_processing import SENSOR_FEATURES


# =============================================================================
# Configuration
# =============================================================================

# Page configuration
st.set_page_config(
    page_title="CNC-01 Live Console",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Constants
MACHINE_ID = "CNC-01"
MODEL_VERSION = "iforest_v1"
DEFAULT_BATCH_SIZE = 5
REFRESH_INTERVAL_MS = 2000  # 2 seconds

# Fault type options
FAULT_TYPES = [
    "Unknown",
    "Vibration spike",
    "Force anomaly",
    "AE burst",
    "Tool wear",
    "Other",
]


# =============================================================================
# Caching and Initialization
# =============================================================================


@st.cache_data
def get_training_data():
    """Load and cache the training dataset."""
    return load_dataset()


@st.cache_resource
def get_scoring_engine(_df: pd.DataFrame):
    """Train and cache the scoring engine."""
    engine = ScoringEngine()
    engine.train(_df)
    return engine


def init_session_state():
    """Initialize session state variables."""
    if "cursor" not in st.session_state:
        st.session_state.cursor = 0
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "last_ingest_ts" not in st.session_state:
        st.session_state.last_ingest_ts = None
    if "batch_size" not in st.session_state:
        st.session_state.batch_size = DEFAULT_BATCH_SIZE


# =============================================================================
# Helper Functions
# =============================================================================


def get_status_indicator(stats: dict) -> tuple:
    """
    Get status indicator based on current state.

    Returns:
        Tuple of (emoji, status_text, color)
    """
    unreviewed = stats.get("unreviewed_count", 0)

    if unreviewed == 0:
        return "🟢", "Normal", "green"
    elif unreviewed <= 3:
        return "🟠", "Warning", "orange"
    else:
        return "🔴", "Anomaly Active", "red"


def format_timestamp(ts: str) -> str:
    """Format ISO timestamp for display."""
    if not ts or pd.isna(ts):
        return "-"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%H:%M:%S")
    except:
        return ts


def create_score_trend_chart(
    events_df: pd.DataFrame, visible_points: int = 50
) -> go.Figure:
    """Create a line chart of anomaly scores over all cycles, with initial view of recent ones."""
    if events_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Anomaly Score Trend (Live)",
            xaxis_title="Cycle",
            yaxis_title="Score",
            height=250,
        )
        return fig

    # Use ALL data (not just tail) so zoom out works
    chart_df = events_df.copy()

    fig = go.Figure()

    # Ensure boolean type for filtering
    if "is_anomaly_pred" in chart_df.columns:
        chart_df["is_anomaly_pred"] = chart_df["is_anomaly_pred"].astype(bool)

    # Add score line for normal points
    normal_df = chart_df[~chart_df["is_anomaly_pred"]]
    if not normal_df.empty:
        fig.add_trace(
            go.Scatter(
                x=normal_df["cycle_index"],
                y=normal_df["anomaly_score"],
                mode="lines+markers",
                name="Normal",
                line=dict(color="#3498db", width=2),
                marker=dict(size=6, color="#3498db"),
                connectgaps=True,
            )
        )

    # Connect all points with a line
    fig.add_trace(
        go.Scatter(
            x=chart_df["cycle_index"],
            y=chart_df["anomaly_score"],
            mode="lines",
            name="Trend",
            line=dict(color="#3498db", width=1, dash="dot"),
            showlegend=False,
        )
    )

    # Highlight anomalies with X markers
    anomalies = chart_df[chart_df["is_anomaly_pred"] == True]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["cycle_index"],
                y=anomalies["anomaly_score"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#e74c3c", size=12, symbol="x", line=dict(width=2)),
            )
        )

    # Add threshold line with better styling
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#dc3545",
        line_width=2,
        opacity=0.7,
        annotation_text="⚠️ Anomaly Threshold",
        annotation_position="right",
        annotation_font_size=11,
        annotation_font_color="#dc3545",
    )

    # Add shaded anomaly zone
    fig.add_hrect(
        y0=chart_df["anomaly_score"].min() - 0.1,
        y1=0,
        fillcolor="rgba(220, 53, 69, 0.1)",
        line_width=0,
        annotation_text="Anomaly Zone",
        annotation_position="bottom left",
        annotation_font_size=10,
        annotation_font_color="#999",
    )

    # Calculate initial x-axis range to show last `visible_points` cycles
    max_cycle = chart_df["cycle_index"].max()
    min_visible_cycle = max(chart_df["cycle_index"].min(), max_cycle - visible_points)

    fig.update_layout(
        title=dict(
            text="📈 Anomaly Score Trend (Live)",
            font=dict(size=18),
            x=0,
            xanchor="left",
        ),
        xaxis_title="Cycle Index",
        yaxis_title="Anomaly Score",
        height=320,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        template="plotly_dark",  # Works well in both themes
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot area
        hovermode="x unified",
        # Set initial view range but allow zooming out to see all data
        xaxis=dict(
            range=[min_visible_cycle, max_cycle + 2],
            rangeslider=dict(
                visible=True,
                thickness=0.05,
            ),
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            gridwidth=1,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            gridwidth=1,
            zeroline=False,
        ),
    )

    return fig


def process_new_batch(df: pd.DataFrame, engine: ScoringEngine):
    """Process a new batch of cycles and add to events log."""
    if df.empty:
        return

    # Score the batch
    scored_df = engine.score_cycles(df)

    # Append to events log
    append_events(
        scored_df,
        machine_id=MACHINE_ID,
        model_version=MODEL_VERSION,
        path=DEFAULT_EVENTS_LOG_PATH,
    )

    st.session_state.last_ingest_ts = datetime.now().isoformat()


# =============================================================================
# Live Console Tab
# =============================================================================


def render_live_console(full_df: pd.DataFrame, engine: ScoringEngine):
    """Render the Live Console tab."""

    # =========================================================================
    # Process new batch FIRST if running (so UI shows fresh data)
    # =========================================================================
    should_continue_running = False
    if st.session_state.is_running and not is_end_of_stream(
        full_df, st.session_state.cursor
    ):
        # Process next batch using selected batch size
        batch, new_cursor = get_next_batch(
            full_df, st.session_state.cursor, st.session_state.batch_size
        )
        if not batch.empty:
            process_new_batch(batch, engine)
            st.session_state.cursor = new_cursor
        should_continue_running = True

    # Load current events and stats AFTER processing
    events_df = load_events(DEFAULT_EVENTS_LOG_PATH)
    stats = get_event_statistics(DEFAULT_EVENTS_LOG_PATH)
    unreviewed_queue = get_unreviewed_anomalies(DEFAULT_EVENTS_LOG_PATH)

    # =========================================================================
    # Professional Header with Status Badge
    # =========================================================================
    last_update = (
        format_timestamp(st.session_state.last_ingest_ts)
        if st.session_state.last_ingest_ts
        else "Not started"
    )
    render_header(last_update, stats)

    # =========================================================================
    # Control Bar
    # =========================================================================
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns([1, 1, 1, 1, 2])

    with ctrl_col1:
        if st.button(
            "▶️ Start" if not st.session_state.is_running else "⏸️ Pause",
            width="stretch",
            type="primary" if not st.session_state.is_running else "secondary",
        ):
            st.session_state.is_running = not st.session_state.is_running
            st.rerun()

    with ctrl_col2:
        if st.button("⏭️ Next Batch", width="stretch"):
            batch, new_cursor = get_next_batch(
                full_df, st.session_state.cursor, st.session_state.batch_size
            )
            if not batch.empty:
                process_new_batch(batch, engine)
                st.session_state.cursor = new_cursor
            st.rerun()

    with ctrl_col3:
        if st.button("🔄 Reset", width="stretch"):
            st.session_state.cursor = reset_cursor()
            st.session_state.is_running = False
            clear_events_log(DEFAULT_EVENTS_LOG_PATH)
            st.rerun()

    with ctrl_col4:
        batch_options = [1, 5, 10, 20]
        current_index = (
            batch_options.index(st.session_state.batch_size)
            if st.session_state.batch_size in batch_options
            else 1
        )
        st.session_state.batch_size = st.selectbox(
            "Batch Size",
            options=batch_options,
            index=current_index,
            label_visibility="collapsed",
        )

    with ctrl_col5:
        progress = get_progress(full_df, st.session_state.cursor)
        st.progress(
            progress / 100,
            text=f"Progress: {st.session_state.cursor}/{len(full_df)} cycles ({progress:.0f}%)",
        )

    st.divider()

    # =========================================================================
    # Live Chart at TOP (most important visual)
    # =========================================================================
    if not events_df.empty:
        fig = create_score_trend_chart(events_df)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("📊 Anomaly chart will appear here once processing starts.")

    st.divider()

    # =========================================================================
    # KPI Cards
    # =========================================================================
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

    with kpi_col1:
        st.metric(
            "Cycles Processed",
            stats["total_cycles"],
            delta=f"+{st.session_state.batch_size}"
            if st.session_state.is_running
            else None,
        )

    with kpi_col2:
        st.metric(
            "Anomalies Detected",
            stats["anomalies_detected"],
        )

    with kpi_col3:
        st.metric(
            "Unreviewed Queue",
            stats["unreviewed_count"],
            delta=None,
            delta_color="inverse",
        )

    st.divider()

    # =========================================================================
    # Main Content: Live Feed + Review Queue
    # =========================================================================
    main_col1, main_col2 = st.columns([7, 3])

    with main_col1:
        st.markdown("#### 📊 Live Feed (Latest 20 Cycles)")

        if events_df.empty:
            st.info("No cycles processed yet. Click 'Start' or 'Next Batch' to begin.")
        else:
            # Show last 20 cycles
            display_df = events_df.tail(20)[
                [
                    "cycle_index",
                    "ingest_ts",
                    "anomaly_score",
                    "is_anomaly_pred",
                    "review_status",
                ]
            ].copy()
            display_df = display_df.iloc[::-1]  # Reverse to show newest first
            display_df["ingest_ts"] = display_df["ingest_ts"].apply(format_timestamp)
            display_df["anomaly_score"] = display_df["anomaly_score"].round(4)
            display_df.columns = ["Cycle", "Time", "Score", "Anomaly", "Status"]

            # Style the dataframe
            def highlight_anomalies(row):
                if row["Anomaly"]:
                    return ["background-color: #ffcdd2"] * len(row)
                return [""] * len(row)

            styled_df = display_df.style.apply(highlight_anomalies, axis=1)
            st.dataframe(styled_df, width="stretch", height=300)

    with main_col2:
        st.markdown("#### ⚠️ Review Queue")

        if unreviewed_queue.empty:
            st.markdown(
                """
            <div class="queue-success-box">
                <span style="font-size: 2rem;">✅</span>
                <p>All anomalies reviewed!</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            # Queue header with count badge
            st.markdown(
                f"""
            <div class="queue-warning-box">
                <span>🔔 Needs Review</span>
                <span class="queue-count-badge">
                    {len(unreviewed_queue)}
                </span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Show each unreviewed anomaly with styled cards
            for idx, (_, row) in enumerate(unreviewed_queue.head(8).iterrows()):
                render_anomaly_card(row, engine, idx)

    # =========================================================================
    # Auto-refresh at END (after UI is rendered)
    # =========================================================================
    if should_continue_running:
        import time

        time.sleep(1)  # Brief pause for UI visibility
        st.rerun()


# =============================================================================
# History & Analysis Tab
# =============================================================================


def render_history_tab():
    """Render the History & Analysis tab."""

    st.markdown("### 📜 History & Analysis")

    # Load all events
    all_events = load_events(DEFAULT_EVENTS_LOG_PATH)

    if all_events.empty:
        st.info("No events recorded yet. Start the live monitoring to generate data.")
        return

    # =========================================================================
    # Filters Row
    # =========================================================================
    st.markdown("#### 🔍 Filters")

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        show_anomalies_only = st.checkbox("Anomalies Only", value=False)

    with filter_col2:
        show_unreviewed_only = st.checkbox("Unreviewed Only", value=False)

    with filter_col3:
        review_status_filter = st.selectbox(
            "Review Status",
            options=["All", "UNREVIEWED", "CONFIRMED_FAULT", "FALSE_ALARM"],
            index=0,
        )

    with filter_col4:
        if "cycle_index" in all_events.columns:
            min_cycle = int(all_events["cycle_index"].min())
            max_cycle = int(all_events["cycle_index"].max())
            cycle_range = st.slider(
                "Cycle Range",
                min_value=min_cycle,
                max_value=max_cycle,
                value=(min_cycle, max_cycle),
            )
        else:
            cycle_range = None

    # Apply filters
    filtered_events = all_events.copy()

    if show_anomalies_only:
        filtered_events = filtered_events[filtered_events["is_anomaly_pred"] == True]

    if show_unreviewed_only:
        filtered_events = filtered_events[
            filtered_events["review_status"] == "UNREVIEWED"
        ]

    if review_status_filter != "All":
        filtered_events = filtered_events[
            filtered_events["review_status"] == review_status_filter
        ]

    if cycle_range:
        filtered_events = filtered_events[
            (filtered_events["cycle_index"] >= cycle_range[0])
            & (filtered_events["cycle_index"] <= cycle_range[1])
        ]

    st.divider()

    # =========================================================================
    # Events Table
    # =========================================================================
    st.markdown(f"#### 📋 Events ({len(filtered_events)} records)")

    # Select columns to display
    display_cols = [
        "event_id",
        "cycle_index",
        "ingest_ts",
        "anomaly_score",
        "is_anomaly_pred",
        "review_status",
        "notes",
    ]
    display_cols = [c for c in display_cols if c in filtered_events.columns]

    if not filtered_events.empty:
        # Create display dataframe
        display_df = filtered_events[display_cols].copy()
        display_df["anomaly_score"] = display_df["anomaly_score"].round(4)

        st.dataframe(display_df, width="stretch", height=300)
    else:
        st.warning("No events match the current filters.")

    st.divider()

    # =========================================================================
    # Detail Panel - Edit Selected Event
    # =========================================================================
    st.markdown("#### ✏️ Review/Edit Event")

    if not filtered_events.empty:
        # Event selector
        event_options = filtered_events["event_id"].tolist()
        selected_event_id = st.selectbox(
            "Select Event to Review",
            options=event_options,
            format_func=lambda x: f"Cycle {filtered_events[filtered_events['event_id'] == x]['cycle_index'].values[0]} - {x[:20]}...",
        )

        if selected_event_id:
            event_row = filtered_events[
                filtered_events["event_id"] == selected_event_id
            ].iloc[0]

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("**Event Details**")
                st.write(f"- **Cycle Index:** {event_row['cycle_index']}")
                st.write(f"- **Ingest Time:** {event_row['ingest_ts']}")
                st.write(f"- **Anomaly Score:** {event_row['anomaly_score']:.4f}")
                st.write(f"- **Is Anomaly:** {event_row['is_anomaly_pred']}")
                st.write(f"- **Current Status:** {event_row['review_status']}")

            with detail_col2:
                st.markdown("**Update Review**")

                new_status = st.selectbox(
                    "Review Status",
                    options=["UNREVIEWED", "CONFIRMED_FAULT", "FALSE_ALARM"],
                    index=["UNREVIEWED", "CONFIRMED_FAULT", "FALSE_ALARM"].index(
                        event_row["review_status"]
                    )
                    if event_row["review_status"]
                    in ["UNREVIEWED", "CONFIRMED_FAULT", "FALSE_ALARM"]
                    else 0,
                    key="detail_status",
                )

                notes = st.text_area(
                    "Notes", value=event_row.get("notes", "") or "", key="detail_notes"
                )

                if st.button("💾 Save Changes", type="primary"):
                    update_event(
                        selected_event_id,
                        {"review_status": new_status, "notes": notes},
                        DEFAULT_EVENTS_LOG_PATH,
                    )

                    # Also append to feedback if status changed
                    if new_status != "UNREVIEWED":
                        append_feedback(
                            cycle_index=int(event_row["cycle_index"]),
                            is_true_anomaly=(new_status == "CONFIRMED_FAULT"),
                            event_id=selected_event_id,
                            notes=notes,
                            path=DEFAULT_FEEDBACK_PATH,
                        )

                    st.success("Event updated!")
                    st.rerun()

    st.divider()

    # =========================================================================
    # Charts
    # =========================================================================
    st.markdown("#### 📊 Analytics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Review status pie chart
        if not all_events.empty:
            status_counts = all_events["review_status"].value_counts()

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=status_counts.index,
                        values=status_counts.values,
                        hole=0.4,
                        marker_colors=["#f59e0b", "#22c55e", "#ef4444"],
                    )
                ]
            )
            fig.update_layout(
                title="Review Status Distribution",
                height=300,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, width="stretch")

    with chart_col2:
        # Anomalies over time
        if not all_events.empty and "cycle_index" in all_events.columns:
            # Group by cycle batches of 100
            all_events["cycle_batch"] = (all_events["cycle_index"] // 100) * 100
            batch_anomalies = (
                all_events.groupby("cycle_batch")["is_anomaly_pred"].sum().reset_index()
            )

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=batch_anomalies["cycle_batch"],
                        y=batch_anomalies["is_anomaly_pred"],
                        marker_color="#ef4444",
                    )
                ]
            )
            fig.update_layout(
                title="Anomalies per 100 Cycles",
                xaxis_title="Cycle Batch",
                yaxis_title="Anomaly Count",
                height=300,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, width="stretch")

    st.divider()

    # =========================================================================
    # Exports
    # =========================================================================
    st.markdown("#### 📥 Export Data")

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        if not all_events.empty:
            csv_events = all_events.to_csv(index=False)
            st.download_button(
                "📄 Download Events Log",
                data=csv_events,
                file_name="events_log_export.csv",
                mime="text/csv",
                width="stretch",
            )

    with export_col2:
        feedback_df = load_feedback(DEFAULT_FEEDBACK_PATH)
        if not feedback_df.empty:
            csv_feedback = feedback_df.to_csv(index=False)
            st.download_button(
                "📄 Download Feedback",
                data=csv_feedback,
                file_name="anomaly_feedback_export.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.button("📄 Download Feedback", disabled=True, width="stretch")
            st.caption("No feedback recorded yet")


# =============================================================================
# Custom CSS Styles
# =============================================================================

CUSTOM_CSS = """
<style>
    /* ========================================
       Theme-aware CSS Variables
       ======================================== */
    :root {
        --card-bg: rgba(248, 249, 250, 0.95);
        --card-bg-hover: rgba(233, 236, 239, 1);
        --card-border: rgba(0, 102, 204, 0.8);
        --card-shadow: rgba(0, 0, 0, 0.1);
        --card-shadow-hover: rgba(0, 0, 0, 0.15);
        --text-primary: #1a1a2e;
        --text-secondary: #4a4a5a;
        --text-muted: #6c757d;
        --surface-bg: rgba(240, 242, 246, 0.9);
        --border-color: rgba(200, 200, 210, 0.5);
        --success-color: #22c55e;
        --success-bg: rgba(34, 197, 94, 0.12);
        --warning-color: #eab308;
        --warning-bg: rgba(234, 179, 8, 0.12);
        --danger-color: #ef4444;
        --danger-bg: rgba(239, 68, 68, 0.12);
        --accent-color: #3b82f6;
    }
    
    /* Dark mode overrides - detect Streamlit's dark theme */
    @media (prefers-color-scheme: dark) {
        :root {
            --card-bg: rgba(30, 32, 40, 0.95);
            --card-bg-hover: rgba(40, 44, 55, 1);
            --card-border: rgba(59, 130, 246, 0.7);
            --card-shadow: rgba(0, 0, 0, 0.3);
            --card-shadow-hover: rgba(0, 0, 0, 0.4);
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --surface-bg: rgba(45, 50, 60, 0.9);
            --border-color: rgba(100, 110, 130, 0.4);
            --success-color: #4ade80;
            --success-bg: rgba(74, 222, 128, 0.15);
            --warning-color: #facc15;
            --warning-bg: rgba(250, 204, 21, 0.15);
            --danger-color: #f87171;
            --danger-bg: rgba(248, 113, 113, 0.15);
            --accent-color: #60a5fa;
        }
    }
    
    /* Streamlit dark theme detection via data attribute */
    [data-testid="stAppViewContainer"][data-theme="dark"],
    .stApp[data-theme="dark"],
    [data-theme="dark"] {
        --card-bg: rgba(30, 32, 40, 0.95);
        --card-bg-hover: rgba(40, 44, 55, 1);
        --card-border: rgba(59, 130, 246, 0.7);
        --card-shadow: rgba(0, 0, 0, 0.3);
        --card-shadow-hover: rgba(0, 0, 0, 0.4);
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --surface-bg: rgba(45, 50, 60, 0.9);
        --border-color: rgba(100, 110, 130, 0.4);
        --success-color: #4ade80;
        --success-bg: rgba(74, 222, 128, 0.15);
        --warning-color: #facc15;
        --warning-bg: rgba(250, 204, 21, 0.15);
        --danger-color: #f87171;
        --danger-bg: rgba(248, 113, 113, 0.15);
        --accent-color: #60a5fa;
    }

    /* KPI Card Styling - Theme Aware */
    [data-testid="stMetric"] {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid var(--accent-color);
        box-shadow: 0 2px 4px var(--card-shadow);
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        background: var(--card-bg-hover);
        box-shadow: 0 4px 8px var(--card-shadow-hover);
    }
    [data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }
    
    /* Tab Styling - Theme Aware */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--surface-bg);
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        font-weight: 500;
        color: var(--text-secondary);
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-color) !important;
        color: white !important;
    }
    
    /* Anomaly Card Styling - Theme Aware */
    .anomaly-card {
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
        background: var(--card-bg);
        color: var(--text-primary);
    }
    .anomaly-card:hover {
        box-shadow: 0 4px 12px var(--card-shadow-hover);
        background: var(--card-bg-hover);
    }
    .anomaly-card-critical {
        border-left: 5px solid var(--danger-color);
        background: linear-gradient(90deg, var(--danger-bg) 0%, var(--card-bg) 100%);
    }
    .anomaly-card-warning {
        border-left: 5px solid var(--warning-color);
        background: linear-gradient(90deg, var(--warning-bg) 0%, var(--card-bg) 100%);
    }
    .anomaly-card .card-title {
        color: var(--text-primary);
    }
    .anomaly-card .card-subtitle {
        color: var(--text-secondary);
    }
    
    /* Status Badge - Theme Aware */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .status-healthy {
        background-color: var(--success-bg);
        border: 2px solid var(--success-color);
        color: var(--success-color);
    }
    .status-warning {
        background-color: var(--warning-bg);
        border: 2px solid var(--warning-color);
        color: var(--warning-color);
    }
    .status-critical {
        background-color: var(--danger-bg);
        border: 2px solid var(--danger-color);
        color: var(--danger-color);
    }
    
    /* Header Styling - Works in both themes */
    .console-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.25rem 1.5rem;
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #1e3a5f 100%);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px var(--card-shadow);
    }
    .console-header-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .console-header-icon {
        font-size: 2.5rem;
    }
    .console-header-title {
        margin: 0;
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .console-header-subtitle {
        margin: 0;
        color: #a0c4e8;
        font-size: 0.9rem;
    }
    .console-header-right {
        text-align: right;
        color: #ffffff;
    }
    .console-header-time {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    /* Queue Box Styling - Theme Aware */
    .queue-success-box {
        background: linear-gradient(135deg, var(--success-bg) 0%, var(--card-bg) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid var(--success-color);
    }
    .queue-success-box p {
        color: var(--success-color);
        margin: 0.5rem 0 0 0;
        font-weight: 600;
    }
    .queue-warning-box {
        background: linear-gradient(135deg, var(--warning-bg) 0%, var(--card-bg) 100%);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .queue-warning-box span {
        color: var(--warning-color);
        font-weight: 600;
    }
    
    /* Progress bar enhancement */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent-color) 0%, #00a8cc 100%);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Footer Styling - Theme Aware */
    .console-footer {
        margin-top: 2rem;
        padding: 1rem;
        background: var(--surface-bg);
        border-radius: 8px;
        text-align: center;
        border-top: 2px solid var(--border-color);
    }
    .console-footer span {
        color: var(--text-muted);
        font-size: 0.85rem;
    }
    .console-footer strong {
        color: var(--text-primary);
    }
    
    /* Version Badge */
    .version-badge {
        background: var(--accent-color);
        color: white !important;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Severity Badges */
    .severity-badge-high {
        background: var(--danger-color);
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-left: 0.5rem;
        font-weight: 600;
    }
    .severity-badge-medium {
        background: var(--warning-color);
        color: #1a1a2e;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-left: 0.5rem;
        font-weight: 600;
    }
    
    /* Score Badge */
    .score-badge {
        background: var(--surface-bg);
        color: var(--text-primary);
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-family: monospace;
        border: 1px solid var(--border-color);
    }
    
    /* Queue Count Badge */
    .queue-count-badge {
        background: var(--danger-color);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.9rem;
    }
</style>
"""


# =============================================================================
# UI Helper Components
# =============================================================================


def render_header(last_update: str, stats: dict):
    """Render the professional header with branding."""
    # Calculate machine health status
    total = max(stats.get("total_cycles", 0), 1)
    anomalies = stats.get("anomalies_detected", 0)
    anomaly_rate = (anomalies / total) * 100

    if anomaly_rate < 5:
        status_class, status_text, status_icon = "status-healthy", "HEALTHY", "✅"
    elif anomaly_rate < 15:
        status_class, status_text, status_icon = "status-warning", "WARNING", "⚠️"
    else:
        status_class, status_text, status_icon = "status-critical", "CRITICAL", "🚨"

    st.markdown(
        f"""
    <div class="console-header">
        <div class="console-header-left">
            <span class="console-header-icon">🏭</span>
            <div>
                <h1 class="console-header-title">{MACHINE_ID} Live Console</h1>
                <p class="console-header-subtitle">Tool Wear Anomaly Detection System</p>
            </div>
        </div>
        <div class="console-header-right">
            <span style="font-size: 0.85rem; opacity: 0.8;">Last Update</span><br/>
            <span class="console-header-time">{last_update}</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Status badge row
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown(
            f"""
        <div style="text-align: right;">
            <span class="status-badge {status_class}">
                <span>{status_icon}</span>
                <span>{status_text}</span>
                <span style="font-weight: normal; opacity: 0.8;">({anomaly_rate:.1f}% anomaly rate)</span>
            </span>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_anomaly_card(row, engine: "ScoringEngine", idx: int):
    """Render a styled anomaly review card."""
    score = row["anomaly_score"]
    severity_class = "anomaly-card-critical" if score < -0.3 else "anomaly-card-warning"
    severity_label = "HIGH" if score < -0.3 else "MEDIUM"
    badge_class = "severity-badge-high" if score < -0.3 else "severity-badge-medium"

    # Get top deviating features
    feature_text = ""
    try:
        top_features = engine.get_top_deviating_features(row, k=2)
        if top_features:
            feature_text = ", ".join([f"{f[0]}" for f in top_features])
    except:
        pass

    st.markdown(
        f"""
    <div class="anomaly-card {severity_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div>
                <span class="card-title" style="font-size: 1.1rem; font-weight: 700;">Cycle #{int(row["cycle_index"])}</span>
                <span class="{badge_class}">
                    {severity_label}
                </span>
            </div>
            <span class="score-badge">
                Score: {score:.3f}
            </span>
        </div>
        {f'<div class="card-subtitle" style="font-size: 0.85rem;">📈 Elevated: {feature_text}</div>' if feature_text else ""}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Action buttons (must be outside markdown for interactivity)
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button(
            "✅ Confirm Fault",
            key=f"confirm_{row['event_id']}",
            use_container_width=True,
        ):
            update_event(
                row["event_id"],
                {"review_status": "CONFIRMED_FAULT"},
                DEFAULT_EVENTS_LOG_PATH,
            )
            append_feedback(
                cycle_index=int(row["cycle_index"]),
                is_true_anomaly=True,
                event_id=row["event_id"],
                fault_type="Unknown",
                path=DEFAULT_FEEDBACK_PATH,
            )
            st.rerun()

    with btn_col2:
        if st.button(
            "❌ False Alarm", key=f"false_{row['event_id']}", use_container_width=True
        ):
            update_event(
                row["event_id"],
                {"review_status": "FALSE_ALARM"},
                DEFAULT_EVENTS_LOG_PATH,
            )
            append_feedback(
                cycle_index=int(row["cycle_index"]),
                is_true_anomaly=False,
                event_id=row["event_id"],
                path=DEFAULT_FEEDBACK_PATH,
            )
            st.rerun()


# =============================================================================
# Main Application
# =============================================================================


def main():
    """Main application entry point."""

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize
    init_session_state()
    init_events_log(DEFAULT_EVENTS_LOG_PATH)

    # Load data and model
    full_df = get_training_data()
    engine = get_scoring_engine(full_df)

    # Create tabs
    tab_live, tab_history = st.tabs(["🖥️ Live Console", "📜 History & Analysis"])

    with tab_live:
        render_live_console(full_df, engine)

    with tab_history:
        render_history_tab()

    # Footer with enhanced styling
    st.markdown(
        """
    <div class="console-footer">
        <span>
            🏭 <strong>CNC Tool Wear Detection</strong> - Operator Console | 
            <span class="version-badge">v3.2</span>
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
