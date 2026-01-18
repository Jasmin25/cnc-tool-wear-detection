import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import time

# --- CONFIGURATION ---
DATA_FILE = 'data/tool_wear_dataset.csv'
FEEDBACK_FILE = 'anomaly_feedback.csv'
FEATURES = [
    'CF_Feature_1', 'CF_Feature_2', 'CF_Feature_3', 'CF_Feature_4', 'CF_Feature_5',
    'Vib_Feature_1', 'Vib_Feature_2', 'Vib_Feature_3', 'Vib_Feature_4', 'Vib_Feature_5',
    'AE_Feature_1', 'AE_Feature_2', 'AE_Feature_3', 'AE_Feature_4', 'AE_Feature_5'
]

st.set_page_config(page_title="CNC Tool Wear Detector", layout="wide")

# --- BACKEND FUNCTIONS ---

@st.cache_data
def load_data():
    """Loads and preprocesses the dataset."""
    if not os.path.exists(DATA_FILE):
        st.error(f"File {DATA_FILE} not found. Please upload it.")
        return None
    
    df = pd.read_csv(DATA_FILE)
    # create a cycle index (simulating time)
    df['Cycle_Index'] = df.index 
    return df

@st.cache_resource
def train_model(df):
    """
    Trains Isolation Forest on 'Healthy' data only (Semi-supervised).
    Returns the model and the scaler.
    """
    # Filter for healthy data to learn "Normal" behavior
    healthy_data = df[df['Wear_Class'] == 'Healthy'][FEATURES]
    
    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(healthy_data)
    
    # Train Isolation Forest
    # contamination='auto' or low value because training data is pure healthy
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(X_train)
    
    return model, scaler

def get_explanation(row_scaled, feature_names):
    """
    Simple explainability: Find feature with max absolute deviation (Z-score).
    """
    # row_scaled is a numpy array. We find the index of the max absolute value.
    import numpy as np
    max_idx = np.argmax(np.abs(row_scaled))
    feature_name = feature_names[max_idx]
    score = row_scaled[0][max_idx]
    return f"{feature_name} ({score:.2f} σ)"

def save_feedback(cycle_id, prediction, user_label, comments=""):
    """Appends user feedback to a CSV file."""
    feedback_entry = {
        'Cycle_Index': cycle_id,
        'Model_Prediction': prediction,
        'User_Label': user_label, # 'True Anomaly' or 'False Alarm'
        'Comments': comments,
        'Timestamp': pd.Timestamp.now()
    }
    
    df_feedback = pd.DataFrame([feedback_entry])
    
    if not os.path.exists(FEEDBACK_FILE):
        df_feedback.to_csv(FEEDBACK_FILE, index=False)
    else:
        df_feedback.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)

# --- FRONTEND UI ---

def main():
    # 1. Header & Description
    st.title("⚙️ Live CNC Tool Wear Anomaly Detection")
    st.markdown("""
    **Scenario:** Simulating a live sensor feed from a CNC machine.
    The model was trained *only* on healthy data. It flags deviations as anomalies.
    **Task:** Monitor the feed. If an anomaly is detected, confirm if it's a **True Fault** or **False Alarm**.
    """)
    
    # 2. Initialization
    df = load_data()
    if df is None: return

    model, scaler = train_model(df)

    # Session State for Simulation Loop
    if 'stream_index' not in st.session_state:
        st.session_state.stream_index = 0
    if 'history' not in st.session_state:
        st.session_state.history = pd.DataFrame()
    if 'is_paused' not in st.session_state:
        st.session_state.is_paused = False # Pauses on anomaly
    if 'simulation_active' not in st.session_state:
        st.session_state.simulation_active = False

    # 3. Sidebar Controls
    with st.sidebar:
        st.header("Control Panel")
        speed = st.slider("Simulation Speed (sec/step)", 0.1, 2.0, 0.5)
        
        col1, col2 = st.columns(2)
        if col1.button("▶ Start"):
            st.session_state.simulation_active = True
            st.session_state.is_paused = False
        if col2.button("⏹ Stop"):
            st.session_state.simulation_active = False

        st.divider()
        st.write("📊 **Data Stats**")
        st.write(f"Total Rows: {len(df)}")
        st.write(f"Current Cycle: {st.session_state.stream_index}")

    # 4. Main Dashboard Area
    placeholder_chart = st.empty()
    placeholder_feedback = st.empty()

    # 5. Simulation Logic
    # We run the loop only if active and not paused (waiting for feedback)
    if st.session_state.simulation_active and not st.session_state.is_paused:
        
        # Get next row
        if st.session_state.stream_index < len(df):
            row = df.iloc[[st.session_state.stream_index]]
            features_raw = row[FEATURES]
            
            # Preprocess & Predict
            features_scaled = scaler.transform(features_raw)
            # anomaly_score: lower is more anomalous. Negative is usually anomaly.
            score = model.decision_function(features_scaled)[0] 
            prediction = -1 if score < 0 else 1 # -1 is Anomaly, 1 is Normal
            
            # Explainability (Simple Max Deviation)
            explanation = get_explanation(features_scaled, FEATURES)
            
            # Update History
            new_entry = {
                'Cycle_Index': row['Cycle_Index'].values[0],
                'Anomaly_Score': score,
                'Prediction': 'Anomaly' if prediction == -1 else 'Normal',
                'Color': 'red' if prediction == -1 else 'blue',
                'Top_Feature': explanation,
                'True_Label': row['Wear_Class'].values[0] # For reference
            }
            st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_entry])], ignore_index=True)
            
            # Check for Anomaly trigger
            if prediction == -1:
                st.session_state.is_paused = True # PAUSE for Feedback
                st.rerun() # Force reload to show feedback UI
            
            # Increment and continue
            st.session_state.stream_index += 1
            time.sleep(speed)
            st.rerun()
        else:
            st.success("Simulation Complete")
            st.session_state.simulation_active = False

    # 6. Render Chart (Always visible)
    if not st.session_state.history.empty:
        # We plot the Anomaly Score over time
        fig = px.scatter(
            st.session_state.history, 
            x='Cycle_Index', 
            y='Anomaly_Score',
            color='Prediction',
            color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
            hover_data=['Top_Feature', 'True_Label'],
            title="Real-time Tool Health Monitor (Anomaly Score)",
            height=400
        )
        # Add a threshold line
        fig.add_hline(y=0, line_dash="dash", line_color="green", annotation_text="Threshold")
        placeholder_chart.plotly_chart(fig, use_container_width=True)

    # 7. Render Feedback UI (Only if Paused on Anomaly)
    if st.session_state.is_paused:
        with placeholder_feedback.container():
            st.warning(f"🚨 **ANOMALY DETECTED at Cycle {st.session_state.stream_index}**")
            
            # Show details
            last_row = st.session_state.history.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Anomaly Score", f"{last_row['Anomaly_Score']:.4f}")
            c2.metric("Primary Cause", last_row['Top_Feature'])
            c3.metric("Ground Truth (Hidden)", last_row['True_Label']) # Optional, for demo
            
            st.write("### 📝 Operator Feedback Required")
            st.write("Please inspect the machine and confirm:")
            
            col_confirm, col_false = st.columns(2)
            
            if col_confirm.button("✅ Confirm: True Anomaly", type="primary"):
                save_feedback(last_row['Cycle_Index'], 'Anomaly', 'True Anomaly')
                st.success("Logged as True Anomaly.")
                time.sleep(0.5)
                st.session_state.is_paused = False # Unpause
                st.session_state.stream_index += 1 # Move to next
                st.rerun()
                
            if col_false.button("❌ Mark as False Alarm"):
                save_feedback(last_row['Cycle_Index'], 'Anomaly', 'False Alarm')
                st.info("Logged as False Alarm.")
                time.sleep(0.5)
                st.session_state.is_paused = False # Unpause
                st.session_state.stream_index += 1
                st.rerun()

if __name__ == "__main__":
    main()