### **Project Title: Live Tool Wear Anomaly Detection & Feedback Loop**

#### **1. Objective**

Create an interactive web application that simulates a live production line, monitoring sensor data to detect tool wear anomalies in real-time. The app allows operators (the user) to inspect anomalies with explainable AI (XAI) and provide feedback to improve the system.

#### **2. Tech Stack (The "Least Effort" Stack)**

* **Language:** Python 3.9+
* **Frontend & App Framework:** **Streamlit**
* *Why:* It turns Python scripts into shareable web apps in minutes. No HTML/CSS/JS knowledge required. Built-in support for live updates, caching, and widgets.


* **Machine Learning:** **Scikit-learn**
* *Algorithm:* **Isolation Forest** (Unsupervised) or **One-Class SVM**.
* *Why:* Fast, effective for high-dimensional anomaly detection, and standard industry practice.


* **Explainability (XAI):** **SHAP (SHapley Additive exPlanations)**
* *Why:* Provides state-of-the-art "local" explanations (e.g., "Vibration Sensor 1 was too high") which is perfect for hover-over tooltips.


* **Visualization:** **Plotly Express**
* *Why:* Interactive charts (zoom, pan, hover) that integrate natively with Streamlit.


* **Data Handling:** **Pandas**

---

#### **3. Functional Requirements**

**A. Data Ingestion & Simulation**

* **Offline Training:** The system must train an Isolation Forest model on the historical dataset (`tool_wear_dataset.csv`) upon startup.
* **Live Simulation:** The app must simulate a "live stream" by feeding rows from the dataset one by one (or in small batches) every few seconds.

**B. Detection Engine**

* **Anomaly Scoring:** For every new data point, calculate an "Anomaly Score" (0 to 1).
* **Thresholding:** Automatically flag data points as "Normal," "Warning," or "Critical (Anomaly)" based on the score.

**C. User Interface (The "Demo" View)**

* **Status Dashboard:** A prominent "Traffic Light" indicator showing the current status of the tool (Green/Yellow/Red).
* **Live Chart:** A dynamic line chart or scatter plot showing the Anomaly Score over time. New points appear in real-time.
* **Explainability Panel:**
* When an anomaly is detected, show a "Why?" section.
* Display a simple SHAP bar chart showing which sensors (e.g., `Vib_Feature_1` or `CF_Feature_5`) contributed most to the alarm.


* **Feedback Loop:**
* Provide buttons: `[Confirm Anomaly]` and `[False Alarm]`.
* Store this feedback in a temporary session log to showcase "Human-in-the-loop" capability.



---

#### **4. Application Flow (Step-by-Step for Demo)**

1. **Sidebar Setup:** User adjusts simulation speed (e.g., 1 sec/update) and Anomaly Sensitivity (threshold slider).
2. **Start Monitoring:** User clicks "Start".
3. **Visuals:**
* Data points start plotting on the chart.
* Most are blue (Normal).
* Suddenly, points turn red (Anomaly).


4. **Interaction:**
* The app pauses (or user pauses it).
* User selects the red point.
* **Explainability:** A plot appears showing "Vibration Feature 3 is +20% higher than normal".
* **Action:** User clicks "Confirm Anomaly". System logs "Label saved".



---

#### **5. Implementation Plan**

**Phase 1: Backend (Data & Model)**

* Load `tool_wear_dataset.csv`.
* Preprocess: Scale features (StandardScaler).
* Train `IsolationForest` on the data.
* Calculate SHAP values for the dataset (pre-calculate for speed in demo).

**Phase 2: Frontend (Streamlit)**

* Set up the layout: `st.columns` for KPI metrics (Current Wear, Status).
* Create the main `st.plotly_chart` that updates in a loop.
* Add `st.button` for feedback.
