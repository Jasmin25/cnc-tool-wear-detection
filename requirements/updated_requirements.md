# Target UX: Simple Operator Console + History/Analysis Tab

## User personas (keep it simple)

1. **Operator (primary)**

   * Wants: “Is the machine OK right now?” + a small queue of items to review + 1-click tagging.
   * Does *not* want: model internals, feature selection, calibration controls, lots of plots.

2. **Supervisor/Engineer (secondary)**

   * Wants: trend + history of events + ability to re-tag / correct + export.

## Core workflows (what the app must do)

### Workflow A — Live Monitoring (Operator)

1. System is “running” (simulated live stream).
2. New cycle records arrive every N seconds.
3. Each record is scored by the anomaly model.
4. If anomalous → show a red highlight + add to “Review Queue”.
5. Operator clicks:

   * **Confirm fault** / **False alarm**
   * (Optional) fault category + notes
6. Feedback is persisted for later retraining.

### Workflow B — History & Analysis (Supervisor/Engineer)

1. View a table of all historical events/cycles.
2. Filter (flagged only, unresolved only, date range, etc.).
3. Select an entry and tag it (even if it wasn’t flagged).
4. See simple trend charts (counts, rates).
5. Export feedback/events.

---

# Information Architecture: Two Tabs Only

Implement as **two tabs** (not a complex multi-page app):

* **Tab 1: Live Console**
* **Tab 2: History & Analysis**

This matches what you described (“operator view + another tab for analysis”).

---

# Data Model (what to store) — Keep it explicit

You already have `anomaly_feedback.csv` described in README (`cycle_index`, `is_true_anomaly`, `timestamp`, `notes`). ([GitHub][1])
For the operator-console experience, add an **events log** so the UI can show “past events” independent of the original dataset.

## A. Event record (one row per incoming cycle)

Store in: `data/events_log.csv` (or parquet if preferred)

Columns (minimum viable):

* `event_id` (uuid or `{machine_id}_{cycle_index}`)
* `machine_id` (string; can default to “CNC-01”)
* `cycle_index` (int)
* `ingest_ts` (timestamp when seen in “live feed”)
* `anomaly_score` (float)
* `is_anomaly_pred` (bool)
* `model_version` (string, e.g. “iforest_v1”)
* `review_status` (enum string: `UNREVIEWED`, `CONFIRMED_FAULT`, `FALSE_ALARM`)
* `review_ts` (timestamp or empty)
* `reviewer` (optional; can be empty for POC)
* `notes` (optional)
* plus the raw features (CF_*, Vib_*, AE_*, and optionally VB_mm if present)

Why this matters:

* Tab 2 can show *what happened*, even if you restart the app.
* Tagging “past events” becomes easy because every row is an entity.

## B. Feedback record (operator decisions)

Option 1: Keep using your existing `anomaly_feedback.csv` format (fastest). ([GitHub][1])
Option 2 (better): Make feedback append to the **events_log** by updating `review_status` fields, and also export a feedback file for ML training later.

For “fast build,” do both:

* Write feedback into `events_log.csv` (update the row)
* Append a row into `anomaly_feedback.csv` (audit trail)

---

# Backend Behavior: “Live system” simulation

You have a static dataset (2,000 cycles; 15 sensor features + VB_mm + Wear_Class) per README. ([GitHub][1])
To simulate live ingestion you have two implementation options. I’m specifying both; the developer should pick Option 1 for speed.

## Option 1 (Fastest): Simulated stream inside Streamlit using session state

* On each refresh tick, the app “reveals” the next row from the dataset.
* Pros: single process, easiest.
* Cons: not truly decoupled, but good for demo.

### Mechanics

* Load dataset into memory.
* Keep pointer: `st.session_state["cursor"]` (starts at 0).
* Every refresh, read next `batch_size` rows, score them, append to `events_log.csv`, increment cursor.
* Provide “Start / Pause / Reset” controls.

## Option 2 (More realistic): Separate `simulator.py` that appends to `data/live_feed.csv`

* Pros: looks like real streaming ingestion.
* Cons: requires two processes (simulator + app).

### Mechanics

* `simulator.py` runs a loop: every N seconds append next row to `data/live_feed.csv`.
* Streamlit reads `live_feed.csv`, detects new rows since last read, scores them, writes to `events_log.csv`.

For a take-home demo, Option 1 is totally fine.

---

# Model Behavior (keep it invisible to the operator)

Your README states the system uses Isolation Forest trained only on “Healthy” cycles, then scores cycles and flags deviations; also mentions a sensitivity slider for contamination. ([GitHub][1])

For an operator console:

* **No sensitivity slider on Live Console** (operators shouldn’t tune models).
* Use a fixed threshold/config in code.
* Keep model explanation minimal (“Detected anomaly” + “score” + “top contributing sensors”).

Implementation requirement:

* A function `score_cycles(df_cycles) -> df_with_score_and_flag`

Optional helpful explanation (lightweight):

* Show “Top 3 deviating features” using z-score vs healthy mean:

  * `z = abs((x - mu_healthy) / sigma_healthy)`
  * Show feature names with highest z (simple, fast, understandable)

---

# UX Spec: Live Console Tab

## Layout (wireframe)

**Top bar**

* Left: “CNC-01 Live Console”
* Right: Status pill + last update time

  * 🟢 Normal / 🟠 Warning / 🔴 Anomaly Active

**Row 1 (3 KPI cards)**

* Card 1: “Cycles processed (today/session)”
* Card 2: “Anomalies detected (session)”
* Card 3: “Unreviewed anomalies (queue count)”

**Row 2 (main content: 2 columns)**
**Left (70%)**

* “Live Feed (latest 20 cycles)” table

  * columns: cycle_index, ingest_ts, anomaly_score, is_anomaly_pred, review_status
  * anomalous rows highlighted
* One small trend plot: anomaly_score over last 50 cycles (simple line + red markers)

**Right (30%)**

* “Review Queue (Unreviewed)” list (most important UI)

  * Each item shows:

    * Cycle ID
    * Score
    * “Why flagged” (top features)
    * Buttons:

      * ✅ Confirm Fault
      * ❌ False Alarm
    * Optional: dropdown “Fault Type” (e.g., Vibration spike, Force anomaly, AE burst, Unknown)
    * Optional: Notes

### Interaction rules

* When an anomaly is detected:

  * It appears at top of queue
  * A toast/banner appears: “New anomaly detected at cycle 123”
* When user clicks Confirm/FALSE:

  * Update that event row: `review_status`, `review_ts`
  * Append to `anomaly_feedback.csv` (audit)
  * Remove it from queue display (or move to “Reviewed” collapsible)

### Minimal operator settings (sidebar)

* Start / Pause
* Speed (records per refresh or seconds per refresh)
* Machine selector (optional; even if only one machine exists in demo)

Do **not** show:

* Feature explorer dropdown
* Multiple plots
* Wear class coloring (that’s ground truth leakage and feels “data science-y”)

---

# UX Spec: History & Analysis Tab

Goal: show “what happened” + allow back-tagging.

## Layout

**Filters row**

* Date/time range (if ingest_ts exists; otherwise cycle index range)
* Toggle: Show anomalies only
* Toggle: Show unreviewed only
* Search box: cycle_index / event_id

**Main table**

* Events table (paginated or “show first N”)
* Columns: ingest_ts, cycle_index, anomaly_score, is_anomaly_pred, review_status, notes

**Detail panel**

* When a row is selected:

  * Show event details (key values + top features)
  * Actions:

    * Set review_status (UNREVIEWED / CONFIRMED_FAULT / FALSE_ALARM)
    * Add/edit notes
  * “Save” button

**Charts (keep minimal)**

* Anomalies per 100 cycles (bar or line)
* Pie: reviewed vs unreviewed
* Optional: confirmed faults vs false alarms

**Exports**

* “Download events_log.csv”
* “Download anomaly_feedback.csv”

---

# Step-by-step build plan for your engineer

This is written as execution steps + deliverables.

## Step 0 — Create a new “Operator Console” entry point (don’t fight existing app.py)

Repo currently has `app.py` (complex demo) and a `v1` folder (simpler version per your description). ([GitHub][1])
To avoid breaking your existing demo:

**Deliverable**

* New file: `operator_console.py` (or `app_operator.py`)
* Keep existing `app.py` untouched (that remains your “deep demo / analyst view”).

## Step 1 — Implement event storage layer

**Create** `src/event_store.py`

Functions:

* `init_events_log(path)`: create file with headers if missing
* `append_events(df_new, path)`: append rows
* `load_events(path, filters=None)`: read full log and apply filters
* `update_event(event_id, updates, path)`: update row (for review tagging)

Design choices:

* For speed, use CSV.
* For correct updates, use one of:

  * read CSV → modify → write CSV (fine for POC small data)
  * or store parquet + update (more complex)
    For POC: read/modify/write CSV is acceptable.

**Acceptance**

* You can append and update events reliably.
* No duplicate `event_id`.

## Step 2 — Implement feedback persistence

**Create** `src/feedback_store.py`

Functions:

* `append_feedback(cycle_index, verdict_bool, notes, ts, path="anomaly_feedback.csv")`
* optionally include `event_id`, `fault_type`

Match your README-described columns at minimum. ([GitHub][1])
If you extend schema, keep backward compatible by adding columns (CSV can handle it).

## Step 3 — Implement model/scoring wrapper (thin layer)

You already have `src/model.py` per README structure. ([GitHub][1])
Even without changing it, wrap it with a stable interface:

**Create** `src/scoring.py`

* `load_or_train_model(df_train) -> model`
* `score(df_features) -> np.array(scores)`
* `predict_is_anomaly(scores, threshold) -> bool array`

Also compute “top deviating features” (optional):

* Save mean/std from healthy training set
* function `top_feature_deviations(row, mu, sigma, k=3)`

**Acceptance**

* Given N new cycles, it returns scores and flags.
* Runs fast enough for UI refresh.

## Step 4 — Implement “stream simulation” engine (Option 1: session cursor)

**Create** `src/stream_simulator.py`

Functions:

* `load_dataset(path) -> df`
* `get_next_batch(df, cursor, batch_size) -> (df_batch, new_cursor)`
* (Optional) `reset_cursor()`

**Acceptance**

* Cursor increments and never resets unless user clicks Reset.
* Doesn’t crash at end of file (loops or stops gracefully).

## Step 5 — Build Streamlit UI (operator_console.py)

### 5.1 App setup

* `st.set_page_config(layout="wide")`
* Initialize session_state:

  * `cursor`
  * `is_running`
  * `machine_id`
  * `last_ingest_ts`
* Load model (cached)
* Initialize events log (create file if missing)

### 5.2 Tabs

Use:

* `tab_live, tab_history = st.tabs(["Live Console", "History & Analysis"])`

### 5.3 Live Console tab implementation

**Components to build in order:**

1. Header row + status pill
2. KPI cards (use `st.metric`)
3. Live feed table (last 20)
4. Score trend chart (last 50)
5. Review queue list:

   * `unreviewed = events[(is_anomaly_pred==True) & (review_status=="UNREVIEWED")]`
   * render newest first
   * each row has Confirm / False buttons
   * On click:

     * `update_event(...)`
     * `append_feedback(...)`
     * rerun

**Refresh behavior**

* Use Streamlit’s auto refresh:

  * Either `st_autorefresh(interval=2000, key="tick")` (if allowed by your deps)
  * Or a simple “Refresh” button for demo (less real-time but simplest)
* On each tick, if `is_running`:

  * get next batch from dataset
  * score it
  * convert to events rows (with ingest_ts)
  * append to events_log

### 5.4 History tab implementation

1. Filters row
2. Full events table (filtered)
3. Selection + details pane
4. Save review updates
5. Charts
6. Export buttons

**Selection**
Streamlit’s native dataframe selection is limited depending on version. For speed:

* Provide a dropdown/selectbox of `event_id` from filtered list
* Or use `st.dataframe` and a separate input “Enter Cycle ID to review”

Keep it dead simple.

## Step 6 — Polish to look like an operator console

This is mostly UX decisions:

* Default to full-width layout
* Remove sidebars full of ML knobs
* Use clear statuses: UNREVIEWED / CONFIRMED / FALSE ALARM
* Make the “Review Queue” visually dominant
* Limit charts to one per tab
* Use plain language:

  * “Potential tool wear anomaly detected”
  * “Confirm issue?” / “False alarm?”

**Acceptance**

* A non-technical operator can understand what to do in <30 seconds.

## Step 7 — Demo script (what you’ll show in the interview)

Provide a 60–90 second “click path”:

1. Start Live Console
2. Let it run until an anomaly appears
3. Click Confirm Fault
4. Switch to History tab and show it’s saved + trend updated
5. Export feedback CSV

---

# What to remove/simplify from the current app for “operator mode”

Your README describes multiple components (overview dashboard, wear progression, anomaly score visualization, sensor explorer, etc.). ([GitHub][1])
For an operator console, keep only:

* Live status + last N cycles table
* Anomaly queue
* Tagging
* Simple history

Everything else is “Analysis mode” (fine to keep in your existing `app.py`).

---

# Definition of Done (copy/paste acceptance criteria)

1. **Live Console**

   * New records appear automatically (or via “Next” button if auto-refresh isn’t used).
   * Anomalous records are clearly highlighted.
   * Unreviewed anomalies appear in a review queue.
   * Operator can mark each queued anomaly as Confirm Fault or False Alarm.
   * Tagging persists to disk (events log updated + feedback CSV appended).

2. **History & Analysis**

   * Shows all past events/cycles from events log.
   * Filters work (flagged only, unreviewed only, cycle range).
   * Operator/supervisor can tag any historical record.
   * Exports download correctly.

3. **Maintainability**

   * UI code does not contain model training logic directly; it calls `src/` functions.
   * File paths configurable in one place (top of app or config file).

