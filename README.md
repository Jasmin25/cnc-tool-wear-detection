# CNC Tool Wear Detection

Real-time anomaly detection for CNC machining operations using Isolation Forest.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run operator_console.py
```

Open http://localhost:8501

### Docker

```bash
docker build -t cnc-tool-wear .
docker run -p 8501:8501 cnc-tool-wear
```

## Features

- **Live Console** — Stream machining cycles and flag anomalies in real-time
- **Operator Feedback** — Confirm or dismiss detected anomalies with fault classification
- **History & Export** — Review past events and export data for analysis

## Project Structure

```
├── operator_console.py     # Streamlit application
├── src/
│   ├── data_processing.py  # Feature definitions and data loading
│   ├── scoring.py          # Isolation Forest scoring engine
│   ├── stream_simulator.py # Batch streaming from dataset
│   ├── event_store.py      # Event persistence (CSV)
│   └── feedback_store.py   # Operator feedback storage
├── data/
│   ├── tool_wear_dataset.csv   # Source dataset
│   └── events_log.csv          # Generated events
└── anomaly_feedback.csv    # Operator feedback log
```

## How It Works

1. **Training** — Isolation Forest trains on "Healthy" cycles to learn normal sensor patterns
2. **Scoring** — Each incoming cycle is scored; deviations from normal are flagged as anomalies
3. **Review** — Operators review flagged cycles and provide feedback (confirm/dismiss + fault type)

### Sensor Features

15 features across 3 sensor types:
- Cutting Force (CF_Feature_1–5)
- Vibration (Vib_Feature_1–5)
- Acoustic Emission (AE_Feature_1–5)

## Requirements

- Python 3.8+
- See [requirements.txt](requirements.txt) for dependencies