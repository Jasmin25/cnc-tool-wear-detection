# CNC Tool Wear Anomaly Detection System

A machine learning-powered system for detecting anomalous tool wear patterns in CNC machining operations using multi-sensor data. Built with Streamlit and scikit-learn.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

## 🎯 Overview

This application provides:

1. **Real-time Anomaly Detection** - Uses Isolation Forest trained on healthy tool operation data to detect unusual patterns
2. **Interactive Visualizations** - Time-series charts showing wear progression and sensor signals with anomaly highlighting
3. **User Feedback Collection** - Interface for operators to confirm or reject detected anomalies
4. **Multi-sensor Analysis** - Supports Cutting Force, Vibration, and Acoustic Emission sensor features

## 📊 Dataset

The system uses the [Multi-Sensor CNC Tool Wear Dataset](https://www.kaggle.com/datasets/ziya07/multi-sensor-cnc-tool-wear-dataset) containing:

- **2,000 machining cycle samples**
- **15 sensor features** across 3 types:
  - Cutting Force (CF_Feature_1-5)
  - Vibration (Vib_Feature_1-5)
  - Acoustic Emission (AE_Feature_1-5)
- **VB_mm** - Continuous flank wear measurement
- **Wear_Class** - Categorical labels (Healthy, Moderate, Worn)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/cnc-tool-wear-detection.git
cd cnc-tool-wear-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

### Using Docker

```bash
# Build the image
docker build -t cnc-tool-wear-detection .

# Run the container
docker run -p 8501:8501 cnc-tool-wear-detection
```

## 📁 Project Structure

```
cnc-tool-wear-detection/
├── app.py                  # Demo-first Streamlit application
├── v2/
│   └── app.py              # Original analytics-heavy app
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # Data loading and preprocessing
│   └── model.py            # Anomaly detection model
├── data/
│   └── tool_wear_dataset.csv
├── notebooks/
│   └── data_exploration.ipynb  # Data analysis notebook
├── requirements/
│   └── *.pdf               # Project requirements
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
└── README.md
```

## 🔧 How It Works

### Anomaly Detection Approach

The system uses a **semi-supervised learning approach**:

1. **Training Phase**: An Isolation Forest model is trained exclusively on "Healthy" tool operation cycles to learn normal sensor patterns
2. **Detection Phase**: The trained model evaluates all cycles and flags those that deviate significantly from normal patterns as anomalies
3. **Scoring**: Each cycle receives an anomaly score - lower scores indicate more unusual patterns

### Key Features

- **Adjustable Sensitivity**: Use the sidebar slider to control how many anomalies are flagged
- **Interactive Charts**: Hover over points to see details, zoom and pan on Plotly charts
- **Feedback Loop**: Confirm or reject anomalies to build a feedback dataset for future model improvement

## 📈 Application Components

The application includes:

1. **Overview Dashboard** - Summary metrics showing total cycles, detected anomalies, and rates
2. **Wear Progression Chart** - VB_mm over machining cycles with anomaly highlighting
3. **Anomaly Score Visualization** - Scores colored by actual wear class for validation
4. **Sensor Feature Explorer** - View any sensor signal with anomaly markers
5. **Feedback Interface** - Review and label each detected anomaly

## 🛠️ Configuration

### Sidebar Options

| Setting | Description | Default |
|---------|-------------|---------|
| Anomaly Sensitivity | Expected proportion of anomalies (contamination) | 0.10 |

### Model Parameters (in code)

| Parameter | Description | Default |
|-----------|-------------|---------|
| n_estimators | Number of trees in Isolation Forest | 100 |
| contamination | Expected anomaly fraction | 0.05-0.30 |
| random_state | Random seed for reproducibility | 42 |

## 📝 Feedback Data

User feedback is stored in `anomaly_feedback.csv` with columns:
- `cycle_index` - The machining cycle ID
- `is_true_anomaly` - Boolean indicating user's verdict
- `timestamp` - When feedback was provided
- `notes` - Optional user comments

This data can be used for:
- Evaluating model performance
- Retraining with labeled examples
- Identifying common false positive patterns

## 🔮 Future Enhancements

- [ ] Automated model retraining with user feedback
- [ ] Deep learning (Autoencoder) anomaly detection
- [ ] Real-time streaming data integration
- [ ] Threshold adjustment from UI
- [ ] Export reports and alerts
- [ ] Multi-tool comparison views

## 📚 References

- Dataset: [Multi-Sensor CNC Tool Wear Dataset - Kaggle](https://www.kaggle.com/datasets/ziya07/multi-sensor-cnc-tool-wear-dataset)
- Isolation Forest: [Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)

## 📄 License

This project is for educational and demonstration purposes.

---

Built with ❤️ using [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/)
