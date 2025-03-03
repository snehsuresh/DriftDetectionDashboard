
# Model Monitoring and Drift Detection Dashboard

This project is an **end-to-end Model Monitoring Dashboard** that tracks performance degradation, detects drift (concept & covariate drift), and allows you to retrain models directly from the UI.

### Watch the Dashboard in Action

![Dashboard Demo](./demo/demo.gif)

Note: The data used in this project is synthetically generated for demonstration purposes. Therefore, the performance metrics (MSE, R²) may not reflect realistic model performance and should not be interpreted as indicators of actual predictive power. The goal of this project is to demonstrate the monitoring and drift detection workflow, not model accuracy.

We use **Streamlit** for the frontend and **Evidently AI** for drift detection. This is a complete monitoring workflow designed for real-world production pipelines.

---

## 📊 Project Structure

```
.
├── dashboard.py            # Streamlit Dashboard
├── generate_data.py         # Synthetic Data Generator
├── train_model.py           # Model Training Script
├── monitor.py                # Generates Evidently Report & Feature Importance Drift
├── reference_data.csv       # Historical (Training) Data
├── current_data.csv         # Current (Production) Data
├── plots/                    # Contains images for drift visualizations
│   ├── feature_importance_drift.png
├── demo/                     # Contains demo video
│   ├── dashboard-demo.mp4
└── README.md                 # This file
```

---

## 🛠️ Setup

### Clone this repository

```bash
git clone https://github.com/your-username/ModelMonitoringDashboard.git
cd ModelMonitoringDashboard
```

### Generate synthetic data

```bash
python generate_data.py
```

### 5️⃣ Train the initial model

```bash
python train_model.py
```

### 6️⃣ Generate Evidently report and feature importance drift

```bash
python monitor.py
```

### 7️⃣ Run the dashboard

```bash
streamlit run dashboard.py
```

---

## 📈 Key Features

### 1️⃣ Real-Time Performance Monitoring

Compare performance (MSE, R²) between reference and current datasets across a selected date range.

### 2️⃣ Evidently AI Report Embedded

Visualize the detailed **Evidently AI Drift Report** directly within the dashboard.

---

## 🧰 Features

### Performance Metrics Section
See real-time performance degradation alerts based on drift.

### Time Series Plot
Compare target vs prediction across time.

### Evidently Report Embed
See detailed drift metrics directly in the dashboard.

### Feature Importance Drift
Track how the importance of features changed between reference and current data.

---

## 📅 Date Range Filtering

Select **any date range** to analyze how performance and drift evolve over time.

---

## 🔄 One-Click Retrain

If drift or degradation is detected, you can trigger retraining **directly from the dashboard**.

---

## 🏁 Workflow Diagram

```text
+-------------------+    +-------------------+    +------------------+
| Reference Data    |    | Current Data      |    | Model (RandomForest) |
| (Historical)      |    | (Production)      |    | Trained on Reference |
+-------------------+    +-------------------+    +------------------+
          \                     /
           \                   /
            \                 /
        +---------------------+
        |    Evidently AI      |
        |  Drift Detection     |
        +---------------------+
                     |
                     v
        +---------------------+
        |    Streamlit UI      |
        | Real-Time Monitoring |
        +---------------------+
                     |
                     v
        +---------------------+
        | Retrain Trigger      |
        +---------------------+
```

---

## 📜 License

MIT License

---

## 👤 Author

**Sneh Pillai**  

---

## ⭐️ Contributions

Feel free to open issues and pull requests for improvements.

