import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import subprocess

# Configure the page
st.set_page_config(page_title="Model Monitoring Dashboard", layout="wide")
st.title("Model Monitoring and Drift Detection Dashboard")

@st.cache_data
def load_data(filename, parse_dates=['date']):
    return pd.read_csv(filename, parse_dates=parse_dates)

def add_predictions(df, model):
    # If there is no data, skip predictions.
    if df.empty:
        return df
    X = df[['feature1', 'feature2']]
    df = df.copy()  # Avoid modifying the original DataFrame
    df['prediction'] = model.predict(X)
    return df

# Load full datasets and model once
ref = load_data('reference_data.csv')
curr = load_data('current_data.csv')
model = joblib.load('model.pkl')

# Sidebar: Date Range Filter
st.sidebar.header("Data Filters")
min_date = min(ref['date'].min(), curr['date'].min())
max_date = max(ref['date'].max(), curr['date'].max())

# The default is provided as a list of dates.
date_range = st.sidebar.date_input(
    "Select date range", 
    [min_date, max_date], 
    min_value=min_date, 
    max_value=max_date
)

# Ensure we always have two dates.
if not isinstance(date_range, (list, tuple)):
    date_range = [date_range]
if len(date_range) < 2:
    start_date = end_date = date_range[0]
else:
    start_date, end_date = date_range

# Convert to pandas datetime (in case they are python date objects)
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter the data based on the selected date range
ref_filtered = ref[(ref['date'] >= start_date) & (ref['date'] <= end_date)]
curr_filtered = curr[(curr['date'] >= start_date) & (curr['date'] <= end_date)]

# Check if filtered data is empty.
if ref_filtered.empty or curr_filtered.empty:
    st.warning("No data available in the selected date range.")
else:
    # Apply predictions AFTER filtering
    ref_filtered = add_predictions(ref_filtered, model)
    curr_filtered = add_predictions(curr_filtered, model)

    # --- Performance Metrics Section ---
    st.header("Performance Metrics")
    mse_ref = mean_squared_error(ref_filtered['target'], ref_filtered['prediction'])
    mse_curr = mean_squared_error(curr_filtered['target'], curr_filtered['prediction'])
    r2_ref = r2_score(ref_filtered['target'], ref_filtered['prediction'])
    r2_curr = r2_score(curr_filtered['target'], curr_filtered['prediction'])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reference MSE", f"{mse_ref:.2f}")
    col2.metric("Current MSE", f"{mse_curr:.2f}")
    col3.metric("Reference R²", f"{r2_ref:.2f}")
    col4.metric("Current R²", f"{r2_curr:.2f}")

    if mse_curr > 1.5 * mse_ref:
        st.error("ALERT: Significant performance degradation detected!")
    else:
        st.success("Model performance is stable.")

    # --- Time Series Plot ---
    st.subheader("Time Series: Target vs Prediction (Current Data)")
    fig, ax = plt.subplots()
    ax.plot(curr_filtered['date'], curr_filtered['target'], label='Target')
    ax.plot(curr_filtered['date'], curr_filtered['prediction'], label='Prediction', linestyle='--')
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Evidently AI Report Section ---
    st.header("Evidently AI Report")
    try:
        with open("evidently_report.html", "r", encoding="utf-8") as f:
            report_html = f.read()
        st.components.v1.html(report_html, height=800, scrolling=True)
    except Exception as e:
        st.error("Evidently report not found. Please run monitor.py first.")

    # --- Feature Importance Drift Section ---
    st.header("Feature Importance Drift")
    try:
        st.image("feature_importance_drift.png", caption="Feature Importance: Reference vs Current")
    except Exception as e:
        st.error("Feature importance plot not found. Please run monitor.py first.")

# --- Optional: Retrain Model Trigger ---
if st.sidebar.button("Retrain Model"):
    with st.spinner("Retraining the model..."):
        result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
        st.sidebar.success("Model retrained successfully!")
        st.sidebar.text(result.stdout)
